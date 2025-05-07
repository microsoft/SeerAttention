import torch
import triton
import triton.language as tl
import argparse

import math
import time
from einops import rearrange, einsum
from .utils import num_splits_heuristic


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4]\
        for num_stages in [1, 2, 3, 4, 7]
    ],
    key=['BLOCK_H', 'BLOCK_N', 'BLOCK_D'],
)
@triton.jit
def _split_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    cache_seqlens_ptr,
    max_seqlen,
    o_partial_ptr,
    metadata_ptr, #[b, h, 2, split] [lse, mi]
    sm_scale,
    num_splits,
    gqa_group_size,
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_b, stride_k_s, stride_k_h, stride_k_d,
    stride_v_b, stride_v_s, stride_v_h, stride_v_d,
    stride_o_b, stride_o_h, stride_o_split, stride_o_d,
    stride_meta_b, stride_meta_h, stride_meta_2, stride_meta_split,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx_kv = tl.program_id(1)
    split_idx = tl.program_id(2)


    head_idx_q = head_idx_kv * gqa_group_size
    offs_h = tl.arange(0, BLOCK_H)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_H], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

    cache_seqlens = tl.load(cache_seqlens_ptr + batch_idx)
    cache_leftpad = max_seqlen - cache_seqlens

    leftpad_block_offset = cache_leftpad // BLOCK_N
    num_blocks = (max_seqlen + BLOCK_N - 1) // BLOCK_N - leftpad_block_offset

    blocks_per_split = num_blocks // num_splits
    remaining_blocks = num_blocks % num_splits
    if split_idx < remaining_blocks:
        loop_range = blocks_per_split + 1
    else:
        loop_range = blocks_per_split

    q_ptr += batch_idx * stride_q_b + head_idx_q * stride_q_h
    k_cache_ptr += batch_idx * stride_k_b + head_idx_kv * stride_k_h + offs_n[None, :] * stride_k_s + offs_d[:, None] * stride_k_d
    v_cache_ptr += batch_idx * stride_v_b + head_idx_kv * stride_v_h + offs_n[:, None] * stride_v_s + offs_d[None, :] * stride_v_d

    q = tl.load(q_ptr + offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d, mask=offs_h[:, None] < gqa_group_size)
    start = blocks_per_split * split_idx + tl.minimum(split_idx, remaining_blocks) + leftpad_block_offset
    for block_idx in range(loop_range):
        cur_block_idx = (start + block_idx)
        start_n = cur_block_idx * BLOCK_N
        k_ptr = k_cache_ptr + start_n * stride_k_s 
        v_ptr = v_cache_ptr + start_n * stride_v_s

        k_mask = (start_n + offs_n[None, :] < max_seqlen) & (start_n + offs_n[None, :] >= cache_leftpad)
        v_mask = (start_n + offs_n[:, None] < max_seqlen) & (start_n + offs_n[:, None] >= cache_leftpad)
        k = tl.load(k_ptr, mask=k_mask, other=0.0)
        v = tl.load(v_ptr, mask=v_mask, other=0.0)
        
        qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk = qk * sm_scale
        qk = tl.where(start_n + offs_n[None, :] < max_seqlen, qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.exp(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        p = p.to(v.type.element_ty)
        acc += tl.dot(p, v)
        m_i = m_ij


    lse = m_i + tl.math.log(l_i)
    # m_i += tl.math.log(l_i)
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    acc = acc.to(o_partial_ptr.dtype.element_ty) 

    metadata_ptr += batch_idx * stride_meta_b + (head_idx_q + offs_h) * stride_meta_h + split_idx * stride_meta_split
    tl.store(metadata_ptr, lse, mask=offs_h < gqa_group_size)
    tl.store(metadata_ptr + stride_meta_2, m_i, mask=offs_h < gqa_group_size)
    
    o_partial_ptr += batch_idx * stride_o_b + (head_idx_q + offs_h[:, None]) * stride_o_h + split_idx * stride_o_split + offs_d[None, :] * stride_o_d
    tl.store(o_partial_ptr, acc, mask=offs_h[:, None] < gqa_group_size)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4]\
        for num_stages in [1, 2, 3, 4, 7]
    ],
    key=['BLOCK_D'],
)
@triton.jit
def _merge_kernel(
    o_partial_ptr,
    metadata_ptr,
    o_ptr,
    meta_stride_b, meta_stride_h, meta_stride_2, meta_stride_split,
    o_partial_stride_b, o_partial_stride_h, o_partial_stride_split, o_partial_stride_d,
    o_stride_b, o_stride_h, o_stride_d,
    BLOCK_D: tl.constexpr,
    num_splits: tl.constexpr,
    num_splits_pow2: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    offs_splits = tl.arange(0, num_splits_pow2)
    offs_d = tl.arange(0, BLOCK_D)

    lse_offsets = metadata_ptr + batch_idx * meta_stride_b + head_idx * meta_stride_h + offs_splits * meta_stride_split
    lse = tl.load(lse_offsets, mask=offs_splits < num_splits, other=float("-inf"))
    m_i = tl.load(lse_offsets + meta_stride_2, mask=offs_splits < num_splits, other=float("-inf"))

    global_max = tl.max(m_i)

    o_offsets = o_partial_ptr + batch_idx * o_partial_stride_b + head_idx * o_partial_stride_h
    o_partial = tl.load(o_offsets + offs_splits[:, None] * o_partial_stride_split + offs_d[None, :] * o_partial_stride_d, mask=offs_splits[:, None] < num_splits)
    
    sumexp_normalized_splitk = tl.exp(lse - global_max)
    sumexp_normalized = tl.sum(sumexp_normalized_splitk, axis=0)
    numerator_normalized = tl.sum(o_partial * sumexp_normalized_splitk[:, None], axis=0)
    acc = numerator_normalized / sumexp_normalized
    acc = acc.to(o_ptr.dtype.element_ty)
    o_ptr += batch_idx * o_stride_b + head_idx * o_stride_h
    tl.store(o_ptr + offs_d * o_stride_d, acc)

## force cache_leftpad 
def flash_decode_leftpad(
    q,
    k_cache,
    v_cache,
    cache_seqlens,
    block_size=32,
    sm_scale=None,
):

    if q.dim() == 4:
        assert q.shape[1] == 1, "q length should be 1"
        q = q.squeeze(1)

    batch, heads, dim = q.shape

    if sm_scale is None:
        sm_scale = 1 / math.sqrt(dim)

    _, max_cache_seqlen, heads_kv, dim_v = v_cache.shape
    # assert max_cache_seqlen == max_cache_seqlen_cache, "max_cache_seqlen mismatch"
    group_size = heads // heads_kv

    block_H = 16

    max_selected_blocks = (max_cache_seqlen + block_size - 1) // block_size
    num_m_blocks = 1 * (heads // heads_kv + block_H - 1) // block_H
    num_n_blocks = max_selected_blocks

    size_one_kv_head = max_selected_blocks * block_size * (
        dim + dim_v) * 2  #kv_seqlen * (dim + dim_v) * 2
    total_mblocks = batch * heads_kv * num_m_blocks
    num_sm = 64
    num_splits = num_splits_heuristic(
        total_mblocks,
        num_sm,
        num_n_blocks,
        num_m_blocks,
        size_one_kv_head,
        is_causal_or_local=True,
        max_splits=128)

    # print("num_splits:", num_splits, "num_blocks:", num_n_blocks)

    # num_splits = 1
    num_splits_pow2 = triton.next_power_of_2(num_splits)
    # num_splits = num_splits_pow2
    


    o_partial = torch.empty((batch, heads, num_splits, dim_v), device=q.device, dtype=q.dtype)
    meta_data = torch.empty((batch, heads, 2, num_splits), device=q.device, dtype=torch.float32)

    BLOCK_D = dim
    BLOCK_H = group_size if group_size > 16 else 16
    grid = (batch, heads_kv, num_splits)

    with torch.cuda.device(q.device.index): 
        _split_kernel[grid](
            q,
            k_cache,
            v_cache,
            cache_seqlens,
            max_cache_seqlen,
            o_partial,
            meta_data,
            sm_scale,
            num_splits,
            group_size,
            q.stride(0), q.stride(1), q.stride(2),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
            o_partial.stride(0), o_partial.stride(1), o_partial.stride(2), o_partial.stride(3),
            meta_data.stride(0), meta_data.stride(1), meta_data.stride(2), meta_data.stride(3),
            BLOCK_H=BLOCK_H,
            BLOCK_N=block_size,
            BLOCK_D=BLOCK_D,
        )

    output = torch.zeros((batch, heads, dim_v), device=q.device, dtype=q.dtype)
    grid = (batch, heads)
    with torch.cuda.device(q.device.index): 
        _merge_kernel[grid](
            o_partial,
            meta_data,
            output,
            meta_data.stride(0), meta_data.stride(1), meta_data.stride(2), meta_data.stride(3),
            o_partial.stride(0), o_partial.stride(1), o_partial.stride(2), o_partial.stride(3),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_D=dim_v,
            num_splits=num_splits,
            num_splits_pow2=num_splits_pow2,
        )

    return output


def ref_program_fa(query, key, value, cache_seqlens, cache_leftpad=None):
    # latency reference
    # from flash_attn_interface import flash_attn_with_kvcache # fa3

    if cache_leftpad is not None: 
        cache_seqlens = cache_seqlens + cache_leftpad ## different definition from fa in this case
    
    from flash_attn import flash_attn_with_kvcache  #fa2
    query = query.unsqueeze(1)
    output = flash_attn_with_kvcache(query, key, value, cache_seqlens=cache_seqlens, cache_leftpad=cache_leftpad)
    output = output.squeeze(1)
    return output


def ref_program_torch_leftpad(
        query, 
        key, 
        value, 
        cache_seqlens,
        max_cache_seqlen, 
    ):

    batch, heads, dim = query.shape
    heads_kv = key.shape[2]

    num_head_groups = query.shape[1] // key.shape[2]
    scale = dim**0.5
    key = rearrange(key, 'b n h d -> b h n d')  # [batch_size, heads_kv, seqlen_kv, dim]
    value = rearrange(value, 'b n h d -> b h n d')  # [batch_size, heads_kv, seqlen_kv, dim]

    query = rearrange(
        query, 'b (h g) d -> b g h d',
        g=num_head_groups)  # [batch_size, num_head_groups, heads_kv, dim]

    scores = einsum(
        query, key,
        'b g h d, b h s d -> b g h s')  # [batch_size, num_head_groups, heads_kv, seqlen_kv]



    range_len = torch.arange(scores.shape[-1], device='cuda').unsqueeze(0)
    # cache_seqlens_expanded = cache_seqlens.unsqueeze(1) + cache_leftpad.unsqueeze(1)
    # print(cache_seqlens_expanded)
    pad_mask = range_len < (max_cache_seqlen - cache_seqlens).unsqueeze(1)
    pad_mask = pad_mask[:, None, None, :]
    # print("pad_mask", pad_mask)
    # print("left_pad_mask", left_pad_mask)
    scores = scores.masked_fill(pad_mask, float('-inf'))
    attention = torch.nn.functional.softmax(
        scores / scale, dim=-1)  # [batch_size, num_head_groups, heads_kv, seqlen_kv]

    out = einsum(attention, value,
                 'b g h s, b h s d -> b g h d')  # [batch_size, num_head_groups, heads_kv, dim]
    out = rearrange(out, 'b g h d -> b (h g) d')  # [batch_size, heads, dim]
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--heads', type=int, default=32, help='heads')
    parser.add_argument('--heads_kv', type=int, default=8, help='heads_kv')
    parser.add_argument(
        '--max_cache_seqlen', type=int, default=8192, help='kvcache sequence length')
    parser.add_argument('--dim', type=int, default=128, help='dim')
    parser.add_argument('--dim_v', type=int, default=128, help='dim_v')
    parser.add_argument('--block_size', type=int, default=32, help='block_size')
    args = parser.parse_args()

    batch, heads, heads_kv, max_cache_seqlen, dim, dim_v = args.batch, args.heads, args.heads_kv, args.max_cache_seqlen, args.dim, args.dim_v
    block_size = args.block_size
    qk_flops = 2 * batch * heads * max_cache_seqlen * dim
    pv_flops = 2 * batch * heads * max_cache_seqlen * dim_v
    total_flops = qk_flops + pv_flops

    dtype = torch.float16
    block_H = 64
    num_blocks = (max_cache_seqlen + block_size - 1) // block_size

    Q = torch.randn((batch, heads, dim), dtype=dtype, device='cuda')
    K = torch.randn((batch, max_cache_seqlen, heads_kv, dim), dtype=dtype, device='cuda')
    V = torch.randn((batch, max_cache_seqlen, heads_kv, dim_v), dtype=dtype, device='cuda')
    cache_seqlens = torch.randint(1, max_cache_seqlen, (batch,), dtype=torch.int32, device='cuda')
    # Ensure at least one element equals cache_seqlen
    random_index = torch.randint(0, batch, (1,), device='cuda').item()  # Select a random index
    # cache_seqlens[
    #     random_index] = max_cache_seqlen  # Assign cache_seqlen to ensure at least one occurrence


    ref = ref_program_torch_leftpad(Q, K, V, cache_seqlens, max_cache_seqlen)

    triton_out = flash_decode_leftpad(
        Q,
        K,
        V,
        cache_seqlens,
        block_size,
    )


    print("ref:", ref)
    print("triton_out:", triton_out)
    print("max diff:", torch.max(torch.abs(ref - triton_out)))
    assert torch.allclose(ref, triton_out, atol=1e-2), "Output mismatch between Triton and reference implementation"
    print("Pass test reference implementation.")


    # # Measure performance
    # torch.cuda.synchronize()
    # start = time.time()
    # for _ in range(1000):
    #     flash_decode_leftpad(
    #         Q,
    #         K,
    #         V,
    #         cache_seqlens,
    #         max_cache_seqlen,
    #         block_size,
    #         cache_leftpad=cache_leftpad,
    #     )
    # torch.cuda.synchronize()
    # end = time.time()
    # elapsed_time = end - start
    # avg_time = elapsed_time / 1000
    # avg_flops = total_flops / avg_time
    # print(f"Average time: {avg_time:.6f} seconds")



    # # Measure performance of reference implementation
    # start = time.time()
    # for _ in range(1000):
    #     ref_program_fa(Q, K, V, cache_seqlens)
    # torch.cuda.synchronize()
    # end = time.time()
    # elapsed_time_ref = end - start
    # avg_time_ref = elapsed_time_ref / 1000
    # avg_flops_ref = total_flops / avg_time_ref
    # print(f"Average time of ref: {avg_time_ref:.6f} seconds")

    # print(f"Speedup: {avg_time_ref / avg_time:.2f}x")
