import torch
import triton
import triton.language as tl
import argparse

import math
import time
from einops import rearrange, einsum
from seer_attn.modules.common import repeat_kv
import torch.nn.functional as F


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
    cache_seqlens_ptr,
    max_seqlen,
    o_partial_ptr, ## reuse for qk max
    po_ptr, 
    sm_scale,
    gqa_group_size,
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_b, stride_k_s, stride_k_h, stride_k_d,
    stride_o_b, stride_o_h, stride_o_s,
    stride_po_b, stride_po_h, stride_po_s,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx_kv = tl.program_id(1)

    head_idx_q = head_idx_kv * gqa_group_size
    offs_h = tl.arange(0, BLOCK_H)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_H], 1.0, dtype=tl.float32)


    
    cache_seqlens = tl.load(cache_seqlens_ptr + batch_idx)
    cache_leftpad = max_seqlen - cache_seqlens
    leftpad_block_offset = cache_leftpad // BLOCK_N
    num_blocks = (max_seqlen + BLOCK_N - 1) // BLOCK_N - leftpad_block_offset

    q_ptr += batch_idx * stride_q_b + head_idx_q * stride_q_h
    k_cache_ptr += batch_idx * stride_k_b + head_idx_kv * stride_k_h + offs_n[None, :] * stride_k_s + offs_d[:, None] * stride_k_d
    o_partial_ptr += batch_idx * stride_o_b + head_idx_q * stride_o_h
    po_ptr += batch_idx * stride_po_b + head_idx_kv * stride_po_h

    q = tl.load(q_ptr + offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d, mask=offs_h[:, None] < gqa_group_size)
    for block_idx in range(num_blocks):
        start_n = (leftpad_block_offset + block_idx) * BLOCK_N
        k_ptr = k_cache_ptr + start_n * stride_k_s 
        
        k_mask = (start_n + offs_n[None, :] < max_seqlen) & (start_n + offs_n[None, :] >= cache_leftpad)
        k = tl.load(k_ptr, mask=k_mask, other=0.0)
        qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk = qk * sm_scale
        qk = tl.where(start_n + offs_n[None, :] < max_seqlen, qk, float("-inf"))
        qk_max = tl.max(qk, 1)
        o_ptrs = o_partial_ptr + offs_h * stride_o_h + (leftpad_block_offset + block_idx) * stride_o_s
        # store qk_max
        tl.store(o_ptrs, qk_max, mask=offs_h < gqa_group_size)

        m_ij = tl.maximum(m_i, qk_max)
        qk -= m_ij[:, None]
        p = tl.exp(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        m_i = m_ij


    for block_idx in range(num_blocks):
        o_ptrs = o_partial_ptr + offs_h * stride_o_h + (leftpad_block_offset + block_idx) * stride_o_s
        po_ptrs = po_ptr + (leftpad_block_offset + block_idx) * stride_po_s
        # rescale qk_max
        local_max = tl.load(o_ptrs, mask=offs_h < gqa_group_size, other=float("-inf"))
        local_max -= m_i
        local_max = tl.exp(local_max) / l_i
        # tl.store(o_ptrs, local_max, mask=offs_h < gqa_group_size)
        head_pooled = tl.max(local_max, 0)
        tl.store(po_ptrs, head_pooled)



## force cache_leftpad 
def oracle_sparse(
    q,
    k_cache,
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

    _, max_cache_seqlen, heads_kv, dim_v = k_cache.shape
    # assert max_cache_seqlen == max_cache_seqlen_cache, "max_cache_seqlen mismatch"
    group_size = heads // heads_kv

    max_selected_blocks = (max_cache_seqlen + block_size - 1) // block_size

    # print("num_splits:", num_splits, "num_blocks:", num_n_blocks)
    # o_partial = torch.full((batch, heads, max_selected_blocks), -1e10, device=q.device, dtype=torch.float32)
    o_partial = torch.zeros((batch, heads, max_selected_blocks), device=q.device, dtype=torch.float32)  
    po = torch.zeros((batch, heads_kv, max_selected_blocks), device=q.device, dtype=q.dtype)

    with torch.cuda.device(q.device.index): 
        BLOCK_D = dim
        BLOCK_H = group_size if group_size > 16 else 16
        grid = (batch, heads_kv)
        _split_kernel[grid](
            q,
            k_cache,
            cache_seqlens,
            max_cache_seqlen,
            o_partial,
            po,
            sm_scale,
            group_size,
            q.stride(0), q.stride(1), q.stride(2),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
            o_partial.stride(0), o_partial.stride(1), o_partial.stride(2), 
            po.stride(0), po.stride(1), po.stride(2),
            BLOCK_H=BLOCK_H,
            BLOCK_N=block_size,
            BLOCK_D=BLOCK_D,
        )

    return po


def ref_program(q, k, attention_mask, block_size):
    if q.dim() == 4:
        batch_size, q_len, num_q_heads, head_dim = q.shape
        assert q_len == 1, "q length should be 1"
        q = q.squeeze(1)
    else:
        batch_size, num_q_heads, head_dim = q.shape
        q_len = 1
    _, kv_len, num_kv_heads, _ = k.shape
    num_gqa_groups = num_q_heads // num_kv_heads
    
    q = q.contiguous()
    k = k.transpose(1, 2).contiguous()

    # Repeat K heads for GQA compatibility
    if num_gqa_groups > 1:
        k = repeat_kv(k, num_gqa_groups)

    attn_weights = torch.einsum('bhd, bhdj -> bhj', q, k.transpose(-1, -2)) * (head_dim**-0.5) # (b, num_q_heads, q_len, kv_len)

    attention_mask=attention_mask.unsqueeze(1)

    attn_weights = attn_weights.masked_fill(~attention_mask.bool(), float('-inf'))
    
    attn_weights_qhead = F.softmax(attn_weights, dim=-1, dtype=torch.float32)

    attn_weights = F.max_pool2d(attn_weights_qhead, kernel_size=(num_gqa_groups, block_size), stride=(num_gqa_groups, block_size), ceil_mode=True)
    # attn_weights_qhead = F.max_pool2d(attn_weights_qhead, kernel_size=(1, block_size), stride=(1, block_size), ceil_mode=True)
    # attn_weights = F.max_pool2d(attn_weights_qhead, kernel_size=(num_gqa_groups, 1), stride=(num_gqa_groups, 1), ceil_mode=True)
    # attn_weights = attn_weights.to(q.dtype)
    # attn_weights_qhead = attn_weights_qhead.to(q.dtype)

    return attn_weights


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
    num_blocks = (max_cache_seqlen + block_size - 1) // block_size

    Q = torch.randn((batch, heads, dim), dtype=dtype, device='cuda')
    K = torch.randn((batch, max_cache_seqlen, heads_kv, dim), dtype=dtype, device='cuda')
    V = torch.randn((batch, max_cache_seqlen, heads_kv, dim_v), dtype=dtype, device='cuda')
    cache_seqlens = torch.randint(1, max_cache_seqlen, (batch,), dtype=torch.int32, device='cuda')
    # Ensure at least one element equals cache_seqlen
    random_index = torch.randint(0, batch, (1,), device='cuda').item()  # Select a random index
    # cache_seqlens[
    #     random_index] = max_cache_seqlen  # Assign cache_seqlen to ensure at least one occurrence
    seq_range = torch.arange(max_cache_seqlen, device='cuda')
    attention_mask = seq_range[None, :].ge(max_cache_seqlen - cache_seqlens[:, None])
    print("attention_mask:", attention_mask, "cache_seqlens:", cache_seqlens)
    ref = ref_program(Q, K, attention_mask, block_size)

    triton_out = oracle_sparse(
        Q,
        K,
        cache_seqlens,
        block_size,
    )

    print("ref:", ref)
    print("triton_out:", triton_out)
    print("max diff:", torch.max(torch.abs(ref - triton_out)))

    print(torch.where(torch.abs(ref - triton_out) > 0.1))
    
    # assert torch.allclose(ref, triton_out, atol=1e-2, rtol=1e-2), "Output mismatch between Triton and reference implementation"
    # print("Pass test reference implementation.")


    # warm up
    for _ in range(10):
        oracle_sparse(
            Q,
            K,
            cache_seqlens,
            block_size,
        )
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        oracle_sparse(
            Q,
            K,
            cache_seqlens,
            block_size,
        )
    torch.cuda.synchronize()
    end = time.time()
    triton_time = (end - start) / 100
    print("Triton kernel time:", triton_time, "s")

    for _ in range(10):
        ref_program(Q, K, attention_mask, block_size)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        ref_program(Q, K, attention_mask, block_size)
    torch.cuda.synchronize()
    end = time.time()
    ref_time = (end - start) / 100
    print("Reference kernel time:", ref_time, "s")
    print("Speedup:" , ref_time / triton_time)
