import torch
import triton
import triton.language as tl
import argparse
from einops import rearrange, einsum
import torch.nn.functional as F
from .utils import num_splits_heuristic
import math
import time
import os


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
    mask_ptr,
    sm_scale,
    num_splits,
    gqa_group_size,
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_b, stride_k_s, stride_k_h, stride_k_d,
    stride_v_b, stride_v_s, stride_v_h, stride_v_d,
    stride_o_b, stride_o_h, stride_o_split, stride_o_d,
    stride_meta_b, stride_meta_h, stride_meta_2, stride_meta_split,
    stride_mask_b, stride_mask_h, stride_mask_s,
    first_block_unmasked: tl.constexpr,
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
    mask_ptr += batch_idx * stride_mask_b + head_idx_kv * stride_mask_h

    q = tl.load(q_ptr + offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d, mask=offs_h[:, None] < gqa_group_size)
    start = blocks_per_split * split_idx + tl.minimum(split_idx, remaining_blocks) + leftpad_block_offset
    for block_idx in range(loop_range):
        cur_block_idx = (start + block_idx)
        start_n = cur_block_idx * BLOCK_N
        mask_val = tl.load(mask_ptr + cur_block_idx * stride_mask_s)
        if mask_val == 1 or (first_block_unmasked == True and cur_block_idx == leftpad_block_offset):
            k_ptr = k_cache_ptr + start_n * stride_k_s 
            v_ptr = v_cache_ptr + start_n * stride_v_s


            k_mask = (start_n + offs_n[None, :] < max_seqlen) & (start_n + offs_n[None, :] >= cache_leftpad)
            v_mask = (start_n + offs_n[:, None] < max_seqlen) & (start_n + offs_n[:, None] >= cache_leftpad)
            k = tl.load(k_ptr, mask=k_mask, other=0.0)
            v = tl.load(v_ptr, mask=v_mask, other=0.0)

            qk = tl.dot(q, k)
            qk = qk * sm_scale
            qk = tl.where(k_mask, qk, float("-inf"))
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



def block_sparse_flash_decode_leftpad_gqa_mask(
    q,
    k_cache,
    v_cache,
    cache_seqlens,
    block_mask,
    block_size=32,
    sm_scale=None,
    first_block_unmasked=False,
    debug=False,
):
    
    """
        Compute block sparse flash decode for left-padded batch of data.

        Args:
            q (torch.Tensor): Query tensor with shape (batch, heads, dim) or (batch, 1, heads, dim). 
                If provided as a 4D tensor, the function expects the second dimension (length) to be 1 and 
                will squeeze that dimension.
            k_cache (torch.Tensor): Key cache tensor with shape (batch, max_cache_seqlen, heads_kv, dim_v).
                It stores the keys used for the attention computation.
            v_cache (torch.Tensor): Value cache tensor with shape (batch, max_cache_seqlen, heads_kv, dim_v).
                It stores the values used for the attention computation.
            cache_seqlens (torch.Tensor): Tensor containing the cumulative sequence lengths for each batch.
                It is used to index into the key and value caches.
            block_mask (torch.Tensor): Mask tensor indicating valid blocks for attention computation.
                It is applied during the kernel operations to filter out or mask specific blocks.
            block_size (int, optional): The size of each block, defaulting to 32. This determines how the 
                cache sequences are divided into blocks.
            sm_scale (float, optional): Scaling factor used for the softmax operation. If not provided, 
                it is computed as 1/sqrt(dim) where dim is the feature dimension of q.
            first_block_unmasked: (bool, optional): If True, the first block is always true. This is useful when
                the gate can not have a precise estimate of the first block.
            debug (bool, optional): If set to True, the function returns additional debugging information 
                (the intermediate partial output tensor and meta data). Defaults to False.
        Returns:
            torch.Tensor: The final attention output tensor with shape (batch, heads, dim_v).
        If debug is True, also returns:
            o_partial (torch.Tensor): The intermediate partial output tensor from the split kernel.
            meta_data (torch.Tensor): Meta data tensor containing auxiliary information used in the merging step.
    
    """


    if q.dim() == 4:
        assert q.shape[1] == 1, "q length should be 1"
        q = q.squeeze(1)
    batch, heads, dim = q.shape

    if sm_scale is None:
        sm_scale = 1 / math.sqrt(dim)

    _, max_cache_seqlen, heads_kv, dim_v = v_cache.shape
    group_size = heads // heads_kv

    block_H = 16
    
    max_selected_blocks = (max_cache_seqlen + block_size - 1) // block_size
    num_m_blocks = 1 * (heads // heads_kv + block_H - 1) // block_H
    num_n_blocks = max_selected_blocks

    size_one_kv_head = max_selected_blocks * block_size * (
        dim + dim_v) * 2  #kv_seqlen * (dim + dim_v) * 2
    total_mblocks = batch * heads_kv * num_m_blocks
    num_sm = 64
    # num_sm = self.num_sm
    num_splits = num_splits_heuristic(
        total_mblocks,
        num_sm,
        num_n_blocks,
        num_m_blocks,
        size_one_kv_head,
        is_causal_or_local=True,
        max_splits=128)

    # print("num_splits:", num_splits, "num_blocks:", num_n_blocks)

    num_splits_pow2 = triton.next_power_of_2(num_splits)

    o_partial = torch.empty((batch, heads, num_splits, dim_v), device=q.device, dtype=q.dtype)
    meta_data = torch.empty((batch, heads, 2, num_splits), device=q.device, dtype=torch.float32)

    with torch.cuda.device(q.device.index):
        BLOCK_D = dim
        BLOCK_H = group_size if group_size > 16 else 16
        grid = (batch, heads_kv, num_splits)
        _split_kernel[grid](
            q,
            k_cache,
            v_cache,
            cache_seqlens,
            max_cache_seqlen,
            o_partial,
            meta_data,
            block_mask,
            sm_scale,
            num_splits,
            group_size,
            q.stride(0), q.stride(1), q.stride(2),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
            o_partial.stride(0), o_partial.stride(1), o_partial.stride(2), o_partial.stride(3),
            meta_data.stride(0), meta_data.stride(1), meta_data.stride(2), meta_data.stride(3),
            block_mask.stride(0), block_mask.stride(1), block_mask.stride(2),
            first_block_unmasked=first_block_unmasked,
            BLOCK_H=BLOCK_H,
            BLOCK_N=block_size,
            BLOCK_D=BLOCK_D,
        )
        output = torch.zeros((batch, heads, dim_v), device=q.device, dtype=q.dtype)
        grid = (batch, heads)
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
    if debug:
        return output, o_partial, meta_data
    return output

def ref_program_torch(
        query, 
        key, 
        value, 
        block_mask, 
        cache_seqlens, 
        max_cache_seqlen, 
        num_blocks,
        block_size,
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

    sparse_mask = torch.zeros_like(scores)
    # Assign mask values
    for b in range(batch):
        for h in range(heads_kv):
            for idx in range(num_blocks):
                if block_mask[b, h, idx]:
                    sparse_mask[b, :, h, idx * block_size:(idx + 1) * block_size] = 1

    scores = scores.masked_fill(sparse_mask == 0, float('-inf'))

    range_len = torch.arange(scores.shape[-1], device='cuda').unsqueeze(0)
    pad_mask = range_len < (max_cache_seqlen - cache_seqlens).unsqueeze(1)
    pad_mask = pad_mask[:, None, None, :]
    scores = scores.masked_fill(pad_mask, float('-inf'))


    attention = F.softmax(
        scores / scale, dim=-1)  # [batch_size, num_head_groups, heads_kv, seqlen_kv]
    
    out = einsum(attention, value,
                 'b g h s, b h s d -> b g h d')  # [batch_size, num_head_groups, heads_kv, dim]
    out = rearrange(out, 'b g h d -> b (h g) d')  # [batch_size, heads, dim]
    return out

def ref_program_fa(query, key, value, cache_seqlens, cache_leftpad=None):

    if cache_leftpad is not None: 
        cache_seqlens = cache_seqlens + cache_leftpad ## different definition from fa in this case
    from flash_attn import flash_attn_with_kvcache  #fa2
    query = query.unsqueeze(1)
    output = flash_attn_with_kvcache(query, key, value, cache_seqlens=cache_seqlens, cache_leftpad=cache_leftpad)
    output = output.squeeze(1)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--heads', type=int, default=32, help='heads')
    parser.add_argument('--heads_kv', type=int, default=8, help='heads_kv')
    parser.add_argument(
        '--max_cache_seqlen', type=int, default=8192, help='kvcache sequence length')
    parser.add_argument('--dim', type=int, default=128, help='dim')
    parser.add_argument('--dim_v', type=int, default=128, help='dim_v')
    parser.add_argument('--sparse_ratio', type=float, default=0.8, help='sparse ratio')
    parser.add_argument('--block_size', type=int, default=32, help='block_size')
    args = parser.parse_args()

    batch, heads, heads_kv, max_cache_seqlen, dim, dim_v = args.batch, args.heads, args.heads_kv, args.max_cache_seqlen, args.dim, args.dim_v
    block_size = args.block_size
    sparse_ratio = args.sparse_ratio
    qk_flops = 2 * batch * heads * max_cache_seqlen * dim
    pv_flops = 2 * batch * heads * max_cache_seqlen * dim_v
    total_flops = qk_flops + pv_flops

    dtype = torch.float16
    block_H = 64

    Q = torch.randn((batch, heads, dim), dtype=dtype, device='cuda')
    K = torch.randn((batch, max_cache_seqlen, heads_kv, dim), dtype=dtype, device='cuda')
    V = torch.randn((batch, max_cache_seqlen, heads_kv, dim_v), dtype=dtype, device='cuda')
    cache_seqlens = torch.randint(1, max_cache_seqlen, (batch,), dtype=torch.int32, device='cuda')
    # Ensure at least one element equals cache_seqlen
    random_index = torch.randint(0, batch, (1,), device='cuda').item()  # Select a random index
    cache_seqlens[
        random_index] = max_cache_seqlen  # Assign cache_seqlen to ensure at least one occurrence

    # cache_seqlens = torch.full((batch,), max_cache_seqlen, dtype=torch.int32, device='cuda')

    num_blocks = (max_cache_seqlen + block_size - 1) // block_size

    valid_num_blocks = torch.ceil(cache_seqlens * (1 - sparse_ratio) / block_size).int()
    print("valid_num_blocks: ", valid_num_blocks)
    max_valid_num_blocks = torch.ceil(cache_seqlens / block_size).int()
    print("max_valid_num_blocks: ", max_valid_num_blocks)
    block_mask = torch.zeros((batch, heads_kv, num_blocks), dtype=torch.bool, device='cuda')

    # Assign valid indices while ensuring no duplicates within each batch-group
    for b in range(batch):
        leftpad = max_cache_seqlen - cache_seqlens[b].item()
        leftpad_block = leftpad // block_size
        max_valid_block = max_valid_num_blocks[b].item()  # Max valid blocks for this batch
        valid_num_block = valid_num_blocks[b].item()  # Valid blocks for this batch
        if valid_num_block > 0:  # Ensure there's at least one valid block
            for h in range(heads_kv):
                perm = torch.randperm(max_valid_block, device='cuda')[0:valid_num_block] + leftpad_block
                block_mask[b, h, perm] = True

    ref = ref_program_torch(
        Q, 
        K, 
        V, 
        block_mask, 
        cache_seqlens, 
        max_cache_seqlen, 
        num_blocks,
        block_size,
    )

    triton_out = block_sparse_flash_decode_leftpad_gqa_mask(
        Q,
        K,
        V,
        cache_seqlens,
        block_mask,
        block_size,
    )

    print("max difference: ", torch.max(torch.abs(ref - triton_out)))
    assert torch.allclose(ref, triton_out, atol=1e-2), "Output mismatch between Triton and reference implementation"
    print("Passed the ref test!")


    ## latency reference
    for _ in range(10):
        ref = ref_program_fa(Q, K, V,cache_seqlens)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        ref = ref = ref_program_fa(Q, K, V, cache_seqlens)
    torch.cuda.synchronize()
    # print("dense time: ", (time.time() - start) / 100 * 1000)
    dense_time = (time.time() - start) / 100 * 1000



    for _ in range(10):
        # out = sparse_gqa_decode_varlen_mask(Q, K, V, block_mask, cache_seqlens, block_size)
        out = block_sparse_flash_decode_leftpad_gqa_mask(
            Q,
            K,
            V,
            cache_seqlens,
            block_mask,
            block_size,
        )  
        
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        # out = sparse_gqa_decode_varlen_mask(Q, K, V, block_mask, cache_seqlens, block_size)
        out = block_sparse_flash_decode_leftpad_gqa_mask(
            Q,
            K,
            V,
            cache_seqlens,
            block_mask,
            block_size,
        )    
    torch.cuda.synchronize()
    # print("sparse time: ", (time.time() - start) / 100 * 1000)
    triton_sparse_time = (time.time() - start) / 100 * 1000

    ## save results to file
    file_dir = "results"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = f"{file_dir}/sparse_gqa_decode_varlen_mask.txt"
    # append the results to the file
    with open(file_name, "a") as f:
        f.write(
            f"batch={batch}, heads={heads}, heads_kv={heads_kv}, max_cache_seqlen={max_cache_seqlen}, dim={dim}, dim_v={dim_v}, block_size={block_size}, "
            f"dense_time={dense_time:.2f}ms, triton_sparse_time={triton_sparse_time:.2f}ms\n"
        )