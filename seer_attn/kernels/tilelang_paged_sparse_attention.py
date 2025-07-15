# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
from einops import rearrange, einsum
import argparse
import time
import math
import random
from seer_attn.kernels.varlen.utils import num_splits_heuristic


tilelang.disable_cache()

def flashattn(batch, heads, heads_kv, dim, dim_v):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    dtype = "float16"
    accum_dtype = "float"
    kv_group_num = heads // heads_kv

    @tilelang.jit(out_idx=[-1])
    def kernel_func(block_N, block_H, page_block_size, num_split, num_stages, threads, num_pages, 
                    max_num_blocks_per_seq, max_selected_blocks):
        shape_q = [batch, heads, dim]
        shape_k = [num_pages, page_block_size, heads_kv, dim]
        shape_v = [num_pages, page_block_size, heads_kv, dim_v]
        shape_indices = [batch, heads_kv, max_selected_blocks]
        shape_block_table = [batch, max_num_blocks_per_seq]
        shape_o = [batch, heads, dim_v]
        part_shape = [batch, heads, num_split, dim_v]
        valid_block_H = min(block_H, kv_group_num)
        assert block_N <= page_block_size and page_block_size % block_N == 0 
        block_ratio = page_block_size // block_N

        @T.macro
        def flash_attn_split(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_v, dtype),
                block_indices: T.Tensor(shape_indices, "int32"),
                cache_seqlens: T.Tensor([batch], "int32"),
                block_table: T.Tensor(shape_block_table, "int32"),
                glse: T.Tensor([batch, heads, num_split], accum_dtype),
                Output_partial: T.Tensor(part_shape, accum_dtype),
        ):
            with T.Kernel(
                    batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim_v], dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
                acc_o = T.alloc_fragment([block_H, dim_v], accum_dtype)

                scores_max = T.alloc_fragment([block_H], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)
                has_valid_block = T.alloc_var("bool")

                bid = bx
                hid = by
                sid = bz
                cur_kv_head = hid // (kv_group_num // valid_block_H)

                T.copy(Q[bid, hid * valid_block_H:hid * valid_block_H + block_H, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                num_blocks = max_selected_blocks
                blocks_per_split = T.floordiv(num_blocks, num_split)
                remaining_blocks = T.floormod(num_blocks, num_split)
                loop_range = (blocks_per_split + T.if_then_else(sid < remaining_blocks, 1, 0))
                start = blocks_per_split * sid + T.min(sid, remaining_blocks)
                has_valid_block = False
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    logical_block_idx = block_indices[bid, cur_kv_head, start + k]
                    if logical_block_idx >= 0:
                        has_valid_block = True
                        block_table_idx = T.floordiv(logical_block_idx, block_ratio)
                        block_tile_idx = T.floormod(logical_block_idx, block_ratio)
                        physical_block_idx = block_table[bid, block_table_idx]
                        T.copy(K[physical_block_idx, 
                                 block_tile_idx * block_N: (block_tile_idx + 1) * block_N, cur_kv_head, :], K_shared)
                        T.clear(acc_s)
                        T.gemm(
                            Q_shared,
                            K_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow)
                        if k == 0:  # assume block_indices is sorted in reverse order, otherwise, remove this if condition
                            for i, j in T.Parallel(block_H, block_N):
                                acc_s[i,
                                      j] = T.if_then_else(logical_block_idx * block_N + j >= cache_seqlens[bid],
                                                          -T.infinity(accum_dtype), acc_s[i, j])
                        T.copy(scores_max, scores_max_prev)
                        T.fill(scores_max, -T.infinity(accum_dtype))
                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        for i in T.Parallel(block_H):
                            scores_max[i] = T.if_then_else(scores_max[i] > scores_max_prev[i],
                                                           scores_max[i], scores_max_prev[i])
                            scores_scale[i] = T.exp2(scores_max_prev[i] * scale -
                                                     scores_max[i] * scale)
                        for i, j in T.Parallel(block_H, block_N):
                            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(block_H):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                        T.copy(acc_s, acc_s_cast)
                        for i, j in T.Parallel(block_H, dim_v):
                            acc_o[i, j] *= scores_scale[i]
                        T.copy(V[physical_block_idx, 
                                 block_tile_idx * block_N: (block_tile_idx + 1) * block_N, cur_kv_head, :], V_shared)
                        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                if has_valid_block:
                    for i, j in T.Parallel(block_H, dim_v):
                        acc_o[i, j] /= logsum[i]

                    for i in T.Parallel(block_H):
                        logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

                for i in T.Parallel(block_H):
                    if i < valid_block_H:
                        glse[bid, hid * valid_block_H + i, sid] = logsum[i]

                for i, j in T.Parallel(block_H, dim_v):
                    if i < valid_block_H:
                        Output_partial[bid, hid * valid_block_H + i, sid, j] = acc_o[i, j]

        @T.macro
        def combine(
                glse: T.Tensor([batch, heads, num_split], accum_dtype),
                Output_partial: T.Tensor(part_shape, accum_dtype),
                Output: T.Tensor(shape_o, dtype),
        ):
            with T.Kernel(heads, batch, threads=128) as (by, bz):
                po_local = T.alloc_fragment([dim_v], accum_dtype)
                o_accum_local = T.alloc_fragment([dim_v], accum_dtype)
                lse_local_split = T.alloc_local([1], accum_dtype)
                lse_logsum_local = T.alloc_local([1], accum_dtype)
                lse_max_local = T.alloc_local([1], accum_dtype)
                scale_local = T.alloc_local([1], accum_dtype)
                max_split = T.alloc_local([1], "int32")

                T.annotate_layout({
                    lse_logsum_local:
                        T.Fragment(lse_logsum_local.shape, forward_thread_fn=lambda i: i),
                })

                T.clear(lse_logsum_local)
                T.clear(o_accum_local)
                lse_max_local[0] = -T.infinity(accum_dtype)
                for k in T.serial(num_split):
                    lse_local_split[0] = glse[bz, by, k]
                    if (lse_local_split[0] != 0):
                        max_split[0] = k
                        lse_max_local[0] = T.max(lse_max_local[0], glse[bz, by, k])

                for k in T.Pipelined(num_split, num_stages=1):
                    if k <= max_split[0]:
                        lse_local_split[0] = glse[bz, by, k]
                        lse_logsum_local[0] += T.exp2(lse_local_split[0] - lse_max_local[0])
                lse_logsum_local[0] = T.log2(lse_logsum_local[0]) + lse_max_local[0]
                for k in T.serial(num_split):
                    if k <= max_split[0]:
                        for i in T.Parallel(dim_v):
                            po_local[i] = Output_partial[bz, by, k, i]
                        lse_local_split[0] = glse[bz, by, k]
                        scale_local[0] = T.exp2(lse_local_split[0] - lse_logsum_local[0])
                        for i in T.Parallel(dim_v):
                            o_accum_local[i] += po_local[i] * scale_local[0]
                for i in T.Parallel(dim_v):
                    Output[bz, by, i] = o_accum_local[i]

        @T.prim_func
        def main(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_v, dtype),
                block_indices: T.Tensor(shape_indices, "int32"),
                cache_seqlens: T.Tensor([batch], "int32"),
                block_table: T.Tensor(shape_block_table, "int32"),
                glse: T.Tensor([batch, heads, num_split], accum_dtype),
                Output_partial: T.Tensor(part_shape, accum_dtype),
                Output: T.Tensor(shape_o, dtype),
        ):
            flash_attn_split(Q, K, V, block_indices, cache_seqlens, block_table, glse, Output_partial)
            combine(glse, Output_partial, Output)

        return main

    return kernel_func


class SparseFlashAttn(torch.nn.Module):

    def __init__(self, batch, heads, heads_kv, dim, dim_v, page_block_size, block_N, num_pages):
        super(SparseFlashAttn, self).__init__()
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.dim_v = dim_v
        self.block_N = block_N
        self.page_block_size = page_block_size
        self.num_pages = num_pages
        self.block_H = 64

        self.kernel = flashattn(batch, heads, heads_kv, dim, dim_v)(
            block_N=block_N,
            block_H=self.block_H,
            page_block_size=page_block_size,
            num_split=T.symbolic("num_split"),
            num_stages=2,
            threads=128,
            num_pages=num_pages,
            max_num_blocks_per_seq=T.symbolic("max_num_blocks_per_seq"),
            max_selected_blocks=T.symbolic("max_selected_blocks"),
        )

        props = torch.cuda.get_device_properties(torch.device("cuda:0"))
        self.num_sm = props.multi_processor_count

    def forward(self, query, key, value, block_indices, cache_seqlens, block_table):
        batch = self.batch
        heads = self.heads
        heads_kv = self.heads_kv
        dim_v = self.dim_v
        dim = self.dim
        block_size = self.block_N
        max_selected_blocks = block_indices.shape[-1]

        # Compute static scheduling parameters
        num_m_blocks = 1 * (heads // heads_kv + self.block_H - 1) // self.block_H
        num_n_blocks = max_selected_blocks
        size_one_kv_head = max_selected_blocks * block_size * (dim + dim_v) * 2
        total_mblocks = batch * heads_kv * num_m_blocks

        num_sm = self.num_sm

        num_split = num_splits_heuristic(
            total_mblocks,
            num_sm,
            num_n_blocks,
            num_m_blocks,
            size_one_kv_head,
            is_causal_or_local=True,
            max_splits=128)

        glse = torch.empty((batch, heads, num_split), dtype=torch.float32, device='cuda')
        output_partial = torch.empty((batch, heads, num_split, dim_v),
                                     dtype=torch.float32,
                                     device='cuda')

        
        output = self.kernel(
            query, key, value, block_indices, cache_seqlens, block_table, glse, output_partial,
        )
        return output



def ref_program_fa(query, kcache, vcache, cache_seqlens, block_table):
    # latency reference
    # from flash_attn_interface import flash_attn_with_kvcache # fa3
    from flash_attn import flash_attn_with_kvcache  #fa2
    query = query.unsqueeze(1)
    output = flash_attn_with_kvcache(query, kcache, vcache, cache_seqlens=cache_seqlens, block_table=block_table)
    output = output.squeeze(1)
    return output


def main(args):

    batch, heads, heads_kv, max_cache_seqlen, dim, dim_v = args.batch, args.heads, args.heads_kv, args.max_cache_seqlen, args.dim, args.dim_v
    sparse_ratio = args.sparse_ratio
    block_N = args.block_N
    page_block_size = args.page_block_size
    num_blocks = args.num_pages  # Use num_pages from args
    
    # For dense case verification, set sparse_ratio to 0 to select all blocks
    max_selected_blocks = int(math.ceil(max_cache_seqlen / block_N))
    print("max_selected_blocks: ", max_selected_blocks)
    dtype = torch.float16

    # Generate random inputs
    Q = torch.randn((batch, heads, dim), dtype=dtype, device='cuda')
    cache_seqlens = torch.randint(max_cache_seqlen // 2, max_cache_seqlen + 1, (batch,), dtype=torch.int32, device='cuda')
    print("cache_seqlens: ", cache_seqlens)

    K = torch.randn((batch, max_cache_seqlen, heads_kv, dim), dtype=dtype, device='cuda')
    V = torch.randn((batch, max_cache_seqlen, heads_kv, dim_v), dtype=dtype, device='cuda')


    # Create paged KV cache
    K_cache = torch.zeros((num_blocks, page_block_size, heads_kv, dim), dtype=dtype, device='cuda')
    V_cache = torch.zeros((num_blocks, page_block_size, heads_kv, dim_v), dtype=dtype, device='cuda')
    
    # Create block table and block indices for dense case (all blocks selected)
    max_num_blocks_per_seq = int(math.ceil(max_cache_seqlen / page_block_size))
    print("max_num_blocks_per_seq: ", max_num_blocks_per_seq)
    block_table = torch.zeros((batch, max_num_blocks_per_seq), dtype=torch.int32, device='cuda')
    block_indices = torch.zeros((batch, heads_kv, max_selected_blocks), dtype=torch.int32, device='cuda')
    

    # Fill block table and block indices and cache
    
    # Fill block table with sequential physical block indices
    for seq_idx in range(batch):
        seq_len = cache_seqlens[seq_idx].item()
        num_blocks_needed = int(math.ceil(seq_len / page_block_size))
        
        # Assign physical blocks sequentially for each sequence
        for block_idx in range(num_blocks_needed):
            physical_block_idx = seq_idx * max_num_blocks_per_seq + block_idx
            block_table[seq_idx, block_idx] = physical_block_idx
    
    print(f"Block table: {block_table}")

    block_table_v2 = torch.arange(
        batch * max_num_blocks_per_seq, dtype=torch.int32,
        device='cuda').view(batch, max_num_blocks_per_seq)
    print(f"Block table v2: {block_table_v2}")


    # Fill K_cache and V_cache with data from original K and V tensors
    for seq_idx in range(batch):
        seq_len = cache_seqlens[seq_idx].item()
        num_blocks_needed = int(math.ceil(seq_len / page_block_size))
        
        for block_idx in range(num_blocks_needed):
            physical_block_idx = seq_idx * max_num_blocks_per_seq + block_idx
            
            # Calculate the range of tokens for this block
            start_token = block_idx * page_block_size
            end_token = min(start_token + page_block_size, seq_len)
            actual_block_size = end_token - start_token
            
            # Copy K and V data to the paged cache
            K_cache[physical_block_idx, :actual_block_size, :, :] = K[seq_idx, start_token:end_token, :, :]
            V_cache[physical_block_idx, :actual_block_size, :, :] = V[seq_idx, start_token:end_token, :, :]
        
    # Fill block_indices for sparse attention
    # For dense case (verification), we select all blocks in reverse order
    # For sparse case, we select a subset of blocks based on sparse_ratio
    for seq_idx in range(batch):
        seq_len = cache_seqlens[seq_idx].item()
        num_logical_blocks = int(math.ceil(seq_len / block_N))
        
        if sparse_ratio == 0.0:
            # Dense case: select all blocks in reverse order
            selected_blocks = min(num_logical_blocks, max_selected_blocks)
            for head_idx in range(heads_kv):
                for i in range(selected_blocks):
                    # Select blocks in reverse order (most recent first)
                    block_indices[seq_idx, head_idx, i] = num_logical_blocks - 1 - i
                # Fill remaining slots with -1 (invalid)
                for i in range(selected_blocks, max_selected_blocks):
                    block_indices[seq_idx, head_idx, i] = -1
        else:
            # Sparse case: select subset of blocks based on sparse_ratio
            num_selected = int(num_logical_blocks * (1.0 - sparse_ratio))
            num_selected = max(1, min(num_selected, max_selected_blocks))
            
            # Create a list of all valid block indices
            all_blocks = list(range(num_logical_blocks))
            
            # For demonstration, select the most recent blocks and some random earlier blocks
            selected_blocks = []
            
            # Always include the most recent blocks (important for attention)
            recent_blocks = min(num_selected // 2, num_logical_blocks)
            for i in range(recent_blocks):
                selected_blocks.append(num_logical_blocks - 1 - i)
            
            # Randomly select some earlier blocks
            if num_selected > recent_blocks:
                remaining_blocks = [b for b in all_blocks if b not in selected_blocks]
                if remaining_blocks:
                    import random
                    random.seed(42)  # For reproducibility
                    additional_blocks = random.sample(remaining_blocks, 
                                                    min(num_selected - recent_blocks, len(remaining_blocks)))
                    selected_blocks.extend(additional_blocks)
            
            # Sort selected blocks in reverse order (most recent first)
            selected_blocks.sort(reverse=True)
            
            # Fill block_indices for all KV heads
            for head_idx in range(heads_kv):
                for i in range(len(selected_blocks)):
                    block_indices[seq_idx, head_idx, i] = selected_blocks[i]
                # Fill remaining slots with -1 (invalid)
                for i in range(len(selected_blocks), max_selected_blocks):
                    block_indices[seq_idx, head_idx, i] = -1
    
    print(f"Block indices shape: {block_indices.shape}")
    print(f"Sample block indices for seq 0, head 0: {block_indices[0, 0, :10]}")
    print(f"Block table shape: {block_table.shape}")
    print(f"Sample block table for seq 0: {block_table[0, :10]}")


    # Initialize sparse attention module
    sparse_attn = SparseFlashAttn(batch, heads, heads_kv, dim, dim_v, page_block_size, block_N, num_blocks)
    output_sparse = sparse_attn.forward(Q, K_cache, V_cache, block_indices, cache_seqlens, block_table)
    

    # Run reference implementation

    output_ref = ref_program_fa(Q, K_cache, V_cache, cache_seqlens, block_table)
    
    print("Kernel Output: ", output_sparse)
    print("Output: ", output_ref)


    # Check correctness
    max_diff = torch.max(torch.abs(output_sparse - output_ref)).item()
    mean_diff = torch.mean(torch.abs(output_sparse - output_ref)).item()
    
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    
    if max_diff < 1e-2:
        print("✓ Verification PASSED: Results match within tolerance")
    else:
        print("✗ Verification FAILED: Results differ significantly")
        
    return output_sparse, output_ref



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--heads', type=int, default=32, help='heads')
    parser.add_argument('--heads_kv', type=int, default=8, help='heads_kv')
    parser.add_argument(
        '--max_cache_seqlen', type=int, default=8192, help='kvcache sequence length')
    parser.add_argument('--dim', type=int, default=128, help='dim')
    parser.add_argument('--dim_v', type=int, default=128, help='dim_v')
    parser.add_argument('--sparse_ratio', type=float, default=0.0, help='sparse ratio')
    parser.add_argument('--block_N', type=int, default=64, help='block_N')
    parser.add_argument('--page_block_size', type=int, default=256, help='page_block_size')
    parser.add_argument('--num_pages', type=int, default=1024, help='total number of pages')
    args = parser.parse_args()
    main(args)