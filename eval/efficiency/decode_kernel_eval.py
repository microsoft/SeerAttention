import torch
import argparse
import os
import subprocess
import time
import math
from tilelang.autotuner import *
from seer_attn.kernels.varlen.triton_sparse_gqa_decode_varlen_indice import block_sparse_flash_decode_gqa_indice_triton
from seer_attn.kernels.varlen.tilelang_sparse_gqa_decode_varlen_indice import SparseFlashAttn


def ref_program_fa(query, key, value, block_indices, cache_seqlens, max_cache_seqlen, num_blocks,
                   block_size):
    # latency reference
    # from flash_attn_interface import flash_attn_with_kvcache # fa3
    from flash_attn import flash_attn_with_kvcache  #fa2
    query = query.unsqueeze(1)
    output = flash_attn_with_kvcache(query, key, value, cache_seqlens=cache_seqlens)
    output = output.squeeze(1)
    return output


def main(batch=8,
         heads=32,
         heads_kv=8,
         max_cache_seqlen=8192,
         dim=128,
         dim_v=128,
         sparse_ratio=0.8,
         block_size=32,
        ):
    batch, heads, heads_kv, max_cache_seqlen, dim, dim_v = batch, heads, heads_kv, max_cache_seqlen, dim, dim_v
    sparse_ratio = sparse_ratio
    block_size = block_size

    sparse_kernel = SparseFlashAttn(batch, heads, heads_kv, dim, dim_v, block_size, max_cache_seqlen, sparse_ratio)

    max_selected_blocks = int(math.ceil(max_cache_seqlen * (1 - sparse_ratio) / block_size))
    print("max_selected_blocks: ", max_selected_blocks)
    dtype = torch.float16

    Q = torch.randn((batch, heads, dim), dtype=dtype, device='cuda')
    K = torch.randn((batch, max_cache_seqlen, heads_kv, dim), dtype=dtype, device='cuda')
    V = torch.randn((batch, max_cache_seqlen, heads_kv, dim_v), dtype=dtype, device='cuda')
    

    cache_seqlens = torch.full((batch,), max_cache_seqlen, dtype=torch.int32, device='cuda')

    print("cache_seqlens: ", cache_seqlens)

    max_valid_num_blocks = torch.ceil(cache_seqlens / block_size).int()
    print("max_valid_num_blocks: ", max_valid_num_blocks)
    # Initialize block_indices with -1 (for padding blocks)
    block_indices = torch.full((batch, heads_kv, max_selected_blocks),
                               -1,
                               dtype=torch.int32,
                               device='cuda')

    # Assign valid indices while ensuring no duplicates within each batch-group
    for b in range(batch):
        max_valid_block = max_valid_num_blocks[b].item()  # Max valid blocks for this batch
        if max_valid_block > 0:  # Ensure there's at least one valid block
            for h in range(heads_kv):
                valid_indices = torch.randperm(
                    max_valid_block, device='cuda', dtype=torch.int32)[:max_selected_blocks]
                # valid_indices = torch.randperm(max_valid_block, device='cuda', dtype=torch.int32)[:max_num_blocks]
                block_indices[b, h, :len(valid_indices)] = valid_indices

    # Sort indices within each batch-group for consistency
    block_indices, _ = block_indices.sort(dim=-1, descending=True)
    # print("block_indices: ", block_indices)
    actual_num_blocks = torch.sum(block_indices != -1, dim=-1).to(torch.int32)[:, 0]
    print("actual_num_blocks: ", actual_num_blocks)
    # print(block_indices.shape, actual_num_blocks.shape)

    max_num_blocks = torch.max(max_valid_num_blocks).item()
    print("max_num_blocks: ", max_num_blocks)
    
    
    ## latency reference
    for _ in range(10):
        ref = ref_program_fa(Q, K, V, block_indices, cache_seqlens, max_cache_seqlen,
                             max_num_blocks, block_size)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        ref = ref_program_fa(Q, K, V, block_indices, cache_seqlens, max_cache_seqlen,
                             max_num_blocks, block_size)
    torch.cuda.synchronize()
    fa2_dense_time = (time.time() - start) / 100 * 1000
    print("fa3 dense time: ", fa2_dense_time)

    for _ in range(10):
        block_sparse_flash_decode_gqa_indice_triton(
            Q,
            K,
            V,
            cache_seqlens,
            max_cache_seqlen,
            max_selected_blocks,
            block_indices,
            block_size,
        )
        
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        block_sparse_flash_decode_gqa_indice_triton(
            Q,
            K,
            V,
            cache_seqlens,
            max_cache_seqlen,
            max_selected_blocks,
            block_indices,
            block_size,
        )   
    torch.cuda.synchronize()
    triton_sparse_time = (time.time() - start) / 100 * 1000
    print("triton sparse time: ", triton_sparse_time)

    for _ in range(10):
        out = sparse_kernel(Q, K, V, block_indices, cache_seqlens)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        out = sparse_kernel(Q, K, V, block_indices, cache_seqlens)
    torch.cuda.synchronize()
    tilelang_sparse_time = (time.time() - start) / 100 * 1000
    print("tilelang sparse time: ", tilelang_sparse_time)

    

    # save results to file
    file_dir = "results"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = f"{file_dir}/kernel_test_{block_size}_gqa{heads}_{heads_kv}.csv"

    data_line = (
        f"{batch},{max_cache_seqlen},{sparse_ratio},"
        f"{fa2_dense_time:.4f},{triton_sparse_time:.4f},{tilelang_sparse_time:.4f}\n"
    )

    with open(file_name, "a") as f:
        # if os.path.getsize(file_name) == 0:
        #     f.write(
        #         "batch,max_cache_seqlen,sparse_ratio,fa2_dense_time,triton_sparse_time,tilelang_sparse_time\n"
        #     )
        f.write(data_line)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--heads', type=int, default=64, help='heads')
    parser.add_argument('--heads_kv', type=int, default=8, help='heads_kv')
    parser.add_argument(
        '--max_cache_seqlen', type=int, default=32768, help='kvcache sequence length')
    parser.add_argument('--dim', type=int, default=128, help='dim')
    parser.add_argument('--dim_v', type=int, default=128, help='dim_v')
    parser.add_argument('--sparse_ratio', type=float, default=0.9, help='sparse ratio')
    parser.add_argument('--block_size', type=int, default=64, help='block_size')
    args = parser.parse_args()
    main(args.batch, args.heads, args.heads_kv, args.max_cache_seqlen, args.dim, args.dim_v,
         args.sparse_ratio, args.block_size)