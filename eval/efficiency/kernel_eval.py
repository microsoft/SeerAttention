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
from seer_attn.kernels.varlen.block_sparse_flash_decode_varlen_mask_leftpad import block_sparse_flash_decode_leftpad_gqa_mask
from seer_attn.kernels.varlen.tilelang_sparse_gqa_decode_varlen_mask import SparseFlashAttn


def ref_program_fa(query, key, value, cache_seqlens):
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
         test_leftpad=False):
    batch, heads, heads_kv, max_cache_seqlen, dim, dim_v = batch, heads, heads_kv, max_cache_seqlen, dim, dim_v
    sparse_ratio = sparse_ratio
    block_size = block_size
    test_leftpad = test_leftpad

    max_selected_blocks = int(math.ceil(max_cache_seqlen * (1 - sparse_ratio) / block_size))
    print("max_selected_blocks: ", max_selected_blocks)
    dtype = torch.float16

    Q = torch.randn((batch, heads, dim), dtype=dtype, device='cuda')
    K = torch.randn((batch, max_cache_seqlen, heads_kv, dim), dtype=dtype, device='cuda')
    V = torch.randn((batch, max_cache_seqlen, heads_kv, dim_v), dtype=dtype, device='cuda')
    
    # Ensure at least one element equals cache_seqlen
    if test_leftpad == 1:
        cache_seqlens = torch.randint(1, max_cache_seqlen, (batch,), dtype=torch.int32, device='cuda')
        random_index = torch.randint(0, batch, (1,), device='cuda').item()  # Select a random index
        cache_seqlens[
            random_index] = max_cache_seqlen  # Assign cache_seqlen to ensure at least one occurrence
    else:
        cache_seqlens = torch.full((batch,), max_cache_seqlen, dtype=torch.int32, device='cuda')

    print("cache_seqlens: ", cache_seqlens)

    num_blocks = (max_cache_seqlen + block_size - 1) // block_size

    valid_num_blocks = torch.ceil(cache_seqlens * (1 - sparse_ratio) / block_size).int()
    print("valid_num_blocks: ", valid_num_blocks)
    max_valid_num_blocks = torch.ceil(cache_seqlens / block_size).int()
    print("max_valid_num_blocks: ", max_valid_num_blocks)
    # Initialize block_mask with false (for padding blocks)
    block_mask_tilelang = torch.zeros((batch, heads_kv, num_blocks), dtype=torch.bool, device='cuda')
    block_mask_triton = torch.zeros((batch, heads_kv, num_blocks), dtype=torch.bool, device='cuda')

    for b in range(batch):
        leftpad = max_cache_seqlen - cache_seqlens[b].item()
        leftpad_block = leftpad // block_size
        max_valid_block = max_valid_num_blocks[b].item()  # Max valid blocks for this batch
        valid_num_block = valid_num_blocks[b].item()  # Valid blocks for this batch
        if valid_num_block > 0:  # Ensure there's at least one valid block
            for h in range(heads_kv):
                perm_nopad = torch.randperm(max_valid_block, device='cuda')[:valid_num_block]
                perm_withpad = torch.randperm(max_valid_block, device='cuda')[0:valid_num_block] + leftpad_block
                block_mask_tilelang[b, h, perm_nopad] = True
                block_mask_triton[b, h, perm_withpad] = True
    
    model = SparseFlashAttn(batch, heads, heads_kv, dim, dim_v, block_size)

    ## latency reference
    for _ in range(10):
        ref = ref_program_fa(Q, K, V, cache_seqlens)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        ref = ref_program_fa(Q, K, V, cache_seqlens)
    torch.cuda.synchronize()
    fa2_dense_time = (time.time() - start) / 100 * 1000
    print("fa2 dense time: ", fa2_dense_time)

    for _ in range(10):
        out = block_sparse_flash_decode_leftpad_gqa_mask(
            Q,
            K,
            V,
            cache_seqlens,
            block_mask_triton,
            block_size,
        )  
        
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        out = block_sparse_flash_decode_leftpad_gqa_mask(
            Q,
            K,
            V,
            cache_seqlens,
            block_mask_triton,
            block_size,
        )    
    torch.cuda.synchronize()
    triton_sparse_time = (time.time() - start) / 100 * 1000
    print("triton sparse time: ", triton_sparse_time)

    for _ in range(10):
        out = model(Q, K, V, block_mask_tilelang, cache_seqlens)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        out = model(Q, K, V, block_mask_tilelang, cache_seqlens)
    torch.cuda.synchronize()
    tilelang_sparse_time = (time.time() - start) / 100 * 1000
    print("tilelang sparse time: ", tilelang_sparse_time)

    

    ## save results to file
    file_dir = "results_all"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = f"{file_dir}/kernel_test_gqa{heads}_{heads_kv}_leftpad{test_leftpad}.txt"
    # append the results to the file
    with open(file_name, "a") as f:
        f.write(
            f"batch={batch}, max_cache_seqlen={max_cache_seqlen}, block_size={block_size}, sparse_ratio={sparse_ratio}, "
            f"fa2_dense_time={fa2_dense_time:.2f}ms, triton_sparse_time={triton_sparse_time:.2f}ms, tilelang_sparse_time={tilelang_sparse_time:.2f}ms\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--heads', type=int, default=64, help='heads')
    parser.add_argument('--heads_kv', type=int, default=8, help='heads_kv')
    parser.add_argument(
        '--max_cache_seqlen', type=int, default=131072, help='kvcache sequence length')
    parser.add_argument('--dim', type=int, default=128, help='dim')
    parser.add_argument('--dim_v', type=int, default=128, help='dim_v')
    parser.add_argument('--sparse_ratio', type=float, default=0.9, help='sparse ratio')
    parser.add_argument('--block_size', type=int, default=16, help='block_size')
    parser.add_argument('--test_leftpad', type=int, default=0, help='0 for no leftpad, 1 for leftpad')
    args = parser.parse_args()
    main(args.batch, args.heads, args.heads_kv, args.max_cache_seqlen, args.dim, args.dim_v,
         args.sparse_ratio, args.block_size, args.test_leftpad)