import torch
import time
import math
import argparse
from tabulate import tabulate
from flash_attn import flash_attn_func as flash_attn_func_offical
from block_sparse_seer_attn import block_sparse_attention

# Helper function to initialize tensors
def get_tensors(bs, num_heads, seq_len, head_dim, dtype=torch.float16):
    q = (torch.empty((bs, num_heads, seq_len, head_dim), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((bs, num_heads, seq_len, head_dim), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((bs, num_heads, seq_len, head_dim), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    return q, k, v

# Benchmarking function
def run_benchmark(epoch, warmup, func, *args, **kwargs):
    # Warmup phase
    for _ in range(warmup):
        _ = func(*args, **kwargs)
    torch.cuda.synchronize()
    time_s = time.time()
    for _ in range(epoch):
        _ = func(*args, **kwargs)
        torch.cuda.synchronize()
    time_e = time.time() - time_s
    return time_e

# Generate sparsity mask
def generate_base_sparsity_mask(max_seqlen_q, max_seqlen_k, round_base, m_block_dim, n_block_dim, sparsity, causal=False, device="cuda"):
    def round_to_multiple(x, base):
        return ((x + base - 1) // base) * base
    nrow, ncol = round_to_multiple(max_seqlen_q, round_base) // m_block_dim, round_to_multiple(max_seqlen_k, round_base) // n_block_dim
    base_mask = torch.zeros(1, nrow, ncol, device=device, dtype=torch.bool)
    total_block_num = 0

    density = 1.0 - sparsity
    if not density == 0.0 and not density == 1.0:
        for i in range(nrow): # do in reverse order
            idx = nrow - i - 1
            if causal:
                available_col_num = max(0, ncol - i)
                total_block_num += available_col_num
                num_one = max(1, int(density * available_col_num))
                base_mask[0][idx, torch.randperm(available_col_num)[:num_one]] = True
            else:
                available_col_num = ncol
                total_block_num += available_col_num
                num_one = max(1, int(density * available_col_num))
                base_mask[0][idx, torch.randperm(available_col_num)[:num_one]] = True
    elif density == 1.0:
        base_mask[0] = torch.ones_like(base_mask[0])
        total_block_num = nrow * ncol
    else:
        total_block_num = nrow * ncol
    
    calculated_block_num = base_mask.sum().item()
    return base_mask

# Main benchmark function
def benchmark(args, is_causal=True, warmup=10, epoch=10, dtype=torch.float16):
    # Extract parameters from args
    bs = args.batch_size
    num_heads = args.num_heads
    seq_len = args.seq_len
    head_dim = args.head_dim
    block_size = args.block_size
    sparsity_list = args.sparsity
    sm_scale = 1 / math.sqrt(head_dim)

    # Initialize tensors
    q, k, v = get_tensors(bs, num_heads, seq_len, head_dim, dtype=dtype)
    results = {}

    # Benchmark block-sparse attention
    for sparsity in sparsity_list:
        base_blockmask = generate_base_sparsity_mask(seq_len, seq_len, block_size, block_size, block_size, sparsity, is_causal, device="cuda")
        base_blockmask = base_blockmask.unsqueeze(0).repeat(bs, num_heads, 1, 1)
        sparse_time = run_benchmark(epoch, warmup, block_sparse_attention, q, k, v, base_blockmask, is_causal, sm_scale)
        results[f"block_sparse_attn ({sparsity:.2f})"] = sparse_time * 1000 / epoch

    # Benchmark FlashAttention-2
    fq = q.transpose(1, 2)
    fk = k.transpose(1, 2)
    fv = v.transpose(1, 2)
    official_time = run_benchmark(epoch, warmup, flash_attn_func_offical, fq, fk, fv, causal=is_causal, softmax_scale=sm_scale)
    results["FlashAttention-2 (full)"] = official_time * 1000 / epoch

    return results

# Argument parser for user input
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Block-Sparse Attention vs FlashAttention-2")
    parser.add_argument("--seq_len", type=int, required=True, help="Sequence length")
    parser.add_argument("--sparsity", type=float, nargs="+", required=True, help="List of sparsity levels to test")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--head_dim", type=int, default=128, help="Dimension of each attention head")
    parser.add_argument("--block_size", type=int, default=64, help="Block size for block-sparse attention")
    return parser.parse_args()

# Main function
def main():
    args = parse_args()
    print("Running benchmark with:")
    print(f"- batch_size: {args.batch_size}, num_heads: {args.num_heads}, seq_len: {args.seq_len}, head_dim: {args.head_dim}")
    print(f"- sparsity: {args.sparsity}")

    results = benchmark(args)

    # Print results in a table format
    table = []
    flash_time = results["FlashAttention-2 (full)"]  # Get FlashAttention-2 time
    for key, value in results.items():
        if key.startswith("block_sparse_attn"):
            speedup = flash_time / value  # Calculate speedup: Flash / Block-Sparse
            table.append([key, f"{value:.2f} ms", f"{speedup:.2f}x"])
        else:
            table.append([key, f"{value:.2f} ms", "-"])

    print("\nBenchmark results:")
    print(tabulate(table, headers=["Method (Sparsity)", "Time (ms)", "Speedup"], tablefmt="pretty"))

if __name__ == "__main__":
    main()