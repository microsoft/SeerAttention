# cd eval_cuda_kernels/efficiency
# python3 benchmark_sparse_attn.py

'''
official_ref: 
 1.3885498046875
my_flash_attn_time: 
 1.4486312866210938
real sparsity 0.000, time: 1.7940998077392578
real sparsity 0.114, time: 1.6274452209472656
real sparsity 0.212, time: 1.495361328125
real sparsity 0.412, time: 1.2025833129882812
real sparsity 0.612, time: 0.9119510650634766
real sparsity 0.816, time: 0.6268024444580078
real sparsity 0.912, time: 0.48542022705078125
'''

import torch
import time
# offical flash attention implement
from flash_attn import flash_attn_func as flash_attn_func_offical
from block_sparse_seer_attn import block_sparse_attention, my_flash_attn

def get_tensors(bs, seq_len, num_heads_q, num_heads_kv, head_dim, dtype=torch.float16):
    q = (torch.empty((bs, seq_len, num_heads_q, head_dim), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((bs, seq_len, num_heads_kv, head_dim), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((bs, seq_len, num_heads_kv, head_dim), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    return q, k, v

def self_attention(q, k, v, causal=True, sm_scale=1):
    SEQLEN = q.shape[-2]
    M = torch.tril(torch.ones((SEQLEN, SEQLEN), device="cuda"))
    p = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)
    return ref_out

def run_benchmark(epoch, warmup, func, *args, **kwargs):
    # warmup phase
    for _ in range(warmup):
        _ = func(*args, **kwargs)
    torch.cuda.synchronize()
    time_s = time.time()
    for _ in range(epoch):
        _ = func(*args, **kwargs)
        torch.cuda.synchronize()
    time_e = time.time() - time_s
    return time_e

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
    real_sparsity = 1.0 - calculated_block_num / total_block_num
    return base_mask, real_sparsity

def get_sparsity_list(sampling_steps, seqlen, causal):
    blockmask_element_num = (seqlen // block_size) ** 2 // (2 if causal else 1)
    stride = max(blockmask_element_num // sampling_steps, 1)
    actual_steps = (blockmask_element_num + stride - 1) // stride
    sparsity_list = []
    for i in range(actual_steps):
        sparse_rate = (1 + i * stride) / blockmask_element_num
        if sparse_rate > 0.95 or sparse_rate < 0.0:
            continue
        sparsity_list.append(sparse_rate)
    return sparsity_list

warmup = 10
epoch = 10

BS, SEQLEN, DIM = 4, 4096, 64
HEAD_Q = 32
HEAD_KV = 4
block_size = 64

sm_scale = 1.0 # 1 / math.sqrt(DIM)
is_causal = True
device = 'cuda'
dtype = torch.float16

q,k,v = get_tensors(BS, SEQLEN, HEAD_Q, HEAD_KV, DIM, dtype=dtype)

def main():
    ## only HEAD_Q == HEAD_KV
    # fq = q.transpose(1, 2)
    # fk = k.transpose(1, 2)
    # fv = v.transpose(1, 2)
    # baseline = self_attention(fq, fk, fv, causal=is_causal, sm_scale=sm_scale)
    # baseline = baseline.transpose(1, 2)    
    # base_time = run_benchmark(epoch, warmup, self_attention, fq, fk, fv, causal=is_causal, sm_scale=sm_scale)
    # print("baseline: \n", base_time * 1000 / epoch)

    official_ref_time = run_benchmark(epoch, warmup, flash_attn_func_offical, q, k, v, causal=is_causal, softmax_scale=sm_scale)
    print("official_ref: \n", official_ref_time * 1000 / epoch)

    my_flash_attn_time = run_benchmark(epoch, warmup, my_flash_attn, q, k, v, causal=is_causal, softmax_scale=sm_scale)
    print("my_flash_attn_time: \n", my_flash_attn_time * 1000 / epoch)    

    for sparsity in [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9]:
        base_blockmask, real_sparsity = generate_base_sparsity_mask(SEQLEN, SEQLEN, block_size, block_size, block_size, sparsity, is_causal, device='cuda')
        base_blockmask = base_blockmask.unsqueeze(0).repeat(BS, HEAD_Q, 1, 1)
        sparse_time = run_benchmark(epoch, warmup, block_sparse_attention, q, k, v, base_blockmask, is_causal, sm_scale)
        print(f"real sparsity {real_sparsity:.3f}, time: {sparse_time * 1000 / epoch}")

if __name__ == "__main__":
    epoch = 1
    for _ in range(epoch):
        main()

