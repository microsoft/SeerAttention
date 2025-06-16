# cd /root/jianying/seer-pr/SeerAttention/seer_attn/kernels/eval_cuda_kernels/accuracy
# python3 benchmark_sparse_attn.py

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

BS, SEQLEN, DIM = 4, 4095, 64
HEAD_Q = 32
# HEAD_KV = 4
HEAD_KV = 32
block_size = 64

sm_scale = 1.0 # 1 / math.sqrt(DIM)
is_causal = True
device = 'cuda'
dtype = torch.float16

q,k,v = get_tensors(BS, SEQLEN, HEAD_Q, HEAD_KV, DIM, dtype=dtype)

def main():
    # fq = q.transpose(1, 2)
    # fk = k.transpose(1, 2)
    # fv = v.transpose(1, 2)
    # baseline = self_attention(fq, fk, fv, causal=is_causal, sm_scale=sm_scale)
    # baseline = baseline.transpose(1, 2)    
    # base_time = run_benchmark(epoch, warmup, self_attention, fq, fk, fv, causal=is_causal, sm_scale=sm_scale)
    # print("baseline: \n", base_time * 1000 / epoch)

    flash_attn_output = flash_attn_func_offical(q, k, v, causal=is_causal, softmax_scale=sm_scale)

    my_flash_attn_output, *_ = my_flash_attn(q, k, v, is_causal, sm_scale)
    # my_flash_attn_time = run_benchmark(epoch, warmup, my_flash_attn, q, k, v, causal=is_causal, softmax_scale=sm_scale)
    assert torch.allclose(flash_attn_output, my_flash_attn_output, rtol=0, atol=1e-2)   

    # CHECK correctness, use sparsity 0.0
    base_blockmask, real_sparsity = generate_base_sparsity_mask(SEQLEN, SEQLEN, block_size, block_size, block_size, 0.0, is_causal, device='cuda')
    base_blockmask = base_blockmask.unsqueeze(0).repeat(BS, HEAD_Q, 1, 1)
    out, _ = block_sparse_attention(q, k, v, base_blockmask, is_causal, sm_scale) 
    assert torch.allclose(flash_attn_output, out, rtol=1e-2, atol=1e-2)   
    print("Passed correctness check with sparsity 0.0")
    # assert torch.allclose(baseline, flash_attn_output, rtol=0, atol=1e-2)   
    # assert torch.allclose(baseline, out, rtol=0, atol=1e-2) 


if __name__ == "__main__":
    main()

