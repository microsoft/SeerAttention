# cd eval/efficiency
# python3 benchmark_varlen_sparse_attn.py

'''
official_ref: 
 1.4579296112060547
my_varlen_flash_attn_time: 
 1.4536380767822266
real sparsity 0.000, time: 1.7731189727783203
real sparsity 0.114, time: 1.608133316040039
real sparsity 0.212, time: 1.4986991882324219
real sparsity 0.412, time: 1.2331008911132812
real sparsity 0.612, time: 0.8945465087890625
real sparsity 0.816, time: 0.6072521209716797
real sparsity 0.912, time: 0.4620552062988281
'''

import torch
import time
from einops import rearrange
from flash_attn import flash_attn_varlen_func as flash_attn_func_offical
from block_sparse_seer_attn import varlen_block_sparse_attention, my_varlen_flash_attn

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

def generate_qkv(
        q, k, v
    ):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
    batch_size, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    assert k.shape == (batch_size, seqlen_k, nheads_k, d)
    assert v.shape == (batch_size, seqlen_k, nheads_k, d)

    q_unpad = rearrange(q, "b s h d -> (b s) h d")
    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q_unpad.device
    )
    max_seqlen_q = seqlen_q
    output_pad_fn = lambda output_unpad: rearrange(
        output_unpad, "(b s) h d -> b s h d", b=batch_size
    )

    k_unpad = rearrange(k, "b s h d -> (b s) h d")
    v_unpad = rearrange(v, "b s h d -> (b s) h d")
    cu_seqlens_k = torch.arange(
        0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=k_unpad.device
    )
    max_seqlen_k = seqlen_k

    return (
        q_unpad.detach().requires_grad_(),
        k_unpad.detach().requires_grad_(),
        v_unpad.detach().requires_grad_(),
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
    )

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
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
    ) = generate_qkv(q, k, v)

    official_ref_time = run_benchmark(epoch, warmup, flash_attn_func_offical, q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=is_causal, softmax_scale=sm_scale)
    print("official_ref: \n", official_ref_time * 1000 / epoch)

    my_varlen_flash_attn_time = run_benchmark(epoch, warmup, my_varlen_flash_attn, q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, is_causal, sm_scale)
    print("my_varlen_flash_attn_time: \n", my_varlen_flash_attn_time * 1000 / epoch)    

    for sparsity in [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9]:
        base_blockmask, real_sparsity = generate_base_sparsity_mask(SEQLEN, SEQLEN, block_size, block_size, block_size, sparsity, is_causal, device='cuda')
        base_blockmask = base_blockmask.unsqueeze(0).repeat(BS, HEAD_Q, 1, 1)
        #row_mask = convert_blockmask_row_reverse(base_blockmask, is_causal)
        sparse_time = run_benchmark(epoch, warmup, varlen_block_sparse_attention, q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, base_blockmask, max_seqlen_q, max_seqlen_k, is_causal, sm_scale)
        print(f"real sparsity {real_sparsity:.3f}, time: {sparse_time * 1000 / epoch}")

if __name__ == "__main__":
    epoch = 1
    for _ in range(epoch):
        main()

