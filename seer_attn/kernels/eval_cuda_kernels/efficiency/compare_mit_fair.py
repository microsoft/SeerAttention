# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py

# cd SeerAttention/eval/efficiency/
# python3 compare_mit_fair.py

import torch

from block_sparse_attn import (
    block_sparse_attn_func,
    flash_attn_varlen_func,
)

from utils import (
    time_fwd,
    flops,
    efficiency,
)

# from my_block_sparse_attn import block_sparse_attention
from block_sparse_seer_attn import block_sparse_attention, varlen_block_sparse_attention

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

block_size = 128

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
    
def get_tensors(bs, seq_len, num_heads, head_dim, dtype=torch.float16):
    q = (torch.empty((bs, seq_len, num_heads, head_dim), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((bs, seq_len, num_heads, head_dim), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((bs, seq_len, num_heads, head_dim), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    return q, k, v

def profile_blocksparse_fwd():
    repeats = 3
    device = 'cuda:0'
    dtype = torch.float16
    causal = True
    batch_size = 4
    sparsity_sampling_steps = 20
    # seqlen_vals = [1024, 4096, 8192, 16384, 32768, 32768 * 2]
    seqlen_vals = [1024, 4096, 8192]    
    headdim = 128
    dim = 4096
    dropout_p = 0.0
    method = ("block_sparse_attention")
    time_f = {}
    speed_f = {}

    for seqlen in seqlen_vals:
        results = {}
        nheads = dim // headdim
        shape = (batch_size * seqlen, nheads, headdim)
        q = torch.randn(shape, device=device, dtype=dtype)
        k = torch.randn(shape, device=device, dtype=dtype)
        v = torch.randn(shape, device=device, dtype=dtype)
        cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=device)
        base_f = time_fwd(flash_attn_varlen_func, q, k, v, cu_seqlens, cu_seqlens, seqlen, seqlen, dropout_p, None, causal, repeats=repeats, verbose=False)
        base_speed = efficiency(flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"), base_f)
        results["base"] = [[base_f], [base_speed]]
        ##### Tiny F
        # fq = q.view(batch_size, seqlen, nheads, headdim).transpose(1,2).contiguous()
        # fk = k.view(batch_size, seqlen, nheads, headdim).transpose(1,2).contiguous()
        # fv = v.view(batch_size, seqlen, nheads, headdim).transpose(1,2).contiguous()
        # tiny_f = time_fwd(flash_attention_v2_cutlass, fq, fk, fv, causal, 1, repeats=repeats, verbose=False)
        # tiny_speed = efficiency(flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"), tiny_f)

        print("MIT BLOCK SPARSE ATTENTION")
        sparsity_list = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        for sparsity in sparsity_list:
            cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=device)
            head_mask_type = torch.tensor([1] * nheads, device=device, dtype=torch.int32)
            base_blockmask, real_sparsity = generate_base_sparsity_mask(seqlen, seqlen, block_size, block_size, block_size, sparsity, causal = causal, device=device)
            base_blockmask = base_blockmask.unsqueeze(0).repeat(batch_size, nheads, 1, 1)
            config = (causal, headdim, nheads, batch_size, seqlen, sparsity, real_sparsity)
            print(q.shape, k.shape, v.shape, base_blockmask.shape)
            f = time_fwd(block_sparse_attn_func, q, k, v, cu_seqlens, cu_seqlens, head_mask_type, None, base_blockmask, seqlen, seqlen, dropout_p, is_causal=causal, exact_streaming=False, repeats=repeats, verbose=False)
            time_f[config, method] = f
            print(f"### causal={causal}, headdim={headdim}, nheads = {nheads}, batch_size={batch_size}, seqlen={seqlen}, real_sparsity={real_sparsity} ###")
            speed_f[config, method] = efficiency(flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"), time_f[config, method])
            print(
                f"{method}"
                f"MIT: {speed_f[config, method]:.2f} TFLOPs/s, {(time_f[config, method]*1000):.2f} ms, "
                # f"fwd base: {base_speed:.2f} TFLOPs/s, {base_f*1000:.2f} ms",
                # f"fwd tiny: {tiny_speed:.2f} TFLOPs/s, {tiny_f*1000:.2f} ms",
                )
        
        fq, fk, fv = get_tensors(batch_size, seqlen, nheads, headdim)
        print("MY BLOCK SPARSE ATTENTION")
        for sparsity in sparsity_list:
            # cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=device)
            head_mask_type = torch.tensor([1] * nheads, device=device, dtype=torch.int32)
            base_blockmask, real_sparsity = generate_base_sparsity_mask(seqlen, seqlen, block_size, block_size, block_size, sparsity, causal = causal, device=device)
            base_blockmask = base_blockmask.unsqueeze(0).repeat(batch_size, nheads, 1, 1)
            config = (causal, headdim, nheads, batch_size, seqlen, sparsity, real_sparsity)
            # f = time_fwd(block_sparse_attention, fq, fk, fv, base_blockmask, causal, 1.0, repeats=repeats, verbose=False)
            import math
            sm_scale = 1 / math.sqrt(headdim)
            # f = time_fwd(block_sparse_attention, fq, fk, fv, base_blockmask, causal, sm_scale, repeats=repeats, verbose=False)
            f = time_fwd(varlen_block_sparse_attention, q, k, v, cu_seqlens, cu_seqlens, base_blockmask, seqlen, seqlen, causal, sm_scale, repeats=repeats, verbose=False)
            time_f[config, method] = f
            print(f"### causal={causal}, headdim={headdim}, nheads = {nheads}, batch_size={batch_size}, seqlen={seqlen}, real_sparsity={real_sparsity} ###")
            speed_f[config, method] = efficiency(flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"), time_f[config, method])
            print(
                f"{method}"
                f"my: {speed_f[config, method]:.2f} TFLOPs/s, {(time_f[config, method]*1000):.2f} ms, "
                )

        print("--------------------------------------------------------------------------------------------------")
        print("--------------------------------------------------------------------------------------------------")
        print("--------------------------------------------------------------------------------------------------")

profile_blocksparse_fwd()
