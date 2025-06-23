# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py

import math
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

torch.set_printoptions(profile="full")
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

def generate_base_sparsity_mask(max_seqlen_q, max_seqlen_k, round_base, m_block_dim, n_block_dim, batch_size, num_blocksparse_heads, sparsity, causal=False, device="cuda"):
    def round_to_multiple(x, base):
        return ((x + base - 1) // base) * base
    
    nrow, ncol = round_to_multiple(max_seqlen_q, round_base) // m_block_dim, round_to_multiple(max_seqlen_k, round_base) // n_block_dim
    base_mask = torch.zeros(batch_size, num_blocksparse_heads, nrow, ncol, device=device, dtype=torch.bool)
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

    base_blockmask = base_mask.unsqueeze(0).repeat(batch_size, num_blocksparse_heads, 1, 1)
            
    return base_blockmask
