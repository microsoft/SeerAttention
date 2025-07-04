import torch
from block_sparse_seer_attn_cutclass import flash_attention_v2_cutlass, varlen_flash_attention_v2_cutlass
from block_sparse_seer_attn_cutclass import flash_attention_block_v2_cutlass, varlen_flash_attention_block_v2_cutlass

def convert_blockmask_row_reverse(blockmask, causal=True, kM=64, kN=64):
    M, N = blockmask.shape[-2], blockmask.shape[-1]
    # TO be added: support arbitrary block size
    def apply_causal_mask():
        max_block_col_ids = ((torch.arange(M, device=blockmask.device) + 1) * kM + kN - 1) // kN
        max_block_col_ids = torch.minimum(max_block_col_ids, torch.tensor(N))
        mask = torch.arange(N, device=blockmask.device).unsqueeze(0) < max_block_col_ids.unsqueeze(1)
        return blockmask * mask

    # Sort does not support bool on CUDA
    if causal:
        blockmask = apply_causal_mask()
    blockmask = blockmask.to(dtype=torch.uint8)
    # print("blockmask: ", blockmask)
    nonzero_val, nonzero_sorted_rowidx = blockmask.sort(dim=-1, stable=True, descending=False)
    
    nonzero_idx = nonzero_sorted_rowidx
    # print("nonzero_idx: ", nonzero_idx)    
    nonzero_idx[nonzero_val == 0] = -1
    # print("nonzero_idx: ", nonzero_idx)
    nonzero_idx = torch.flip(nonzero_idx, dims=[-1])
    # print("nonzero_idx: ", nonzero_idx)
    
    return nonzero_idx.contiguous().to(dtype=torch.int32)

def my_flash_attn(q, k, v, causal=True, softmax_scale=1.0):
    return flash_attention_v2_cutlass(q, k, v, causal, softmax_scale)

def my_varlen_flash_attn(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=True, softmax_scale=1.0):
    return varlen_flash_attention_v2_cutlass(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, softmax_scale)
    
def block_sparse_attention(q, k, v, mask, is_causal=True, sm_scale=1.0):
    row_mask = convert_blockmask_row_reverse(mask, is_causal)
    return flash_attention_block_v2_cutlass(q, k, v, row_mask, is_causal, sm_scale)

def varlen_block_sparse_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, mask, max_seqlen_q, max_seqlen_k, causal=True, softmax_scale=1.0):
    row_mask = convert_blockmask_row_reverse(mask, causal)
    return varlen_flash_attention_block_v2_cutlass(q, k, v, cu_seqlens_q, cu_seqlens_k, row_mask, max_seqlen_q, max_seqlen_k, causal, softmax_scale)

