import os
from typing import Optional, TypedDict

import torch
import torch.nn.functional as F

from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
from seer_attn.kernels.varlen.flash_decode_varlen_left_pad_max_v2 import flash_decode_leftpad
from seer_attn.kernels.varlen.block_sparse_attn_varlen_2d_leftpad import blocksparse_flash_attn_varlen_leftpad
from seer_attn.kernels.block_sparse_attn import block_sparse_triton_fn
import os
import math

from seer_attn.modules.common import (
    repeat_kv_varlen,
    repeat_kv,
    pad_input,
    _upad_input,
    get_sparse_attn_mask_from_nz_ratio,
    get_sparse_attn_mask_from_threshold,
)

def sparse_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    softmax_scale: Optional[float] = None,
    attn_gate_score: Optional[torch.Tensor] = None,
    sparsity_method: Optional[str] = None,
    threshold: Optional[float] = None,
    nz_ratio: Optional[float] = None,
    last_block_dense: Optional[bool] = None,
    block_size: Optional[int] = None,
    num_key_value_groups: Optional[int] = None,
    profile_file: Optional[str] = None,
    **kwargs,
):


    if query_length > 1:
        if sparsity_method == "nz_ratio":
            downsampled_len = math.ceil(key_states.shape[-2] / block_size)
            gate_mask = get_sparse_attn_mask_from_nz_ratio(attn_gate_score, nz_ratio, last_block_dense)
        elif sparsity_method == "threshold":
            gate_mask = get_sparse_attn_mask_from_threshold(attn_gate_score, threshold, last_block_dense)
            if profile_file is not None:
                downsampled_len = gate_mask.shape[-1]
                total_causal_size = ((1 + downsampled_len) * downsampled_len / 2) * gate_mask.shape[0] * gate_mask.shape[1]
                with open(profile_file, "a") as f:
                    f.write(f"{query_length}: {gate_mask.sum().item() / total_causal_size}\n")

        if attention_mask is None:
            query_states = query_states.transpose(1, 2).contiguous()
            key_states = key_states.transpose(1, 2).contiguous()
            value_states = value_states.transpose(1, 2).contiguous()
            key_states = repeat_kv(key_states, num_key_value_groups)
            value_states = repeat_kv(value_states, num_key_value_groups)
            attn_output = block_sparse_triton_fn( 
                query_states,
                key_states,
                value_states,
                block_sparse_mask=gate_mask,
                sm_scale=softmax_scale,
                BLOCK_M=block_size,
                BLOCK_N=block_size,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()

        else:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            key_states = repeat_kv_varlen(key_states, num_key_value_groups)
            value_states = repeat_kv_varlen(value_states, num_key_value_groups)

            attn_output_unpad = blocksparse_flash_attn_varlen_leftpad( ## assume leftpad of gate_mask
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen=max_seqlen_in_batch_q,
                block_mask=gate_mask,
                sm_scale=softmax_scale,
                output_lse=False,
            )
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)

    else:

        if attention_mask is None:
            cache_seqlens = torch.full(
                (key_states.shape[0],), key_states.shape[1], dtype=torch.int32, device=key_states.device)
        else:
            cache_seqlens = torch.sum(attention_mask.to(torch.int32), dim=-1, dtype=torch.int32) 

        attn_output = flash_decode_leftpad(
            query_states, 
            key_states,
            value_states, 
            cache_seqlens=cache_seqlens, 
            block_size=block_size,
            sm_scale=softmax_scale,
        )

    return attn_output


