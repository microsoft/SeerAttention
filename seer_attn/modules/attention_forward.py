import os
from typing import Optional, TypedDict

import torch
import torch.nn.functional as F

from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
from seer_attn.kernels.varlen.flash_decode_varlen_leftpad import flash_decode_leftpad
from seer_attn.kernels.varlen.block_sparse_attn_varlen_2d import blocksparse_flash_attn_varlen_fwd
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

        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            k = repeat_kv_varlen(k, num_key_value_groups)
            v = repeat_kv_varlen(v, num_key_value_groups)

            attn_output_unpad = blocksparse_flash_attn_varlen_fwd(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen=max_seqlen_in_batch_q,
                block_mask=gate_mask,
                sm_scale=softmax_scale,
            )
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            print("query_states:", query_states.shape)
            print("key_states:", key_states.shape)
            print("value_states:", value_states.shape)
            print("gate_mask:", gate_mask.shape)
            print("sm_scale:", softmax_scale)

            query_states = query_states.transpose(1, 2).contiguous()
            key_states = key_states.transpose(1, 2).contiguous()
            value_states = value_states.transpose(1, 2).contiguous()
            key_states = repeat_kv(key_states, num_key_value_groups)
            value_states = repeat_kv(value_states, num_key_value_groups)


            causal_mask = torch.kron(
                gate_mask, torch.ones((block_size, block_size), device=gate_mask.device, dtype=gate_mask.dtype)
            )
           
            causal_mask = torch.tril(causal_mask, diagonal=0)
            print("causal_mask:", causal_mask)
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                is_causal=True,
                attn_mask=causal_mask,
            )

            # attn_output = block_sparse_triton_fn(
            #     query_states,
            #     key_states,
            #     value_states,
            #     block_sparse_mask=gate_mask,
            #     sm_scale=softmax_scale,
            #     BLOCK_M=block_size,
            #     BLOCK_N=block_size,
            # )
    else:

        cache_seqlens = torch.sum(attention_mask.to(torch.int32), dim=-1, dtype=torch.int32) 
        if max_cache_len is None:
            max_cache_len = cache_seqlens.max().item()

        attn_output = flash_decode_leftpad(
            query_states, 
            key_states,
            value_states, 
            cache_seqlens=cache_seqlens, 
            block_size=block_size,
            sm_scale=softmax_scale,
        )

    return attn_output


