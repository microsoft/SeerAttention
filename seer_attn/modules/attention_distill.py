from typing import Optional, TypedDict

import torch
import torch.nn.functional as F
from seer_attn.kernels.attn_pooling_kernel import attn_with_pooling

from seer_attn.modules.common import (
    repeat_kv,
)

def attention_distill_forward(
    query_states: torch.Tensor, ## [batch, seq_len, num_heads, head_dim]
    key_states: torch.Tensor, ## [batch, seq_len, num_heads, head_dim]
    value_states: torch.Tensor, ## [batch, seq_len, num_heads, head_dim]
    softmax_scale: Optional[float] = None,
    block_size: Optional[int] = None,
    num_key_value_groups: Optional[int] = 1,
    **kwargs,
):

    query_states = query_states.transpose(1, 2).contiguous()
    key_states = key_states.transpose(1, 2).contiguous()
    value_states = value_states.transpose(1, 2).contiguous()


    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)
    

    attn_output, mask_ground_truth = attn_with_pooling(
        query_states,
        key_states,
        value_states,
        True, 
        softmax_scale,
        block_size,      
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, mask_ground_truth




