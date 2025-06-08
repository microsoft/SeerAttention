import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from itertools import combinations
from flash_attn.bert_padding import index_put_first_axis 
from ..kernels.varlen.pooling_varlen import maxpool_varlen, avgpool_varlen
from ..modules.common import apply_rotary_pos_emb_single, RMSNorm

import os
import math


def minpool_varlen(x, cu_seqlen, max_seqlen, block_size):
    return -maxpool_varlen(-x, cu_seqlen, max_seqlen, block_size)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    if hidden_states.dim() == 3:
        num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, None, :, :].expand(num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(num_key_value_heads * n_rep, slen, head_dim)
        
    if hidden_states.dim() == 4:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)



class HeadPoolingLinear(nn.Module):
    def __init__(self, num_k_head, gqa_group_size, model_hidden_size, gate_hidden_size):
        super(HeadPoolingLinear, self).__init__()
        self.num_k_head = num_k_head
        self.gqa_group_size = gqa_group_size
        self.model_hidden_size = model_hidden_size
        self.gate_hidden_size = gate_hidden_size
        self.weight = nn.Parameter(torch.Tensor(self.num_k_head, gqa_group_size, self.model_hidden_size, self.gate_hidden_size))
        self._init_weight()

    def _init_weight(self):
        init.xavier_uniform_(self.weight)

    def forward(self, x): ## x shape (seq_length, num_q_head, channel_size)
        assert x.dim() == 3
        x = x.view(x.shape[0], self.num_k_head, self.gqa_group_size, x.shape[2])
        return torch.einsum('skgi,kgio->sko', x, self.weight)



class MultiHeadLinear(nn.Module):
    def __init__(self, in_channel_size, hidden_size, num_head):
        super(MultiHeadLinear, self).__init__()
        self.in_channel = in_channel_size
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.weight = nn.Parameter(torch.Tensor(self.num_head, self.in_channel, self.hidden_size))
        self._init_weight()
    

    def _init_weight(self):
        init.xavier_uniform_(self.weight)

    def forward(self, x): # x shape (seq_length, head, channel_size)
        # return torch.matmul(x, self.weight) 
        if x.dim() == 3:
            return torch.einsum('shi,hio->sho', x, self.weight)
        elif x.dim() == 4:
            return torch.einsum('bhsi,hio->bhso', x, self.weight)
        else:
            raise ValueError("x dim should be 3 or 4")

class AttnGate(nn.Module):
    def __init__(
            self, 
            block_size, 
            model_hidden_size, 
            gate_hidden_size, 
            num_k_head, 
            num_q_head, 
            q_head_pooling_type, 
            k_pooling_funcs,
            use_qk_norm,
        ):
        super(AttnGate, self).__init__()
        self.block_size = block_size
        self.model_hidden_size = model_hidden_size   
        self.gate_hidden_size = gate_hidden_size
        self.num_k_head = num_k_head
        self.num_q_head = num_q_head
        self.gqa_group_size = int(num_q_head // num_k_head)
        self.k_pooling_funcs = k_pooling_funcs
        self.use_qk_norm = use_qk_norm
    

        self.k_dup_size = len(k_pooling_funcs)
        k_in_channel_size = model_hidden_size * self.k_dup_size
        
        self.q_head_pooling_type = q_head_pooling_type
        
        if self.q_head_pooling_type == "Qproj":
            self.attngate_linear_q = HeadPoolingLinear(self.num_k_head, self.gqa_group_size, self.model_hidden_size, self.gate_hidden_size)
        elif self.q_head_pooling_type == "Qavgproj":
            self.attngate_linear_q = MultiHeadLinear(self.model_hidden_size, self.gate_hidden_size, self.num_k_head)
        else:
            self.attngate_linear_q = None
        self.attngate_linear_k = MultiHeadLinear(k_in_channel_size, self.gate_hidden_size, self.num_k_head)

        if self.use_qk_norm:
            self.attngate_qnorm = RMSNorm(self.gate_hidden_size, eps=1e-06)
            self.attngate_knorm = RMSNorm(self.gate_hidden_size, eps=1e-06)
        

    # q shape (nnz, num_q_head, channel_size)
    # k shape (nnz, num_k_head, channel_size)
    def forward(self, 
            q, 
            k,
            unpadded_lengths, 
            attention_mask,
            position_embeddings_gate_q=None,
            block_position_embeddings=None, 
            block_slice_mode=False,
        ):  
        
        cu_seqlens, max_seqlen, unpad_indices = unpadded_lengths
        max_seqlen_compressed = (max_seqlen + self.block_size - 1) // self.block_size

        bsz = cu_seqlens.shape[0] - 1


        if self.q_head_pooling_type == "Qavgproj" or self.q_head_pooling_type == "Qavg":
            q = F.avg_pool2d(q, kernel_size=[self.gqa_group_size, 1], stride=[self.gqa_group_size, 1])
        if self.q_head_pooling_type == "Qavgproj" or self.q_head_pooling_type == "Qproj":
            q = self.attngate_linear_q(q)

        ## transform layout to [b, head, seqlen, hidden_size]
        q = index_put_first_axis(q, unpad_indices, bsz * max_seqlen)
        q = q.view(bsz, max_seqlen, -1, q.size(-1))
        q = q.permute(0, 2, 1, 3).contiguous()

        if self.use_qk_norm:
            q = self.attngate_qnorm(q)

        ## when use Qorg, no need to reapply rope by passing the roped Q into gate.
        ## here for simplicity, we still apply the rotary embedding to Q but use the noped Q for gate.
        if position_embeddings_gate_q is not None:  
            cos, sin = position_embeddings_gate_q
            q = apply_rotary_pos_emb_single(q, cos, sin, unsqueeze_dim=1)

        k_pooled = [pool_func(k, cu_seqlens, max_seqlen, self.block_size) for pool_func in self.k_pooling_funcs] ## pooling change to batch layout
        k = torch.cat(k_pooled, dim=-1)        
        k = self.attngate_linear_k(k) ## [b, num_k_head, seqlen, hidden_size]

        if self.use_qk_norm:
            k = self.attngate_knorm(k)

        if block_position_embeddings is not None:
            cos, sin = block_position_embeddings
            k = apply_rotary_pos_emb_single(k, cos, sin, unsqueeze_dim=1)
        
        if self.q_head_pooling_type == "Qorig":
            k = repeat_kv(k, self.gqa_group_size)

        if block_slice_mode:
            q = q[:, :, self.block_size-1::self.block_size, :].contiguous()
            attention_mask = attention_mask[:, :, self.block_size-1::self.block_size, :].contiguous()

        attn = torch.matmul(q, k.transpose(-1, -2)) * (1 / math.sqrt(self.gate_hidden_size))

        if self.q_head_pooling_type == "Qorig":
            attn = attn.view(bsz, self.num_k_head, self.gqa_group_size, max_seqlen, max_seqlen_compressed).sum(dim=2, keepdim=False)
            attn.div_(self.gqa_group_size)

        if attention_mask.dtype == torch.bool:
            attn.masked_fill_(~attention_mask, -1e20)
        else:
            attn.add_(attention_mask)

        return attn



POOL_FUNCS = {
    'max': maxpool_varlen,
    'min': minpool_varlen,
    'avg': avgpool_varlen,
}


def _create_generic_attngate_class(base_class, suffix, k_pooling_names):
    k_pooling_funcs = [POOL_FUNCS[name] for name in k_pooling_names]
    class_name = f"K{''.join(k_pooling_names)}{suffix}"

    class NewAttnGate(base_class):
        def __init__(self, block_size, model_hidden_size, gate_hidden_size, num_k_head, num_q_head, q_head_pooling_type, use_qk_norm=False):
            super(NewAttnGate, self).__init__(
                block_size=block_size,
                model_hidden_size=model_hidden_size,
                gate_hidden_size=gate_hidden_size,
                num_k_head=num_k_head,
                num_q_head=num_q_head,
                q_head_pooling_type=q_head_pooling_type,
                k_pooling_funcs=k_pooling_funcs,
                use_qk_norm=use_qk_norm,
            )
    NewAttnGate.__name__ = class_name
    return class_name, NewAttnGate


def generate_combinations():
    new_classes = {}
    pool_types = ['max', 'min', 'avg']

    for k_comb in range(1, 4):
        for k_pooling_comb in combinations(pool_types, k_comb):
            class_name, new_class = _create_generic_attngate_class(AttnGate, '', k_pooling_comb)
            new_classes[class_name] = new_class
    return new_classes


ATTNGATE_CLASSES = generate_combinations()
print(ATTNGATE_CLASSES)