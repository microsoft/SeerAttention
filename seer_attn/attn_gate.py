
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from itertools import combinations
from seer_attn.modules.common import apply_rotary_pos_emb, repeat_kv, repeat_kv_varlen
from flash_attn.layers.rotary import apply_rotary_emb_func




def min_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
    return -F.max_pool3d(-input, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode)



class MultiHeadLinear(nn.Module):
    def __init__(self, in_channel_size, hidden_size, num_head):
        super(MultiHeadLinear, self).__init__()
        self.in_channel = in_channel_size
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.weight = nn.Parameter(torch.Tensor(self.num_head, self.in_channel, self.hidden_size))
    

    def forward(self, x): # x shape (batch_size, seq_length, head, channel_size)
        if x.shape[2] < self.num_head:
            x = repeat_kv_varlen(x, self.num_head // x.shape[2])
        # print(f"x.shape: {x.shape}, self.weight.shape: {self.weight.shape}")
        return torch.einsum('bshi, hio->bsho', x, self.weight) # torch.matmul(x, self.weight)
        # return torch.matmul(x, self.weight) # torch.einsum('bhsi,hio->bhso', x, self.weight)


class AttnGate(nn.Module):
    def __init__(self, block_size, in_channel_size, hidden_size, num_k_head, num_q_head, q_pooling_funcs, k_pooling_funcs, force_double=False, use_flash_rope=False):
        super(AttnGate, self).__init__()
        self.block_size = block_size
        self.in_channel = in_channel_size
        self.hidden_size = hidden_size
        self.num_k_head = num_k_head
        self.num_q_head = num_q_head
        self.use_flash_rope = use_flash_rope
        self.q_pooling_funcs = q_pooling_funcs
        self.k_pooling_funcs = k_pooling_funcs
        self.scale = self.hidden_size ** -0.5

        self.q_dup_size = len(q_pooling_funcs)
        self.k_dup_size = len(k_pooling_funcs)

        q_in_channel_size = in_channel_size * self.q_dup_size
        k_in_channel_size = in_channel_size * self.k_dup_size
        
        
        if self.q_dup_size > 1 or self.hidden_size != in_channel_size or force_double:
            self.mask_linear_q = MultiHeadLinear(q_in_channel_size, self.hidden_size, self.num_q_head)
            self.mask_linear_k = MultiHeadLinear(k_in_channel_size, self.hidden_size, self.num_k_head)
        else: # Can use a single linear layer if hidden_size = in_channel_size
            self.mask_linear_q = None
            self.mask_linear_k = MultiHeadLinear(k_in_channel_size, self.hidden_size, self.num_q_head)

    
    def forward(
            self, 
            q, # [batch_size, seq_length, num_q_head, channel_size]
            k, # [batch_size, seq_length, num_k_head, channel_size]
            attention_mask, 
            position_embeddings=None, 
            use_softmax=True
        ):  
        q_len = q.shape[1]
        if q_len == 1:
            return None

        q_pooled = [pool_func(q, kernel_size=[self.block_size, 1, 1], stride=[self.block_size, 1, 1], ceil_mode=True) for pool_func in self.q_pooling_funcs]
        q = torch.cat(q_pooled, dim=-1)
        if self.mask_linear_q is not None:
            q = self.mask_linear_q(q)

        k_pooled = [pool_func(k, kernel_size=[self.block_size, 1, 1], stride=[self.block_size, 1, 1], ceil_mode=True) for pool_func in self.k_pooling_funcs]
        k = torch.cat(k_pooled, dim=-1)
        k = self.mask_linear_k(k)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            if self.use_flash_rope:
                q = apply_rotary_emb_func(q, cos, sin, False, True, cu_seqlens=None, max_seqlen=q_len)
                k = apply_rotary_emb_func(k, cos, sin, False, True, cu_seqlens=None, max_seqlen=q_len)
            else:
                q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

        q, k = q.transpose(1, 2), k.transpose(1, 2)

        if k.shape[1] < self.num_q_head:
            k = repeat_kv(k, self.num_q_head // k.shape[1])
        
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # print("attn", attn, "mask", attention_mask)
        if attention_mask.dtype == torch.bool:
            attn = attn.masked_fill(~attention_mask, -1e9)
        else:
            attn = attn + attention_mask
        if use_softmax:
            attn = F.softmax(attn, dim=-1)
        return attn


POOL_FUNCS = {
    'max': F.max_pool3d,
    'min': min_pool3d,
    'avg': F.avg_pool3d
}


def _create_generic_attngate_class(base_class, suffix, q_pooling_names, k_pooling_names):
    q_pooling_funcs = [POOL_FUNCS[name] for name in q_pooling_names]
    k_pooling_funcs = [POOL_FUNCS[name] for name in k_pooling_names]
    class_name = f"Q{''.join(q_pooling_names)}_K{''.join(k_pooling_names)}{suffix}"

    class NewAttnGate(base_class):
        def __init__(self, block_size, in_channel_size, hidden_size, num_k_head, num_q_head, force_double=False, use_flash_rope=False):
            super(NewAttnGate, self).__init__(
                block_size=block_size,
                in_channel_size=in_channel_size,
                hidden_size=hidden_size,
                num_k_head=num_k_head,
                num_q_head=num_q_head,
                q_pooling_funcs=q_pooling_funcs,
                k_pooling_funcs=k_pooling_funcs,
                force_double=force_double,
                use_flash_rope=use_flash_rope
            )
    NewAttnGate.__name__ = class_name
    return class_name, NewAttnGate


def generate_combinations():
    new_classes = {}
    pool_types = ['max', 'min', 'avg']

    for q_comb in range(1, 4):
        for k_comb in range(1, 4):
            for q_pooling_comb in combinations(pool_types, q_comb):
                for k_pooling_comb in combinations(pool_types, k_comb):
                    class_name, new_class = _create_generic_attngate_class(AttnGate, '', q_pooling_comb, k_pooling_comb)
                    new_classes[class_name] = new_class
    return new_classes


ATTNGATE_CLASSES = generate_combinations()