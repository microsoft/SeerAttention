import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from itertools import combinations
from flash_attn.bert_padding import index_put_first_axis 
# from seer_attn.kernels.pooling_varlen_bshd import maxpool_varlen_leftpad, avgpool_varlen_leftpad
from flash_attn.layers.rotary import apply_rotary_emb_func


import os
import math
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_single(x, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

def min_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
    return -F.max_pool3d(-input, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def get_sparse_attn_mask_from_threshold(x, threshold, sliding_window_size, block_attention_mask):
    block_seq_len = x.size(-1)
    
    if block_seq_len <= sliding_window_size:
        full_mask = torch.ones_like(x, dtype=torch.bool, device=x.device)
        full_mask = full_mask & block_attention_mask
        return full_mask

    final_mask = x > threshold

    if sliding_window_size > 0:
        final_mask[..., -sliding_window_size:] = True

    final_mask = final_mask & block_attention_mask

    return final_mask


def get_sparse_attn_mask_from_budget(x, block_budget, sliding_window_size, block_attention_mask):
    block_seq_len = x.size(-1)
    
    if block_seq_len <= block_budget:
        full_mask = torch.ones_like(x, dtype=torch.bool, device=x.device)
        full_mask = full_mask & block_attention_mask
        return full_mask

    k = block_budget - sliding_window_size
    
    mask_sliding = torch.zeros_like(x, dtype=torch.bool, device=x.device)
    if sliding_window_size > 0:
        mask_sliding[..., -sliding_window_size:] = True
    
    if k <= 0:
        final_mask = mask_sliding
    else:
        modified_x = x.masked_fill(mask_sliding, float('-inf'))
        _, topk_indices = torch.topk(modified_x, k=k, dim=-1, sorted=False)
        
        mask_extra = torch.zeros_like(x, dtype=torch.bool, device=x.device)
        mask_extra.scatter_(-1, topk_indices, True)
        
        final_mask = mask_sliding | mask_extra

    final_mask = final_mask & block_attention_mask
    
    return final_mask

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

    def forward(self, x): 
        if x.dim() == 3: ## x shape (seq_length, num_q_head, channel_size)
            x = x.view(x.shape[0], self.num_k_head, self.gqa_group_size, x.shape[2])
            return torch.einsum('skgi,kgio->sko', x, self.weight)
        elif x.dim() == 4: ## x shape (b, seq_length, num_q_head, channel_size)
            x = x.view(x.shape[0], x.shape[1], self.num_k_head, self.gqa_group_size, x.shape[3])
            return torch.einsum('bskgi,kgio->bsko', x, self.weight)
        else:
            raise ValueError("x dim should be 3 or 4")


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
            return torch.einsum('bshi,hio->bsho', x, self.weight)
            # return torch.einsum('bhsi,hio->bhso', x, self.weight)
        else:
            raise ValueError("x dim should be 3 or 4")

class AttnGate(nn.Module):
    def __init__(self, 
                 block_size, 
                 model_hidden_size, 
                 gate_hidden_size, 
                 num_k_head, 
                 num_q_head, 
                 q_head_pooling_type, 
                 k_pooling_funcs,
                 use_flash_rope,

                ):
        super(AttnGate, self).__init__()
        self.block_size = block_size
        self.model_hidden_size = model_hidden_size   
        self.gate_hidden_size = gate_hidden_size
        self.num_k_head = num_k_head
        self.num_q_head = num_q_head
        self.gqa_group_size = int(num_q_head // num_k_head)
        self.k_pooling_funcs = k_pooling_funcs
        self.use_flash_rope = use_flash_rope
    

        self.k_dup_size = len(k_pooling_funcs)
        k_in_channel_size = model_hidden_size * self.k_dup_size
        
        self.q_head_pooling_type = q_head_pooling_type
        
        if self.q_head_pooling_type == "Qproj":
            self.mask_linear_q = HeadPoolingLinear(self.num_k_head, self.gqa_group_size, self.model_hidden_size, self.gate_hidden_size)
        elif self.q_head_pooling_type == "Qavgproj":
            self.mask_linear_q = MultiHeadLinear(self.model_hidden_size, self.gate_hidden_size, self.num_k_head)
        else:
            self.mask_linear_q = None
        self.mask_linear_k = MultiHeadLinear(k_in_channel_size, self.gate_hidden_size, self.num_k_head)


    def forward(self, 
            k, # [b, klen, k_head, head_dim]
            layer_idx,
            k_compressed_cache,
            q=None, #[b, 1, q_head, head_dim]
            attention_mask=None, # [b, 1, klen]
            max_cache_len=None,
            position_embeddings=None,
            block_position_embeddings=None, 
            sparsity_method=None,
            threshold=0.0,
            block_budget=32,
            block_sliding_window_size=0,
        ):  
        """
        This attngate module is only used in inference. 
        Args:
            k (torch.Tensor): Key tensor of shape (batch_size, num_key_heads, seqlen, head_dim).
            layer_idx (int): Layer index.
            k_compressed_cache (Cache): Cache object for key compressed states.
            q (torch.Tensor): Query tensor of shape (batch_size, 1, num_query_heads, head_dim).
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, 1, 1, seqlen).
            max_cache_len (int): Maximum cache length.
            position_embeddings: Position embeddings for query tensor.
            block_position_embeddings: Position embeddings for key tensor.
            threshold: Threshold for attention mask.
        """

        is_decode = k.shape[1] == 1 
        batch_size, _, num_kv_heads, _ = k.shape 
        kv_len = attention_mask.shape[-1]     

        if is_decode:
            assert q.dim() == 4

            if self.q_head_pooling_type == "Qavgproj" or self.q_head_pooling_type == "Qavg":
                q = F.avg_pool2d(q, kernel_size=[self.gqa_group_size, 1], stride=[self.gqa_group_size, 1])
            if self.q_head_pooling_type == "Qavgproj" or self.q_head_pooling_type == "Qproj":
                q = self.mask_linear_q(q)

            cos, sin = position_embeddings
            if self.use_flash_rope:
                q = apply_rotary_emb_func(q, cos, sin, False, True, cu_seqlens=None, max_seqlen=1)
            else:
                q = apply_rotary_pos_emb_single(q, cos, sin, unsqueeze_dim=2)

            k = k_compressed_cache.update(k=k, layer_idx=layer_idx, is_decode=is_decode)

            if max_cache_len % self.block_size == 0:
                remainder = k_compressed_cache.get_k_remainder(layer_idx)
                k_compressed = [pool_func(remainder, kernel_size=[self.block_size, 1, 1], stride=[self.block_size, 1, 1], ceil_mode=True) for pool_func in self.k_pooling_funcs]
                k_compressed = torch.cat(k_compressed, dim=-1)        
                k_compressed = self.mask_linear_k(k_compressed)
                cos, sin = block_position_embeddings
                if self.use_flash_rope:
                    k = apply_rotary_emb_func(k_compressed, cos, sin, False, True, cu_seqlens=None, max_seqlen=1)
                else:
                    k = apply_rotary_pos_emb_single(k_compressed, cos, sin, unsqueeze_dim=2)
                k_compressed = k_compressed_cache.update(k_compressed=k_compressed, layer_idx=layer_idx, is_decode=is_decode)


            q = q.squeeze(1) ## currently only for khead size of q


            attn = torch.einsum('bhd,bshd->bhs', q, k)
            attn = attn * (1 / math.sqrt(self.gate_hidden_size))
            if attention_mask.dtype == torch.bool:
                attn = attn.masked_fill(~attention_mask, -1e20)
            else:
                attn = attn + attention_mask
            attn = F.softmax(attn, dim=-1)
            

            if sparsity_method == "token_budget":
                mask = get_sparse_attn_mask_from_budget(attn, block_budget, block_sliding_window_size, attention_mask)
            elif sparsity_method == "threshold":
                mask = get_sparse_attn_mask_from_threshold(attn, threshold, block_sliding_window_size, attention_mask)
            mask[:, : ,-1] = True

            return mask
        else:
            k_pooled = [pool_func(k, kernel_size=[self.block_size, 1, 1], stride=[self.block_size, 1, 1], ceil_mode=True) for pool_func in self.k_pooling_funcs]
            k_pooled = torch.cat(k_pooled, dim=-1)        
            k_compressed = self.mask_linear_k(k_pooled)
            cos, sin = block_position_embeddings
            if self.use_flash_rope:
                k_compressed = apply_rotary_emb_func(k_compressed, cos, sin, False, True, cu_seqlens=None, max_seqlen=1)
            else:
                k_compressed = apply_rotary_pos_emb_single(k_compressed, cos, sin, unsqueeze_dim=2)
            num_valid_blocks = max_cache_len // self.block_size
            num_remainder = max_cache_len % self.block_size
            if num_remainder > 0:
                k_remainder = k[:, num_valid_blocks * self.block_size :, :, :]
                k_compressed[:, -1, :, :] = 0.0
            else:
                k_remainder = None
            k = k_compressed_cache.update(layer_idx=layer_idx, k_compressed=k_compressed, k_remainder=k_remainder, is_decode=is_decode)
            return None



POOL_FUNCS = {
    'max': F.max_pool3d,
    'min': min_pool3d,
    'avg': F.avg_pool3d,
}


def _create_generic_attngate_class(base_class, suffix, k_pooling_names):
    k_pooling_funcs = [POOL_FUNCS[name] for name in k_pooling_names]
    class_name = f"K{''.join(k_pooling_names)}{suffix}"

    class NewAttnGate(base_class):
        def __init__(self, block_size, model_hidden_size, gate_hidden_size, num_k_head, num_q_head, q_head_pooling_type, use_flash_rope=False):
            super(NewAttnGate, self).__init__(
                block_size=block_size,
                model_hidden_size=model_hidden_size,
                gate_hidden_size=gate_hidden_size,
                num_k_head=num_k_head,
                num_q_head=num_q_head,
                q_head_pooling_type=q_head_pooling_type,
                k_pooling_funcs=k_pooling_funcs,
                use_flash_rope=use_flash_rope,
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
# print(ATTNGATE_CLASSES)