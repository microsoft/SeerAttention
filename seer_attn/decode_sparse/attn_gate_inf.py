import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from itertools import combinations
from flash_attn.bert_padding import index_put_first_axis 
# from seer_attn.kernels.pooling_varlen_bshd import maxpool_varlen_leftpad, avgpool_varlen_leftpad
from flash_attn.layers.rotary import apply_rotary_emb_func
from seer_attn.modules.common import apply_rotary_pos_emb_single, RMSNorm, repeat_kv, repeat_kv_varlen
from seer_attn.kernels.varlen.oracle_sparse import oracle_sparse


import math


def min_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
    return -F.max_pool3d(-input, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode)



def get_sparse_attn_mask_from_threshold(x, threshold):
    dense_mask = x > threshold 
    return  dense_mask

def get_sparse_attn_mask_from_budget(x, block_budget, block_attention_mask):
    block_seq_len = x.size(-1)
    
    if block_seq_len <= block_budget:
        full_mask = torch.ones_like(x, dtype=torch.bool, device=x.device)
        full_mask = full_mask & block_attention_mask
        return full_mask

    k = block_budget 
    
    _, topk_indices = torch.topk(x, k=k, dim=-1, sorted=False)
    
    mask = torch.zeros_like(x, dtype=torch.bool, device=x.device)
    mask.scatter_(-1, topk_indices, True)

    final_mask = mask & block_attention_mask
    
    return final_mask

def compute_oracle_sparse_mask(q, k, cache_seqlens, block_attention_mask, block_size, sparsity_method, threshold=0.0, block_budget=2048):
    #batch_size, q_len, num_q_heads, head_dim = q.shape
    q_len = q.shape[1]
    if q_len > 1: # use dense prefill for sparse decode
        block_sparse_mask = None
    else:

        attn_weights = oracle_sparse(q, k, cache_seqlens, block_size)
        
        if sparsity_method == "token_budget":
            block_sparse_mask = get_sparse_attn_mask_from_budget(attn_weights, block_budget, block_attention_mask)
        elif sparsity_method == "threshold":
            block_sparse_mask = get_sparse_attn_mask_from_threshold(attn_weights, threshold) 

        block_sparse_mask[:, :, -1] = True
    
    return block_sparse_mask


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
        self.use_flash_rope = use_flash_rope
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
        

    def forward(self, 
            k, # [b, klen, k_head, head_dim]
            layer_idx,
            k_compressed_cache,
            q=None, #[b, 1, q_head, head_dim]
            attention_mask=None, # [b, 1, klen]
            max_cache_len=None,
            position_embeddings=None,
            block_position_embeddings=None, 
            threshold=0.0,
            block_budget=None,
            sparsity_method="threshold",
        ):  

        is_decode = k.shape[1] == 1        

        if is_decode:
            assert q.dim() == 4

            if self.q_head_pooling_type == "Qavgproj" or self.q_head_pooling_type == "Qavg":
                q = F.avg_pool2d(q, kernel_size=[self.gqa_group_size, 1], stride=[self.gqa_group_size, 1])
            if self.q_head_pooling_type == "Qavgproj" or self.q_head_pooling_type == "Qproj":
                q = self.attngate_linear_q(q)

            if self.use_qk_norm:
                q = self.attngate_qnorm(q)

            if position_embeddings is not None:
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
                k_compressed = self.attngate_linear_k(k_compressed) ## [b, 1, k_head, dim]
                

                if self.use_qk_norm:
                    k_compressed = self.attngate_knorm(k_compressed)

                if position_embeddings is not None:
                    cos, sin = position_embeddings ## change to positional embedding instead of block_position_embeddings
                    if self.use_flash_rope:
                        k_compressed = apply_rotary_emb_func(k_compressed, cos, sin, False, True, cu_seqlens=None, max_seqlen=1)
                    else:
                        k_compressed = apply_rotary_pos_emb_single(k_compressed, cos, sin, unsqueeze_dim=2)
                k = k_compressed_cache.update(k_compressed=k_compressed, layer_idx=layer_idx, is_decode=is_decode)


            q = q.squeeze(1) 

            if self.q_head_pooling_type == "Qorig":
                q = q.view(q.shape[0], self.num_k_head, self.gqa_group_size, q.shape[2])
                attn = torch.einsum('bkgd,bskd->bks', q, k)
                scale = 1 / (math.sqrt(self.gate_hidden_size) * self.gqa_group_size)
                attn.mul_(scale)
            else: 
                attn = torch.einsum('bhd,bshd->bhs', q, k)
                attn = attn * (1 / math.sqrt(self.gate_hidden_size))


            if attention_mask.dtype == torch.bool:
                attn = attn.masked_fill(~attention_mask, -1e20)
            else:
                attn = attn + attention_mask
            attn = F.softmax(attn, dim=-1)
            if sparsity_method == "token_budget":
                mask = get_sparse_attn_mask_from_budget(attn, block_budget, attention_mask)
            elif sparsity_method == "threshold":
                mask = get_sparse_attn_mask_from_threshold(attn, threshold)
            mask[:, : ,-1] = True
            
            return mask
        else:
            if k.shape[1] >= self.block_size:
                k_pooled = [pool_func(k, kernel_size=[self.block_size, 1, 1], stride=[self.block_size, 1, 1], ceil_mode=True) for pool_func in self.k_pooling_funcs]
            else:
                k_pooled = [pool_func(k, kernel_size=[k.shape[1], 1, 1], stride=[k.shape[1], 1, 1], ceil_mode=True) for pool_func in self.k_pooling_funcs]
            k_pooled = torch.cat(k_pooled, dim=-1)        
            k_compressed = self.attngate_linear_k(k_pooled)
            if self.use_qk_norm:
                k_compressed = self.attngate_knorm(k_compressed)

            if block_position_embeddings is not None:
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
        def __init__(self, block_size, model_hidden_size, gate_hidden_size, num_k_head, num_q_head, q_head_pooling_type, use_flash_rope=False, use_qk_norm=False):
            super(NewAttnGate, self).__init__(
                block_size=block_size,
                model_hidden_size=model_hidden_size,
                gate_hidden_size=gate_hidden_size,
                num_k_head=num_k_head,
                num_q_head=num_q_head,
                q_head_pooling_type=q_head_pooling_type,
                k_pooling_funcs=k_pooling_funcs,
                use_flash_rope=use_flash_rope,
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
# print(ATTNGATE_CLASSES)