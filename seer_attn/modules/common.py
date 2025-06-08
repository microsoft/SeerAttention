import torch
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
import math
import torch.nn.functional as F
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"




def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_single(x, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
 
    rotary_dim = cos.shape[-1]
    if x.shape[-1] != rotary_dim:
        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        x_embed = torch.cat([(x_rot * cos) + (rotate_half(x_rot) * sin), x_pass], dim=-1)
    else:
        x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    if q.shape[-1] != rotary_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_embed = torch.cat([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], dim=-1)
        k_embed = torch.cat([(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], dim=-1)
    else:
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _get_unpad_data(attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Retrieves indexing data required to repad unpadded (ragged) tensors.

    Arguments:
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.

    Return:
        indices (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input sequence.
        cu_seqlens (`torch.Tensor`):
            The cumulative sequence lengths, used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        max_seqlen_in_batch (`int`):
            Maximum sequence length in batch.
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _upad_input(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
):
    """
    Unpads query, key, and values tensors, using a single dimension for all tokens even though they belong to different batches.

    This function is used instead of `flash_attn.bert_padding.unpad_input` in order to avoid the recomputation of the same intermediary
    tensors for query, key, value tensors.

    Arguments:
        query_layer (`torch.Tensor`):
            Query state with padding. Shape: (batch_size, query_length, num_heads, head_dim).
        key_layer (`torch.Tensor`):
            Key state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        value_layer (`torch.Tensor`):
            Value state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.
        query_length (`int`):
            Target length.

    Return:
        query_layer (`torch.Tensor`):
            Query state without padding. Shape: (total_target_length, num_heads, head_dim).
        key_layer (`torch.Tensor`):
            Key state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        value_layer (`torch.Tensor`):
            Value state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        indices_q (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input target sequence.
        (cu_seqlens_q, cu_seqlens_k) (`Tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`Tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query, `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k)
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )
    if query_length == kv_seq_len:
        query_layer = index_first_axis(query_layer.reshape(batch_size * kv_seq_len, -1, head_dim), indices_k)
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        # The -q_len: slice assumes left padding.
        attention_mask = attention_mask[:, -query_length:]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q, *_ = unpad_input(query_layer, attention_mask)

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )



@torch.jit.script
def repeat_kv_varlen(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    final_shape = list(hidden_states.shape[:-2]) + [-1] + [hidden_states.shape[-1]]
    expand_shape = [-1] * (len(hidden_states.shape) - 1) + [n_rep] + [-1]
    hidden_states = hidden_states.unsqueeze(-2).expand(expand_shape)
    return hidden_states.reshape(final_shape)



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



def get_sparse_attn_mask_from_nz_ratio(x, nz_ratio, use_dense_for_last_block=False):
    bsz, num_head, downsample_len, _ = x.shape
    topk_nz_ratio = 1 - math.sqrt(1 -nz_ratio)
    topk = int(topk_nz_ratio * downsample_len)
    topk = 1 if topk == 0 else topk
    sparse_index = torch.topk(x, topk, dim=-1).indices
    dense_mask = torch.full([bsz, num_head, downsample_len, downsample_len], False, dtype=torch.bool, device=x.device)
    dense_mask.scatter_(-1, sparse_index, True)
    if use_dense_for_last_block:
        dense_mask[:, :,-2:,:] = True 
    dense_mask.tril_()
    return  dense_mask


def get_sparse_attn_mask_from_threshold(x, threshold, use_dense_for_last_block=False):
    dense_mask = x > threshold 
    if use_dense_for_last_block:
        dense_mask[:, :,-2:,:] = True 
    dense_mask.tril_()
    return  dense_mask


def get_indice(maxseq, cu_seqlen):
    assert cu_seqlen.dim() == 1, "cu_seqlen must be a 1D tensor"
    bsz = cu_seqlen.size(0) - 1
    lengths = cu_seqlen[1:] - cu_seqlen[:-1]
    maxseq = lengths.max().item()
    device = cu_seqlen.device
    nnz = cu_seqlen[-1].item()
    # Compute batch indices for each element in the input tensor
    batch_idx = torch.repeat_interleave(torch.arange(bsz, device=device), lengths)

    # Compute start indices for each batch and repeat them according to their lengths
    start_indices = cu_seqlen[:-1]
    batch_start = torch.repeat_interleave(start_indices, lengths)

    # Calculate sequence positions within each batch
    seq_pos = torch.arange(nnz, device=device) - batch_start
    # Generate the indices in the flattened padded tensor
    indices = batch_idx * maxseq + seq_pos
    return indices

def compute_oracle_sparse_mask(q, k, attention_mask, block_size, threshold):
    batch_size, q_len, num_q_heads, head_dim = q.shape
    _, kv_len, num_kv_heads, _ = k.shape
    num_gqa_groups = num_q_heads // num_kv_heads
    
    if q_len > 1: # use dense prefill for sparse decode
        block_sparse_mask = None
    else:
        q = q.transpose(1, 2) # (b, num_q_heads, q_len, d)
        k = k.transpose(1, 2) # (b, num_kv_heads, kv_len, d)

        # Repeat K heads for GQA compatibility
        if num_gqa_groups > 1:
            k = repeat_kv(k, num_gqa_groups)

        attn_weights = torch.einsum('bhid, bhdj -> bhij', q, k.transpose(-1, -2)) * (head_dim**-0.5) # (b, num_q_heads, q_len, kv_len)

        #q_len=1, (b, num_q_heads, q_len, kv_len) -> (b, num_q_heads, kv_len), as block_sparse_mask must in bhs
        attn_weights = attn_weights.squeeze(2)
        
        #from 2D to 3D, (b,kv_len) -> (b,1,kv_len)
        attention_mask=attention_mask.unsqueeze(1)
        # note attention_mask is in int/float, with 1 means True.
        # our block attention_mask is in bool.
        attn_weights = attn_weights.masked_fill(~attention_mask.bool(), float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)

        attn_weights = F.max_pool2d(attn_weights, kernel_size=(num_gqa_groups, block_size), stride=(num_gqa_groups, block_size), ceil_mode=True)

        block_sparse_mask = attn_weights > threshold  
    
    return block_sparse_mask

