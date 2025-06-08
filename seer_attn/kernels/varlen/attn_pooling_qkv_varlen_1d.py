import torch
import triton
import triton.language as tl
import math

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


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

@triton.jit
def _fwd_kernel_inner(
    l_i, m_i, acc,
    q,
    k_block_col_idx,
    k_ptrs, v_ptrs,
    R_ptrs,
    offs_m, offs_n,
    stride_kt, stride_vt,
    stride_rn,
    sm_scale,
    seqlen,
    LAST_K_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):

    start_n = (k_block_col_idx * BLOCK_N).to(tl.int64)

    if LAST_K_BLOCK:
        k = tl.load(k_ptrs + start_n * stride_kt,
                    mask=offs_n[None, :] + start_n < seqlen)
        v = tl.load(v_ptrs + start_n * stride_vt,
                    mask=offs_n[:, None] + start_n < seqlen)
    else:
        k = tl.load(k_ptrs + start_n * stride_kt)
        v = tl.load(v_ptrs + start_n * stride_vt)

    qk = tl.dot(q, k)
    qk *= sm_scale

    # the following is needed only when LAST_K_BLOCK or BLOCK_M < BLOCK_N
    if LAST_K_BLOCK :
        qk += tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), 0, float('-inf'))

    row_max = tl.max(qk, 1)
    R_block_ptr = R_ptrs + k_block_col_idx * stride_rn
    tl.store(R_block_ptr, row_max.to(q.dtype), mask=offs_m < seqlen)

    m_ij = tl.maximum(m_i, row_max)
    qk -= m_ij[:, None]
    p = tl.exp(qk)
    l_ij = tl.sum(p, 1)
    alpha = tl.exp(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    m_i = m_ij
    acc *= alpha[:, None]
    p = p.to(v.type.element_ty)
    acc += tl.dot(p, v)
    return l_i, m_i, acc



@triton.jit
def _fwd_kernel_varlen(
    Q, K, V, Pool1D, O,
    sm_scale,
    cu_seqlens,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_p1d_z, stride_p1d_h, stride_p1d_m, stride_p1d_n,
    stride_ot, stride_oh, stride_od,
    q_k_ratio,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):

    start_m = tl.program_id(0).to(tl.int64)
    off_h_q = tl.program_id(1).to(tl.int64)
    off_z = tl.program_id(2).to(tl.int64)

    off_h_for_kv = (off_h_q // q_k_ratio).to(tl.int64)


    cu_q_start = tl.load(cu_seqlens + off_z).to(tl.int64)
    cu_q_end = tl.load(cu_seqlens + off_z + 1).to(tl.int64)
    seqlen = cu_q_end - cu_q_start

    if start_m * BLOCK_M < seqlen:
        offs_m = (start_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
        offs_d = tl.arange(0, BLOCK_D).to(tl.int64)


        Q_ptrs = Q + cu_q_start * stride_qt + off_h_q * stride_qh
        K_ptrs = K + cu_q_start * stride_kt + off_h_for_kv * stride_kh
        V_ptrs = V + cu_q_start * stride_vt + off_h_for_kv * stride_vh
        O_ptrs = O + cu_q_start * stride_ot + off_h_q * stride_oh


        q = tl.load(Q_ptrs + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
                    mask=offs_m[:, None] < seqlen)


        k_block_start = 0
        k_block_end = tl.cdiv((start_m + 1) * BLOCK_M, BLOCK_N)


        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        padding_mask = offs_m >= seqlen
        m_i = tl.where(padding_mask, float("inf"), m_i) ## avoid nan in exp
        

        l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)

        k_ptrs = K_ptrs + offs_n[None, :] * stride_kt + offs_d[:, None] * stride_kd
        v_ptrs = V_ptrs + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd
        R_ptrs = Pool1D + off_z * stride_p1d_z + off_h_q * stride_p1d_h + offs_m * stride_p1d_m

        for k_block_col_idx in range(k_block_start, k_block_end - 1):
            l_i, m_i, acc = _fwd_kernel_inner(
                l_i, m_i, acc,
                q,
                k_block_col_idx,
                k_ptrs, v_ptrs,
                R_ptrs,
                offs_m, offs_n,
                stride_kt, stride_vt,
                stride_p1d_n,
                sm_scale,
                seqlen,
                False,
                BLOCK_M,
                BLOCK_N,
            )

        l_i, m_i, acc = _fwd_kernel_inner(
            l_i, m_i, acc,
            q,
            k_block_end - 1,
            k_ptrs, v_ptrs,
            R_ptrs,
            offs_m, offs_n,
            stride_kt, stride_vt,
            stride_p1d_n,
            sm_scale,
            seqlen,
            True,
            BLOCK_M,
            BLOCK_N,
        )

        Pool1D += off_z * stride_p1d_z + off_h_q * stride_p1d_h 
        for n in range(0, start_m + 1): ## causal only for now
            n = n.to(tl.int64)
            R_block_ptr = Pool1D + offs_m * stride_p1d_m +  n * stride_p1d_n
            row_max = tl.load(R_block_ptr, mask=offs_m < seqlen)
            rescaled_max = tl.exp(row_max - m_i) / l_i
            tl.store(R_block_ptr, rescaled_max.to(q.dtype), mask=offs_m < seqlen)

        acc = acc / l_i[:, None]
        acc = acc.to(O.type.element_ty)
        
        # Store O
        tl.store(O_ptrs + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od, acc,
                mask=offs_m[:, None] < seqlen)



def attn_pooling_qkv_varlen(
    q, k, v,# (#tokens, n_heads, head_size)
    cu_seqlens,
    max_seqlen,
    sm_scale,
    block_size=64,
):
    # split q to blocks
    _, n_heads, head_size = q.shape
    batch = cu_seqlens.size(0) - 1


    # print(f'> {q.shape=}, {k.shape=}')
    assert q.dim() == k.dim() == 3
    assert q.size(1) % k.size(1) == 0
    assert q.size(2) == k.size(2)
    assert cu_seqlens.dim() == 1
    assert cu_seqlens.size(0) == cu_seqlens.size(0)
    assert head_size in {64, 128, 256}

    k_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu()
    if max_seqlen:
        assert k_lens.max() <= max_seqlen

    q_k_ratio = q.size(1) // k.size(1)
    
    cu_seqlens = cu_seqlens.contiguous()
    cu_seqlens = cu_seqlens.contiguous()

    block_d = head_size
    num_blocks = triton.cdiv(max_seqlen, block_size)

    O = torch.zeros_like(q)

    n_blocks = triton.cdiv(max_seqlen, block_size)
    Pool1D = torch.full((batch, n_heads, max_seqlen, n_blocks), -65504.0, device=q.device, dtype=torch.bfloat16)

    grid = (num_blocks, n_heads, batch)
    
    with torch.cuda.device(q.device.index): 
        _fwd_kernel_varlen[grid](
            q, k, v, Pool1D, O,
            sm_scale,
            cu_seqlens,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *Pool1D.stride(),
            *O.stride(),
            q_k_ratio,
            BLOCK_M = block_size,
            BLOCK_N = block_size,
            BLOCK_D = block_d,
            num_warps = 4,
            num_stages = 1
        )
    
    Pool1D.clamp_(min=0.0)

    return O, Pool1D



if __name__ == "__main__":
    batch_size = 20
    n_heads = 32
    kv_heads = 16
    seq_len = 256
    head_size = 128
    block_size = 64
    # Compute scaled dot-product attention using Torch
    sm_scale = 1.0 / math.sqrt(head_size)

    q = torch.randn(batch_size, n_heads, seq_len, head_size, device='cuda', requires_grad=True, dtype=torch.bfloat16)
    k = torch.randn(batch_size, kv_heads, seq_len, head_size, device='cuda', requires_grad=True, dtype=torch.bfloat16)
    v = torch.randn(batch_size, kv_heads, seq_len, head_size, device='cuda', requires_grad=True, dtype=torch.bfloat16)

    q_varlen = q.detach().clone().requires_grad_().transpose(1, 2).reshape(batch_size * seq_len, n_heads, head_size)
    k_varlen = k.detach().clone().requires_grad_().transpose(1, 2).reshape(batch_size * seq_len, kv_heads, head_size)
    v_varlen = v.detach().clone().requires_grad_().transpose(1, 2).reshape(batch_size * seq_len, kv_heads, head_size)

    # Generate cumulative sequence lengths
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * seq_len, seq_len, device='cuda', dtype=torch.int32)
    # cu_seqlens_q = torch.tensor([0, 1024], device='cuda', dtype=torch.int32)
    cu_seqlens_k = cu_seqlens_q.clone()

    max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
    max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()


    attn_out, attn_pooling1d = attn_pooling_qkv_varlen(
        q_varlen, 
        k_varlen, 
        v_varlen,
        cu_seqlens_q, 
        max_seqlen_q, 
        sm_scale, 
        block_size,
    )

    if kv_heads < n_heads:
        k = repeat_kv(k, n_heads // kv_heads)
        v = repeat_kv(v, n_heads // kv_heads)


    attn_mask = torch.ones(seq_len, seq_len, device='cuda', dtype=torch.bool)
    attn_mask = torch.triu(attn_mask, diagonal=1)
    attn_score = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    attn_score.masked_fill_(attn_mask, float('-inf'))
    attn_score = attn_score.to(torch.float32)
    attn_score = attn_score.softmax(dim=-1)
    attn_score = attn_score.to(torch.bfloat16)
    attn_out_torch = torch.matmul(attn_score, v)
    attn_out_torch = attn_out_torch.transpose(1, 2).reshape(batch_size * seq_len, n_heads, head_size)
    attn_out_torch_sdpa = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True, dropout_p=0.0, scale=sm_scale
    )
    attn_out_torch_sdpa = attn_out_torch_sdpa.transpose(1, 2).reshape(batch_size * seq_len, n_heads, head_size)
    attn_out_torch_sdpa = attn_out_torch_sdpa.to(torch.bfloat16)
    max_diff_sdpa = torch.max(torch.abs(attn_out - attn_out_torch_sdpa))
    print(f"Max diff sdpa: {max_diff_sdpa.item()}")

    max_diff = torch.max(torch.abs(attn_out - attn_out_torch))
    print(f"Max diff: {max_diff.item()}")
    
    torch_pooling_1d = torch.max_pool2d(attn_score, kernel_size=(1, block_size), stride=(1, block_size), ceil_mode=True)
    max_diff_pool = torch.max(torch.abs(torch_pooling_1d - attn_pooling1d))
    print(f"Max diff: {max_diff_pool.item()}")
