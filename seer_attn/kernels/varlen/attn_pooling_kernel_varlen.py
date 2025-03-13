import torch
import triton
import triton.language as tl

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

@triton.jit
def _fwd_kernel_inner(
    l_i, m_i,
    q,
    k_block_col_idx,
    k_ptrs,
    R_ptrs,
    offs_m, offs_n,
    stride_kt, 
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
    else:
        k = tl.load(k_ptrs + start_n * stride_kt)

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
    return l_i, m_i



@triton.jit
def _fwd_kernel_varlen(
    Q, K, Po,
    sm_scale,
    cu_seqlens,
    d_R_ptrs,
    d_R_sizes,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_pz, stride_ph, stride_pm, stride_pn,
    q_k_ratio,
    batch_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):

    start_m = tl.program_id(0).to(tl.int64)
    off_h_q = tl.program_id(1).to(tl.int64)

    off_h_for_kv = (off_h_q // q_k_ratio).to(tl.int64)

    for off_z in range(batch_size):
        R = tl.load(d_R_ptrs + off_z).to(tl.pointer_type(tl.bfloat16))
        stride_rh = tl.load(d_R_sizes + off_z * 3).to(tl.int64)
        stride_rm = tl.load(d_R_sizes + off_z * 3 + 1).to(tl.int64)
        stride_rn = tl.load(d_R_sizes + off_z * 3 + 2).to(tl.int64)

        cu_q_start = tl.load(cu_seqlens + off_z).to(tl.int64)
        cu_q_end = tl.load(cu_seqlens + off_z + 1).to(tl.int64)
        seqlen = cu_q_end - cu_q_start

        if start_m * BLOCK_M < seqlen:
            offs_m = (start_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
            offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
            offs_d = tl.arange(0, BLOCK_D).to(tl.int64)


            Q_ptrs = Q + cu_q_start * stride_qt + off_h_q * stride_qh
            K_ptrs = K + cu_q_start * stride_kt + off_h_for_kv * stride_kh


            q = tl.load(Q_ptrs + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
                        mask=offs_m[:, None] < seqlen)


            k_block_start = 0
            k_block_end = tl.cdiv((start_m + 1) * BLOCK_M, BLOCK_N)


            m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

            padding_mask = offs_m >= seqlen
            m_i = tl.where(padding_mask, float("inf"), m_i) ## avoid nan in exp
            

            l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)

            k_ptrs = K_ptrs + offs_n[None, :] * stride_kt + offs_d[:, None] * stride_kd
            R_ptrs = R + off_h_q * stride_rh + offs_m * stride_rm

            for k_block_col_idx in range(k_block_start, k_block_end - 1):
                l_i, m_i = _fwd_kernel_inner(
                    l_i, m_i,
                    q,
                    k_block_col_idx,
                    k_ptrs,
                    R_ptrs,
                    offs_m, offs_n,
                    stride_kt,
                    stride_rn,
                    sm_scale,
                    seqlen,
                    False,
                    BLOCK_M,
                    BLOCK_N,
                )

            l_i, m_i = _fwd_kernel_inner(
                l_i, m_i,
                q,
                k_block_end - 1,
                k_ptrs,
                R_ptrs,
                offs_m, offs_n,
                stride_kt,
                stride_rn,
                sm_scale,
                seqlen,
                True,
                BLOCK_M,
                BLOCK_N,
            )

            ## col max on R and store in Po
            Po_ptrs = Po + off_z * stride_pz + off_h_q * stride_ph + start_m * stride_pm        

            for n in range(0, start_m + 1): ## causal only for now
                n = n.to(tl.int64)
                Po_block_ptr = Po_ptrs + n * stride_pn
                R_block_ptr = R + off_h_q * stride_rh + offs_m * stride_rm +  n * stride_rn
                row_max = tl.load(R_block_ptr, mask=offs_m < seqlen)
                rescaled_max = tl.exp(row_max - m_i) / l_i
                tl.store(R_block_ptr, rescaled_max.to(q.dtype), mask=offs_m < seqlen)
                col_max = tl.max(rescaled_max, 0).to(q.dtype)
                tl.store(Po_block_ptr, col_max)



def attn_pooling_qk_varlen(
    q, k, # (#tokens, n_heads, head_size)
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

    Po = torch.zeros((batch, n_heads, num_blocks, num_blocks), device=q.device, dtype=torch.bfloat16)

    
    group_R = []
    R_prts = []
    R_sizes = []
    for i in range(batch):
        seq_len = cu_seqlens[i+1] - cu_seqlens[i]
        n_blocks = triton.cdiv(seq_len, block_size)
        R = torch.full((1, n_heads, seq_len, n_blocks), -65504.0, device=q.device, dtype=torch.bfloat16)
        group_R.append(R)
        R_prts.append(R.data_ptr())
        R_sizes += [R.stride(1), R.stride(2), R.stride(3)]
    
    d_R_ptrs = torch.tensor(R_prts, device=q.device, dtype=torch.int64)
    d_R_sizes = torch.tensor(R_sizes, device=q.device, dtype=torch.int64)

    grid = (num_blocks, n_heads, )
    
    with torch.cuda.device(q.device.index): 
        _fwd_kernel_varlen[grid](
            q, k, Po,
            sm_scale,
            cu_seqlens,
            d_R_ptrs,
            d_R_sizes,
            *q.stride(),
            *k.stride(),
            *Po.stride(),
            q_k_ratio,
            batch,
            BLOCK_M = block_size,
            BLOCK_N = block_size,
            BLOCK_D = block_d,
            num_warps = 4,
            num_stages = 1
        )


    Sum = torch.sum(Po, dim=-1, keepdim=True)
    Po.div_(Sum + 1e-6)
    
    for r in group_R:
        r.clamp_(min=0.0)

    return Po, group_R

