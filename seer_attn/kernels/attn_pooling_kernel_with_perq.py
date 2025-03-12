"""
    Original code from Triton's official fused attention example (https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html).
"""
"""
    Modified by Zhichen Zeng,
    Self-attention output with 2D maxpooling attention map.
"""

import torch

import triton
import triton.language as tl


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    R_block_ptr,  #
                    A_block_ptr,  #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk += tl.where(mask, 0, -1.0e6)

        max = tl.max(qk, 1) * qk_scale
        m_ij = tl.maximum(m_i, max)
        qk = qk * qk_scale - m_ij[:, None]
        
        tl.store(tl.advance(R_block_ptr, (0, start_n // BLOCK_N)), max[:, None].to(q.dtype))

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(q.dtype)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    
    # -- update Po --
    if STAGE == 2:
        for start_n in range(0, (start_m + 1) * BLOCK_N, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            row_max = tl.load(R_block_ptr)
            xi = row_max - m_i[:, None]
            row_max = tl.exp2(xi)/l_i[:, None]
            tl.store(R_block_ptr, row_max.to(q.dtype))
            col_max = tl.max(row_max, 0)
            col_max = col_max[:, None].to(q.dtype)
            tl.store(A_block_ptr, col_max)
            A_block_ptr = tl.advance(A_block_ptr, (0, 1))
            R_block_ptr = tl.advance(R_block_ptr, (0, 1))

    elif STAGE == 3: 
        for start_n in range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            row_max = tl.load(R_block_ptr)
            xi = row_max - m_i[:, None]
            row_max = tl.exp2(xi)/l_i[:, None]
            tl.store(R_block_ptr, row_max.to(q.dtype))
            col_max = tl.max(row_max, 0)
            col_max = col_max[:, None].to(q.dtype)
            tl.store(A_block_ptr, col_max)
            A_block_ptr = tl.advance(A_block_ptr, (0, 1))
            R_block_ptr = tl.advance(R_block_ptr, (0, 1))

    return acc, l_i, m_i


@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out,  #
              R, Po,
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              stride_rz, stride_rh, stride_rm, stride_rn,  #
              stride_poz, stride_poh, stride_pom, stride_pon,  #
              Z, H, N_CTX,  #
              n_rep,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              N_DOWNSAMPLE: tl.constexpr,  #
              STAGE: tl.constexpr  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_kvh = off_h // n_rep
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_kvh.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_kvh.to(tl.int64) * stride_vh
    r_offset = off_z.to(tl.int64) * stride_rz + off_h.to(tl.int64) * stride_rh
    po_offset = off_z.to(tl.int64) * stride_poz + off_h.to(tl.int64) * stride_poh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    
    R_block_ptr = tl.make_block_ptr(
        base=R + r_offset,
        shape=(N_CTX, N_DOWNSAMPLE),
        strides=(stride_rm, stride_rn),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, 1),
        order=(0, 1),

    )
    A_block_ptr = tl.make_block_ptr(
        base=Po + po_offset,
        shape=(N_DOWNSAMPLE, N_DOWNSAMPLE),
        strides=(stride_pom, stride_pon),
        offsets=(start_m, 0),
        block_shape=(1, 1),
        order=(0, 1),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        R_block_ptr,  #
                                        A_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        R_block_ptr,  #
                                        A_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))

class _attention_pooling(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, block_size, norm=True):
        assert block_size in {32, 64, 128}
        # shape constraints
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        NUM_HEADS_Q, NUM_HEADS_K, NUM_HEADS_V = q.shape[1], k.shape[1], v.shape[1]
        assert NUM_HEADS_K == NUM_HEADS_V
        n_rep = NUM_HEADS_Q // NUM_HEADS_K
        o = torch.empty_like(q)
        BLOCK_N = block_size
        n_d = triton.cdiv(q.shape[2], BLOCK_N)
        R = torch.full((q.shape[0], q.shape[1], q.shape[2], n_d), -65504.0, device=q.device, dtype=torch.bfloat16)
        Po = torch.zeros((q.shape[0], q.shape[1], n_d, n_d), device=q.device, dtype=torch.bfloat16)
        stage = 3 if causal else 1
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        _attn_fwd[grid](
            q, k, v, sm_scale, M, o,  #
            R, Po,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            R.stride(0), R.stride(1), R.stride(2), R.stride(3),  #
            Po.stride(0), Po.stride(1), Po.stride(2), Po.stride(3),  #
            q.shape[0], q.shape[1],  #
            N_CTX=q.shape[2],  #
            n_rep=n_rep,  #
            HEAD_DIM=HEAD_DIM_K,  #
            STAGE=stage,  #
            BLOCK_M=block_size,
            BLOCK_N=block_size,
            N_DOWNSAMPLE=n_d,
            num_stages=3,
            num_warps=4,
            **extra_kern_args)

        Sum = torch.sum(Po, dim=-1, keepdim=True)
        Po.div_(Sum + 1e-9)
        # ignore the last block because of padding 64
        R = R[:, :, :, :-1]
        R.clamp_(min=0.0)
        R_sum = torch.sum(R, dim=-1, keepdim=True)
        R.div_(R_sum + 1e-9)
        return o, Po, R

attn_with_pooling = _attention_pooling.apply