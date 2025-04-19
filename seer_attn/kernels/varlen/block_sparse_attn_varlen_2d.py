"""
Author: Yizhao Gao
"""

from typing import TypeVar
from functools import lru_cache
import math
import torch
import numpy as np

import triton
import triton.language as tl

import os

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def get_sparse_attn_mask_from_threshold(x, threshold, block_mask=None, use_dense_for_last_block=False):
    dense_mask = x > threshold 
    
    if use_dense_for_last_block:
        dense_mask[:,:,-1:,:] = True 

    if block_mask is not None:
        dense_mask = dense_mask & block_mask
    else:
        dense_mask.tril_()

    return  dense_mask



@triton.jit
def _fwd_kernel_inner(
    acc, l_i, m_i,
    q,
    k_block_col_idx,
    block_mask_ptr,
    k_ptrs, v_ptrs,
    offs_m, offs_n,
    stride_kt, stride_vt, stride_bmask_n,
    sm_scale,
    seqlen_k,
    LAST_K_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):



    mask_val = tl.load(block_mask_ptr + k_block_col_idx * stride_bmask_n)
    if mask_val == True:
        start_n = k_block_col_idx * BLOCK_N
        # -- compute qk ----

        if LAST_K_BLOCK:
            k = tl.load(k_ptrs + start_n * stride_kt,
                        mask=offs_n[None, :] + start_n < seqlen_k)

        else:
            k = tl.load(k_ptrs + start_n * stride_kt)


        qk = tl.dot(q, k)

        qk *= sm_scale

        # the following is needed only when LAST_K_BLOCK or BLOCK_M < BLOCK_N
        if LAST_K_BLOCK :
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), 0, float('-inf'))


        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.exp(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        # update acc
        if LAST_K_BLOCK:
            v = tl.load(v_ptrs + start_n * stride_vt,
                        mask=offs_n[:, None] + start_n < seqlen_k)
        else:
            v = tl.load(v_ptrs + start_n * stride_vt)

        p = p.to(v.type.element_ty)

        acc += tl.dot(p, v)
        # update m_i and l_i
        m_i = m_ij
    return acc, l_i, m_i


# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in [1, 2, 4]\
#         for num_stages in [1, 2, 3, 4, 7]
#     ],
#     key=['BLOCK_M', 'BLOCK_N', 'BLOCK_D'],
# )
@triton.jit
def _fwd_kernel_varlen(
    Q, K, V, Out, L,
    sm_scale,
    cu_seqlens_q,
    cu_seqlens_k,
    block_mask_ptr,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    stride_lt, stride_lh,
    stride_bmask_z, stride_bmask_h, stride_bmask_m, stride_bmask_n,
    q_k_ratio,
    output_lse: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):

    start_m = tl.program_id(0).to(tl.int64)
    off_h_q = tl.program_id(1).to(tl.int64)
    off_z = tl.program_id(2).to(tl.int64)

    off_h_for_kv = (off_h_q // q_k_ratio)

    cu_q_start = tl.load(cu_seqlens_q + off_z).to(tl.int64)
    cu_q_end = tl.load(cu_seqlens_q + off_z + 1).to(tl.int64)
    seqlen_q = cu_q_end - cu_q_start

    cu_k_start = tl.load(cu_seqlens_k + off_z).to(tl.int64)
    cu_k_end = tl.load(cu_seqlens_k + off_z + 1).to(tl.int64)
    seqlen_k = cu_k_end - cu_k_start
    assert seqlen_q == seqlen_k 

    if start_m * BLOCK_M >= seqlen_q:
        return
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)


    Q += cu_q_start * stride_qt + off_h_q * stride_qh
    K += cu_k_start * stride_kt + off_h_for_kv * stride_kh
    V += cu_k_start * stride_vt + off_h_for_kv * stride_vh
    Out += cu_q_start * stride_ot + off_h_q * stride_oh


    q = tl.load(Q + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
                mask=offs_m[:, None] < seqlen_q)


    block_mask_ptr += off_z * stride_bmask_z + off_h_q * stride_bmask_h + start_m * stride_bmask_m


    k_block_start = 0
    k_block_end = tl.cdiv((start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N)


    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    k_ptrs = K + offs_n[None, :] * stride_kt + offs_d[:, None] * stride_kd
    v_ptrs = V + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd


    for k_block_col_idx in range(k_block_start, k_block_end - 1):
        acc, l_i, m_i = _fwd_kernel_inner(
            acc, l_i, m_i,
            q,
            k_block_col_idx,
            block_mask_ptr,
            k_ptrs, v_ptrs,
            offs_m, offs_n,
            stride_kt, stride_vt, stride_bmask_n,
            sm_scale,
            seqlen_k,
            False,
            BLOCK_M,
            BLOCK_N,
        )

    acc, l_i, m_i = _fwd_kernel_inner(
        acc, l_i, m_i,
        q,
        k_block_end - 1,
        block_mask_ptr,
        k_ptrs, v_ptrs,
        offs_m, offs_n,
        stride_kt, stride_vt, stride_bmask_n,
        sm_scale,
        seqlen_k,
        True,
        BLOCK_M,
        BLOCK_N,
    )


    m_i += tl.math.log(l_i)
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    acc = acc.to(Out.dtype.element_ty) 

    if output_lse:
        l_ptrs = L + off_h_q * stride_lh + cu_q_start * stride_lt + offs_m * stride_lt
        end_m_idx = (start_m + 1) * BLOCK_M
        overflow_size = end_m_idx - seqlen_q
        if overflow_size > 0:
            boundary = tl.full((BLOCK_M, ), BLOCK_M - overflow_size, dtype=tl.int64)
            l_ptrs_mask = tl.arange(0, BLOCK_M) < boundary
            tl.store(l_ptrs, m_i, mask=l_ptrs_mask) # the log of the normalization constant
        else:
            tl.store(l_ptrs, m_i) # the log of the normalization constant


    tl.store(Out + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od, acc,
            mask=offs_m[:, None] < seqlen_q)



def blocksparse_flash_attn_varlen_fwd(
    q, k, v, 
    cu_seqlens_k,
    cu_seqlens_q,
    max_seqlen,
    sm_scale,
    block_mask,
    block_size=64,
    output_lse=True
):

    # split q to blocks
    _, n_heads, head_size = q.shape
    batch = cu_seqlens_k.size(0) - 1


    assert q.dim() == k.dim() == v.dim() == 3
    assert q.size(1) % k.size(1) == 0
    assert q.size(2) == k.size(2)
    assert k.shape == v.shape # TODO: allow diff head_size for k, v
    assert cu_seqlens_k.dim() == 1
    assert cu_seqlens_k.size(0) == cu_seqlens_q.size(0)
    assert head_size in {64, 128, 256}

    k_lens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).cpu()
    if max_seqlen:
        assert k_lens.max() <= max_seqlen

    q_k_ratio = q.size(1) // k.size(1)
    
    out = q.new_empty(q.shape)
    cu_seqlens_q = cu_seqlens_q.contiguous()
    cu_seqlens_k = cu_seqlens_k.contiguous()

    block_d = head_size

    if is_hip():
        extra_kern_args = {"waves_per_eu": 1}
    else:
        extra_kern_args = {}


    grid = lambda META: (triton.cdiv(max_seqlen, META['BLOCK_M']), n_heads, batch)

    L = torch.empty((q.shape[0], n_heads), device=q.device, dtype=torch.float32)


    with torch.cuda.device(q.device.index): 
        _fwd_kernel_varlen[grid](
            q, k, v, out, L,
            sm_scale,
            cu_seqlens_q,
            cu_seqlens_k,
            block_mask,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *out.stride(),
            *L.stride(),
            *block_mask.stride(),
            q_k_ratio,
            BLOCK_M = block_size,
            BLOCK_N = block_size,
            BLOCK_D = block_d,
            num_warps = 4,
            num_stages = 2,
            output_lse=output_lse,
            **extra_kern_args
        )
    if output_lse:
        return out, L
    else:
        return out


@triton.jit
def _bwd_preprocess_use_o(
    Out,
    DO,
    Delta,
    stride_om, stride_oh, stride_ok,
    stride_dom, stride_doh, stride_dok,
    stride_deltam, stride_deltah, 
    cu_seqlens_q,
    cu_seqlens_k,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    N_CTX_Q: tl.constexpr,
    H: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_bh = tl.program_id(1).to(tl.int64)

    # Compute batch and head indices
    off_z = pid_bh // H
    off_h = pid_bh % H


    q_start = tl.load(cu_seqlens_q + off_z).to(tl.int64)
    q_end = tl.load(cu_seqlens_q + off_z + 1).to(tl.int64)
    k_start = tl.load(cu_seqlens_k + off_z).to(tl.int64)
    k_end = tl.load(cu_seqlens_k + off_z + 1).to(tl.int64)

    # Compute actual sequence lengths
    N_CTX_Q = q_end - q_start
    N_CTX_K = k_end - k_start

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_DMODEL)

    mask_m = off_m < N_CTX_Q

    if pid_m * BLOCK_M >= N_CTX_Q:
        return


    o_offset = Out + off_h * stride_oh + q_start * stride_om
    do_offset = DO + off_h * stride_oh + q_start * stride_om

    out_ptrs = o_offset + off_m[:, None] * stride_om + off_d[None, :] * stride_ok
    do_ptrs = do_offset + off_m[:, None] * stride_dom + off_d[None, :] * stride_dok

    # load
    o = tl.load(out_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    delta = tl.sum(o * do, axis=1)

    delta_offset = Delta + off_h * stride_deltah + q_start * stride_deltam
    delta_ptrs = delta_offset + off_m * stride_deltam
    tl.store(delta_ptrs, delta, mask=mask_m)


@triton.jit
def _bwd_dkdv(
    Q, K, V, sm_scale,
    block_mask_ptr,
    DO, DK, DV,
    L, D,
    stride_qm, stride_qh, stride_qd,
    stride_kn, stride_kh, stride_kd,
    stride_vn, stride_vh, stride_vd,
    stride_lm, stride_lh,
    stride_deltam, stride_deltah, 
    stride_bmask_z, stride_bmask_h, stride_bmask_m, stride_bmask_n,
    cu_seqlens_q,  
    cu_seqlens_k,
    H, 
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    start_n = tl.program_id(0).to(tl.int64)
    off_hz = tl.program_id(1).to(tl.int64)
    off_z = off_hz // H
    off_h = off_hz % H

    q_start = tl.load(cu_seqlens_q + off_z).to(tl.int64)
    q_end = tl.load(cu_seqlens_q + off_z + 1).to(tl.int64)
    k_start = tl.load(cu_seqlens_k + off_z).to(tl.int64)
    k_end = tl.load(cu_seqlens_k + off_z + 1).to(tl.int64)

    # Compute actual sequence lengths
    N_CTX = k_end - k_start

    if start_n * BLOCK_N >= N_CTX:
        return


    q_offset = Q + off_h * stride_qh + q_start * stride_qm
    k_offset = K + off_h * stride_kh + k_start * stride_kn
    v_offset = V + off_h * stride_vh + k_start * stride_vn
    do_offset = DO + off_h * stride_qh + q_start * stride_qm
    l_offset = L + off_h * stride_lh + q_start * stride_lm
    d_offset = D + off_h * stride_deltah + q_start * stride_deltam


    dk_offset = DK + off_h * stride_kh + k_start * stride_kn
    dv_offset = DV + off_h * stride_vh + k_start * stride_vn

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    mask_n = offs_n < N_CTX
    kv_mask = mask_n[:, None]

    # initialize pointers to value-like data
    k_ptrs = k_offset + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    v_ptrs = v_offset + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)

    # initialize dv amd dk
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    # k and v stay in SRAM throughout
    k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
    v = tl.load(v_ptrs, mask=kv_mask, other=0.0)

    block_mask_ptr += off_z * stride_bmask_z + off_h * stride_bmask_h + start_n * stride_bmask_n


    num_block = tl.cdiv(N_CTX, BLOCK_N)
    start_l = start_n
    end_l = num_block

    for row_idx in range(start_l, end_l):
        row_idx = row_idx.to(tl.int64)
        mask_val = tl.load(block_mask_ptr + row_idx * stride_bmask_m)    
        if mask_val == True:
            start_m = row_idx * BLOCK_M
            offs_m = start_m + tl.arange(0, BLOCK_M)
            q_ptrs =  q_offset + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
            do_ptrs = do_offset + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)

            mask_m = offs_m < N_CTX

            q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
            do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, tl.trans(k))
            
            qk += tl.where(offs_m[:, None] >= (offs_n[None, :]), 0, float('-inf'))
            l_i = tl.load(l_offset + offs_m * stride_lm, mask=mask_m)
            p = tl.exp(qk * sm_scale - l_i[:, None])
            
            p_mask = mask_m[:, None] & mask_n[None, :]
            p = tl.where(p_mask, p, 0.0)

            dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
            dp = tl.dot(do, tl.trans(v))


            d_ptrs = d_offset + offs_m * stride_deltam
            Di = tl.load(d_ptrs, mask=mask_m)
            ds = (p * (dp - Di[:, None])) * sm_scale
            ds = tl.where(p_mask, ds, 0.0).to(Q.dtype.element_ty)

            dk += tl.dot(tl.trans(ds), q)

    # write-back
    dk_ptrs = dk_offset + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    dv_ptrs = dv_offset + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
    tl.store(dk_ptrs, dk.to(K.dtype.element_ty), mask=kv_mask)
    tl.store(dv_ptrs, dv.to(V.dtype.element_ty), mask=kv_mask)




@triton.jit
def _bwd_dq(
    Q, K, V, sm_scale,
    block_mask_ptr,
    DO, DQ,
    L, D,
    stride_qm, stride_qh, stride_qd,
    stride_kn, stride_kh, stride_kd,
    stride_vn, stride_vh, stride_vd,
    stride_lm, stride_lh,
    stride_deltam, stride_deltah, 
    stride_bmask_z, stride_bmask_h, stride_bmask_m, stride_bmask_n,
    cu_seqlens_q,  
    cu_seqlens_k,
    H, 
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):


    start_m = tl.program_id(0).to(tl.int64)  
    off_hz = tl.program_id(1).to(tl.int64)
    off_z = off_hz // H
    off_h = off_hz % H


    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)  
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_start = tl.load(cu_seqlens_q + off_z).to(tl.int64)
    q_end = tl.load(cu_seqlens_q + off_z + 1).to(tl.int64)
    k_start = tl.load(cu_seqlens_k + off_z).to(tl.int64)
    k_end = tl.load(cu_seqlens_k + off_z + 1).to(tl.int64)


    # Compute actual sequence lengths
    N_CTX = q_end - q_start
    mask_m = offs_m < N_CTX


    if start_m * BLOCK_M >= N_CTX:
        return


    q_offset = Q + off_h * stride_qh + (q_start + offs_m[:, None]) * stride_qm + offs_d[None, :] * stride_qd
    do_offset = DO + off_h * stride_qh + (q_start + offs_m[:, None]) * stride_qm + offs_d[None, :] * stride_qd
    l_offset = L + off_h * stride_lh + (q_start + offs_m) * stride_lm 
    d_offset = D + off_h * stride_deltah + (q_start + offs_m) * stride_deltam 

    do = tl.load(do_offset, mask=mask_m[:, None], other=0.0)
    l_i = tl.load(l_offset, mask=mask_m, other=0.0)
    Di = tl.load(d_offset, mask=mask_m, other=0.0)

    
    q = tl.load(q_offset, mask=mask_m[:, None], other=0.0)
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)  

    block_mask_ptr += off_z * stride_bmask_z + off_h * stride_bmask_h + start_m * stride_bmask_m

    start_l = 0
    end_l = tl.cdiv((start_m + 1) * BLOCK_M, BLOCK_N)

    for col_idx in range(start_l, end_l):
        mask_val = tl.load(block_mask_ptr + col_idx * stride_bmask_n)
        if mask_val == True:
            start_n = col_idx * BLOCK_N
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N_CTX

            k_ptrs = K + off_h * stride_kh + (k_start + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kd
            v_ptrs = V + off_h * stride_vh + (k_start + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vd
            k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, tl.trans(k))
            
            qk += tl.where(offs_m[:, None] >= (offs_n[None, :]), 0, float('-inf'))
            p = tl.exp(qk * sm_scale - l_i[:, None])
            
            p_mask = mask_m[:, None] & mask_n[None, :]
            p = tl.where(p_mask, p, 0.0)

            dp = tl.dot(do, v.T).to(tl.float32)

            ds = (p * (dp - Di[:, None])) * sm_scale
            ds = tl.where(p_mask, ds, 0.0).to(Q.dtype.element_ty)

            dq += tl.dot(ds, k)  

    dq_offset = DQ + off_h * stride_qh + (q_start + offs_m[:, None]) * stride_qm + offs_d[None, :] * stride_qd
    tl.store(dq_offset, dq, mask=mask_m[:, None])


def blocksparse_flash_attn_varlen_bwd(
    do, q, k, v, o, L, 
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen,
    sm_scale,
    block_mask,
    block_size=64,
):
    do = do.contiguous()
    L = L.contiguous()
    assert q.dim() == k.dim() == v.dim() == 3

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    o = o.contiguous()
    cu_seqlens_k = cu_seqlens_k.contiguous()
    cu_seqlens_q = cu_seqlens_q.contiguous()


    num_block = triton.cdiv(max_seqlen, block_size)
    head_size = q.size(-1)
    block_d = head_size


    dq = torch.zeros_like(q)
    dk = torch.empty_like(k) 
    dv = torch.empty_like(v)

    delta = torch.empty_like(L)

    batch = cu_seqlens_k.size(0) - 1
    n_heads = q.size(1)
    head_size = q.size(2)
    batch_headsize = batch * n_heads

    if is_hip():
        extra_kern_args = {"waves_per_eu": 1}
    else:
        extra_kern_args = {}

    block_mask = block_mask.contiguous()
    
    with torch.cuda.device(q.device.index): 
        _bwd_preprocess_use_o[(num_block, batch_headsize)](
            o, do, delta,
            *o.stride(),
            *do.stride(),
            *delta.stride(),
            cu_seqlens_q,
            cu_seqlens_k,
            BLOCK_M = block_size,
            BLOCK_DMODEL = block_d,
            N_CTX_Q = q.size(0),
            H = n_heads,
        )


        _bwd_dkdv[(num_block, batch_headsize)](
            q, k, v, sm_scale,
            block_mask,
            do, dk, dv,
            L, delta,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *L.stride(),
            *delta.stride(),
            *block_mask.stride(),
            cu_seqlens_q,
            cu_seqlens_k,
            H = n_heads,
            BLOCK_M = block_size,
            BLOCK_N = block_size,
            BLOCK_DMODEL = block_d,
            num_warps = 4,
            num_stages = 1,
            **extra_kern_args
        )

        _bwd_dq[(num_block, batch_headsize)](
            q, k, v, sm_scale,
            block_mask,
            do, dq,
            L, delta,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *L.stride(),
            *delta.stride(),
            *block_mask.stride(),
            cu_seqlens_q,
            cu_seqlens_k,
            H = n_heads,
            BLOCK_M = block_size,
            BLOCK_N = block_size,
            BLOCK_DMODEL = block_d,
            num_warps = 4,
            num_stages = 1,
            **extra_kern_args
        )

    return dq, dk, dv

class _block_sparse_attn_varlen(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens_k, cu_seqlens_q, max_seqlen, sm_scale, dense_block_mask, block_size=64):
        out, L = blocksparse_flash_attn_varlen_fwd(q, k, v, cu_seqlens_k, cu_seqlens_q, max_seqlen, sm_scale, dense_block_mask, block_size=block_size)
        ctx.save_for_backward(out, L, q, k, v, cu_seqlens_k, cu_seqlens_q, dense_block_mask)
        ctx.sm_scale = sm_scale
        ctx.block_size = block_size
        ctx.max_seqlen = max_seqlen
        return out

    @staticmethod
    def backward(ctx, do):
        out, L, q, k, v, cu_seqlens_k, cu_seqlens_q, dense_block_mask = ctx.saved_tensors
        dq, dk, dv = blocksparse_flash_attn_varlen_bwd(
            do, 
            q, 
            k, 
            v, 
            out, 
            L, 
            cu_seqlens_q, 
            cu_seqlens_k, 
            ctx.max_seqlen, 
            ctx.sm_scale, 
            dense_block_mask,
            ctx.block_size
        )
        return dq, dk, dv, None, None, None, None, None, None



block_2d_sparse_attn_varlen_func = _block_sparse_attn_varlen.apply
