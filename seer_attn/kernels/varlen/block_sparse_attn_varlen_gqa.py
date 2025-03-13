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

    return  dense_mask

def get_sparse_attn_mask_from_threshold_grouped(soft_masks, threshold, max_seqlen, block_size):
    bsz = len(soft_masks)
    n_heads = soft_masks[0].size(0)
    num_blocks = math.ceil(max_seqlen / block_size)

    dense_mask = torch.zeros(bsz, n_heads, max_seqlen, num_blocks, dtype=torch.bool, device=soft_masks[0].device)

    for i, mask in enumerate(soft_masks):
        mask_m, mask_n = mask.size(-2), mask.size(-1)
        dense_mask[i, :, 0:mask_m, 0:mask_n] = mask > threshold

    return  dense_mask



@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=['gqa_group_size', 'BLOCK_H', 'BLOCK_N', 'BLOCK_D'],
)
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
    gqa_group_size: tl.constexpr,
    BLOCK_H: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):

    start_t = tl.program_id(0).to(tl.int64) ## sequence index
    off_h_for_kv = tl.program_id(1).to(tl.int64)
    off_h_q = off_h_for_kv * gqa_group_size
    off_z = tl.program_id(2).to(tl.int64)

    cu_q_start = tl.load(cu_seqlens_q + off_z).to(tl.int64)
    cu_q_end = tl.load(cu_seqlens_q + off_z + 1).to(tl.int64)
    seqlen_q = cu_q_end - cu_q_start

    cu_k_start = tl.load(cu_seqlens_k + off_z).to(tl.int64)
    cu_k_end = tl.load(cu_seqlens_k + off_z + 1).to(tl.int64)
    seqlen_k = cu_k_end - cu_k_start

    if start_t >= seqlen_q:
        return
    
    offs_m = tl.arange(0, BLOCK_H) ## head 
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)


    Q += (cu_q_start + start_t) * stride_qt + off_h_q * stride_qh
    K += cu_k_start * stride_kt + off_h_for_kv * stride_kh
    V += cu_k_start * stride_vt + off_h_for_kv * stride_vh
    Out += (cu_q_start + start_t) * stride_ot + off_h_q * stride_oh

    q = tl.load(Q + offs_m[:, None] * stride_qh + offs_d[None, :] * stride_qd,
                mask=offs_m[:, None] < gqa_group_size) ## padding to min 16

    block_mask_ptr += off_z * stride_bmask_z + off_h_for_kv * stride_bmask_h + start_t * stride_bmask_m


    k_block_start = 0
    k_block_end = tl.cdiv(start_t + 1 + seqlen_k - seqlen_q, BLOCK_N)

    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_H], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

    k_ptrs = K + offs_n[None, :] * stride_kt + offs_d[:, None] * stride_kd
    v_ptrs = V + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd


    for k_block_col_idx in range(k_block_start, k_block_end):
        mask_val = tl.load(block_mask_ptr + k_block_col_idx * stride_bmask_n)
        if mask_val == True:
            start_n = k_block_col_idx * BLOCK_N
            # -- compute qk ----

            k = tl.load(k_ptrs + start_n * stride_kt, mask=offs_n[None, :] + start_n < seqlen_k)

            qk = tl.dot(q, k)
            qk *= sm_scale
            qk = tl.where(start_t >= (start_n + offs_n[None, :]), qk, float('-inf'))

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
            p = tl.exp(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            
            v = tl.load(v_ptrs + start_n * stride_vt, mask=offs_n[:, None] + start_n < seqlen_k)
            p = p.to(v.type.element_ty)

            acc += tl.dot(p, v)
            m_i = m_ij

    m_i += tl.math.log(l_i)
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    acc = acc.to(Out.dtype.element_ty) 

    l_ptrs = L + (off_h_q + offs_m) * stride_lh + (cu_q_start + start_t) * stride_lt
    tl.store(l_ptrs, m_i, mask=offs_m < gqa_group_size) 

    O_ptrs = Out + offs_m[:, None] * stride_oh + offs_d[None, :] * stride_od
    tl.store(O_ptrs, acc, mask=offs_m[:, None] < gqa_group_size)



def blocksparse_flash_attn_varlen_fwd(
    q, k, v, # (#tokens, n_heads, head_size)
    cu_seqlens_k,
    cu_seqlens_q,
    max_seqlen,
    sm_scale,
    block_mask,
    block_size=64,
):
    # split q to blocks
    _, n_heads, head_size = q.shape
    batch = cu_seqlens_k.size(0) - 1

    assert q.dim() == k.dim() == v.dim() == 3
    assert q.size(2) == k.size(2)
    assert k.shape == v.shape 
    assert cu_seqlens_k.dim() == 1
    assert cu_seqlens_k.size(0) == cu_seqlens_q.size(0)
    assert head_size in {64, 128, 256}

    k_lens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).cpu()
    q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).cpu()
    if max_seqlen:
        assert k_lens.max() <= max_seqlen
    else:
        max_seqlen = q_lens.max()

    q_h = q.size(1)
    k_h = k.size(1)
    gqa_group_size = q_h // k_h
    
    out = q.new_empty(q.shape)
    cu_seqlens_q = cu_seqlens_q.contiguous()
    cu_seqlens_k = cu_seqlens_k.contiguous()

    block_d = head_size

    if is_hip():
        extra_kern_args = {"waves_per_eu": 1}
    else:
        extra_kern_args = {}

    block_h = gqa_group_size if gqa_group_size > 16 else 16

    grid = lambda META: (max_seqlen, k_h, batch)

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
            gqa_group_size,
            BLOCK_H = block_h,
            BLOCK_N = block_size,
            BLOCK_D = block_d,
            **extra_kern_args
        )

    return out, L


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

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=['gqa_group_size', 'BLOCK_H', 'BLOCK_N', 'BLOCK_DMODEL'],
)
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
    num_kv_heads: tl.constexpr,
    gqa_group_size: tl.constexpr,
    BLOCK_H: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    start_n = tl.program_id(0).to(tl.int64)
    off_hz = tl.program_id(1).to(tl.int64)
    off_z = off_hz // num_kv_heads
    off_h_for_kv = off_hz % num_kv_heads
    off_h_q = off_h_for_kv * gqa_group_size

    q_start = tl.load(cu_seqlens_q + off_z).to(tl.int64)
    q_end = tl.load(cu_seqlens_q + off_z + 1).to(tl.int64)
    k_start = tl.load(cu_seqlens_k + off_z).to(tl.int64)
    k_end = tl.load(cu_seqlens_k + off_z + 1).to(tl.int64)

    # Compute actual sequence lengths
    seqlen_k = k_end - k_start
    seqlen_q = q_end - q_start

    if start_n * BLOCK_N >= seqlen_k:
        return


    q_offset = Q + off_h_q * stride_qh + q_start * stride_qm
    k_offset = K + off_h_for_kv * stride_kh + k_start * stride_kn
    v_offset = V + off_h_for_kv * stride_vh + k_start * stride_vn
    do_offset = DO + off_h_q * stride_qh + q_start * stride_qm
    l_offset = L + off_h_q * stride_lh + q_start * stride_lm
    d_offset = D + off_h_q * stride_deltah + q_start * stride_deltam


    dk_offset = DK + off_h_for_kv * stride_kh + k_start * stride_kn
    dv_offset = DV + off_h_for_kv * stride_vh + k_start * stride_vn

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    mask_n = offs_n < seqlen_k
    kv_mask = mask_n[:, None]

    # initialize pointers to value-like data
    k_ptrs = k_offset + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    v_ptrs = v_offset + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)

    # initialize dv and dk
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    # k and v stay in SRAM throughout
    k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
    v = tl.load(v_ptrs, mask=kv_mask, other=0.0)
    # loop over rows


    block_mask_ptr += off_z * stride_bmask_z + off_h_for_kv * stride_bmask_h + start_n * stride_bmask_n

    for start_m in range(start_n * BLOCK_N, seqlen_q):
        start_m = start_m.to(tl.int64)
        mask_val = tl.load(block_mask_ptr + start_m * stride_bmask_m)    
        if mask_val == True:
            # offs_m = start_m + tl.arange(0, BLOCK_M)
            offs_h = tl.arange(0, BLOCK_H)
            q_ptrs =  q_offset + (start_m * stride_qm + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd)
            do_ptrs = do_offset + (start_m * stride_qm + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd)

            mask_h = offs_h < gqa_group_size
            # mask_m = offs_m < N_CTX

            q = tl.load(q_ptrs, mask=mask_h[:, None], other=0.0)
            do = tl.load(do_ptrs, mask=mask_h[:, None], other=0.0)

            qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, tl.trans(k))
            
            qk += tl.where(start_m >= (offs_n[None, :]), 0, float('-inf'))
            l_i = tl.load(l_offset + start_m * stride_lm + offs_h * stride_lh, mask=mask_h)
            p = tl.exp(qk * sm_scale - l_i[:, None])
            
            p_mask = mask_n[None, :]
            p = tl.where(p_mask, p, 0.0)

            dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
            dp = tl.dot(do, tl.trans(v))


            d_ptrs = d_offset + start_m * stride_deltam + offs_h * stride_deltah
            Di = tl.load(d_ptrs, mask=mask_h)
            ds = (p * (dp - Di[:, None])) * sm_scale
            ds = tl.where(p_mask, ds, 0.0).to(Q.dtype.element_ty)

            dk += tl.dot(tl.trans(ds), q)

    # write-back
    dk_ptrs = dk_offset + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    dv_ptrs = dv_offset + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
    tl.store(dk_ptrs, dk.to(K.dtype.element_ty), mask=kv_mask)
    tl.store(dv_ptrs, dv.to(V.dtype.element_ty), mask=kv_mask)



@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=['gqa_group_size', 'BLOCK_H', 'BLOCK_N', 'BLOCK_DMODEL'],
)
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
    num_kv_heads: tl.constexpr,
    gqa_group_size: tl.constexpr,
    BLOCK_H: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):


    start_m = tl.program_id(0).to(tl.int64)  
    off_hz = tl.program_id(1).to(tl.int64)
    off_z = off_hz // num_kv_heads
    off_h_for_kv = off_hz % num_kv_heads
    off_h_q = off_h_for_kv * gqa_group_size

    offs_h = tl.arange(0, BLOCK_H)  
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_start = tl.load(cu_seqlens_q + off_z).to(tl.int64)
    q_end = tl.load(cu_seqlens_q + off_z + 1).to(tl.int64)
    k_start = tl.load(cu_seqlens_k + off_z).to(tl.int64)
    k_end = tl.load(cu_seqlens_k + off_z + 1).to(tl.int64)
    seqlen_k = k_end - k_start


    # Compute actual sequence lengths
    seqlen_q = q_end - q_start

    if start_m >= seqlen_q:
        return

    mask_h = offs_h < gqa_group_size
    q_offset = Q + (off_h_q + offs_h[:, None]) * stride_qh + (q_start + start_m) * stride_qm + offs_d[None, :] * stride_qd
    do_offset = DO + (off_h_q + offs_h[:, None]) * stride_qh + (q_start + start_m) * stride_qm + offs_d[None, :] * stride_qd
    l_offset = L + (off_h_q + offs_h) * stride_lh + (q_start + start_m) * stride_lm 
    d_offset = D + (off_h_q + offs_h) * stride_deltah + (q_start + start_m) * stride_deltam 

    do = tl.load(do_offset, mask=mask_h[:, None], other=0.0)
    l_i = tl.load(l_offset, mask=mask_h, other=0.0)
    Di = tl.load(d_offset, mask=mask_h, other=0.0)

    
    q = tl.load(q_offset, mask=mask_h[:, None], other=0.0)
    dq = tl.zeros([BLOCK_H, BLOCK_DMODEL], dtype=tl.float32)  

    block_mask_ptr += off_z * stride_bmask_z + off_h_for_kv * stride_bmask_h + start_m * stride_bmask_m

    start_l = 0
    end_l = tl.cdiv(start_m + 1 + seqlen_k - seqlen_q, BLOCK_N)

    for col_idx in range(start_l, end_l):
        mask_val = tl.load(block_mask_ptr + col_idx * stride_bmask_n)
        if mask_val == True:
            start_n = col_idx * BLOCK_N
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < seqlen_k

            k_ptrs = K + off_h_for_kv * stride_kh + (k_start + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kd
            v_ptrs = V + off_h_for_kv * stride_vh + (k_start + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vd
            k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

            qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, tl.trans(k))
            
            qk += tl.where(start_m >= (offs_n[None, :]), 0, float('-inf'))
            p = tl.exp(qk * sm_scale - l_i[:, None])
            
            p_mask = mask_n[None, :]
            p = tl.where(p_mask, p, 0.0)

            dp = tl.dot(do, v.T).to(tl.float32)

            ds = (p * (dp - Di[:, None])) * sm_scale
            ds = tl.where(p_mask, ds, 0.0).to(Q.dtype.element_ty)

            dq += tl.dot(ds, k)  

    dq_offset = DQ + (off_h_q + offs_h[:, None]) * stride_qh + (q_start + start_m) * stride_qm + offs_d[None, :] * stride_qd
    tl.store(dq_offset, dq, mask=mask_h[:, None])


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
    num_q_heads = q.size(1)
    num_kv_heads = k.size(1)
    gpq_group_size = num_q_heads // num_kv_heads
    block_d = q.size(-1)


    dq = torch.zeros_like(q)
    dk = torch.empty_like(k) 
    dv = torch.empty_like(v)

    delta = torch.empty_like(L)

    batch = cu_seqlens_k.size(0) - 1
    batch_q_head_size = batch * num_q_heads
    batch_kv_head_size = batch * num_kv_heads

    if is_hip():
        extra_kern_args = {"waves_per_eu": 1}
    else:
        extra_kern_args = {}

    block_mask = block_mask.contiguous()
    
    block_h = gpq_group_size if gpq_group_size > 16 else 16

    with torch.cuda.device(q.device.index): 
        _bwd_preprocess_use_o[(num_block, batch_q_head_size)](
            o, do, delta,
            *o.stride(),
            *do.stride(),
            *delta.stride(),
            cu_seqlens_q,
            cu_seqlens_k,
            BLOCK_M = block_size,
            BLOCK_DMODEL = block_d,
            N_CTX_Q = q.size(0),
            H = num_q_heads,
        )


        _bwd_dkdv[(num_block, batch_kv_head_size)](
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
            num_kv_heads,
            gpq_group_size,
            BLOCK_H = block_h,
            BLOCK_N = block_size,
            BLOCK_DMODEL = block_d,
            **extra_kern_args
        )

        _bwd_dq[(max_seqlen, batch_kv_head_size)](
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
            num_kv_heads,
            gpq_group_size,
            BLOCK_H = block_h,
            BLOCK_N = block_size,
            BLOCK_DMODEL = block_d,
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



block_1d_gqa_sparse_attn_varlen_func = _block_sparse_attn_varlen.apply
