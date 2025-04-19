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
    leftpad_block_offset,
    k_ptrs, v_ptrs,
    offs_m, offs_n,
    stride_kt, stride_vt, stride_bmask_n,
    sm_scale,
    seqlen_k,
    LAST_K_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):



    mask_val = tl.load(block_mask_ptr + (k_block_col_idx + leftpad_block_offset) * stride_bmask_n)
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
    cu_seqlens,
    max_seqlen,
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

    cu_start = tl.load(cu_seqlens + off_z).to(tl.int64)
    cu_end = tl.load(cu_seqlens + off_z + 1).to(tl.int64)
    seqlen = cu_end - cu_start
    leftpad_block_offset = (max_seqlen - seqlen) // BLOCK_M

    if start_m * BLOCK_M >= seqlen:
        return
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)


    Q += cu_start * stride_qt + off_h_q * stride_qh
    K += cu_start * stride_kt + off_h_for_kv * stride_kh
    V += cu_start * stride_vt + off_h_for_kv * stride_vh
    Out += cu_start * stride_ot + off_h_q * stride_oh


    q = tl.load(Q + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
                mask=offs_m[:, None] < seqlen)


    block_mask_ptr += off_z * stride_bmask_z + off_h_q * stride_bmask_h + (start_m + leftpad_block_offset) * stride_bmask_m


    k_block_start = 0
    k_block_end = tl.cdiv((start_m + 1) * BLOCK_M, BLOCK_N)


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
            leftpad_block_offset,
            k_ptrs, v_ptrs,
            offs_m, offs_n,
            stride_kt, stride_vt, stride_bmask_n,
            sm_scale,
            seqlen,
            False,
            BLOCK_N,
        )

    acc, l_i, m_i = _fwd_kernel_inner(
        acc, l_i, m_i,
        q,
        k_block_end - 1,
        block_mask_ptr,
        leftpad_block_offset,
        k_ptrs, v_ptrs,
        offs_m, offs_n,
        stride_kt, stride_vt, stride_bmask_n,
        sm_scale,
        seqlen,
        True,
        BLOCK_N,
    )


    m_i += tl.math.log(l_i)
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    acc = acc.to(Out.dtype.element_ty) 

    if output_lse:
        l_ptrs = L + off_h_q * stride_lh + cu_start * stride_lt + offs_m * stride_lt
        end_m_idx = (start_m + 1) * BLOCK_M
        overflow_size = end_m_idx - seqlen
        if overflow_size > 0:
            boundary = tl.full((BLOCK_M, ), BLOCK_M - overflow_size, dtype=tl.int64)
            l_ptrs_mask = tl.arange(0, BLOCK_M) < boundary
            tl.store(l_ptrs, m_i, mask=l_ptrs_mask) # the log of the normalization constant
        else:
            tl.store(l_ptrs, m_i) # the log of the normalization constant


    tl.store(Out + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od, acc,
            mask=offs_m[:, None] < seqlen)


## This kernel assume leftpadding of block_mask from gate
def blocksparse_flash_attn_varlen_leftpad(
    q, k, v, 
    cu_seqlens,
    max_seqlen,
    sm_scale,
    block_mask,
    block_size=64,
    output_lse=True
):

    # split q to blocks
    _, n_heads, head_size = q.shape
    batch = cu_seqlens.size(0) - 1


    assert q.dim() == k.dim() == v.dim() == 3
    assert q.size(1) % k.size(1) == 0
    assert q.size(2) == k.size(2)
    assert k.shape == v.shape # TODO: allow diff head_size for k, v
    assert cu_seqlens.dim() == 1
    assert cu_seqlens.size(0) == cu_seqlens.size(0)
    assert head_size in {64, 128, 256}

    k_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu()
    if max_seqlen:
        assert k_lens.max() <= max_seqlen

    q_k_ratio = q.size(1) // k.size(1)
    
    out = q.new_empty(q.shape)
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
            cu_seqlens,
            max_seqlen,
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


