
"""
    Original Author: Eric Lin (xihlin) (https://huggingface.co/microsoft/Phi-3-small-8k-instruct/blob/main/triton_flash_blocksparse_attn.py)
"""
"""
    Modified by Yizhao Gao
    Use binary block mask for simplicity. Need to be updated to varlen version for batched inference.
"""


from typing import TypeVar
from functools import lru_cache
import math
import torch
import numpy as np

import triton
import triton.language as tl
import torch.nn.functional as F
import os

import dataclasses


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"



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

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)

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



@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    block_mask_ptr,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_bmz, stride_bmh, stride_bmm, stride_bmn,
    stride_oz, stride_oh, stride_om, stride_od,
    H, N_CTX,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    Q_LEN = N_CTX
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_h = off_hz % H
    off_z = off_hz // H
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    block_mask_ptr += off_z * stride_bmz + off_h * stride_bmh

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    # off_k = offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    off_k = offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
    off_v = offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    mask_ptrs = block_mask_ptr + start_m * stride_bmm

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    q = tl.load(q_ptrs, mask=offs_m[:, None] < Q_LEN)

    k_block_start = 0
    k_block_end = tl.cdiv((start_m + 1) * BLOCK_M, BLOCK_N)

    # loop over k, v and update accumulator
    for col_idx in range(k_block_start, k_block_end-1):
        acc, l_i, m_i = _fwd_kernel_inner(
            acc, l_i, m_i,
            q,
            col_idx,
            mask_ptrs,
            k_ptrs, v_ptrs,
            offs_m, offs_n,
            stride_kn, stride_vn, stride_bmn,
            sm_scale,
            N_CTX,
            False,
            BLOCK_M,
            BLOCK_N,
        )

    # last block
    acc, l_i, m_i = _fwd_kernel_inner(
        acc, l_i, m_i,
        q,
        k_block_end-1,
        mask_ptrs,
        k_ptrs, v_ptrs,
        offs_m, offs_n,
        stride_kn, stride_vn, stride_bmn,
        sm_scale,
        N_CTX,
        True,
        BLOCK_M,
        BLOCK_N,
    )

    m_i += tl.math.log(l_i)
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    acc = acc.to(Out.dtype.element_ty) 


    off_o = off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < N_CTX)



def get_stride_from_layout(x, layout):
    if layout == 'bhsd':
        return x.stride()
    elif layout == 'bshd':
        return (x.stride(0), x.stride(2), x.stride(1), x.stride(3),)
    else:
        raise ValueError(f"Unsupported layout: {layout}")


def block_sparse_triton_fn(
        q, 
        k, 
        v, 
        block_sparse_mask, 
        sm_scale, 
        BLOCK_M=64, 
        BLOCK_N=64, 
        layout='bhsd',
    ):



    o = torch.empty_like(q).contiguous()
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and block_sparse_mask.is_contiguous()
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    BLOCK_DMODEL = q.shape[-1]

    if is_hip():
        num_warps, num_stages = 8, 1
    else:
        num_warps, num_stages = 4, 2

    if layout == 'bhsd':
        N_CTX = k.shape[2]
        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1])
        H = q.shape[1]
    elif layout == 'bshd':
        N_CTX = k.shape[1]
        grid = (triton.cdiv(q.shape[1], BLOCK_M), q.shape[0] * q.shape[2])
        H = q.shape[2]
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    with torch.cuda.device(q.device.index): 
        _fwd_kernel[grid](
            q, k, v, sm_scale,
            block_sparse_mask,
            o,
            *get_stride_from_layout(q, layout),
            *get_stride_from_layout(k, layout),
            *get_stride_from_layout(v, layout),
            *block_sparse_mask.stride(), 
            *get_stride_from_layout(o, layout),
            H, N_CTX,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return o




