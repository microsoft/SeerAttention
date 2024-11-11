"""
Training kernel modified from Phi-3-small implementation (https://huggingface.co/microsoft/Phi-3-small-8k-instruct/blob/main/triton_flash_blocksparse_attn.py)
Currently only support batch size 1
"""

from typing import TypeVar
from functools import lru_cache
import math
import pytest
import torch
import numpy as np

import triton
import triton.language as tl

import os

import dataclasses


def dense_to_crow_col(x):
    ''' Turning a 2D/3D torch tensor (x) to CSR rows/cols indexing.
    param:
    TODO:
        1. improve efficiency, is it faster if done in CPU, or customize a cuda kernel for it?
    NOTE: col_indices padded -1
    '''
    pad = -1
    dim = x.dim()
    assert x.dim() in (2, 3)
    if x.dim() == 2:
        x = x[None]
    x = [xi.to_sparse_csr() for xi in x]
    crows = torch.vstack([xi.crow_indices() for xi in x])
    cols = [xi.col_indices() for xi in x]
    max_cols = max(len(xi) for xi in cols)
    cols = [torch.cat([xi, pad + xi.new_zeros(max_cols - xi.shape[0])]) for xi in cols]
    cols = torch.vstack(cols)
    if dim == 2:
        crows = crows[0]
        cols = cols[0]
    return crows, cols


def crow_col_to_dense(crows, cols, dtype=torch.float16):
    dim = crows.dim()
    if dim == 1:
        crows = crows[None]
        cols = cols[None]
    device = crows.device
    crows, cols = crows.cpu(), cols.cpu()  # faster in cpu
    shape = (crows.shape[0], crows.shape[1] - 1, cols.max() + 1)
    x = torch.zeros(shape, dtype=dtype)
    for i in range(shape[0]):
        for j in range(shape[1]):
            x[i, j, cols[i, crows[i, j]:crows[i, j+1]]] = 1
    if dim == 1:
        x = x[0]
    return x.to(device)


def dense_to_ccol_row(x):
    '''Similar, but to CSC format
    '''
    x = x.transpose(-2, -1)
    return dense_to_crow_col(x)


def ccol_row_to_dense(ccol, rows, dtype=torch.float16):
    return crow_col_to_dense(ccol, rows, dtype).permute(0, 2, 1).contiguous()


def gen_causal_mask(N_CTX, device):
    attention_mask = torch.full((N_CTX, N_CTX), True, dtype=torch.bool, device=device)
    attention_mask.triu_(diagonal=1)
    attention_mask.bitwise_not_()
    return attention_mask


def get_sparse_attn_mask_from_topk(x, causal_mask, topk, dtype, device, BLOCK=64):
    x = x.to(device)
    _, num_head, downsample_len, _ = x.shape
    # N_CTX = downsample_len * BLOCK
    sparse_index = torch.topk(x, topk, dim=-1).indices
    dense_mask = torch.full([1, num_head, downsample_len, downsample_len], False, dtype=torch.bool, device=device)
    dense_mask.scatter_(-1, sparse_index, True)
    dense_mask &= causal_mask
    dense_mask.squeeze_(0)
    dense_mask = dense_mask.to(dtype)
    return  dense_mask



@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    layout_crow_ptr,
    layout_col_ptr,
    layout_crow_stride_h, layout_crow_stride_m,
    layout_col_stride_h, layout_col_stride_m,
    TMP, L, M,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug. TMP, L, M are assumed to have contiguous layouts
    Out,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    Z, H, N_CTX,
    PAST_LEN,
    Q_ROUNDED_LEN,
    BLOCK_M: tl.constexpr, 
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    Q_LEN = N_CTX - PAST_LEN
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_h = off_hz % H
    off_z = off_hz // H
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
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
    # initialize pointer to m and l
    t_ptrs = TMP + off_hz * Q_ROUNDED_LEN + offs_m
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)


    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)

    layout_ptr = layout_crow_ptr + off_h * layout_crow_stride_h + start_m * layout_crow_stride_m
    start_l = tl.load(layout_ptr).to(tl.int32)
    end_l = tl.load(layout_ptr + layout_crow_stride_m).to(tl.int32)

    # loop over k, v and update accumulator
    for col_idx_idx in range(start_l, end_l):
        col_idx = tl.load(layout_col_ptr +  off_h * layout_col_stride_h + col_idx_idx * layout_col_stride_m).to(tl.int32)
        start_n = col_idx * BLOCK_N
        # -- compute qk ----
        k = tl.load(k_ptrs + start_n * stride_kn)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)


        qk *= sm_scale
        qk += tl.where(offs_m[:, None] + PAST_LEN >= (start_n + offs_n[None, :]), 0, float('-inf'))
        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        # tl.store(t_ptrs, acc_scale)
        # acc_scale = tl.load(t_ptrs)  # BUG: have to store and immediately load
        acc = acc * acc_scale[:, None]

        p = p.to(Q.dtype.element_ty)
        # update acc
        v = tl.load(v_ptrs + start_n * stride_vn)
        acc += tl.dot(p, v)


        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    # rematerialize offsets to save registers
    # start_m = tl.program_id(0)
    # offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    l_ptrs = L + off_hz * N_CTX + offs_m
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, l_i)
    tl.store(m_ptrs, m_i)
    # initialize pointers to output
    # offs_n = tl.arange(0, BLOCK_DMODEL)
    off_o = off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc,  mask=offs_m[:, None] < Q_LEN)


@triton.jit
def _bwd_preprocess(
    Out, DO, L, # assume contiguous for Out, DO, L, NewDO, Delta layout.
    NewDO, Delta,
    N_CTX,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, D_HEAD)
    # load
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_d[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_d[None, :]).to(tl.float32)
    denom = tl.load(L + off_m).to(tl.float32)
    # compute
    do = do / denom[:, None]
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(NewDO + off_m[:, None] * D_HEAD + off_d[None, :], do)
    tl.store(Delta + off_m, delta)

@triton.jit
def _bwd_kernel(
    Q, K, V, sm_scale,
    layout_ccol_ptr,
    layout_row_ptr,
    layout_ccol_stride_h, layout_ccol_stride_m,
    layout_row_stride_h, layout_row_stride_m,
    Out, DO,  # assume contigous: Out, Do, DQ, DK, DV, L, M, D, seq(q) == seq(k), with stride_oz, stride_oh, stride_om, stride_od,
    DQ, DK, DV,
    L, M,
    D,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    # stride_dz, stride_dh, stride_dm, stride_dd,
    Z, H, N_CTX,
    num_block,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    # offset pointers for batch/head
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    DO += off_z * stride_oz + off_h * stride_oh
    DQ += off_z * stride_oz + off_h * stride_oh
    DK += off_z * stride_oz + off_h * stride_oh
    DV += off_z * stride_oz + off_h * stride_oh
    # Look like this loop can be parallelled
    # for start_n in range(0, num_block):

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    # initialize pointers to value-like data
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)

    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_hz * N_CTX
    m_ptrs = M + off_hz * N_CTX
    # initialize dv amd dk
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    # k and v stay in SRAM throughout
    k = tl.load(k_ptrs)
    v = tl.load(v_ptrs)
    # loop over rows

    layout_ptr = layout_ccol_ptr + off_h * layout_ccol_stride_h + start_n * layout_ccol_stride_m
    start_l = tl.load(layout_ptr).to(tl.int32)
    end_l = tl.load(layout_ptr + layout_ccol_stride_m).to(tl.int32)

    for row_idx_idx in range(start_l, end_l):
        row_idx = tl.load(layout_row_ptr + off_h * layout_row_stride_h + row_idx_idx * layout_row_stride_m).to(tl.int32)
        start_m = row_idx * BLOCK_M

        # offs_qm = start_m + tl.arange(0, BLOCK_M)
        offs_m_curr = start_m + offs_m
        q_ptrs =   Q + (offs_m_curr[:, None] * stride_qm + offs_d[None, :] * stride_qd)
        do_ptrs = DO + (offs_m_curr[:, None] * stride_om + offs_d[None, :] * stride_od)
        dq_ptrs = DQ + (offs_m_curr[:, None] * stride_om + offs_d[None, :] * stride_od)

        # load q, k, v, do on-chip
        q = tl.load(q_ptrs)
        # re-compute p = softmax(qk, dim=-1).T
        # NOTE: `do` is pre-divided by `l`; no normalization here
        qk = tl.dot(q, tl.trans(k))

        qk += tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), 0, float('-inf'))

        m = tl.load(m_ptrs + offs_m_curr)

        p = tl.exp(qk * sm_scale - m[:, None])

        # compute dv
        do = tl.load(do_ptrs)


        dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)



        Di = tl.load(D_ptrs + offs_m_curr)

        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
        dp += tl.dot(do, tl.trans(v))

        # compute ds = p * (dp - delta[:, None])
        ds = p * dp * sm_scale
        # compute dk = dot(ds.T, q)
        dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)


        # # compute dq
        dq = tl.dot(ds.to(Q.dtype.element_ty), k)
        tl.atomic_add(dq_ptrs, dq)


    # write-back
    dv_ptrs = DV + (offs_n[:, None] * stride_om + offs_d[None, :] * stride_od)
    dk_ptrs = DK + (offs_n[:, None] * stride_om + offs_d[None, :] * stride_od)
    tl.store(dv_ptrs, dv)
    tl.store(dk_ptrs, dk)


def _forward(ctx, q, k, v, block_sparse_dense, sm_scale, num_stages=1, out=None):
    '''
    :param q, k, v: [batch, n_heads, seq_len, model_dim]. len of q is allowed to be different than k/v.
    :param layout_crow_indices, layout_col_indices: same as CSR.crow_indices, and CSR.col_indices used to preresent a sparse tensor.
        Each element represent a block, i.e, all elements in a block to be attentdd, or not attended at all..
    '''

    layout_crow_indices, layout_col_indices = dense_to_crow_col(block_sparse_dense)

    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert k.shape[2] == v.shape[2]
    o = out if out is not None else torch.empty_like(q).contiguous()
    BLOCK_M, BLOCK_N, = 64, 64
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1])

    q_rounded_len = grid[0] * BLOCK_M
    tmp = torch.empty((q.shape[0] * q.shape[1], q_rounded_len), device=q.device, dtype=torch.float32)

    L = torch.empty((q.shape[0] * q.shape[1], q_rounded_len), device=q.device, dtype=torch.float32)
    m = torch.empty((q.shape[0] * q.shape[1], q_rounded_len), device=q.device, dtype=torch.float32)

    if layout_col_indices.dim() == 1:
        layout_crow_indices = layout_crow_indices[None].expand(q.shape[1] , -1)
        layout_col_indices = layout_col_indices[None].expand(q.shape[1] , -1)

    assert q.shape[-1] in [64, 128]
    

    MIN_D = min(BLOCK_M, BLOCK_N)
    num_warps = max(1, 2 ** int(math.log2(MIN_D / 16)))
    # num_warps = 4


    _fwd_kernel[grid](
        q, k, v, sm_scale,
        layout_crow_indices,
        layout_col_indices,
        layout_crow_indices.stride(0), layout_crow_indices.stride(1),
        layout_col_indices.stride(0), layout_col_indices.stride(1),
        tmp, L, m,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], k.shape[2],
        k.shape[2] - q.shape[2],
        q_rounded_len,
        BLOCK_M=BLOCK_M, 
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=q.shape[-1],
        num_warps=num_warps,
        num_stages=num_stages,
    )

    ctx.save_for_backward(q, k, v, o, L, m, block_sparse_dense)
    ctx.BLOCK_M = BLOCK_M
    ctx.BLOCK_N = BLOCK_N
    ctx.grid = grid
    ctx.sm_scale = sm_scale
    ctx.num_warps = num_warps
    ctx.num_stages = num_stages
    return o


def _backward(ctx, do, dq=None, dk=None, dv=None):
    # q, k, v, o, l, m = ctx.saved_tensors
    q, k, v, o, l, m, block_sparse_dense = ctx.saved_tensors

    ## this following too slow to do online, so get it from inputs, which is cached.
    # layout_ccol_indices, layout_row_indices = dense_to_ccol_row(crow_col_to_dense(ctx.layout_crow_indices, ctx.layout_col_indices))

    layout_ccol_indices, layout_row_indices = dense_to_ccol_row(block_sparse_dense)

    if not do.is_contiguous():
        do = do.contiguous()
        ## for debugging
        # print(f'----> do is not contiguous: {do.stride()=}')
        # raise ValueError(f'>>>> output grad is not contiguous: {do.stride()=}')

    if not o.is_contiguous():
        # TODO: currently only work with contiguous q/k/v.
        raise ValueError(f'--> output is not contiguous: {o.stride()=}. This is maybe caused by q/k/v not being contiguous.')


    if layout_ccol_indices.dim() == 1:
        layout_ccol_indices = layout_ccol_indices[None].expand(q.shape[1], -1)
        layout_row_indices = layout_row_indices[None].expand(q.shape[1], -1)

    # do = do.contiguous()
    dq = dq if dq is not None else torch.zeros_like(q, dtype=torch.float32)
    dk = dk if dk is not None else torch.empty_like(k)
    dv =dv if dv is not None else  torch.empty_like(v)
    do_scaled = torch.empty_like(do)
    delta = torch.empty_like(l)

    assert o.stride() == dq.stride() == dk.stride() == dv.stride() == do_scaled.stride()

    _bwd_preprocess[(ctx.grid[0] * ctx.grid[1], )](
        o, do, l,
        do_scaled, delta,
        k.shape[2],
        BLOCK_M=ctx.BLOCK_M, D_HEAD=q.shape[-1],
    )

    grid = (triton.cdiv(q.shape[2], ctx.BLOCK_N), ctx.grid[1])

    _bwd_kernel[grid](
        q, k, v, ctx.sm_scale,
        layout_ccol_indices,
        layout_row_indices,
        layout_ccol_indices.stride(0), layout_ccol_indices.stride(1),
        layout_row_indices.stride(0), layout_row_indices.stride(1),
        o, do_scaled,
        dq, dk, dv,
        l, m,
        delta,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        ctx.grid[0],
        BLOCK_M=ctx.BLOCK_M,
        BLOCK_N=ctx.BLOCK_N,
        BLOCK_DMODEL=q.shape[-1],
        num_warps=ctx.num_warps,
        num_stages=1,
    )
    return dq, dk, dv, None, None, None


class _sparse_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, block_sparse_dense, sm_scale):
        # shape constraints
        return _forward(ctx, q, k, v, block_sparse_dense, sm_scale)

    @staticmethod
    def backward(ctx, do):
        return _backward(ctx, do)




def sparse_attention_factory(BLOCK_M=64, BLOCK_N=64, **kwargs):
    class _sparse_attention_config(_sparse_attention):
        @staticmethod
        def forward(ctx, q, k, v, block_sparse_dense, sm_scale):
            # shape constraints
            return _forward(ctx, q, k, v, block_sparse_dense, sm_scale,
                            **kwargs
                        )
    return _sparse_attention_config.apply

block_sparse_triton_fn = sparse_attention_factory()

