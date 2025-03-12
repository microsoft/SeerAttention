import torch
import triton
import triton.language as tl
import math
import time

import os


def is_cuda():
    return triton.runtime.driver.cu_driver.get_current_target().backend == "cuda"


def is_hip_mi200():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

configs = [
    triton.Config({'BLOCK_SIZE_D': BD}, num_stages=s, num_warps=w) \
    for BD in [32, 64, 128]\
    for s in [3, 4, 7]\
    for w in [4, 8]\
]


def keep(conf):
    BLOCK_D = conf.kwargs["BLOCK_SIZE_D"]
    if BLOCK_D < 128 and conf.num_warps == 8:
        return False
    return True


# @triton.autotune(list(filter(keep, configs)), key=["B", "H", "D", "S"])
@triton.jit
def compute_pv_kernel(
    Q_ptr, K_ptr, V_ptr, PV_ptr, SE_ptr, MAX_ptr, MASK_ptr,
    stride_q_b, stride_q_h, stride_q_1, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_pv_b, stride_pv_h, stride_pv_s, stride_pv_d,
    stride_se_b, stride_se_h, stride_se_s,
    stride_max_b, stride_max_h, stride_max_s,
    stride_mask_b, stride_mask_h, stride_mask_s,
    B, H, D, S, sm_scale,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
):
    n_bh = tl.program_id(0)
    pid = tl.program_id(1)
    offset_b = n_bh // H
    offset_h = n_bh % H
    q_offset = offset_b.to(tl.int64) * stride_q_b + offset_h.to(tl.int64) * stride_q_h
    k_offset = offset_b.to(tl.int64) * stride_k_b + offset_h.to(tl.int64) * stride_k_h
    v_offset = offset_b.to(tl.int64) * stride_v_b + offset_h.to(tl.int64) * stride_v_h
    pv_offset = offset_b.to(tl.int64) * stride_pv_b + offset_h.to(tl.int64) * stride_pv_h
    se_offset = offset_b.to(tl.int64) * stride_se_b + offset_h.to(tl.int64) * stride_se_h + pid * stride_se_s
    max_offset = offset_b.to(tl.int64) * stride_max_b + offset_h.to(tl.int64) * stride_max_h + pid * stride_max_s
    mask_offset = offset_b.to(tl.int64) * stride_mask_b + offset_h.to(tl.int64) * stride_mask_h + pid * stride_mask_s

    output_dtype = PV_ptr.dtype.element_ty
    
    mask = tl.load(MASK_ptr + mask_offset)
    if mask == False:
        return

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + q_offset,
        shape=(1, D),
        strides=(stride_q_1, stride_q_d),
        offsets=(0, 0),
        block_shape=(1, BLOCK_SIZE_D),
        order=(0, 1),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + k_offset,
        shape=(D, S),
        strides=(stride_k_d, stride_k_s),
        offsets=(0, pid * BLOCK_SIZE_S),
        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_S),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + v_offset,
        shape=(S, D),
        strides=(stride_v_s, stride_v_d),
        offsets=(pid * BLOCK_SIZE_S, 0),
        block_shape=(BLOCK_SIZE_S, BLOCK_SIZE_D),
        order=(0, 1),
    )

    PV_block_ptr = tl.make_block_ptr(
        base=PV_ptr + pv_offset,
        shape=(1, D),
        strides=(stride_pv_s, stride_pv_d),
        offsets=(pid, 0),
        block_shape=(1, BLOCK_SIZE_D),
        order=(0, 1),
    )

    qk = tl.zeros((1, BLOCK_SIZE_S), dtype=tl.float32)
    for m in range(0, tl.cdiv(D, BLOCK_SIZE_D)):
        q = tl.load(Q_block_ptr)
        k = tl.load(K_block_ptr, boundary_check=(1, ))
        q = q * sm_scale
        q = q.to(tl.float32)
        k = k.to(tl.float32)
        qk += tl.sum(q[:, :, None] * k, 1) 
        Q_block_ptr = tl.advance(Q_block_ptr, (0, BLOCK_SIZE_D))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_SIZE_D, 0))

    qk = tl.where(tl.arange(0, BLOCK_SIZE_S) + pid * BLOCK_SIZE_S < S, qk, float("-inf"))
    m = tl.max(qk, axis=1)
    qk = qk - m
    p = tl.math.exp2(qk * 1.44269504)
    exp_sum = tl.sum(p, axis=1)
    p = p / exp_sum

    offset_zero = tl.arange(0, 1)
    tl.store(SE_ptr + se_offset + offset_zero, exp_sum)
    tl.store(MAX_ptr + max_offset + offset_zero, m)

    for m in range(0, tl.cdiv(D, BLOCK_SIZE_D)):
        v = tl.load(V_block_ptr, boundary_check=(0, ))
        v = v.to(tl.float32)
        o = tl.sum(p[:, :, None] * v, 1)
        o = o.to(output_dtype)
        tl.store(PV_block_ptr, o)
        V_block_ptr = tl.advance(V_block_ptr, (0, BLOCK_SIZE_D))
        PV_block_ptr = tl.advance(PV_block_ptr, (0, BLOCK_SIZE_D))



# @triton.autotune(list(filter(keep, configs)), key=["B", "H", "D", "S"])
@triton.jit
def merge_pv_kernel(
    PV_ptr, SE_ptr, MAX_ptr, O_ptr,
    stride_pv_b, stride_pv_h, stride_pv_s, stride_pv_d,
    stride_se_b, stride_se_h, stride_se_s,
    stride_max_b, stride_max_h, stride_max_s,
    stride_o_b, stride_o_h, stride_o_1, stride_o_d,
    B, H, D, S,
    SPLIT_S: tl.constexpr, SPLIT_S_POW2: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
):

    n_bh = tl.program_id(0)
    offset_b = n_bh // H
    offset_h = n_bh % H
    pv_offset = offset_b.to(tl.int64) * stride_pv_b + offset_h.to(tl.int64) * stride_pv_h
    se_offset = offset_b.to(tl.int64) * stride_se_b + offset_h.to(tl.int64) * stride_se_h
    max_offset = offset_b.to(tl.int64) * stride_max_b + offset_h.to(tl.int64) * stride_max_h
    o_offset = offset_b.to(tl.int64) * stride_o_b + offset_h.to(tl.int64) * stride_o_h


    offset_split_s = tl.arange(0, SPLIT_S_POW2)
    mask_split_s = offset_split_s < SPLIT_S
    se_vector = tl.load(SE_ptr + se_offset + offset_split_s * stride_se_s, mask=mask_split_s, other=0)
    m_vector = tl.load(MAX_ptr + max_offset + offset_split_s * stride_max_s, mask=mask_split_s, other=float("-inf"))

    max = tl.max(m_vector)
    up_scale = tl.math.exp2((m_vector - max) * 1.44269504)
    sum = tl.sum(se_vector * up_scale)
    scale = se_vector / sum * up_scale


    PV_block_ptr = tl.make_block_ptr(
        base=PV_ptr + pv_offset,
        shape=(SPLIT_S, D),
        strides=(stride_pv_s, stride_pv_d),
        offsets=(0, 0),
        block_shape=(SPLIT_S_POW2, BLOCK_SIZE_D),
        order=(0, 1),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + o_offset,
        shape=(1, D),
        strides=(stride_o_1, stride_o_d),
        offsets=(0, 0),
        block_shape=(1, BLOCK_SIZE_D),
        order=(0, 1),
    )

    
    for m in range(0, tl.cdiv(D, BLOCK_SIZE_D)):
        pv = tl.load(PV_block_ptr, boundary_check=(0, ))
        pv = pv.to(tl.float32)
        o = tl.sum(pv * scale[:, None], 0)
        o = o.to(O_ptr.dtype.element_ty)
        tl.store(O_block_ptr, o[None, :])
        PV_block_ptr = tl.advance(PV_block_ptr, (0, BLOCK_SIZE_D))
        O_block_ptr = tl.advance(O_block_ptr, (0, BLOCK_SIZE_D))


def block_sparse_decode(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, nnz_id: torch.Tensor, sm_scale: float) -> torch.Tensor:
    # assert Q.is_cuda and K.is_cuda and V.is_cuda and nnz_id.is_cuda
    # assert Q.shape[0] == K.shape[0] == V.shape[0]
    # assert Q.shape[1] == K.shape[1] == V.shape[1]
    # assert Q.shape[2] == 1
    # assert Q.shape[3] == K.shape[3] == V.shape[3]

    B, H, S, D = K.shape

    BLOCK_SIZE_D = 32
    BLOCK_SIZE_S = 64
    num_stages = 4
    num_warps = 4
    
    device = Q.device

    dtype = K.dtype
    P = torch.empty((B, H, 1, S), device=device, dtype=dtype)
    SPLIT_S = math.ceil(S / BLOCK_SIZE_S) 
    SPLIT_S_POW2 = triton.next_power_of_2(SPLIT_S)
    SE = torch.zeros((B, H, SPLIT_S), device=device, dtype=dtype)
    PV = torch.zeros((B, H, SPLIT_S, D), device=device, dtype=dtype)
    MAX = torch.full((B, H, SPLIT_S), float('-inf'), device=device, dtype=dtype)
    Scale = torch.empty((B, H, SPLIT_S), device=device, dtype=dtype)
    Output = torch.empty((B, H, 1, D), device=device, dtype=dtype)

    grid_0 = lambda META: (B * H, triton.cdiv(S, META['BLOCK_SIZE_S']))
    grid_1 = (B * H, 1, 1)


    with torch.cuda.device(Q.device.index): 
        compute_pv_kernel[grid_0](
            Q, K, V, PV, SE, MAX, nnz_id,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            PV.stride(0), PV.stride(1), PV.stride(2), PV.stride(3),
            SE.stride(0), SE.stride(1), SE.stride(2),
            MAX.stride(0), MAX.stride(1), MAX.stride(2),
            nnz_id.stride(0), nnz_id.stride(1), nnz_id.stride(2),
            B, H, D, S, sm_scale,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            num_stages=num_stages,
            num_warps=num_warps,
        )

    

        merge_pv_kernel[grid_1](
            PV, SE, MAX, Output,
            PV.stride(0), PV.stride(1), PV.stride(2), PV.stride(3),
            SE.stride(0), SE.stride(1), SE.stride(2),
            MAX.stride(0), MAX.stride(1), MAX.stride(2),
            Output.stride(0), Output.stride(1), Output.stride(2), Output.stride(3),
            B, H, D, S,
            SPLIT_S=SPLIT_S,
            SPLIT_S_POW2=SPLIT_S_POW2,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            num_stages=4,
            num_warps=num_warps,
        )

    
    return Output

def block_sparse_decode_pt(q, k, v, nnz_id, block_size, sm_scale):
    # Initialize output tensor
    o = torch.empty_like(q)

    q = q * sm_scale

    # Compute attention scores
    attn_scores = torch.matmul(q, k.transpose(-1, -2))

    # Apply sparse mask
    mask = torch.zeros_like(attn_scores)
    for batch_idx in range(nnz_id.shape[0]):
        for head_idx in range(nnz_id.shape[1]):
            for block_idx in range(nnz_id.shape[2]):
                if nnz_id[batch_idx, head_idx, block_idx] == 1:
                    mask[batch_idx, head_idx, 0, block_idx * block_size:(block_idx + 1) * block_size] = 1

    attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

    # Compute attention output
    attn_probs = torch.softmax(attn_scores, dim=-1)
    o = torch.matmul(attn_probs, v)
    return o
