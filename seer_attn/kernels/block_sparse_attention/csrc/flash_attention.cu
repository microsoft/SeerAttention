#include "attention_api.cuh"
#include <cassert>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <torch/extension.h>
#include <torch/python.h>
#include <vector>

#include "static_switch.h"
#include "kernel_traits.h"
#include "flash.h"
#include "utils.h"
#include "block_info.h"

namespace flash {

using namespace cute;

// // bool Is_causal, bool Is_even_MN, 
// template <int kBlockM, int kBlockN, int kNWarps,typename Engine, typename Layout>
// inline __device__ void mask_within_nblock(Tensor<Engine, Layout> &tensor, const int m_block, const int nbi) {
//     // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
//     static_assert(Layout::rank == 2, "Only support 2D Tensor");
//     // NOTE:
//     // Determining the index within an MMA is also a challenge
//     // (nrow=(2, MMA_M), ncol=(2, MMA_N)) looks like:
//     //    T1.V0 T1.V1
//     //    T1.V0 T1.V1
//     // Determine col and row values based on the mma_tile diagram

//     // NOTE:
//     // Calculate the processing range of the thread, mask out the parts beyond the range
//     //
//     // NOTE:
//     // % 32 means grouping by 32, because the maximum thread id in SM80_16x8x16_F32F16F16F32_TN _1_2_1 is 32
//     // (lane_id % 4) * 2 indicates which "color" col(thread) it is in, *2 is to align to the right (i.e., which value2 is being processed)
//     // Therefore, col_idx_offset represents which column in the 4 columns of the single Atom the current thread is processing

//     // lane_id represents a "thread group" in an MMA tile
//     const int lane_id = threadIdx.x % 32;
//     const int col_idx_offset = kBlockN * nbi + (lane_id % 4) * 2;

//     const int nrow_group = threadIdx.x / 32;
//     const int row_idx_offset = kBlockM * m_block + lane_id / 4 + nrow_group * 16 /* 2*8 */;
//     // (2, nrow), 2*8 for each
//     const int group_stride = kNWarps * 16;

//     #pragma unroll
//     for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
//         // In SM80_16x8x16_F32F16F16F32_TN, a group of 4 threads processes 8 values in a row
//         const int col_idx_base = col_idx_offset + nj * 8;
//         #pragma unroll
//         for (int j = 0; j < size<1, 0>(tensor); ++j) {
//             // j is used to calculate the col for value 1 and value 2
//             // col_idx ultimately represents the column number of the value being processed by the current thread
//             const int col_idx = col_idx_base + j;

//             // Mask out the parts of the scores (result after QK) that are beyond the range
//             // Compare column and row numbers

//             // Without the "make_coord" we get wrong results
//             // for nrow(2, MMA_M)
//             #pragma unroll
//             for (int mi = 0; mi < size<0, 0>(tensor); ++mi) {

//               #pragma unroll
//               for (int mj = 0; mj < size<0, 1>(tensor); ++mj) {
//                 const int row_idx = row_idx_offset + mi * 8 + mj * group_stride;
//                 if (col_idx > row_idx) {
//                   tensor(make_coord(mi, mj), make_coord(j, nj)) = -INFINITY;
//                 }
//               }

//             }

//         }
//     }
// }

template <
    int kBlockM, int kBlockN, int kNWarps,
    bool Is_causal, bool Is_even_MN,
    typename Engine, typename Layout
>
inline __device__ void mask_within_nblock(
    Tensor<Engine, Layout> &tensor,
    const int m_block, const int nbi,
    const int max_seqlen_k = -1  // 仅当 Is_even_MN=false 时需要
) {
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    static constexpr bool Need_masking = Is_causal || !Is_even_MN;
    if constexpr (Need_masking) {
        const int lane_id = threadIdx.x % 32;
        const int col_idx_offset = kBlockN * nbi + (lane_id % 4) * 2;
        const int nrow_group = threadIdx.x / 32;
        const int row_idx_offset = kBlockM * m_block + lane_id / 4 + nrow_group * 16;
        const int group_stride = kNWarps * 16;

        #pragma unroll
        for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
            const int col_idx_base = col_idx_offset + nj * 8;
            #pragma unroll
            for (int j = 0; j < size<1, 0>(tensor); ++j) {
                const int col_idx = col_idx_base + j;
                #pragma unroll
                for (int mi = 0; mi < size<0, 0>(tensor); ++mi) {
                    #pragma unroll
                    for (int mj = 0; mj < size<0, 1>(tensor); ++mj) {
                        const int row_idx = row_idx_offset + mi * 8 + mj * group_stride;
                        // Causal mask
                        if constexpr (Is_causal) {
                            if (col_idx > row_idx) {
                                tensor(make_coord(mi, mj), make_coord(j, nj)) = -INFINITY;
                                continue;
                            }
                        }
                        // Padding mask (for uneven MN)
                        if constexpr (!Is_even_MN) {
                            if (col_idx >= max_seqlen_k) {
                                tensor(make_coord(mi, mj), make_coord(j, nj)) = -INFINITY;
                            }
                        }
                    }
                }
            }
        }
    }
}



// NOTE: GEMM encapsulation with matrix A already in registers
template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
inline __device__ void gemm_A_in_regs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                                      TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                                      ThrCopy smem_thr_copy_B) {
    // NOTE: Conforms to M N K description: A[M, K] @ B[N, K] = C[M, N]
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    // NOTE: Retile to the size needed for copying
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

template<typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
inline __device__ void gemm_smem(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    // NOTE: s -> reg
    cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

// Blocks until all but N previous cp.async.commit_group operations have committed.
// This differs from cute::cp_async_wait in that when N = 0 we don't call cp.async.wait_all
// (which is equivalent to commit_group then wait_group 0).
// Instead we just call cp.async.wait_group 0, which is slightly faster.
// https://github.com/NVIDIA/cutlass/blob/master/include/cute/arch/copy_sm80.hpp#L113
template <int N>
CUTE_HOST_DEVICE
void cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

// copy from S to D with tiled_copy
// TODO: Need to support skipping copy in causal mode
template <bool Is_even_MN=true, bool Is_even_K=true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
inline __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                            Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                }
            }
        }
    }
}


// Convert rowcol_layout from (nrow=(2, MMA_M), ncol=(2, MMA_N)) to ((2, 2, 2), MMA_M, MMA_N / 2)
// if using m16n8k16, or to ((2, 2, 1), MMA_M, MMA_N) if using m16n8k8.
template<typename MMA_traits, typename Layout>
inline __device__ auto convert_layout_rowcol_Aregs(Layout rowcol_layout) {
    using X = Underscore;
    static_assert(decltype(size<0, 0>(rowcol_layout))::value == 2);
    static_assert(decltype(size<1, 0>(rowcol_layout))::value == 2);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    constexpr int MMA_N_divisor = mma_shape_K == 8 ? 1 : 2;
    auto l = logical_divide(rowcol_layout, Shape<X, Shape<X, Int<MMA_N_divisor>>>{});  // ((2, MMA_M), (2, (2, MMA_N / 2)))
    // TD [2023-08-13]: Same error as above on Cutlass 3.2
    // return make_layout(make_layout(get<1, 0>(l), get<0, 0>(l), get<1, 1, 0>(l)),
    //                    get<0, 1>(l),
    //                    get<1, 1, 1>(l));
    return make_layout(make_layout(get<0>(get<1>(l)), get<0>(get<0>(l)), get<0>(get<1>(get<1>(l)))),
                       get<1>(get<0>(l)),
                       get<1>(get<1>(get<1>(l))));
};


// TODO: not work
template <typename To_type, typename Engine, typename Layout>
inline __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}


template <typename Fragment>
inline __device__ auto convert_type_f32_to_f16(Fragment const &acc_fp32) {
  Tensor acc_fp16 = make_tensor<cute::half_t>(shape(acc_fp32));
  {
    Tensor acc_fp32x2 = recast< float2>(acc_fp32);
    Tensor acc_fp16x2 = recast<__half2>(acc_fp16);
    for (int i = 0; i < size(acc_fp32x2); ++i) { acc_fp16x2(i) = __float22half2_rn(acc_fp32x2(i)); }
  }
  return acc_fp16;
}

// Apply the exp to all the elements.
template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            tensor(mi, ni) = expf(tensor(mi, ni) * scale - max_scaled);
        }
    }
}

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
// TODO: Understand the mathematical meaning after convert_layout_acc_rowcol
// A visual explanation is that it transforms:
//    T1.V0
//    T1.V1
//    T1.V0
//    T1.V1
// into:
//    T1.V0 T1.V1
//    T1.V0 T1.V1
// This aligns with the row-column intuition of the MMA tile.
template<typename Layout>
inline __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
    // TD [2023-08-13]: Idk why but get<0, 1>(l) doesn't work for Cutlass 3.2, I'm getting
    // "int_tuple.hpp(74): error: conversion to inaccessible base class"
    // return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
    return make_layout(make_layout(get<1>(get<0>(l)), get<1>(l)), make_layout(get<0>(get<0>(l)), get<2>(l)));
};

template<bool Is_first, typename Tensor0, typename Tensor1, typename Tensor2>
inline __device__ void softmax_rescale_o(Tensor0 &scores, Tensor1 &scores_max, Tensor1 &scores_sum,
                                         Tensor2 &acc_o, float softmax_scale_log2) {
    // NOTE: scores come from acc_s: Q@K.T
    // acc_s is used to store the result of QK and softmax [seqlen, seqlen]
    // acc_o is used to store the numerator part of the softmax(QK) result, for rescaling
    // Streaming computation continuously rescales with the current block computation result scores

    if (Is_first) {
        // NOTE: Optimization, the first softmax does not need rescaling, only needs to record the numerator, max, sum
        reduce_max</*zero_init=*/true>(scores, scores_max);
        flash::scale_apply_exp2(scores, scores_max, softmax_scale_log2);
        reduce_sum(scores, scores_sum);
    } else {
        // Record the previous max
        Tensor scores_max_prev = make_fragment_like(scores_max);
        cute::copy(scores_max, scores_max_prev);
        // TODO: Learn the implementation of reduce
        // NOTE: Calculate the new max into scores_max
        // reduce_max includes steps:
        //  1. Calculate the max within the current thread: iterate
        //  2. Reduce the max across threads: use shift trick to reduce
        reduce_max</*zero_init=*/false>(scores, scores_max);
        // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
        // Convert acc_o into a shape that aligns with 2D intuition (nrow, ncol)
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        #pragma unroll
        for (int mi = 0; mi < size(scores_max); ++mi) {
            // NOTE: Auxiliary variable: current max
            float scores_max_cur = scores_max(mi);
            // NOTE: Calculate the rescale value for the old score
            // NOTE: Since QK (affecting max) was calculated without considering softmax_scale, we need to compensate here
            float scores_scale = expf((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
            // NOTE: Rescale the old denominator part
            scores_sum(mi) *= scores_scale;
            // NOTE: Rescale the old numerator part
            // acc_o_rowcol.shape = (nrow, ncol)
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
        }
        // NOTE: Calculate the new numerator part: rescale all scores
        flash::scale_apply_exp2(scores, scores_max, softmax_scale_log2);

        // NOTE: Accumulate the new denominator
        Tensor scores_sum_cur = make_fragment_like(scores_sum);
        // NOTE: Use the new numerator to accumulate the new denominator
        //  1. Accumulate within the thread: iterate
        //  2. Accumulate across threads: use shift trick to reduce
        reduce_sum(scores, scores_sum_cur);
        // NOTE: Add the new denominator to the old denominator
        #pragma unroll
        for (int mi = 0; mi < size(scores_sum); ++mi) { scores_sum(mi) += scores_sum_cur(mi); }
    }
};

} // namespace flash

void set_params_fprop(Flash_fwd_params &params,
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *softmax_lse_d,
                      float softmax_scale,
                      bool is_causal) {

  memset(&params, 0, sizeof(params));

  params.b = b;
  params.h = h;
  params.h_k = h_k;
  params.h_h_k_ratio = h / h_k;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.seqlen_q_rounded = seqlen_q_rounded;
  params.seqlen_k_rounded = seqlen_k_rounded;
  params.d = d;
  params.d_rounded = d_rounded;
  
  // params.bs = q.size(0);
  // params.head = q.size(1);
  // params.q_seqlen = q.size(2);
  // params.dim = q.size(3);

  // params.k_head = k.size(1);
  // params.k_seqlen = k.size(2);

  // params.bs_stride = q.stride(0);
  // params.head_stride = q.stride(1);
  // params.seqlen_stride = q.stride(2);
  // params.dim_stride = q.stride(3);

  params.scale_softmax = softmax_scale;
  // TODO: Use log2 for scaling
  params.scale_softmax_log2 = softmax_scale * M_LOG2E;
  params.is_causal = is_causal;
  params.is_bf16 = q.dtype() == torch::kBFloat16;

  // LogSumExp save for backward
  params.softmax_lse_ptr = softmax_lse_d;

  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.o_ptr = out.data_ptr();

  params.q_row_stride = q.stride(-3);
  params.k_row_stride = k.stride(-3);
  params.v_row_stride = v.stride(-3);
  params.o_row_stride = out.stride(-3);

  params.q_head_stride = q.stride(-2);
  params.k_head_stride = k.stride(-2);
  params.v_head_stride = v.stride(-2);
  params.o_head_stride = out.stride(-2);      

  if (cu_seqlens_q_d == nullptr) {
      params.q_batch_stride = q.stride(0);
      params.k_batch_stride = k.stride(0);
      params.v_batch_stride = v.stride(0);
      params.o_batch_stride = out.stride(0);
  }

  params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
  params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);  
  params.seqused_k = static_cast<int *>(seqused_k);

  params.is_causal = is_causal;

  params.is_seqlens_k_cumulative = true;
}


// Shared Storage with Aligned addresses.
template <class ElementType, class SmemLayoutQ, class SmemLayoutK, class SmemLayoutV>
struct SharedStorage {
  // TODO: If aligned, does smem calculation have issues?
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutQ>> smem_q;
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutK>> smem_k;
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutV>> smem_v;
};

template <typename Kernel_traits, bool Is_causal=false, bool Is_even_MN, bool Is_even_K, typename Params>
__global__ void flash_attention_v2_cutlass_kernel(const Params params) {

  using namespace cute;

  const int m_block = blockIdx.x;
  // The block index for the batch.
  const int bidb = blockIdx.y;
  // The block index for the head.
  const int bidh = blockIdx.z;

  // The thread index.
  const int tidx = threadIdx.x;

  // TODO: Pass in generics
  // NOTE: Small trick
  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  // using TiledMMA = typename Kernel_traits::MMA;
  using TiledMMA = typename Kernel_traits::TiledMma;
  using index_t = typename Kernel_traits::index_t;
  using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
  using SmemLayoutK = typename Kernel_traits::SmemLayoutKV;
  using SmemLayoutV = typename Kernel_traits::SmemLayoutKV;
  using SmemLayoutVt = typename Kernel_traits::SmemLayoutVtransposed;
  using SmemLayoutVtNoSwizzle = typename Kernel_traits::SmemLayoutVtransposedNoSwizzle;

  constexpr int kNWarps = Kernel_traits::kNWarps;
  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr int kBlockN = Kernel_traits::kBlockN;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;

  const flash::BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
  if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

  const int n_block_min = 0;
  int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
  if (Is_causal) {
      n_block_max = std::min(n_block_max,
                             cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q, kBlockN));
  }

  // Shared memory.
  extern __shared__ char smem_[];
  using SharedStorage = SharedStorage<Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>;
  SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(smem_);

  // const int bs_head_offset = base_id * params.head_stride;

  const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
      + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
  const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
  const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
      + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
  // We move K and V to the last block.
  const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
      + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
  const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
      + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;

  // if (m_block == 0 && bidb == 0 && bidh == 0){
  //   printf("[calculate mask_ptr 1]: bidb %d, params.q_batch_stride:%" PRId64 ", bidh %d, params.q_head_stride:%" PRId64 ", n_block_max %d  \n", bidb, params.q_batch_stride, bidh, params.q_head_stride, n_block_max);
  // }

  // TODO: base offset for MHA
  // NOTE: convert C pointer to Tensor for convenience
  Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
  Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
  Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
  // Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
  //                               Shape<Int<kBlockM>, Int<kHeadDim>>{},
  //                               make_stride(params.o_row_stride, _1{}));
  // TODO:
  // Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
  //                                 Shape<Int<kBlockM>>{}, Stride<_1>{});

  // Get MMA abstraction
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tidx);

  // Construct SMEM tensors.
  Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
  Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

  // Tensor for V Transpose; used in GEMM-II.
  Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVt{});
  Tensor sVtNoSwizzle = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVtNoSwizzle{});

  // NOTE: Copy abstraction
  // NOTE: QKV gmem -> smem copy abstraction
  typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

  // NOTE: Define src, dst for gmem -> smem copy
  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
  Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
  Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
  Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
  Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);


  // NOTE: Define dst for smem -> reg copy
  // partition_fragment is similar to partition, but returns a register representation
  Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
  Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
  Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

  //
  // Copy Atom retiling
  //

  // TODO: Understand the atom retiling here

  // NOTE: Prepare copy objects for Q, K, V to smem
  auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

  auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
  Tensor tSsK = smem_thr_copy_K.partition_S(sK);

  // TODO: Transpose during copy
  // NOTE: smem->reg copy Vt
  auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
  Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

  Tensor rAccOut = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
  
    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

  // Pipeline loading initial Q, K
  // Load Q to smem
  // flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ);
  flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                       binfo.actual_seqlen_q - m_block * kBlockM);  
  // Load K to smem
  flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k - (n_block_max - 1) * kBlockN);
  // Start async copy
  cute::cp_async_fence();

  // step1: slice-k compute QK block
  // Q[BLOCK_M, BLOCK_N] @ K[BLOCK_M, BLOCK_N].T = O[BLOCK_M, BLOCK_M]
  //
  // step2:
  // advance K, V

  // NOTE: Number of K, V blocks: processing range
  // const int n_block_min = 0;
  // NOTE: 1. mask between N BLOCKs if is causal mode
  int seqlen_start = m_block * kBlockM;
  int seqlen_end = (m_block + 1) * kBlockM;
  // int n_block_max = Is_causal ? cute::ceil_div(seqlen_end, kBlockN) : cute::ceil_div(params.k_seqlen, kBlockN);

  // NOTE: Max to be recorded
  Tensor scores_max = make_tensor<ElementAccum>(Shape<Int<2 * size<1>(rAccOut)>>{});
  // NOTE: Denominator to be recorded
  Tensor scores_sum = make_fragment_like(scores_max);

  clear(rAccOut);

  for (int nbi = n_block_max-1; nbi >= n_block_min; nbi--) {
    auto rAccScore = partition_fragment_C(tiled_mma, make_shape(Int<kBlockM>{}, Int<kBlockN>{}));

    clear(rAccScore);

    // Wait for Q, K gmem -> smem copy to complete, i.e., Q, K ready
    // wait<0> means wait for 0 remaining
    flash::cp_async_wait<0>();
    __syncthreads();

    // Asynchronously load V while doing gemm
    // gV = local_tile(V, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi, _));
    // tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));    
    // tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));      
    if (nbi < n_block_max - 1) {    
      // tVgV.data() = tVgV.data() - int(kBlockN * params.v_row_stride);    
      tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
    }

    // Asynchronously load V to smem
    if (nbi == n_block_max - 1) {
      flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k - nbi * kBlockN);
    }
    else if (nbi < n_block_max - 1) {
      flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
    }

    // Initiate async copy
    cute::cp_async_fence();

    // O = Q@K.T
    // NOTE: Load data from smem into registers before performing GEMM, **retile during loading**
    flash::gemm_smem(rAccScore, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
        smem_thr_copy_Q, smem_thr_copy_K
    );

    Tensor scores = make_tensor(rAccScore.data(), flash::convert_layout_acc_rowcol(rAccScore.layout()));

    // NOTE: 2. Mask within N BLOCKs
    if (Is_causal == true && nbi * kBlockN >= seqlen_start) {
      // flash::mask_within_nblock<kBlockM, kBlockN, kNWarps>(scores, m_block, nbi);
      int max_seqlen_k_value = Is_even_MN ? -1 : params.seqlen_k;
      flash::mask_within_nblock<kBlockM, kBlockN, kNWarps, Is_causal, Is_even_MN>(scores, m_block, nbi, max_seqlen_k_value);
    }

    // NOTE: Wait for V to finish loading, prepare the initial state for the next K load
    flash::cp_async_wait<0>();
    __syncthreads();

    // Advance K
    if (nbi > n_block_min) {
      // gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi + 1, _));
      // tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
      tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));     
      // flash::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
      flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
      cute::cp_async_fence();
    }

    // Compute softmax
    // NOTE: rAccOut records all numerators after softmax
    nbi == n_block_max-1 ? flash::softmax_rescale_o</*Is_first=*/true>(scores, scores_max, scores_sum, rAccOut, params.scale_softmax) :
      flash::softmax_rescale_o</*Is_first=*/false>(scores, scores_max, scores_sum, rAccOut, params.scale_softmax);

    // Perform QK @ V computation
    // (score AKA rAccScore): QK[M, N] @ V[N, dim]
    // NOTE: DABC: F32F16F16F32, convert D type (F32) to A type (F16)
    // TODO: convert_type is currently hardcoded
    Tensor rP = flash::convert_type_f32_to_f16(rAccScore);
    // NOTE: Convert from layout C to layout A
    Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_rowcol_Aregs<TiledMMA>(scores.layout()));

    flash::gemm_A_in_regs(rAccOut, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
  }

  // Epilogue

  // NOTE: Finally, divide by the denominator
  // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
  // AKA reshape to (nrow, ncol) but with specific MMA layout
  Tensor acc_o_rowcol = make_tensor(rAccOut.data(), flash::convert_layout_acc_rowcol(rAccOut.layout()));
  // NOTE: Save lse for backward pass
  Tensor lse = make_fragment_like(scores_sum);
  // For row
  #pragma unroll
  for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
    float sum = scores_sum(mi);
    float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
    // Compute lse
    // NOTE: Here we use max * scale
    lse(mi) = (sum == 0.f || sum != sum) ? INFINITY : scores_max(mi) * params.scale_softmax + __logf(sum);
    float scale = inv_sum;
    // For col
    #pragma unroll
    for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
      acc_o_rowcol(mi, ni) *= scale;
    }
  }

  // Convert acc_o from fp32 to fp16/bf16
  Tensor rO = flash::convert_type_f32_to_f16(rAccOut);
  // Reuse sQ's smem for copying out sO
  Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)

  // Partition sO to match the accumulator partitioning
  // TODO: Review
  auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
  Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
  Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

  // NOTE: Copy to smem first
  cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

  Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                                Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                make_stride(params.o_row_stride, _1{}));
  // Create copy from smem -> gmem
  typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
  Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
  Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

  __syncthreads();

  // NOTE: Copy to gmem

  // TODO: Review the purpose of these two copies
  Tensor tOrO = make_tensor<Element>(shape(tOgO));
  cute::copy(gmem_tiled_copy_O, tOsO, tOrO);
  // flash::copy(gmem_tiled_copy_O, tOrO, tOgO);
    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );

  // NOTE: Write back lse
  Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});  
  Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
  Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
  static_assert(decltype(size<0>(taccOcO))::value == 4);
  // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
  // TODO: Review this shape
  Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
  CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
  // TODO: Understand the logic here
  if (get<1>(taccOcO_row(0)) == 0) {
      #pragma unroll
      for (int mi = 0; mi < size(lse); ++mi) {
          const int row = get<0>(taccOcO_row(mi));
          gLSE(row) = lse(mi);
      }
  }
}

template<typename Kernel_traits, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
  // TODO: check if works: default stream = 0
  using Element = typename Kernel_traits::Element;
  using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
  using SmemLayoutK = typename Kernel_traits::SmemLayoutKV;
  using SmemLayoutV = typename Kernel_traits::SmemLayoutKV;

  const int num_m_block =
      (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;

  // dim3 grid(num_m_block, params.bs * params.head, 1);
  dim3 grid(num_m_block, params.b, params.h);  
  dim3 block(Kernel_traits::kNThreads);

  const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
  const bool is_even_K = params.d == Kernel_traits::kHeadDim;

  int smem_size = int(sizeof(SharedStorage<Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>));

  BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        auto kernel = &flash_attention_v2_cutlass_kernel<Kernel_traits, Is_causal, IsEvenMNConst, IsEvenKConst, Flash_fwd_params>;
        if (smem_size >= 48 * 1024) {
            CUDA_ERROR_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }      
        kernel<<<grid, block, smem_size>>>(params);
    });
  });
}

template<typename T, int Headdim>
void run_flash_fwd_(Flash_fwd_params &params, cudaStream_t stream);

template<typename T, int Headdim>
void run_flash_fwd_(Flash_fwd_params &params, cudaStream_t stream) {
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, /*kBlockM_=*/128, /*kBlockN_=*/128, /*kNWarps_=*/4, T>, Is_causal>(params, stream);

        // TODO: Combination of kBlockM and kBlockN
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, /*kBlockM_=*/64, /*kBlockN_=*/64, /*kNWarps_=*/4, T>, Is_causal>(params, stream);
    });
}

// Entry point of Flash Attention
void run_flash_attn_cutlass(Flash_fwd_params &params, cudaStream_t stream) {
    // FP16_SWITCH yields elem_type namespace
    FP16_SWITCH(!params.is_bf16, [&] {
        // FWD_HEADDIM_SWITCH yields kHeadDim constexpr
        FWD_HEADDIM_SWITCH(params.d, [&] {
            run_flash_fwd_<elem_type, kHeadDim>(params, stream);
        });
    });
}

std::vector<torch::Tensor> flash_attention_v2_cutlass(torch::Tensor q, torch::Tensor k,
  torch::Tensor v, bool is_causal = false, float softmax_scale=1) {

CHECK_INPUT(q);
CHECK_INPUT(k);
CHECK_INPUT(v);

const auto sizes = q.sizes();
const int batch_size = sizes[0];
int seqlen_q = sizes[1];
int num_heads = sizes[2];
const int head_size_og = sizes[3];
const int seqlen_k = k.size(1);
const int num_heads_k = k.size(2);  
auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
const int seqlen_k_rounded = round_multiple(seqlen_k, 128);  
const int head_size = round_multiple(head_size_og, 8);
const int head_size_rounded = round_multiple(head_size, 32);    

auto out = torch::empty_like(q);
auto opts = q.options();
auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(torch::kFloat32));

Flash_fwd_params params;
set_params_fprop(params,
                batch_size,
                seqlen_q, seqlen_k,
                seqlen_q_rounded, seqlen_k_rounded,
                num_heads, num_heads_k,
                head_size, head_size_rounded,
                q, k, v, out,
                /*cu_seqlens_q_d=*/nullptr,
                /*cu_seqlens_k_d=*/nullptr,
                /*seqused_k=*/nullptr,
                softmax_lse.data_ptr(),
                softmax_scale,
                is_causal);

run_flash_attn_cutlass(params, 0);

// Wait until kernel finishes.
cudaDeviceSynchronize();
CUDA_ERROR_CHECK(cudaGetLastError());

return {out, softmax_lse};
}

std::vector<torch::Tensor> varlen_flash_attention_v2_cutlass(torch::Tensor q, torch::Tensor k,
  torch::Tensor v, torch::Tensor cu_seqlens_q, torch::Tensor cu_seqlens_k, int max_seqlen_q, int max_seqlen_k, bool is_causal = false, float softmax_scale=1) {

CHECK_INPUT(q);
CHECK_INPUT(k);
CHECK_INPUT(v);

CHECK_INPUT(cu_seqlens_q);
CHECK_INPUT(cu_seqlens_k);

const auto sizes = q.sizes();
const int total_q = sizes[0];
const int batch_size = cu_seqlens_q.numel() - 1;
const int num_heads = sizes[1];
const int head_size_og = sizes[2];
const int total_k = k.size(0);
const int num_heads_k = k.size(1);
auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
const int head_size = round_multiple(head_size_og, 8);
const int head_size_rounded = round_multiple(head_size, 32);
const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);  

auto out = torch::empty_like(q);
auto opts = q.options();
auto softmax_lse = torch::empty({batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));

Flash_fwd_params params;
set_params_fprop(params,
                batch_size,
                max_seqlen_q, max_seqlen_k,
                seqlen_q_rounded, seqlen_k_rounded,
                num_heads, num_heads_k,
                head_size, head_size_rounded,
                q, k, v, out,
                cu_seqlens_q.data_ptr(),
                cu_seqlens_k.data_ptr(),
                /*seqused_k=*/nullptr,
                softmax_lse.data_ptr(),
                softmax_scale,
                is_causal);

run_flash_attn_cutlass(params, 0);

// Wait until kernel finishes.
cudaDeviceSynchronize();
CUDA_ERROR_CHECK(cudaGetLastError());

return {out, softmax_lse};
}

template <typename Kernel_traits, bool Is_causal=false, bool Is_even_MN, bool Is_even_K, typename Params, int maskM=64, int maskN=64>
__global__ void flash_attention_block_v2_cutlass_kernel(const Params params) {

  using namespace cute;

  const int m_block = blockIdx.x;
  // The block index for the batch.
  const int bidb = blockIdx.y;
  // The block index for the head.
  const int bidh = blockIdx.z;

  // The thread index.
  const int tidx = threadIdx.x;

  // TODO: Pass in generic type
  // NOTE: Small trick
  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  // using TiledMMA = typename Kernel_traits::MMA;
  using TiledMMA = typename Kernel_traits::TiledMma;
  using index_t = typename Kernel_traits::index_t;
  using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
  using SmemLayoutK = typename Kernel_traits::SmemLayoutKV;
  using SmemLayoutV = typename Kernel_traits::SmemLayoutKV;
  using SmemLayoutVt = typename Kernel_traits::SmemLayoutVtransposed;
  using SmemLayoutVtNoSwizzle = typename Kernel_traits::SmemLayoutVtransposedNoSwizzle;

  constexpr int kNWarps = Kernel_traits::kNWarps;
  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr int kBlockN = Kernel_traits::kBlockN;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;

  const flash::BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
  if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

  const int n_block_min = 0;
  int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
  if (Is_causal) {
      n_block_max = std::min(n_block_max,
                             cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q, kBlockN));
  }

  // Shared memory.
  extern __shared__ char smem_[];
  using SharedStorage = SharedStorage<Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>;
  SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(smem_);

  // TODO: Add assert
  const int row_factor = maskM / kBlockM;
  const int col_factor = maskN / kBlockN;
  // if (m_block == 2 && bidb == 2 && bidh == 2){
  //   printf("maskM: %d\n", maskM); // 64 64 1
  //   printf("kBlockM: %d\n", kBlockM); // 64 64 1
  //   printf("rol_factor: %d\n", row_factor); // 64 64 1    
  //   printf("maskM, kBlockM, rol_factor: %d %d %d\n", maskM, kBlockM, row_factor); // 64 64 1
  //   printf("maskN, kBlockN, col_factor: %d %d %d\n", maskN, kBlockN, col_factor); // 64 64 1
  // }
  const int m_mask = m_block / row_factor;
  // printf("m_block, row_factor, m_mask: %d %d %d\n", m_block, row_factor, m_mask);  // x 1 x
  const int num_n_mask = cute::ceil_div(params.seqlen_k, maskN);
  // printf("params.seqlen_k, maskN, num_n_mask: %d  %d  %d\n", params.seqlen_k, maskN, num_n_mask);  // 512  64  8
  // const int num_n_block = cute::ceil_div(params.seqlen_k, kBlockN);
  const int num_n_block = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);  

  // Add mask start pointer
  // int *blockmask_ptr = params.block_mask_ptr + (batch_idx * params.num_blocksparse_heads + mask_type - 1) * int(params.seqlen_q_rounded / m_block_dim) * int(params.seqlen_k_rounded / n_block_dim) + int(loop_step_idx / row_factor) * int(params.seqlen_k_rounded / n_block_dim);
  // int *mask_ptr = params.block_mask_ptr + base_id * params.mask_head_stride + m_mask * num_n_mask;
  // 
  // mask: [params.b, params.h (num_q_heads), cute::ceil_div(params.seqlen_q, maskM) (ceil(maxlen_q/block)), cute::ceil_div(params.seqlen_k, maskN) (ceil(maxlen_k/block))]
  // q:  b, sq, h, d
  // k, v: b, sk, h_k, d
  // mask: b, h, sq / 64, sk / 64
  int *mask_ptr = params.block_mask_ptr + bidb * params.mask_batch_stride + bidh * params.mask_head_stride + m_mask * num_n_mask; 
  // if (m_block == 7 && bidb == 0 && bidh == 0){
  //   printf("[calculate mask_ptr 1]: bidb %d, params.q_batch_stride:%" PRId64 ", bidh %d, params.q_head_stride:%" PRId64 ", n_block_max %d, m_mask %d, num_n_mask %d  \n", bidb, params.q_batch_stride, bidh, params.q_head_stride, n_block_max, m_mask, num_n_mask);
  //   printf("[calculate mask_ptr 2]: mask_ptr-params.block_mask_ptr: %d  \n", mask_ptr-params.block_mask_ptr);        
  //   printf("[calculate mask_ptr 3]: params.mask_batch_stride:%" PRId64 ", params.mask_head_stride:%" PRId64 ", m_mask %d, num_n_mask %d  \n", params.mask_batch_stride, params.mask_head_stride, m_mask, num_n_mask);    
  // }
  int mask_id = 0, nbi = mask_ptr[0] * col_factor;
  int nbi_prev = nbi;
  // if (m_block == 2 && bidb == 2 && bidh == 2){
  //   printf("initialize nbi: mask_id %d, nbi %d, nbi_prev %d  \n", mask_id, nbi, nbi_prev); 
  //   printf("m_mask %d \n", m_mask);
  // }  
  

  // const int bs_head_offset = base_id * params.q_head_stride;
  const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
      + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
  const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
  const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
      + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
  // We move K and V to the last block.
  const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
      + (n_block_min + nbi) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
  const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
      + (n_block_min + nbi) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;

  // if (m_block == 0) {
  // printf("----------------------%d %d %d %d\n", m_block, num_n_block, nbi, mask_ptr[num_n_block - 1]);
  // printf("mask_ptr as follows: \n");
  // for (int i = 0; i <  num_n_block; i++) {
  //   printf("%d   ", mask_ptr[i]);
  //   if (mask_ptr[i] < -1 || mask_ptr[i] > num_n_mask) printf("\n Error mask_ptr[i] %d, ", mask_ptr[i]);
  // }
  // printf("\n");
  // }

  // Empty line
  if (nbi < 0) {
    return;
  }

  // TODO: Base offset for MHA
  // NOTE: Convert C pointer to Tensor for convenience
  Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
  Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
  Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));

  // Get MMA abstraction
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tidx);

  // Construct SMEM tensors.
  Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
  Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

  // Tensor for V Transpose; used in GEMM-II.
  Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVt{});
  Tensor sVtNoSwizzle = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVtNoSwizzle{});

  // NOTE: Copy abstraction
  // NOTE: QKV gmem -> smem copy abstraction
  typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

  // NOTE: Define gmem -> smem copy src, dst
  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
  Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
  Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
  Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
  Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

  // NOTE: Define smem -> reg copy dst
  // partition_fragment is similar to partition, but returns a register representation
  Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
  Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
  Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

  //
  // Copy Atom retiling
  //

  // TODO: Understand atom retiling here

  // NOTE: Prepare the copy objects to copy Q, K, V to smem
  auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

  auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
  Tensor tSsK = smem_thr_copy_K.partition_S(sK);

  // TODO: Transpose during copy
  // NOTE: smem->reg copy of Vt
  auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
  Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

  // Pipeline load initial Q, K
  // // Load Q to smem
  // flash::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
  // // Load K to smem
  // flash::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
  if (nbi == n_block_max - 1) {
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                       binfo.actual_seqlen_q - m_block * kBlockM);  
    // Load K to smem
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k - nbi * kBlockN);
  }
  else {
    flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ);
    flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
  }
  // Start asynchronous copy
  cute::cp_async_fence();

  Tensor rAccOut = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

  // step1: slice-k compute QK block
  // Q[BLOCK_M, BLOCK_N] @ K[BLOCK_M, BLOCK_N].T = O[BLOCK_M, BLOCK_M]
  //
  // step2:
  // advance K, V

  // NOTE: Number of K, V blocks: processing range
  // NOTE: 1. mask between N BLOCKs if in causal mode
  int seqlen_start = m_block * kBlockM;
  int seqlen_end = (m_block + 1) * kBlockM;

  // NOTE: Maximum values to record
  Tensor scores_max = make_tensor<ElementAccum>(Shape<Int<2 * size<1>(rAccOut)>>{});
  // NOTE: Denominator values to record
  Tensor scores_sum = make_fragment_like(scores_max);

  clear(rAccOut);
  
  while (nbi >= 0) {
    auto rAccScore = partition_fragment_C(tiled_mma, make_shape(Int<kBlockM>{}, Int<kBlockN>{}));

    clear(rAccScore);

    // Wait for the gmem -> smem copy of Q and K to complete, meaning Q and K are ready
    // wait<0> indicates waiting for 0 unfinished tasks

    flash::cp_async_wait<0>();
    __syncthreads();

    // Asynchronous loading of V during GEMM
    if (mask_id > 0) {   // nbi > nbi_start
      // printf("advance V: mask_id %d, nbi %d, nbi_prev %d  \n", mask_id, nbi, nbi_prev); 
      // if (nbi < -1 || nbi > num_n_mask) printf("error advance V: mask_id %d, nbi %d, nbi_prev %d  \n", mask_id, nbi, nbi_prev);
      tVgV.data() = tVgV.data() + (nbi - nbi_prev) * int(kBlockN * params.v_row_stride);    
    }    
    if (nbi == n_block_max - 1) {
      flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - nbi * kBlockN);
    }    
    else if (nbi < n_block_max - 1) {
      flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
    }

    // // Asynchronously load V into smem
    // flash::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
    // Initiate asynchronous copy
    cute::cp_async_fence();

    // O = Q@K.T
    // NOTE: Load data from smem to registers and perform gemm, **retile during loading**
    flash::gemm_smem(rAccScore, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
        smem_thr_copy_Q, smem_thr_copy_K
    );

    Tensor scores = make_tensor(rAccScore.data(), flash::convert_layout_acc_rowcol(rAccScore.layout()));

    // NOTE: 2. Mask within N BLOCKs
    if (Is_causal ==  true && nbi * kBlockN >= seqlen_start) {
      // flash::mask_within_nblock<kBlockM, kBlockN, kNWarps>(scores, m_block, nbi);
      int max_seqlen_k_value = Is_even_MN ? -1 : params.seqlen_k;
      flash::mask_within_nblock<kBlockM, kBlockN, kNWarps, Is_causal, Is_even_MN>(scores, m_block, nbi, max_seqlen_k_value);    
    }

    // NOTE: Wait for V loading to complete, prepare initial state for the next K load
    flash::cp_async_wait<0>();
    __syncthreads();

    // Advance K
    mask_id++;
    // if (m_block == 7 && bidb == 0 && bidh == 0){
    //   printf("after mask_id++: mask_id %d  \n", mask_id); 
    // }          
    if (mask_id == num_n_block) {
      nbi = -1;
    }
    else if(mask_id % col_factor == 0) {
      // Load next mask
      nbi_prev = nbi;
      nbi = mask_ptr[mask_id / col_factor] * col_factor;
      // if (m_block == 7 && bidb == 0 && bidh == 0){
      //   printf("update nbi 1: mask_id %d, nbi %d, nbi_prev %d  \n", mask_id, nbi, nbi_prev); 
      // }             
      // if (nbi < -1 || nbi > num_n_mask) printf("error nbi: mask_id %d, nbi %d, nbi_prev %d  \n", mask_id, nbi, nbi_prev);
    } else {
      nbi_prev = nbi;
      nbi++;
      // if (m_block == 7 && bidb == 0 && bidh == 0){
      //   printf("update nbi 2: mask_id %d, nbi %d, nbi_prev %d  \n", mask_id, nbi, nbi_prev); 
      // }    
    }

    if (nbi >= 0) {
      // gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi, _));
      // tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
      // printf("tKgK----------mask_id, nbi, nbi_prev------------%d %d %d\n", mask_id, nbi, nbi_prev);
      tKgK.data() = tKgK.data() + (nbi - nbi_prev) * int(kBlockN * params.k_row_stride);   
      // printf("advance K: mask_id %d, nbi %d, nbi_prev %d  \n", mask_id, nbi, nbi_prev); 
      // if (nbi < -1 || nbi > num_n_mask) printf("error advance K: mask_id %d, nbi %d, nbi_prev %d  \n", mask_id, nbi, nbi_prev);
      // flash::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
      if (nbi < n_block_max - 1) {
        flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
      }
      else {
        flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - nbi * kBlockN);
      }
      cute::cp_async_fence();
    }

    // Compute softmax
    // NOTE: rAccOut records all the numerators after softmax
    (mask_id == 1) ? flash::softmax_rescale_o</*Is_first=*/true>(scores, scores_max, scores_sum, rAccOut, params.scale_softmax) :
      flash::softmax_rescale_o</*Is_first=*/false>(scores, scores_max, scores_sum, rAccOut, params.scale_softmax);

    // Actual QK @ V execution
    // (score AKA rAccScore): QK[M, N] @ V[N, dim]
    // NOTE: DABC: F32F16F16F32, convert D type(F32) to A type(F16)
    // TODO: convert_type is currently hardcoded
    Tensor rP = flash::convert_type_f32_to_f16(rAccScore);
    // NOTE: Convert from layout C to layout A
    Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_rowcol_Aregs<TiledMMA>(scores.layout()));

    flash::gemm_A_in_regs(rAccOut, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
  }

  // Epilogue

  // NOTE: Finally divide by the denominator
  // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
  // AKA reshape to (nrow, ncol) but with specific MMA layout
  Tensor acc_o_rowcol = make_tensor(rAccOut.data(), flash::convert_layout_acc_rowcol(rAccOut.layout()));
  // NOTE: Save lse for backward
  Tensor lse = make_fragment_like(scores_sum);
  // for row
  #pragma unroll
  for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
    float sum = scores_sum(mi);
    float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
    // compute lse
    // NOTE: here we use max * scale 
    lse(mi) = (sum == 0.f || sum != sum) ? INFINITY : scores_max(mi) * params.scale_softmax + __logf(sum);
    float scale = inv_sum;
    // for col
    #pragma unroll
    for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
      acc_o_rowcol(mi, ni) *= scale;
    }
  }

  // Convert acc_o from fp32 to fp16/bf16
  Tensor rO = flash::convert_type_f32_to_f16(rAccOut);
  // Reuse sQ's smem for sO copy out
  Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)

  // Partition sO to match the accumulator partitioning
  // TODO: review
  auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
  Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
  Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

  // NOTE: Copy to smem first
  cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

  // Tensor gO = local_tile(O, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(m_block, _));
  Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
  Shape<Int<kBlockM>, Int<kHeadDim>>{},
  make_stride(params.o_row_stride, _1{}));  

  // Create a copy from smem to gmem
  typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
  Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
  Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

  __syncthreads();

  // NOTE:: Copy again to gmem

  // TODO: review, what is the purpose of these two copy operations?
  Tensor tOrO = make_tensor<Element>(shape(tOgO));
  cute::copy(gmem_tiled_copy_O, tOsO, tOrO);
  // flash::copy(gmem_tiled_copy_O, tOrO, tOgO);
    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );  

  // NOTE: Write back lse
  // Tensor gLSE = local_tile(LSE, make_tile(Int<kBlockM>{}), make_coord(m_block));
  Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});    
  Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
  Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
  static_assert(decltype(size<0>(taccOcO))::value == 4);
  // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
  // TODO: review this shape
  Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
  CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
  // TODO: Clarify the logic here
  if (get<1>(taccOcO_row(0)) == 0) {
      #pragma unroll
      for (int mi = 0; mi < size(lse); ++mi) {
          const int row = get<0>(taccOcO_row(mi));
          // if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
          gLSE(row) = lse(mi);
      }
  }
}

template<typename Kernel_traits, bool Is_causal>
void run_flash_block_fwd(Block_flash_fwd_params &params, cudaStream_t stream) {
  // TODO: check if works: default stream = 0
  using Element = typename Kernel_traits::Element;
  using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
  using SmemLayoutK = typename Kernel_traits::SmemLayoutKV;
  using SmemLayoutV = typename Kernel_traits::SmemLayoutKV;

  const int num_m_block =
      (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;

  // printf("Kernel_traits::kBlockM, Kernel_traits::kBlockN %d %d\n", Kernel_traits::kBlockM, Kernel_traits::kBlockN);    
  // dim3 grid(num_m_block, params.b * params.h, 1);
  dim3 grid(num_m_block, params.b, params.h);  
  dim3 block(Kernel_traits::kNThreads);
  // printf("Launching kernel with gridDim=(%d,%d,%d)\n", grid.x, grid.y, grid.z);

  const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
  const bool is_even_K = params.d == Kernel_traits::kHeadDim;

  int smem_size = int(sizeof(SharedStorage<Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>));

  BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        auto kernel = &flash_attention_block_v2_cutlass_kernel<Kernel_traits, Is_causal, IsEvenMNConst, IsEvenKConst, Block_flash_fwd_params, 64, 64>;
        // auto kernel = &flash_attention_block_v2_cutlass_kernel<Kernel_traits, Is_causal, IsEvenMNConst, IsEvenKConst, Block_flash_fwd_params, 128, 128>;

        if (smem_size >= 48 * 1024) {
            CUDA_ERROR_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }      
        kernel<<<grid, block, smem_size>>>(params);
    });
  });
}

template<typename T, int Headdim>
void run_flash_block_fwd_(Block_flash_fwd_params &params, cudaStream_t stream);

// TODO: Write specialization for each case, currently using general template
// For example, run_flash_fwd_hdim32 for specialization with hdim=32
// This allows adjustment of kBlockN and kBlockM combinations for better compilation speed
template<typename T, int Headdim>
void run_flash_block_fwd_(Block_flash_fwd_params &params, cudaStream_t stream) {
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, /*kBlockM_=*/128, /*kBlockN_=*/128, /*kNWarps_=*/4, T>, Is_causal>(params, stream);

        // TODO: Adjust kBlockM, kBlockN combinations
        run_flash_block_fwd<Flash_fwd_kernel_traits<Headdim, /*kBlockM_=*/64, /*kBlockN_=*/64, /*kNWarps_=*/4, T>, Is_causal>(params, stream);
    });
}

// Entry point for flash attention
void run_flash_attn_block_cutlass(Block_flash_fwd_params &params, cudaStream_t stream) {
    // FP16_SWITCH yield elem_type namespace
    FP16_SWITCH(!params.is_bf16, [&] {
        // FWD_HEADDIM_SWITCH yield kHeadDim constexpr
        FWD_HEADDIM_SWITCH(params.d, [&] {
            run_flash_block_fwd_<elem_type, kHeadDim>(params, stream);
        });
    });
}


std::vector<torch::Tensor> flash_attention_block_v2_cutlass(torch::Tensor q, torch::Tensor k,
    torch::Tensor v, torch::Tensor row_mask, bool is_causal = false, float softmax_scale=1) {

  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);

  const auto sizes = q.sizes();
  const int batch_size = sizes[0];
  int seqlen_q = sizes[1];
  int num_heads = sizes[2];
  const int head_size_og = sizes[3];
  const int seqlen_k = k.size(1);
  const int num_heads_k = k.size(2);  
  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);  
  const int head_size = round_multiple(head_size_og, 8);
  const int head_size_rounded = round_multiple(head_size, 32);  
  
  auto out = torch::empty_like(q);
  auto opts = q.options();
  auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(torch::kFloat32));
  
  Block_flash_fwd_params params;
  set_params_fprop(params,
                  batch_size,
                  seqlen_q, seqlen_k,
                  seqlen_q_rounded, seqlen_k_rounded,
                  num_heads, num_heads_k,
                  head_size, head_size_rounded,
                  q, k, v, out,
                  /*cu_seqlens_q_d=*/nullptr,
                  /*cu_seqlens_k_d=*/nullptr,
                  /*seqused_k=*/nullptr,
                  softmax_lse.data_ptr(),
                  softmax_scale,
                  is_causal);     
  // TODO: get ptr
  params.block_mask_ptr = reinterpret_cast<int*>(row_mask.data_ptr());
  params.mask_batch_stride = row_mask.stride(0);
  params.mask_head_stride = row_mask.stride(1);
  // printf("params.mask_batch_stride:%" PRId64 ", params.mask_head_stride:%" PRId64 "\n", params.mask_batch_stride, params.mask_head_stride); // 2048 64

  run_flash_attn_block_cutlass(params, 0);

  // Wait until kernel finishes
  cudaDeviceSynchronize();
  CUDA_ERROR_CHECK(cudaGetLastError());

  return {out, softmax_lse};
}


std::vector<torch::Tensor> varlen_flash_attention_block_v2_cutlass(torch::Tensor q, torch::Tensor k,
  torch::Tensor v, torch::Tensor cu_seqlens_q, torch::Tensor cu_seqlens_k, torch::Tensor row_mask, int max_seqlen_q, int max_seqlen_k, bool is_causal = false, float softmax_scale=1){
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);

  CHECK_INPUT(cu_seqlens_q);
  CHECK_INPUT(cu_seqlens_k);
  
  const auto sizes = q.sizes();
  const int total_q = sizes[0];
  const int batch_size = cu_seqlens_q.numel() - 1;
  const int num_heads = sizes[1];
  const int head_size_og = sizes[2];
  const int total_k = k.size(0);
  const int num_heads_k = k.size(1);
  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size = round_multiple(head_size_og, 8);
  const int head_size_rounded = round_multiple(head_size, 32);
  const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);  

  auto out = torch::empty_like(q);
  auto opts = q.options();
  auto softmax_lse = torch::empty({batch_size, num_heads, max_seqlen_q}, opts.dtype(torch::kFloat32));

  Block_flash_fwd_params params;
  set_params_fprop(params,
                  batch_size,
                  max_seqlen_q, max_seqlen_k,
                  seqlen_q_rounded, seqlen_k_rounded,
                  num_heads, num_heads_k,
                  head_size, head_size_rounded,
                  q, k, v, out,
                  cu_seqlens_q.data_ptr(),
                  cu_seqlens_k.data_ptr(),
                  /*seqused_k=*/nullptr,
                  softmax_lse.data_ptr(),
                  softmax_scale,
                  is_causal);

  // TODO: get ptr
  params.block_mask_ptr = reinterpret_cast<int*>(row_mask.data_ptr());
  params.mask_batch_stride = row_mask.stride(0);
  params.mask_head_stride = row_mask.stride(1);

  run_flash_attn_block_cutlass(params, 0);

  // Wait until kernel finishes
  cudaDeviceSynchronize();
  CUDA_ERROR_CHECK(cudaGetLastError());

  return {out, softmax_lse};    
}