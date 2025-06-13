#pragma once

#include <cstdint>

struct Qkv_params {
    using index_t = int64_t;
    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;

    // The number of heads.
    int h, h_k;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_h_k_ratio; // precompute h / h_k,

    bool is_bf16;
};


struct Flash_fwd_params : public Qkv_params {
  // The O matrix (output).
  void * __restrict__ o_ptr;
  // void * __restrict__ oaccum_ptr;

  // The stride between rows of O.
  index_t o_batch_stride;
  index_t o_row_stride;
  index_t o_head_stride;

  // The pointer to the P matrix.
  void * __restrict__ p_ptr;

  // The pointer to the softmax sum.
  void * __restrict__ softmax_lse_ptr;
  // void * __restrict__ softmax_lseaccum_ptr;

  // The dimensions.
  int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded;
  // , rotary_dim

  // The scaling factors for the kernel.
  float scale_softmax;
  float scale_softmax_log2;

  // array of length b+1 holding starting offset of each sequence.
  int * __restrict__ cu_seqlens_q;
  int * __restrict__ cu_seqlens_k;

  // If provided, the actual length of each k sequence.
  int * __restrict__ seqused_k;

  int *__restrict__ blockmask;
  // int *__restrict__ streaming_info;
  // int *__restrict__ head_mask_type;
  int m_block_dim, n_block_dim, num_blocksparse_heads;

  // The K_new and V_new matrices.
  // void * __restrict__ knew_ptr;
  // void * __restrict__ vnew_ptr;

  // The stride between rows of the Q, K and V matrices.
  // index_t knew_batch_stride;
  // index_t vnew_batch_stride;
  // index_t knew_row_stride;
  // index_t vnew_row_stride;
  // index_t knew_head_stride;
  // index_t vnew_head_stride;

  bool is_bf16;
  bool is_causal;
  // bool is_exact_streaming;

  // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
  // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
  bool is_seqlens_k_cumulative;
};

struct Block_flash_fwd_params : public Flash_fwd_params {
  int *block_mask_ptr; // of type [B, NHead, sqe/blockM, seq / blockN]
  index_t mask_head_stride;
  index_t mask_batch_stride;
};