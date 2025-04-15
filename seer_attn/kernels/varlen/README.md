# Varlen Flash Sparse Attention 



## **block_sparse_attn_varlen_gqa_simulate.py** 

This is a training kernel that assumes you give a binary block mask where only k's sequence dimension is compressed, B is block size of sparsity. It requires sparse mask within a gqa group to be the same similar to deepseek's NSA. Since some base models might have a GQA group size smaller than 16, to avoid waste computation on padding, we directly multiply the sparse mask on the tensor without actually doing **sparse** computation. 

Use case:
```python
block_mask = repeat_kv(block_mask, self.num_key_value_groups) ## repeat block mask's kv_head to q_head
k = repeat_kv_varlen(k, self.num_key_value_groups)
v = repeat_kv_varlen(v, self.num_key_value_groups)

# q, k, v shapes: [t, num_heads, head_dim]
# block_mask shapes: [bsz, num_heads, t, ceil(t/B)]
attn_output = block_1d_gqa_sparse_attn_varlen_sim_func(
    q, 
    k,
    v,
    cu_seqlens, 
    cu_seqlens,
    max_seqlen,
    1.0 / math.sqrt(self.head_dim),
    block_mask,
    block_size,            
)

```
The current default config is tuned for MI300x. If you run on Nvidia GPUs, you might want to run with autotune. 



## **block_sparse_attn_varlen_gqa.py** 

This kernel is bascially has the same functionaly as the previous one except that it performs real sparse computation. However, it can waste computation when GQA group size < 16.  

Use case:
```python

# q shapes: [t, num_q_heads, head_dim]
# k, v shapes: [t, num_kv_heads, head_dim]
# block_mask shapes: [bsz, num_kv_heads, t, ceil(t/B)]
attn_output = block_1d_gqa_sparse_attn_varlen_func(
    q, 
    k,
    v,
    cu_seqlens, 
    cu_seqlens,
    max_seqlen,
    1.0 / math.sqrt(self.head_dim),
    block_mask,
    block_size,            
)

```


## **block_sparse_attn_varlen_2D.py** 

This kernel assumes you give a binary block mask where q and k's sequence dimension are both compressed, B is the block size of sparsity. 

Use case:
```python

k = repeat_kv_varlen(k, self.num_key_value_groups)
v = repeat_kv_varlen(v, self.num_key_value_groups)

# q, k, v shapes: [t, num_heads, head_dim]
# block_mask shapes: [bsz, num_heads, ceil(t/B), ceil(t/B)]
attn_output = block_2d_sparse_attn_varlen_func(
    q, 
    k,
    v,
    cu_seqlens, 
    cu_seqlens,
    max_seqlen,
    1.0 / math.sqrt(self.head_dim),
    block_mask,
    block_size,            
)

```


## **attn_pooling_kernel_varlen.py**
This kernel generate per-head 2D and 1D maxpooled attention weights for distillation.



```python
# Input Shape:
# q: [t, num_q_heads, head_dim]
# k: [t, num_kv_heads, head_dim]
# v: [t, num_kv_heads, head_dim]

# Output Shape:
# pooling_gt_2d: [bsz, num_q_heads, ceil(t/Block), ceil(t/Block)]
# pooling_gt_1d: a list of tensor [[num_q_heads, ti/Block, ceil(ti/Block)] for ti in seqlens]
# The 1d pooling use a list of tensors as different batch can have very different seqlen. If padded to maxlen, it ca easily cause OOM.
pooling_gt_2d, pooling_gt_1d = attn_pooling(
    q,
    k,
    cu_seqlens,
    max_seqlen,
    1.0 / math.sqrt(self.head_dim),
    self.config.seerattn_gate_block_size,      
) 
```
It should be noted that `pooling_gt_1d` is not normalized to sum=1. Depending on the need, you might want to do an additional pooling on the head dim to train a shared GPA gates. In such case, you need to normalized after maxpool:

```python 
for i in range(len(pooling_gt_1d)):
    gt_i = F.max_pool3d(pooling_gt_1d[i], kernel_size=[gpq_group, 1, 1], stride=[gpq_group, 1, 1])
    sum = torch.sum(gt_i, dim=-1, keepdim=True)
    gt_i.div_(sum + 1e-6)
```


## **block_sparse_flash_decode_varlen kernels**
Those are flash-decoding kernels with block_sparse attention. Masks are required to be the same in a GQA group. 