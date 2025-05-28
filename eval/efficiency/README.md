## KERNEL Efficiency Evaluation

We have customized the CUDA kernel to accelerate the attention computation. First, compile the kernel:

```bash
cd seer_attn/kernels/block_sparse_attention
make build
```
For the performance of sparse attention kernel, please refer to `benchmark_sparse_attn.py`.

```bash
pip install flash-attn --no-build-isolation

cd eval/efficiency
python benchmark_sparse_attn.py --seq_len 8192 --sparsity 0.9 0.8 0.7 0.6 0.5
python benchmark_sparse_attn.py --seq_len 32768 --sparsity 0.9 0.8 0.7 0.6 0.5
python benchmark_sparse_attn.py --seq_len 131072 --sparsity 0.9 0.8 0.7 0.6 0.5
```

## Tilelang and Triton Kernel for Sparse Decoding

Run the following commands to compare the efficiency among FlashAttention-V2 dense, Triton sparse and Tilelang sparse.
```bash
pip install tilelang
bash kernel_eval.sh
```
