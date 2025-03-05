# block-sparse-attention

Custom block-sparse FlashAttention implementation.

## Usage
```bash
make build
```

```
from block_sparse_seer_attn import block_sparse_attention
q = (torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
k = (torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
v = (torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
set block_mask -> a bool tensor of size (B, SEQ_LEN, NHEAD, DIM) 
out, _ = block_sparse_attention(q, k, v, block_mask, is_causal=True, sm_scal=1.0)

```
