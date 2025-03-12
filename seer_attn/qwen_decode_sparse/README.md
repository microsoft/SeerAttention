## Modeling of SeerAttention with Decode Sparsity (BETA)

To perform better performance on decode phase, the seq dim of Q is not compressed/pooled as prefill. The currently implemenetation is unoptimied and can be slow. It is used to demonstrate reasonable sparsity/acc performance for experiment purpose. 