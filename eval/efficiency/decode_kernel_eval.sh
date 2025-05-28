batches=(32)
max_cache_seqlens=(8192 16384 32768 65536)
block_sizes=(16 32 64 128)
sparse_ratios=(0.5 0.6 0.7 0.8 0.9)
heads_heads_kv_combinations=("32,8" "40,8" "64,8")

for batch in "${batches[@]}"; do
  for max_cache_seqlen in "${max_cache_seqlens[@]}"; do
    for block_size in "${block_sizes[@]}"; do
      for sparse_ratio in "${sparse_ratios[@]}"; do
        for heads_heads_kv in "${heads_heads_kv_combinations[@]}"; do
          IFS=',' read -r -a heads_kv <<< "$heads_heads_kv"
          heads=${heads_kv[0]}
          heads_kv=${heads_kv[1]}
          echo "batch_size: $batch, max_cache_seqlen: $max_cache_seqlen, block_size: $block_size, sparse_ratio:$sparse_ratio, heads: $heads, heads_kv: $heads_kv"
          python decode_kernel_eval.py \
            --batch "$batch" \
            --heads "$heads" \
            --heads_kv "$heads_kv" \
            --max_cache_seqlen "$max_cache_seqlen" \
            --dim 128 \
            --dim_v 128 \
            --sparse_ratio "$sparse_ratio" \
            --block_size "$block_size" 
        done
      done
    done
  done
done
