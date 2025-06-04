block_size=64
heads=64
heads_kv=8

configs=(
    "16*8192" "16*16384" "16*32768" "16*65536" "16*131072"
    "8*8192" "8*16384" "8*32768" "8*65536" "8*131072"
    "4*8192" "4*16384" "4*32768" "4*65536" "4*131072"
)

sparse_ratios=("0.5" "0.6" "0.7" "0.8" "0.9")


for config in "${configs[@]}"; do
    
    IFS='*' read -r batch max_cache_seqlen <<< "$config"

    for sr in "${sparse_ratios[@]}"; do
        echo "Running config: batch=$batch, max_cache_seqlen=$max_cache_seqlen, sparse_ratio=$sr"
        python decode_kernel_eval.py \
            --batch $batch \
            --max_cache_seqlen $max_cache_seqlen \
            --sparse_ratio $sr 
    done
done

data_path="results/kernel_test_${block_size}_gqa${heads}_${heads_kv}.csv"
python draw.py \
    --data_path $data_path \
