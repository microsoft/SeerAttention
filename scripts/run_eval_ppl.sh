MODEL_DIR="checkpoints/llama_3.1_8B/Qavg_Kmaxmin_64k_mse_single_layer" 
CACHE_DIR="/dev/shm/cache" 
RESULTS_DIR="./results/blob_ppl.txt"
NZ_RATIOS="0.5"

CUDA_VISIBLE_DEVICES=0 python eval_ppl.py \
    -m checkpoints/llama_3.1_8B/Qavg_Kmaxmin_64k_mse_double_layer_w1e-2 \
    --split test \
    --feature text \
    --dataset-min-tokens 131072 \
    --dataset emozilla/pg19-test \
    --nz_ratios $NZ_RATIOS\
    --use_seer_attn \
    --max-tokens 131072 \
    --min-tokens 131072 \
    --cache_dir $CACHE_DIR \
    --truncate \
    --output-file $RESULTS_DIR

CUDA_VISIBLE_DEVICES=0 python eval_ppl.py \
    -m checkpoints/llama_3.1_8B/Qavg_Kmaxmin_64k_mse_double_layer_w1e-3 \
    --split test \
    --feature text \
    --dataset-min-tokens 131072 \
    --dataset emozilla/pg19-test \
    --nz_ratios $NZ_RATIOS\
    --use_seer_attn \
    --max-tokens 131072 \
    --min-tokens 131072 \
    --cache_dir $CACHE_DIR \
    --truncate \
    --output-file $RESULTS_DIR