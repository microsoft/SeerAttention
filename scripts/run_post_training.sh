# huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir  models/meta-llama/Meta-Llama-3-8B

OUTPUT_MODEL_DIR="checkpoints/llama_3.1_8B/Qavg_Kmaxmin_64k" 
CACHE_DIR="/dev/shm/cache" 

torchrun --nproc_per_node=4 --master_port=10002 post_training.py  \
        --model_name_or_path models/meta-llama/Llama-3.1-8B \
        --attn_gate_type Qavg_Kmaxmin \
        --bf16 True \
        --output_dir $OUTPUT_MODEL_DIR      \
        --cache_dir $CACHE_DIR   \
        --training_max_length 65536 \
        --use_mse_loss True \
        --per_device_train_batch_size 1     \
        --per_device_eval_batch_size 1     \
        --gradient_accumulation_steps 4     \
        --evaluation_strategy "no"     \
        --save_strategy "no"     \
        --learning_rate 1e-3     \
        --weight_decay 1e-2     \
        --warmup_steps 20     \
        --lr_scheduler_type "cosine"     \
        --logging_steps 1     \
        --deepspeed "ds_config/stage2.json" \
        --max_steps 500


CUDA_VISIBLE_DEVICES=0 python eval_ppl.py \
    -m $OUTPUT_MODEL_DIR \
    --split test \
    --feature text \
    --dataset-min-tokens 131072 \
    --dataset emozilla/pg19-test \
    --nz_ratios 1.0,0.5,0.4,0.3,0.2,0.1\
    --use_seer_attn \
    --max-tokens 131072 \
    --min-tokens 8192 \
    --cache_dir $CACHE_DIR \
    --truncate \
    --output-file ./results/pg_ppl.txt

CUDA_VISIBLE_DEVICES=0 python eval_ppl.py \
    -m $OUTPUT_MODEL_DIR \
    --split test \
    --feature text \
    --dataset-min-tokens 131072 \
    --dataset hoskinson-center/proof-pile \
    --nz_ratios 1.0,0.5,0.4,0.3,0.2,0.1\
    --use_seer_attn \
    --max-tokens 131072 \
    --min-tokens 8192 \
    --sample 10 \
    --cache_dir $CACHE_DIR \
    --truncate \
    --output-file ./results/proofpile_ppl.txt

