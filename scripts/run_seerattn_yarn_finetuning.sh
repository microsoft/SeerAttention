
# huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir  models/meta-llama/Meta-Llama-3-8B
OUTPUT_POST_TRAINING_DIR="./checkpoints/llama_3_8B_init/" 
OUTPUT_FINE_TUNING_DIR="./checkpoints/llama_3_8B_32k_seer/"
CACHE_DIR="/dev/shm/cache"

# torchrun --nproc_per_node=4 --master_port=10002 post_training.py  \
#         --model_name_or_path models/meta-llama/Meta-Llama-3-8B \
#         --attn_gate_type Qavg_Kmaxmin \
#         --gate_hidden_size 128 \
#         --bf16 True \
#         --output_dir $OUTPUT_POST_TRAINING_DIR      \
#         --cache_dir $CACHE_DIR   \
#         --training_max_length 8192 \
#         --use_mse_loss True \
#         --per_device_train_batch_size 4     \
#         --per_device_eval_batch_size 1     \
#         --gradient_accumulation_steps 1     \
#         --evaluation_strategy "no"     \
#         --save_strategy "no"     \
#         --learning_rate 1e-3     \
#         --weight_decay 0.0     \
#         --warmup_steps 20     \
#         --lr_scheduler_type "cosine"     \
#         --logging_steps 1     \
#         --deepspeed "ds_config/stage2.json" \
#         --max_steps 1000

# # per_device_train_batch_size can only be 1, limited by the current implementation of the training kernel
# torchrun --nproc_per_node=4 fine_tuning.py  \
#         --model_name_or_path $OUTPUT_POST_TRAINING_DIR   \
#         --output_dir $OUTPUT_FINE_TUNING_DIR    \
#         --bf16 True \
#         --cache_dir $CACHE_DIR   \
#         --train_dense_baseline False \
#         --scaling_factor 4.0 \
#         --model_max_length 32768 \
#         --nz_ratio 0.5 \
#         --gradient_checkpointing True \
#         --per_device_train_batch_size 1     \
#         --per_device_eval_batch_size 1     \
#         --gradient_accumulation_steps 2     \
#         --evaluation_strategy "no"     \
#         --save_strategy "no"     \
#         --save_total_limit 4     \
#         --warmup_steps 20     \
#         --learning_rate 1e-5     \
#         --lr_scheduler_type "linear"     \
#         --logging_steps 1     \
#         --deepspeed "ds_config/stage3.json" \
#         --gate_loss_scale 10 \
#         --max_steps 400


CUDA_VISIBLE_DEVICES=0 python eval_ppl.py \
    -m $OUTPUT_FINE_TUNING_DIR \
    --split test \
    --feature text \
    --dataset-min-tokens 131072 \
    --dataset emozilla/pg19-test \
    --nz_ratios 0.5\
    --use_seer_attn \
    --max-tokens 32768 \
    --min-tokens 32768 \
    --cache_dir $CACHE_DIR \
    --truncate \
    --output-file ./results/ppl_pg19.txt

# CUDA_VISIBLE_DEVICES=0 python eval_ppl.py \
#     -m $OUTPUT_FINE_TUNING_DIR \
#     --split test \
#     --feature text \
#     --dataset-min-tokens 32768 \
#     --dataset hoskinson-center/proof-pile\
#     --nz_ratios 0.5\
#     --use_seer_attn \
#     --max-tokens 32768 \
#     --min-tokens 32768 \
#     --cache_dir $CACHE_DIR \
#     --truncate \
#     --samples 10 \
#     --output-file ./results/proofpile_ppl.txt
