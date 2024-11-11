
# huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir  models/meta-llama/Meta-Llama-3-8B

OUTPUT_MODEL_DIR="./checkpoints/llama-3-8B-yarn-32K" 
CACHE_DIR="/dev/shm/cache" 

# torchrun --nproc_per_node=4 fine_tuning.py  \
#         --model_name_or_path models/meta-llama/Meta-Llama-3-8B        \
#         --output_dir $OUTPUT_MODEL_DIR     \
#         --bf16 True \
#         --cache_dir /dev/shm/cache   \
#         --train_dense_baseline True \
#         --scaling_factor 4.0 \
#         --model_max_length 32768 \
#         --gradient_checkpointing True \
#         --per_device_train_batch_size 1     \
#         --per_device_eval_batch_size 1     \
#         --gradient_accumulation_steps 2     \
#         --evaluation_strategy "no"     \
#         --save_strategy "no"     \
#         --warmup_steps 20     \
#         --save_total_limit 4     \
#         --learning_rate 1e-5     \
#         --lr_scheduler_type "linear"     \
#         --logging_steps 1     \
#         --deepspeed "ds_config/stage3.json" \
#         --max_steps 400



CUDA_VISIBLE_DEVICES=0 python eval_ppl.py \
    -m $OUTPUT_MODEL_DIR \
    --split test \
    --feature text \
    --dataset-min-tokens 131072 \
    --dataset emozilla/pg19-test \
    --max-tokens 32768 \
    --min-tokens 32768 \
    --cache_dir $CACHE_DIR \
    --truncate \
    --output-file ./results/ppl_pg19.txt

# CUDA_VISIBLE_DEVICES=0 python eval_ppl.py \
#     -m $OUTPUT_MODEL_DIR \
#     --split test \
#     --feature text \
#     --dataset-min-tokens 131072 \
#     --dataset hoskinson-center/proof-pile \
#     --max-tokens 32768 \
#     --min-tokens 32768 \
#     --sample 10 \
#     --cache_dir $CACHE_DIR \
#     --truncate \
#     --output-file ./results/proofpile_ppl.txt