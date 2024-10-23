OUTPUT_DIR="./checkpoints/llama_3.1_8B/Qavg_Kmaxmin_64k_mse_double_layer_w1e-1" 
CACHE_DIR="/dev/shm/cache" 

torchrun --nproc_per_node=4 --master_port=10002 train_pt.py  \
        --model_name_or_path models/Llama-3.1-8B \
        --attn_gate_type Qavg_Kmaxmin \
        --bf16 True \
        --output_dir $OUTPUT_DIR      \
        --cache_dir $CACHE_DIR   \
        --training_max_length 65536 \
        --use_mse_loss True \
        --per_device_train_batch_size 1     \
        --per_device_eval_batch_size 1     \
        --gradient_accumulation_steps 4     \
        --evaluation_strategy "no"     \
        --save_strategy "no"     \
        --learning_rate 1e-3     \
        --weight_decay 1e-1     \
        --warmup_steps 20     \
        --lr_scheduler_type "cosine"     \
        --logging_steps 1     \
        --deepspeed "ds_config/stage2.json" \
        --max_steps 500



