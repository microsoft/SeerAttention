# huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --local-dir  models/meta-llama/Meta-Llama-3.1-8B-Instruct

CACHE_DIR=${CACHE_DIR:-"./cache_dir"}
warmup_steps=${warmup_steps:-20}
training_max_length=${training_max_length:-65536}
lr=${lr:-1e-3}
weight_decay=${weight_decay:-0.0}
gate_type=${gate_type:-"Qavg_Kmaxminavg"}
bs=${bs:-16}
gpus=${gpus:-8}
total_data=${total_data:-8000}
gate_loss_scale=${gate_loss_scale:-10.0}
steps=$(($total_data / $bs))
gradient_accumulation_steps=$(($bs / $gpus))
run_name="${gate_type}_lr${lr}_maxlen${training_max_length}_warmup${warmup_steps}_bs${bs}_steps${steps}_gatelossscale${gate_loss_scale}"
echo $run_name
torchrun --nproc_per_node=$gpus --master_port=10003 distillation.py  \
        --model_name_or_path models/meta-llama/Llama-3.1-8B-Instruct \
        --seerattn_gate_type $gate_type \
        --bf16 True \
        --output_dir models/seer_attn_llama_3.1/$run_name      \
        --cache_dir $CACHE_DIR   \
        --training_max_length $training_max_length \
        --per_device_train_batch_size 1     \
        --gradient_accumulation_steps $gradient_accumulation_steps     \
        --gate_loss_scale $gate_loss_scale \
        --evaluation_strategy "no"     \
        --save_strategy "no"     \
        --learning_rate $lr     \
        --weight_decay $weight_decay     \
        --warmup_steps $warmup_steps     \
        --lr_scheduler_type "cosine"     \
        --logging_steps 1     \
        --deepspeed ds_config/stage2.json \
        --max_steps $steps