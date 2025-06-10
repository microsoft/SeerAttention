## Scripts to run AttnGate Distillation on Llama-3.1-8B-Instruct
## Estimated time: 5h for 8xA100 GPUs

warmup_steps=${WARMUP_STEPS:-20}
training_max_length=${TRAINING_MAX_LENGTH:-65536}
lr=${LR:-1e-3}
weight_decay=${WEIGHT_DECAY:-0.0}
gate_type=${GATE_TYPE:-"Qavg_Kmaxminavg"}
bs=${BS:-16}
gpus=${GPUS:-8}
total_data=${TOTAL_DATA:-524288000}
gate_loss_scale=${GATE_LOSS_SCALE:-10.0}
steps=$(($total_data /$training_max_length / $bs))
gradient_accumulation_steps=$(($bs / $gpus))
base_model=${BASE_MODEL:-"meta-llama/Llama-3.1-8B-Instruct"}
run_name="${gate_type}_lr${lr}_maxlen${training_max_length}_warmup${warmup_steps}_bs${bs}_steps${steps}_gatelossscale${gate_loss_scale}"

echo $run_name
torchrun --nproc_per_node=$gpus --master_port=10003 distillation_prefill.py  \
        --base_model ${base_model} \
        --seerattn_gate_type $gate_type \
        --bf16 True \
        --output_dir models/seer_attn_llama_3.1/$run_name      \
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

## Qwen2.5 Instruct:
## Change training_max_length=32768