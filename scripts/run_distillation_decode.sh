#!/bin/bash

headpooling_type=${HEADPOOLING_TYPE:-"Qproj"} # Using Linear layer to aggregate Q heads to K heads
gate_type=${GATE_TYPE:-"Kmaxminavg"}  ## A combination of max, min, and average pooling
gate_hidden_size=${SEERATTN_GATE_HIDDEN_SIZE:-128}     
blocksize=${SEERATTN_BLOCK_SIZE:-64}
warmup_steps=${WARMUP_STEPS:-20}
training_max_length=${TRAINING_MAX_LENGTH:-32768}
lr=${LR:-1e-3}
weight_decay=${WEIGHT_DECAY:-0.0}
gpus=${GPUS:-8}
gate_loss_scale=${GATE_LOSS_SCALE:-10.0}
use_qk_norm=${USE_QK_NORM:-true}
use_rope=${USE_ROPE:-true}
block_slice_mode=${BLOCK_SLICE_MODE:-false}
output_dir=${OUTPUT_DIR:-"./models/"}
steps=${STEPS:-800}
bs=${BS:-16}
loss_slice_ratio=${LOSS_SLICE_RATIO:-0.5}
dataset_name=${DATASET_NAME:-"open-r1/OpenR1-Math-220k"}
base_model=${BASE_MODEL:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"}
base_name=$(basename "$base_model")
prefix=${PREFIX:-"openr1"}
ds_config=${DS_CONFIG:-"ds_config/stage2.json"}
gradient_accumulation_steps=$((bs/gpus))

export WANDB_MODE=offline

run_name="${prefix}_${headpooling_type}_${gate_type}_bs${bs}_gdim${gate_hidden_size}_block${blocksize}_wd${weight_decay}_lr${lr}_slice${loss_slice_ratio}_qknorm${use_qk_norm}"
echo "Running with headpooling type: ${headpooling_type}"
echo "Run name: ${run_name}"

torchrun --nproc_per_node=$gpus --master_port=10003 distillation_decode.py  \
    --base_model $base_model \
    --seerattn_k_seq_pooling_type $gate_type \
    --bf16 True \
    --output_dir ${output_dir}/$base_name/$run_name \
    --dataset_name $dataset_name \
    --training_max_length $training_max_length \
    --num_train_epochs 1     \
    --per_device_train_batch_size 1     \
    --gradient_accumulation_steps $gradient_accumulation_steps     \
    --gate_loss_scale $gate_loss_scale \
    --eval_strategy "no"     \
    --save_strategy "no"     \
    --learning_rate $lr     \
    --weight_decay $weight_decay     \
    --warmup_steps $warmup_steps     \
    --lr_scheduler_type "cosine_with_min_lr"     \
    --lr_scheduler_kwargs '{"min_lr":1e-5}'     \
    --logging_steps 1     \
    --deepspeed $ds_config \
    --seerattn_q_head_pooling_type $headpooling_type \
    --seerattn_gate_hidden_size $gate_hidden_size \
    --seerattn_gate_block_size $blocksize \
    --seerattn_loss_slice_ratio $loss_slice_ratio \
    --seerattn_use_qk_norm $use_qk_norm \
    --seerattn_use_rope $use_rope \
    --max_steps $steps 
