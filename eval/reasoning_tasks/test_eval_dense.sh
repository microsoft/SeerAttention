
model_dir="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
output_dir="./result_dense"
task=math # aime math gpqa olympiadbench

bs=100
limit=100
repeat=1

use_batch_exist=1     
attention_implementation=fa2 # fa2 ours
use_sparse_kernel=0          # must be 0 when use fa2
gpu=0

CUDA_VISIBLE_DEVICES=$gpu python eval_dense.py \
    --model_name_or_path $model_dir \
    --data_name $task \
    --batch_size $bs \
    --limit $limit \
    --repeat $repeat \
    --output_dir $output_dir \
    --attention_implementation $attention_implementation \
    --use_batch_exist $use_batch_exist\
    --use_sparse_kernel $use_sparse_kernel \
    --surround_with_messages \
    --rank $gpu

python get_results.py \
    --model_name_or_path $model_dir \
    --data_name $task \
    --batch_size $bs \
    --limit $limit \
    --repeat $repeat \
    --output_dir $output_dir \
    --attention_implementation $attention_implementation \
    --use_batch_exist $use_batch_exist\
    --use_sparse_kernel $use_sparse_kernel \
    --surround_with_messages \
    --rank $gpu

# CUDA_VISIBLE_DEVICES=1 python eval.py \
#     --model_name_or_path $model_dir \
#     --data_name $task \
#     --batch_size $bs \
#     --limit $limit \
#     --repeat $repeat \
#     --output_dir $output_dir  \
#     --attention_implementation $attention_implementation \
#     --threshold 0.0 \
#     --use_batch_exist \
#     --use_fused_kernel \
#     --surround_with_messages 
