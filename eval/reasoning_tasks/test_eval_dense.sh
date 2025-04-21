# model_dir="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
# model_dir="/mnt/output/models/DeepSeek-R1-Distill-Qwen-14B"
model_dir="/mnt/output/models/DeepSeek-R1-Distill-Qwen-14B"
# output_dir="./result_dense"
output_dir="./result_dense"
task=math # aime math gpqa olympiadbench

bs=4
limit=4
repeat=1

use_batch_exist=1     
attention_implementation=ours # fa2 ours


CUDA_VISIBLE_DEVICES=0 python eval_dense.py \
    --model_name_or_path $model_dir \
    --data_name $task \
    --batch_size $bs \
    --limit $limit \
    --repeat $repeat \
    --output_dir $output_dir \
    --attention_implementation $attention_implementation \
    --use_batch_exist $use_batch_exist\
    --surround_with_messages 
