model_dir="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
output_dir="./result_dense"
task=aime # aime math gpqa olympiadbench

bs=60
limit=-1
repeat=8


# 0 fa2 0; 1 fa2 0; 1 ours 0; 1 ours 1
use_batch_exist=1
attention_implementation=fa2 # fa2 ours
use_sparse_kernel=0          # must be 0 when use fa2

for gpu in 0 1 2 3 4 5 6 7
do
    CUDA_VISIBLE_DEVICES=$gpu \
    python eval_dense.py \
    --model_name_or_path $model_dir \
    --data_name $task \
    --batch_size $bs \
    --limit $limit \
    --repeat $repeat \
    --output_dir $output_dir \
    --attention_implementation $attention_implementation \
    --use_batch_exist $use_batch_exist \
    --use_sparse_kernel $use_sparse_kernel \
    --surround_with_messages  \
    --rank $gpu  & 
done
wait

echo "All generation finished"

for gpu in 0 1 2 3 4 5 6 7 
do
    python get_results.py \
    --model_name_or_path $model_dir \
    --data_name $task \
    --batch_size $bs \
    --limit $limit \
    --repeat $repeat \
    --output_dir $output_dir \
    --attention_implementation $attention_implementation \
    --use_batch_exist $use_batch_exist \
    --use_sparse_kernel $use_sparse_kernel \
    --surround_with_messages  \
    --rank $gpu  
done

echo "All finished"