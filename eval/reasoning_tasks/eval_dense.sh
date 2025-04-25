model_dir="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
output_dir="./result_dense"
task=olympiadbench # aime math gpqa olympiadbench

bs=60
limit=-1
repeat=1

use_batch_exist=1
attention_implementation=seer_sparse # fa2 seer_sparse

for gpu in 0 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu \
    python eval.py \
    --model_name_or_path $model_dir \
    --data_name $task \
    --batch_size $bs \
    --limit $limit \
    --repeat $repeat \
    --output_dir $output_dir \
    --attention_implementation $attention_implementation \
    --threshold 0 \
    --use_batch_exist $use_batch_exist \
    --surround_with_messages  \
    --rank $gpu &
done
wait

echo "All generation finished"

for gpu in 0 1 2 3
do
    python get_results.py \
    --model_name_or_path $model_dir \
    --data_name $task \
    --batch_size $bs \
    --limit $limit \
    --repeat $repeat \
    --output_dir $output_dir \
    --surround_with_messages  \
    --rank $gpu  
done

echo "All finished"