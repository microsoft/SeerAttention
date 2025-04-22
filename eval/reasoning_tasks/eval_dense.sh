 
# model_dir="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
# model_dir="/mnt/output/distilled_model_gates/DeepSeek-R1-Distill-Qwen-14BSFT-selfdata-sharehead/bs16_steps858_Qproj_thres0_gatehidden128_blocksize64"
model_dir="/mnt/output/models/DeepSeek-R1-Distill-Qwen-14B"
# output_dir="./result_dense"
output_dir="/mnt/output/results/lm_eval_batch/dense_test/DeepSeek-R1-Distill-Qwen-14B"
task=gpqa # aime math gpqa olympiadbench

bs=100
limit=-1
repeat=1

use_batch_exist=1     
attention_implementation=ours # fa2 ours
use_sparse_kernel=0           # must be 0 when use fa2

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
    --surround_with_messages  &
done
wait

echo "All finished"
# for gpu in 0 1 2 3
# do
#     CUDA_VISIBLE_DEVICES=$gpu \
#     python eval.py \
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
#     --surround_with_messages  &
# done
# wait
