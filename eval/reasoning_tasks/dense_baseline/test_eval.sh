task=aime # aime math gpqa olympiadbench
bs=30


## Dense Baseline
# model_dir="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
# model_dir="/mnt/output/models/DeepSeek-R1-Distill-Qwen-14B"
model_dir="/mnt/output/models/DeepSeek-R1-Distill-Qwen-14B"
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --model_name_or_path $model_dir \
    --data_name $task \
    --batch_size $bs \
    --limit -1 \
    --output_dir ./results_sparse_aime \
    --attention_implementation fa2 \
    --use_batch_exist \
    --surround_with_messages 
