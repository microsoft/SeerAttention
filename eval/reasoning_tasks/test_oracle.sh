task=aime
bs=30
threshold=1e-4
block_size=64

## Dense Baseline
model_dir="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
CUDA_VISIBLE_DEVICES=3 python eval.py \
    --model_name_or_path $model_dir \
    --data_name $task \
    --batch_size $bs \
    --limit -1 \
    --output_dir ./results_oracle_aime \
    --attention_implementation oracle_sparse \
    --threshold $threshold \
    --block_size $block_size \
    --use_batch_exist \
    --profile_sparsity \
    --surround_with_messages