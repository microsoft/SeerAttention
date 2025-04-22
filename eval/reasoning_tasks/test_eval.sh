task=aime
bs=30
threshold=4e-3

model_dir=SeerAttention/SeerAttention-DeepSeek-R1-Distill-Qwen-14B-Decode-AttnGates
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python eval.py \
    --model_name_or_path $model_dir \
    --data_name $task \
    --batch_size $bs \
    --limit -1 \
    --output_dir ./results_sparse_aime \
    --attention_implementation seer_sparse \
    --threshold $threshold \
    --use_batch_exist \
    --use_fused_kernel \
    --profile_sparsity \
    --surround_with_messages 



## Dense Baseline
model_dir="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --model_name_or_path $model_dir \
    --data_name $task \
    --batch_size $bs \
    --limit -1 \
    --output_dir ./results_sparse_aime \
    --attention_implementation fa2 \
    --use_batch_exist \
    --surround_with_messages 
