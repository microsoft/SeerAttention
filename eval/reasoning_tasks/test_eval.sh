task=aime
bs=30
threshold=4e-3
limit=-1
rank=1

model_dir=SeerAttention/SeerAttention-DeepSeek-R1-Distill-Qwen-14B-Decode-AttnGates
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python eval.py \
    --model_name_or_path $model_dir \
    --rank $rank \
    --data_name $task \
    --batch_size $bs \
    --limit $limit \
    --output_dir ./results_sparse_aime \
    --attention_implementation seer_sparse \
    --threshold $threshold \
    --use_batch_exist true \
    --use_fused_kernel \
    --profile_sparsity \
    --surround_with_messages 

python get_results.py \
    --data_name $task \
    --limit $limit \
    --output_dir ./results_sparse_aime/rank${rank} \
    --profile_sparsity \

