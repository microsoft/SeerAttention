model_dir="SeerAttention/SeerAttention-DeepSeek-R1-Distill-Qwen-14B-Decode-AttnGates"
output_dir="./outputs"
# data_name="math"
# data_name="gpqa"
data_name="aime"


batch_size=1         
max_tokens=32768
limit=1  # -1 for all
threshold=0.001
nz_ratio=0.1

export CUDA_VISIBLE_DEVICES=0

# test for threshold
python eval.py \
--model_name_or_path $model_dir \
--data_name $data_name \
--batch_size $batch_size \
--limit $limit \
--max_tokens $max_tokens \
--output_dir $output_dir \
--threshold $threshold \
--surround_with_messages \


# test for topk
# python eval_topk.py \
# --model_name_or_path $model_dir \
# --data_name $data_name \
# --batch_size $batch_size \
# --limit $limit \
# --max_tokens $max_tokens \
# --output_dir $output_dir \
# --nz_ratio $nz_ratio \
# --surround_with_messages \






