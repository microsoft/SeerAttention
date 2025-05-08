model_dir="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
output_dir="./result_dense"
model_size="14B"
attention_implementation="fa2"
num_gpus=8
limit=-1

tasks="aime,math,gpqa,olympiadbench"

python parallel_run_hf.py \
      --model_dir "$model_dir" \
      --tasks "$tasks" \
      --output_dir "$output_dir" \
      --attention_implementation "$attention_implementation" \
      --sparsity_method "threshold" \
      --threshold "0" \
      --num_gpus "$num_gpus" \
      --limit "$limit" \
