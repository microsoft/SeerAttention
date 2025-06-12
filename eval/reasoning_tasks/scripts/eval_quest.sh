model_dir="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
output_dir="./result_quest"
model_size="14B"
attention_implementation="quest"
max_tokens=32768
limit=-1
num_gpus=8

# tasks="aime24,aime25,math,gpqa,livecodebench"
tasks="math"


block_size="64"
sparsity_method="token_budget"
token_budget="4096"
sliding_window_size="0"
start_layer="0"


python parallel_run_hf.py \
      --model_dir "$model_dir" \
      --model_size "$model_size" \
      --tasks "$tasks" \
      --output_dir "$output_dir" \
      --attention_implementation "$attention_implementation" \
      --block_size "$block_size" \
      --sparsity_method "$sparsity_method" \
      --token_budget "$token_budget" \
      --sliding_window_size "$sliding_window_size" \
      --limit "$limit" \
      --num_gpus "$num_gpus" \
      --max_tokens "$max_tokens" \
      --start_layer "$start_layer" \