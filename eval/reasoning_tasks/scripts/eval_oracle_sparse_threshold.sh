model_dir="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
output_dir="./result_oracle_sparse"
model_size="14B"
attention_implementation="oracle_sparse"
max_tokens=32768
num_gpus=8
limit=-1

# tasks="aime24,aime25,math,gpqa,livecodebench"
tasks="math"

block_size="16,32,64,128"
sparsity_method="threshold"
threshold="5e-4,1e-3,5e-3,1e-2"

python parallel_run_hf.py \
      --model_dir "$model_dir" \
      --model_size "$model_size" \
      --tasks "$tasks" \
      --output_dir "$output_dir" \
      --attention_implementation "$attention_implementation" \
      --block_size "$block_size" \
      --sparsity_method "$sparsity_method" \
      --threshold "$threshold" \
      --profile_sparsity \
      --num_gpus "$num_gpus" \
      --limit "$limit" \
      --max_tokens "$max_tokens" \