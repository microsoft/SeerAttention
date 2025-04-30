model_dir="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
output_dir="./result_oracle_sparse"
model_size="14B"
attention_implementation="oracle_sparse"

# tasks="aime,math,gpqa,olympiadbench"
tasks="math"

block_size="16,32,64,128"
sparsity_method="threshold"
threshold="5e-4,1e-3,5e-3,1e-2"

python parallel_run.py \
      --model_dir "$model_dir" \
      --model_size "$model_size" \
      --tasks "$tasks" \
      --output_dir "$output_dir" \
      --attention_implementation "$attention_implementation" \
      --block_size "$block_size" \
      --sparsity_method "$sparsity_method" \
      --threshold "$threshold" \
      --profile_sparsity \