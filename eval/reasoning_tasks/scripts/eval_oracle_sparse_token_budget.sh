model_dir="/mnt/output/models/DeepSeek-R1-Distill-Qwen-14B"
output_dir="./result_oracle_sparse"
model_size="14B"
attention_implementation="oracle_sparse"

# tasks="aime,math,gpqa,olympiadbench"
tasks="aime"

block_size="16,32,64,128"
sparsity_method="token_budget"
token_budget="1024,2048,4096,8192"
sliding_window_size="0,512,1024"
# threshold="1e-3"

python parallel_run_token_budget.py \
      --model_dir "$model_dir" \
      --model_size "$model_size" \
      --tasks "$tasks" \
      --output_dir "$output_dir" \
      --attention "$attention_implementation" \
      --block_size "$block_size" \
      --sparsity_method "$sparsity_method" \
      --token_budget "$token_budget" \
      --sliding_window_size "$sliding_window_size" \
      --profile_sparsity \