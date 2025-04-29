model_dir="/home/v-shumingguo/gsm_blob/models/DeepSeek-R1-Distill-Qwen-14B"
output_dir="./result_oracle_sparse"
model_size="14B"
attention_implementation="oracle_sparse"

# tasks="aime,math,gpqa,olympiadbench"
tasks="math"
sparsity_method="token_budget"
token_budget="1536"

python parallel_run_token_budget.py \
      --model_dir "$model_dir" \
      --model_size "$model_size" \
      --tasks "$tasks" \
      --output_dir "$output_dir" \
      --attention "$attention_implementation" \
      --sparsity_method "$sparsity_method" \
      --token_budget "$token_budget" \
      --profile_sparsity \