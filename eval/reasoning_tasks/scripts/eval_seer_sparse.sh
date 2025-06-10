model_dir="SeerAttention/SeerAttention-Decode-Qwen3-4B-AttnGates"
output_dir="./result_seer_sparse"
attention_implementation="seer_sparse"
max_tokens=32768
num_gpus=8
limit=-1

# tasks="aime24,aime25,math,gpqa"
tasks="aime24"

block_size="64"    # SeerAttention uses a fixed block size of 64
sparsity_method="token_budget" 
token_budget="2048,4096,6144,8192"
start_layer=0

python parallel_run_hf.py \
      --model_dir "$model_dir" \
      --tasks "$tasks" \
      --output_dir "$output_dir" \
      --attention_implementation "$attention_implementation" \
      --block_size "$block_size" \
      --sparsity_method "$sparsity_method" \
      --token_budget "$token_budget" \
      --profile_sparsity \
      --num_gpus "$num_gpus" \
      --limit "$limit" \
      --max_tokens "$max_tokens" \
      --start_layer "$start_layer" 