# "pip install vllm" on NVIDIA GPUS
# use vllm docker on MI300

model_dir="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
output_dir="./result_dense"

# tasks="aime,math,gpqa,olympiadbench"
tasks="math"

python parallel_run_vllm.py \
      --model_dir "$model_dir" \
      --tasks "$tasks" \
      --output_dir "$output_dir" \

