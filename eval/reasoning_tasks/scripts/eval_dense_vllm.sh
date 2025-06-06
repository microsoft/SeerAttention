# "pip install vllm" on NVIDIA GPUS
# use vllm docker on MI300

model_dir="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
output_dir="./result_dense"
max_tokens=32768
num_gpus=8
limit=-1

# tasks="aime24,aime25,math,gpqa,livecodebench"
tasks="math"


python parallel_run_vllm.py \
      --model_dir "$model_dir" \
      --tasks "$tasks" \
      --output_dir "$output_dir" \
      --num_gpus "$num_gpus" \
      --limit "$limit" \
      --max_tokens "$max_tokens" \

