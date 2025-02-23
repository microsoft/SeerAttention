model_path="SeerAttention/SeerAttention-Llama-3.1-8B-AttnGates"
# change model to the path of your model if needed

export PROFILE_FILE="./results/llama/8B_2e-3.txt" # Comment this line to disable profiling
python run.py \
    --output_dir ./results/llama \
    --model_checkpoint $model_path \
    --threshold 2e-3 


model_path="SeerAttention/SeerAttention-Qwen2.5-7B-AttnGates"
export PROFILE_FILE="./results/qwen/7B_2e-3.txt" 
python run.py \
    --output_dir ./results/qwen \
    --model_checkpoint $model_path \
    --threshold 2e-3 

## Get profiled sparsity
python averaged_sparsity.py --file $PROFILE_FILE