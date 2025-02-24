model_path="SeerAttention/SeerAttention-Llama-3.1-8B-AttnGates"
nz_ratios=(0.1 0.2 0.3 0.4 0.5 1.0)

# Start at GPU 0
gpu=0
num_total_gpus=8

# Process the first dataset: emozilla/pg19-test
dataset="emozilla/pg19-test"
for nz in "${nz_ratios[@]}"; do
    echo "Running dataset $dataset with nz_ratio $nz on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python eval_ppl.py \
        -m $model_path \
        --split test \
        --feature text \
        --dataset-min-tokens 131072 \
        --dataset $dataset \
        --nz-ratios $nz \
        --use_seer_attn \
        --sparsity-method nz_ratio \
        --max-tokens 131072 \
        --min-tokens 8192 \
        --truncate \
        --output-file ./results/llama_pg_ppl.txt &
    gpu=$(( (gpu + 1) % $num_total_gpus ))
done

wait 

## Qwen2.5 Instruct context length is currently 32k. We have not explore dynamic rope scaling yet.
model_path="SeerAttention/SeerAttention-Qwen2.5-7B-AttnGates"
for nz in "${nz_ratios[@]}"; do
    echo "Running dataset $dataset with nz_ratio $nz on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python eval_ppl_seperate.py \
        -m $model_path \
        --split test \
        --feature text \
        --dataset-min-tokens 131072 \
        --dataset $dataset \
        --nz-ratios $nz \
        --use_seer_attn \
        --sparsity-method nz_ratio \
        --max-tokens 32768 \
        --min-tokens 8192 \
        --truncate \
        --output-file ./results/qwen_pg_ppl.txt &
    gpu=$(( (gpu + 1) % $num_total_gpus ))
done
wait



