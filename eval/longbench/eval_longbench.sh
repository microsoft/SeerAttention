model_path="../../models/seer_attn_llama_3.1/Qavg_Kmaxminavg_lr1e-3_maxlen65536_warmup20_bs16_steps500_gatelossscale10.0"
# change model to the path of your model if needed

export PROFILE_FILE="./results/Qavg_Kmaxminavg_2e-3.txt" # Comment this line to disable profiling
python run.py \
    --output_dir ./results/ \
    --model_checkpoint $model_path \
    --threshold 2e-3 

## Get profiled sparsity
python averaged_sparsity.py --file $PROFILE_FILE