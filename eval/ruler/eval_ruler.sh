model=../../models/seer_attn_llama_3.1/Qavg_Kmaxminavg_lr1e-3_maxlen65536_warmup20_bs16_steps500_gatelossscale10.0
# change model to the path of your model if needed
basedir=./results/ruler
threshold=5e-4
export CUDA_VISIBLE_DEVICES=0
export PROFILE_FILE=${basedir}/${threshold}.txt # Comment this line to disable profiling
bash run_seer.sh \
    $model \
    SeerAttn \
    $basedir \
    $threshold

## Get profiled sparsity
python averaged_sparsity.py --file $PROFILE_FILE