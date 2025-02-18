model_path="../../models/seer_attn_llama_3.1/Qavg_Kmaxminavg_lr1e-3_maxlen65536_warmup20_bs16_steps500_gatelossscale10.0"
# change model to the path of your model if needed
CUDA_VISIBLE_DEVICES=0 python eval_ppl.py \
    -m $model_path \
    --split test \
    --feature text \
    --dataset-min-tokens 131072 \
    --dataset emozilla/pg19-test \
    --nz-ratios 0.1,0.2,0.3,0.4,0.5,1.0\
    --use_seer_attn \
    --sparsity-method nz_ratio\
    --max-tokens 131072 \
    --min-tokens 8192 \
    --truncate \
    --output-file ./results/pg_ppl.txt

CUDA_VISIBLE_DEVICES=0 python eval_ppl.py \
    -m $model_path \
    --split test \
    --feature text \
    --dataset-min-tokens 131072 \
    --dataset hoskinson-center/proof-pile \
    --nz-ratios 0.1,0.2,0.3,0.4,0.5,1.0\
    --use_seer_attn \
    --sparsity-method nz_ratio\
    --max-tokens 131072 \
    --min-tokens 8192 \
    --samples 10 \
    --truncate \
    --output-file ./results/proofpile_ppl.txt








