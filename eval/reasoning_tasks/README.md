# Reasoning Tasks Evaluation


This math evaluation framework is modified from [LIMO](https://github.com/GAIR-NLP/LIMO/blob/main/eval). We tuned a very conservative batch size config for 80GB GPU memory, you can adjust it in parallel_run_hf.py

```bash

## Run the script to reproduce results of Qwen3-4B. 
bash scripts/eval_seer_sparse.sh 
```

## Results of Reasoning Tasks

Results of reasoning task with different token budgets. 
All the results are the averaged pass@1 results with 64 sample per query for AIME, 16 samples for GPQA-Diamond, and 8 samples for MATH-500.

### AIME24

| Model                         | 2k    | 4k    | 6k    | 8k    | Full Attention |
|-------------------------------|-------|-------|-------|-------|----------------|
| Qwen3-4B                      | 55.83 | 69.32 | 70.47 | 72.60 | 71.25          |
| Qwen3-8B                      | 58.23 | 71.35 | 74.06 | 74.22 | 74.48          |
| Qwen3-14B                     | 63.65 | 75.73 | 78.49 | 78.85 | 78.91          |
| DeepSeek-R1-Distill-Qwen-14B  | 55.97 | 66.68 | 67.66 | 67.45 | 67.50          |


### AIME25

| Model                         | 2k    | 4k    | 6k    | 8k    | Full Attention |
|-------------------------------|-------|-------|-------|-------|----------------|
| Qwen3-4B                      | 45.16 | 58.59 | 61.88 | 62.40 | 66.41          |
| Qwen3-8B                      | 43.30 | 57.81 | 63.07 | 64.22 | 67.86          |
| Qwen3-14B                     | 48.70 | 64.79 | 68.70 | 69.84 | 70.21          |
| DeepSeek-R1-Distill-Qwen-14B  | 39.37 | 48.12 | 50.10 | 50.42 | 50.00          |


### MATH500

| Model                         | 1k    | 2k    | 4k    | 6k    | Full Attention |
|-------------------------------|-------|-------|-------|-------|----------------|
| Qwen3-4B                      | 84.67 | 91.85 | 94.10 | 94.12 | 93.93          |
| Qwen3-8B                      | 83.57 | 91.67 | 94.00 | 94.53 | 94.43          |
| Qwen3-14B                     | 86.12 | 93.02 | 95.12 | 95.40 | 95.22          |
| DeepSeek-R1-Distill-Qwen-14B  | 87.35 | 92.45 | 93.77 | 93.40 | 93.30          |


### GPQA-Diamond

| Model                         | 1k    | 2k    | 4k    | 6k    | Full Attention |
|-------------------------------|-------|-------|-------|-------|----------------|
| Qwen3-4B                      | 39.84 | 49.94 | 55.40 | 55.90 | 56.19          |
| Qwen3-8B                      | 39.43 | 54.41 | 60.48 | 60.57 | 60.54          |
| Qwen3-14B                     | 45.64 | 61.68 | 63.83 | 64.33 | 65.25          |
| DeepSeek-R1-Distill-Qwen-14B  | 52.02 | 55.46 | 56.34 | 57.20 | 57.80          |