# Reasoning Tasks Evaluation


This math evaluation framework is modified from [LIMO](https://github.com/GAIR-NLP/LIMO/blob/main/eval). We tuned a very conservative batch size config for 80GB GPU memory, you can adjust it in parallel_run_hf.py

```bash
bash scripts/eval_seer_sparse.sh 
```

## Results of Reasoning Tasks

Results of reasoning task with different token budgets. 
All the results are the averaged pass@1 results with 64 sample per query for AIME, 16 samples for GPQA-Diamond, and 8 samples for MATH-500.

### AIME24

| Model                         | 2k    | 4k    | 6k    | 8k    | Full Attention |
|-------------------------------|-------|-------|-------|-------|----------------|
| Qwen3-4B                      | 55.42 | 68.75 | 70.94 | 72.50 | 71.25          |
| Qwen3-8B                      | 56.56 | 72.29 | 74.22 | 75.05 | 74.48          |
| Qwen3-14B                     | 62.24 | 75.78 | 78.02 | 78.65 | 78.91          |
| DeepSeek-R1-Distill-Qwen-14B  | 55.78 | 66.35 | 67.50 | 66.82 | 67.50          |


### AIME25

| Model                         | 2k    | 4k    | 6k    | 8k    | Full Attention |
|-------------------------------|-------|-------|-------|-------|----------------|
| Qwen3-4B                      | 45.73 | 57.60 | 60.20 | 62.90 | 66.41          |
| Qwen3-8B                      | 42.60 | 56.77 | 60.31 | 64.17 | 67.86          |
| Qwen3-14B                     | 46.67 | 62.66 | 67.19 | 69.01 | 70.21          |
| DeepSeek-R1-Distill-Qwen-14B  | 38.44 | 47.19 | 52.25 | 50.05 | 50.00          |


### MATH500

| Model                         | 1k    | 2k    | 4k    | 6k    | Full Attention |
|-------------------------------|-------|-------|-------|-------|----------------|
| Qwen3-4B                      | 84.80 | 92.20 | 93.60 | 93.60 | 93.93          |
| Qwen3-8B                      | 82.82 | 91.53 | 94.17 | 94.53 | 94.43          |
| Qwen3-14B                     | 85.13 | 93.20 | 94.77 | 94.80 | 95.22          |
| DeepSeek-R1-Distill-Qwen-14B  | 87.65 | 92.10 | 93.05 | 93.12 | 93.30          |


### GPQA-Diamond

| Model                         | 1k    | 2k    | 4k    | 6k    | Full Attention |
|-------------------------------|-------|-------|-------|-------|----------------|
| Qwen3-4B                      | 39.61 | 51.20 | 55.20 | 55.90 | 56.19          |
| Qwen3-8B                      | 37.59 | 54.32 | 59.60 | 60.48 | 60.54          |
| Qwen3-14B                     | 44.54 | 59.72 | 63.76 | 64.20 | 65.25          |
| DeepSeek-R1-Distill-Qwen-14B  | 51.26 | 56.79 | 56.41 | 57.48 | 57.80          |