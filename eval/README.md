# Evaluation of SeerAttention

## Results of SeerAttention-R for Reasoning Tasks

Results of reasoning task with different token budgets. 
All the results are the averaged pass@1 results with 64 sample per query for AIME, 16 samples for GPQA-Diamond, and 8 samples for MATH-500.
See `reasoning_tasks` for details. 

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


## Results of SeerAttention for prefill sparsity


## Perplexity on PG19
For the perplexity test, we adopts TopK (nz_ratio) based sparsity method. 
The ppl is evaluated with data > 128k and truncated to test length. 

| Density | 8192  | 16384 | 32768 | 65536 | 131072 |
|---------|-------|-------|-------|-------|--------|
| 1.00    | 10.03 | 9.88  | 9.92  | 9.97  | 10.03  |
| 0.50    | 10.04 | 9.89  | 9.92  | 9.99  | 10.05  |
| 0.40    | 10.06 | 9.89  | 9.93  | 9.99  | 10.07  |
| 0.30    | 10.09 | 9.91  | 9.95  | 10.01 | 10.15  |
| 0.20    | 10.19 | 9.94  | 9.97  | 10.04 | 10.37  |
| 0.10    | 10.61 | 10.08 | 10.04 | 10.09 | 10.88  |

## LongBench

For the LongBench evaluation, we use Threshold based sparsity method. The threshold is set to 2e-3. 

| Task                 | 0-4k  | 4-8k  | 8k+   |
|----------------------|-------|-------|-------|
| 2wikimqa             | 51.1  | 47.85 | 33.36 |
| gov_report           | 35.03 | 35.05 | 34.57 |
| hotpotqa             | 63.97 | 60.0  | 56.7  |
| lcc                  | 67.98 | 73.18 | 65.28 |
| multi_news           | 28.1  | 25.78 | 24.25 |
| multifieldqa_en      | 58.63 | 51.45 | 51.87 |
| passage_count        | 18.0  | 10.15 | 11.88 |
| passage_retrieval_en | 100.0 | 99.0  | 98.0  |
| qasper               | 47.77 | 44.04 | 39.63 |
| repobench-p          | 51.78 | 56.24 | 56.75 |
| samsum               | 43.28 | 41.19 | 45.29 |
| trec                 | 64.0  | 76.0  | 75.0  |
| triviaqa             | 90.91 | 88.45 | 92.43 |
| averaged             | 55.43 | 54.49 | 52.69 |


## RULER
For the RULER benchmark, we use Threshold based sparsity method. The threshold is set to 5e-4. 

|          | niah_single_1 | niah_single_2 | niah_single_3 | niah_multikey_1 | niah_multikey_2 | niah_multikey_3 | niah_multivalue | niah_multiquery | vt   | cwe   | fwe   | qa_1 | qa_2 | avg    |
|----------|---------------|---------------|---------------|-----------------|-----------------|-----------------|-----------------|-----------------|------|-------|-------|------|------|--------|
| 4k       | 100           | 100           | 100           | 100             | 100             | 100             | 100             | 100             | 99.2 | 100   | 90.67 | 84   | 68   | 95.53  |
| 8k       | 100           | 100           | 100           | 100             | 100             | 100             | 100             | 100             | 99.2 | 98    | 88    | 68   | 52   | 92.71  |
| 16k      | 100           | 100           | 100           | 96              | 100             | 100             | 100             | 100             | 100  | 85.6  | 90.67 | 68   | 56   | 92.02  |
| 32k      | 100           | 100           | 100           | 100             | 96              | 100             | 92              | 99              | 99.2 | 50.8  | 93.33 | 72   | 48   | 88.49  |
| 64k      | 100           | 100           | 100           | 100             | 100             | 92              | 94              | 99              | 97.6 | 8     | 82.67 | 64   | 48   | 83.48  |
| 128k     | 100           | 100           | 100           | 100             | 76              | 56              | 94              | 94              | 75.2 | 0     | 66.67 | 64   | 28   | 73.37  |


## Sparse Attention Kernel Efficiency

![Efficiency](../figures/efficiency_prefill.png)