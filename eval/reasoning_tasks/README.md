# Reasoning Tasks Evaluation


This math evaluation framework is modified from [LIMO](https://github.com/GAIR-NLP/LIMO/blob/main/eval). 

```bash
    pip install -r reqirements.txt
    bash test_eval.sh # run one single demo on AIME
```
Change limit=-1 in the scipt to run all the tests.

| threshold| 0.005  | 0.001 | Dense |
|----------|--------|-------|-------|
| Acc      | 73.33  | 73.33 | 70    |
| Sparsity | 86%    | 61.68%| 0     |