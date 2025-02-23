# Ruler Test

The RULER benchmark scripts is modified from [MInference's repo](https://github.com/microsoft/MInference/tree/main/experiments/ruler).

To run the RULER benchmark, you need first install the requirements:
```
    pip install Cython
    pip install -r requirements.txt
```

To reproduce SeerAttn results on RULER, run:
```
    bash eval_ruler.sh
```

Noted that the some package version like huggingface-hub in RULER might confict with others. If you find any errors, feel free to leave an issue.