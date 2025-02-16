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

Noted that the huggingface-hub version in RULER might confict with others. Be mindful if you encounter any error. 