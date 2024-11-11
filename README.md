# SeerAttention

SeerAttention is a learning-based method to enable block-level sparse attention for long-context LLM without using prefined static pattern or heuristic methods. It can be applied in Post-training or Fine-tuning stages. The Attention Gate units learn from the intrinsic sparsity in the pre-trained models.  
[[arxiv paper](https://arxiv.org/abs/2410.13276)] 

![SeerAttn](figures/SeerAttn.png)


The AttnGate units perform pooling in sequence dimension and predict the estimated max attention score of a block. By applying row-wise TopK on top the results, sparse block indices can be generated. 

<div style="text-align: center;">
    <img src="figures/illustration.png" alt="drawing" width="550"/>
</div>


## Environment
```bash
conda create -yn seer python=3.11
conda activate seer
pip install torch==2.4.0
pip install -r requirements.txt
```


## Download the pretrained models for experiments
```bash
mkdir models
huggingface-cli download meta-llama/Llama-3.1-8B --local-dir  models/meta-llama/Llama-3.1-8B
huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir  models/meta-llama/Meta-Llama-3-8B
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir  models/meta-llama/Llama-3.1-8B-Instruct
```


## Post-training with SeerAttention
Only AttnGates are trained in Post-training case. In other words, the original model's weights are untouched. The AttnGates are trained using the 2D maxpooled attention map from the original model in a given training/calibration dataset.

Run the below script to reproduce the results on llama-3.1-8B. Once you obtain the model with AttnGates, different sparsity ratios can be applied. PPL results will be evaluated on pg19 and proof-pile datasets.
```bash
bash scripts/run_post_training.sh
```


## Fine-tuning with SeerAttention
You can fine-tuning a model during long-context extension with SeerAttention. Both original models and AttnGates are tuned. To stabilize the training process, the AttnGates will first be initialized using the Post-training method before context length extension. 

Run the below scripts to reproduce the dense baseline and Seerattention (50% sparsity) results of extending llama-3-8B from 8k to 32k. 
```bash
bash scripts/run_dense_yarn_finetuning.sh
bash scripts/run_seerattn_yarn_finetuning.sh
```

## Experiment with other AttnGate designs
The current AttnGate design is simple, only pooling + linear layers. You are encouraged to contribute your own design and train with our customized attention pooling kernel that generates ground truth. It is a functional self-attention kernel but also outputs the 2D maxpooled (block-size 64) attention map.
```python
from seer_attn.kernels.attn_pooling_kernel import attn_with_pooling
###...

predict_mask = your_gate_design(...)

attn_output, pooling_gt = attn_with_pooling(
    query_states,
    key_states,
    value_states,
    True, 
    1.0 / math.sqrt(self.head_dim)      
)

###...
loss = mse(predict_mask, pooling_gt)   
```
## Inference Kerenel Development
Our current block sparse attention triton kernel is experimental with limited use cases. Currently it does not support external attention masks.


## Citation

If you find SeerAttention useful or want to use in your projects, please kindly cite our paper:

```bibtex
@article{gao2024seerattention,
    title={SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs},
    author={Gao, Yizhao and Zeng, Zhichen and Du, Dayou and Cao, Shijie and So, Hayden Kwok-Hay and Cao, Ting and Yang, Fan and Yang, Mao},
    journal={arXiv preprint arXiv:2410.13276},
    year={2024}
}

```


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
