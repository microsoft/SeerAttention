import os
import math
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling
from torch.distributed import barrier
from seer_attn.modeling_llama_seerattn_ft import LlamaForCausalLMSeerAttnFT

import importlib

from mytrainer import AttnGateTrainer

from datasets import load_dataset
import warnings
from huggingface_hub import login

warnings.simplefilter(action='ignore', category=FutureWarning)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B")
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length."},
    )
    train_dense_baseline: bool = field(
        default=False,
        metadata={"help": "Train dense baseline with yarn extension."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a checkpoint to resume training from."},
    )
    use_mse_loss: bool = field(
        default=True,
        metadata={"help": "Use MSE loss instead of SmoothL1 loss."},
    )
    scaling_factor: float = field(
        default=8.0,
        metadata={"help": "Scaling_factor to scale up the rope."},
    )
    gate_loss_scale: float = field(
        default=1.0,
        metadata={"help": "Mask loss scale."},
    )
    nz_ratio: float = field(
        default=0.1053,
        metadata={"help": "non-zero ratio in training."},
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def tokenize_fn(tokenizer, example):
    context_length = tokenizer.model_max_length
    outputs = tokenizer(
        tokenizer.eos_token.join(example["text"]),
        truncation=False,
        return_tensors="pt",
        pad_to_multiple_of=context_length,
        # pad_to_multiple_of=64,
        padding=True,
    )
    return {"input_ids": outputs["input_ids"].view(-1, context_length)}

def train():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()


    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    config.rope_scaling = {
        "type": "yarn",
        "factor": training_args.scaling_factor,
        "original_max_position_embeddings": config.max_position_embeddings,
    }

    config.max_position_embeddings = int(training_args.scaling_factor * config.max_position_embeddings)
    config.nz_ratio = training_args.nz_ratio

    if training_args.train_dense_baseline:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.bfloat16,
            config=config,
            attn_implementation="flash_attention_2",
        )
    else:
        model = LlamaForCausalLMSeerAttnFT.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.bfloat16,
            config=config,
        )
        


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    for n, p in model.named_parameters():
        p.requires_grad = True

    rank = int(os.environ.get('RANK', -1))
    if rank > 0:
        barrier()

    dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", cache_dir=training_args.cache_dir, trust_remote_code=True)
    dataset = dataset.map(partial(tokenize_fn,tokenizer),batched=True, num_proc=128, remove_columns=["text", "meta"])

    if rank == 0:
        barrier()

    print(dataset)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model.config.use_cache = False        
    model.enable_input_require_grads()    
    model.gradient_checkpointing_enable()  
    
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    if training_args.train_dense_baseline:
        trainer = Trainer(
            model=model, 
            tokenizer=tokenizer, 
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=None,
            data_collator=data_collator
        )
    else:
        trainer = AttnGateTrainer(
            model=model, 
            orig_weight_training=True,
            tokenizer=tokenizer, 
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=None,
            gate_loss_scale=training_args.gate_loss_scale,
            use_mse_loss=training_args.use_mse_loss,
            data_collator=data_collator
        )

    if training_args.resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":

    train()


