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
from seer_attn import LlamaForCausalLMSeerAttnPT

from huggingface_hub import login

from mytrainer import AttnGateTrainer

from datasets import load_dataset, load_from_disk
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

import random

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3.1-8B")
    attn_gate_type: Optional[str] = field(
        default="Qavg_Kmaxmin",
        metadata={"help": "AttnGate pooling type. Currently support combination of max min avg pooling for both q and k."},
    )
    gate_block_size: Optional[int] = field(
        default=64,
        metadata={"help": "AttnGate block size. Currently only support 64."},
    ) 
    gate_hidden_size: Optional[int] = field(
        default=128,
        metadata={"help": "AttnGate hidden size."},
    )
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    training_max_length: int = field(
        default=65536,
        metadata={"help": "Maximum sequence length in training."},
    )
    trainable_params: str = field(
        default="mask_linear",
        metadata={"help": "AttnGate trainable parameters."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a checkpoint to resume training from."},
    )
    use_mse_loss: bool = field(
        default=True,
        metadata={"help": "Use MSE loss instead of SmoothL1 loss."},
    )
    gate_loss_scale: float = field(
        default=10.0,
        metadata={"help": "Gate loss scale."},
    )
    dataset_name: Optional[str] = field(
        default="togethercomputer/RedPajama-Data-1T-Sample",
        metadata={"help": "The name of the dataset to use."},
    )
    toknized_dataset: bool = field(
        default=False,
        metadata={"help": "If the dataset is already toknized."},
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

def tokenize_fn(tokenizer, tranining_max_length, example):
    outputs = tokenizer(
        tokenizer.eos_token.join(example["text"]),
        truncation=False,
        return_tensors="pt",
        pad_to_multiple_of=tranining_max_length,
        padding=True,
    )
    return {"input_ids": outputs["input_ids"].view(-1, tranining_max_length)}


def train():

    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()


    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )


    config.attn_gate_type = model_args.attn_gate_type
    config.gate_block_size = model_args.gate_block_size
    config.gate_hidden_size = model_args.gate_hidden_size

    model = LlamaForCausalLMSeerAttnPT.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
    )
    
    print("Using AttnGate type:", model_args.attn_gate_type)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
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
        if training_args.trainable_params in n:
            p.requires_grad = True
            torch.nn.init.xavier_uniform_(p)
        else:
            p.requires_grad = False

    rank = int(os.environ.get('RANK', -1))
    if rank > 0:
        barrier()

    if training_args.toknized_dataset:
        dataset = load_from_disk(training_args.dataset_name)
        dataset = dataset['input_ids']
    else:
        dataset = load_dataset(training_args.dataset_name, cache_dir=training_args.cache_dir, trust_remote_code=True)
        dataset = dataset.map(partial(tokenize_fn,tokenizer, training_args.training_max_length),batched=True, num_proc=128, remove_columns=["text", "meta", ])
        dataset = dataset['train']

    if rank == 0:
        barrier()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


    trainer = AttnGateTrainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        gate_loss_scale=training_args.gate_loss_scale,
        use_mse_loss=training_args.use_mse_loss,
        data_collator=data_collator)

    if training_args.resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    # login(token=os.getenv('HUGGING_FACE_HUB_TOKEN'))
    train()


