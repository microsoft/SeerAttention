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
from seer_attn import SeerAttnLlamaForCausalLM, SeerAttnQwen2ForCausalLM
from transformers.trainer_utils import get_last_checkpoint
from huggingface_hub import login

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
    base_model: str = field(default="meta-llama/Meta-Llama-3.1-8B")
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "The local model path if any."})
    seerattn_gate_type: Optional[str] = field(
        default="Qavg_Kmaxmin",
        metadata={"help": "AttnGate pooling type. Currently support combination of max min avg pooling for both q and k."},
    )
    seerattn_gate_block_size: Optional[int] = field(
        default=64,
        metadata={"help": "AttnGate block size. Currently only support 64."},
    ) 
    seerattn_gate_hidden_size: Optional[int] = field(
        default=128,
        metadata={"help": "AttnGate hidden size."},
    )
    seerattn_gate_force_double: Optional[bool] = field(
        default=False,
        metadata={"help": "Force using double linear for AttnGate."},
    )
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
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
    save_entire_model: bool = field(
        default=False,
        metadata={"help": "Save entire model."},
    )


class AttnGateTrainer(Trainer):
    def __init__(
            self, 
            orig_weight_training=False, 
            gate_loss_scale=1.0, 
            fix_mask_predictor=False, 
            *args, 
            **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.gate_loss_scale = gate_loss_scale
        self.orig_weight_training = orig_weight_training
        self.fix_mask_predictor = fix_mask_predictor

    def compute_loss(self, model, inputs, **kwargs):
        outputs = model(**inputs)

        original_loss = outputs.get("loss")
        mask_loss = outputs.get("mask_loss")
        
        del outputs
            
        if self.orig_weight_training:
            tok_loss = original_loss + self.gate_loss_scale * mask_loss
        else:
            tok_loss = self.gate_loss_scale * mask_loss
        
        return tok_loss


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

    if model_args.model_name_or_path is None:
        model_args.model_name_or_path = model_args.base_model

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
    )

    original_vocab_size = config.vocab_size

    if "llama" in model_args.model_name_or_path.lower():
        model = SeerAttnLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            load_gate=False,
            seerattn_gate_type = model_args.seerattn_gate_type,
            seerattn_gate_hidden_size = model_args.seerattn_gate_hidden_size,
            seerattn_gate_force_double = model_args.seerattn_gate_force_double,
            torch_dtype=torch.bfloat16,
        )
        model.config.base_model = model_args.base_model
    elif "qwen" in model_args.model_name_or_path.lower():
        model = SeerAttnQwen2ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            load_gate=False,
            seerattn_gate_type = model_args.seerattn_gate_type,
            seerattn_gate_hidden_size = model_args.seerattn_gate_hidden_size,
            seerattn_gate_force_double = model_args.seerattn_gate_force_double,
            torch_dtype=torch.bfloat16,
        )
        model.config.base_model = model_args.base_model
    
    print("Using AttnGate type:", model_args.seerattn_gate_type)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
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
            # torch.nn.init.xavier_uniform_(p)
        else:
            p.requires_grad = False

    rank = int(os.environ.get('RANK', -1))
    if rank > 0:
        barrier()

    if training_args.toknized_dataset:
        dataset = load_from_disk(training_args.dataset_name)
        dataset = dataset['input_ids']
    else:
        dataset = load_dataset(training_args.dataset_name, trust_remote_code=True)
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
        data_collator=data_collator
    )

    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if training_args.resume_from_checkpoint is None and last_checkpoint is not None:
        print(f"Found checkpoint {last_checkpoint}. Resuming training.")
        training_args.resume_from_checkpoint = last_checkpoint

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    if training_args.save_entire_model:
        trainer.save_model(output_dir=training_args.output_dir)
    elif rank == 0:
        if hasattr(trainer.model, 'module'):
            state_dict = trainer.model.module.state_dict()
        else:
            state_dict = trainer.model.state_dict()

        model.config.vocab_size = original_vocab_size
        model.config.save_pretrained(training_args.output_dir)
        attn_gate_state_dict = {k: v for k, v in state_dict.items() if "attn_gate" in k}
        torch.save(attn_gate_state_dict, os.path.join(training_args.output_dir, "attn_gate_weights.pth"))

if __name__ == "__main__":
    train()


