import os
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import deepspeed # without this, there will be a deepspeed.runtime error when transformers >=4.50
import torch
import transformers
# from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling
from transformers.trainer_utils import get_last_checkpoint

from torch.distributed import barrier
from seer_attn.decode_sparse.qwen2.modeling_qwen2_seerattn_training import SeerAttnQwen2ForCausalLM
from seer_attn.decode_sparse.qwen3.modeling_qwen3_seerattn_training import SeerAttnQwen3ForCausalLM

from datasets import load_dataset, load_from_disk, Dataset
import warnings
import json

warnings.simplefilter(action='ignore', category=FutureWarning)


import random


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

    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        torch.cuda.empty_cache()
        step = self.state.global_step
        outputs = model(**inputs)

        original_loss = outputs.get("loss")

        mask_loss = outputs.get("gate_loss")
        if not return_outputs:
            del outputs
            
        if self.orig_weight_training:
            tok_loss = original_loss + self.gate_loss_scale * mask_loss
        else:
            tok_loss = self.gate_loss_scale * mask_loss
        
        return (tok_loss, outputs) if return_outputs else tok_loss


@dataclass
class ModelArguments:
    base_model: str = field(default="meta-llama/Meta-Llama-3.1-8B")
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "The local model path if any."})
    seerattn_k_seq_pooling_type: Optional[str] = field(
        default="Kmaxminavg",
        metadata={"help": "AttnGate pooling type. Currently support combination of max min avg pooling for both q and k."},
    )
    seerattn_gate_block_size: Optional[int] = field(
        default=64,
        metadata={"help": "AttnGate block size. Currently only support 64."},
    ) 
    seerattn_q_head_pooling_type: Optional[str] = field(
        default="Qproj",
        metadata={"help": "AttnGate head pooling type for share head."},
    )
    seerattn_gate_hidden_size: Optional[int] = field(
        default=128,
        metadata={"help": "AttnGate hidden size."},
    )
    seerattn_loss_slice_ratio: Optional[float] = field(
        default=0.0,
        metadata={"help": "loss_slice_ratio for kl loss."},
    )
    seerattn_use_qk_norm: Optional[bool] = field(
        default=False,
        metadata={"help": "use_qk_norm in attn gate."},
    )
    seerattn_use_rope: Optional[bool] = field(
        default=True,
        metadata={"help": "use rope in attn gate."},
    )



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    training_max_length: int = field(
        default=65536,
        metadata={"help": "Maximum sequence length in training."},
    )
    trainable_params: str = field(
        default="attngate",
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
        default="./cache_dir/qwq-sftopenr1-newtemplate",
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

def apply_chat_template(
    example,
    tokenizer,
    add_prefix: bool = False,
    prefix: str = "<think>\n",
) -> dict[str, str]:

    ## when using vllm to generate data, the prompt uses 'add_generation_prompt'. So the \<think> token does not appear in the assistant.

    messages = example["messages"]
    if add_prefix: 
        assert messages[1]["role"] == "assistant"
        assistant_message = messages[1]["content"]
        assistant_message = prefix + assistant_message
        messages[1]["content"] = assistant_message

    messages = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
    )

    return {"text": messages}

def tokenize_fn(tokenizer, examples):
    outputs = tokenizer(
        examples["text"],
        add_special_tokens=False,
        truncation=False,
        return_tensors="pt",
        padding=False,
    )

    return {"input_ids": outputs["input_ids"][0]}

def build_optimized_chunks(dataset, chunk_size):
    samples = []
    for example in dataset:
        input_ids = torch.as_tensor(example["input_ids"])
        seq_len = input_ids.size(0)
        samples.append((input_ids, seq_len))
    
    samples.sort(key=lambda x: -x[1])  
    
    chunks = [] 
    
    for input_ids, seq_len in samples:
        best_idx = -1
        min_remainder = chunk_size + 1  
        

        for i, chunk in enumerate(chunks):
            total = chunk['current_length'] + seq_len
            if total <= chunk_size:
                remainder = chunk_size - total
                if remainder < min_remainder:
                    min_remainder = remainder
                    best_idx = i
        
        if best_idx != -1:
            chunks[best_idx]['input_ids'].append(input_ids)
            chunks[best_idx]['seqlens'].append(seq_len)
            chunks[best_idx]['current_length'] += seq_len
        else:
            chunks.append({
                'input_ids': [input_ids],
                'seqlens': [seq_len],
                'current_length': seq_len
            })
    
    processed_chunks = []
    random.shuffle(chunks)
    for chunk in chunks:
        full_chunk = torch.cat(chunk['input_ids'])
        processed_chunks.append({
            "input_ids": full_chunk.numpy(),
            "seqlens": chunk['seqlens']
        })
    
    return Dataset.from_dict({
        "input_ids": [chunk["input_ids"] for chunk in processed_chunks],
        "seq_lengths": [chunk["seqlens"] for chunk in processed_chunks]
    })


def train():

    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()


    if os.path.exists(os.path.join(training_args.output_dir, "attn_gate_weights.pth")):
        print("Attn gate weights already exist, skip training.")
        return


    if model_args.model_name_or_path is None:
        model_args.model_name_or_path = model_args.base_model

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
    )

    original_vocab_size = config.vocab_size
    config.seerattn_k_seq_pooling_type = model_args.seerattn_k_seq_pooling_type
    config.seerattn_gate_block_size = model_args.seerattn_gate_block_size

    config.seerattn_q_head_pooling_type = model_args.seerattn_q_head_pooling_type
    config.seerattn_training_threshold = model_args.seerattn_training_threshold
    config.seerattn_gate_hidden_size = model_args.seerattn_gate_hidden_size
    config.seerattn_ignore_last_block = model_args.seerattn_ignore_last_block
    config.seerattn_loss_slice_ratio = model_args.seerattn_loss_slice_ratio
    config.seerattn_use_qk_norm = model_args.seerattn_use_qk_norm
    config.seerattn_use_rope = model_args.seerattn_use_rope
    config.seerattn_block_slice_mode = model_args.seerattn_block_slice_mode
    config.base_model = model_args.base_model


    if "qwen2" in model_args.model_name_or_path.lower() or "r1-distill-qwen" in model_args.model_name_or_path.lower():
        print("Using Qwen2 model")
        model_cls = SeerAttnQwen2ForCausalLM
    elif "qwen3" in model_args.model_name_or_path.lower():
        print("Using Qwen3 model")
        model_cls = SeerAttnQwen3ForCausalLM
    else:  
        raise ValueError("Model not supported. Current only support qwen2 or qwen3 model.")
    

    model = model_cls.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
    )

    print("tokenier name:", model_args.model_name_or_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        trust_remote_code=True,
        use_fast=True,
    )
    
    # Some chat template will remove the tokens between <think> and </think>, 
    # so we need modify the chat template to keep the tokens for training.
    with open("chat_template/templates.json", "r") as f:
        chat_template = json.load(f)

    if model_args.base_model in chat_template.keys():
        print("Using modified chat template:", model_args.base_model)
        tokenizer.chat_template = chat_template[model_args.base_model]
    print("Use defult chat_template:", tokenizer.chat_template)


    for n, p in model.named_parameters():
        if training_args.trainable_params in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

    rank = int(os.environ.get('RANK', -1))
    if rank > 0:
        barrier()


    if not os.path.isdir(training_args.dataset_name): 
        add_prefix = False
        dataset = load_dataset(training_args.dataset_name, "default", split="train")
    else:
        add_prefix = True
        dataset = load_from_disk(training_args.dataset_name)
        

    if "phi" in model_args.model_name_or_path.lower():
        add_prefix = False


    dataset = dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, 
                   "add_prefix": add_prefix},
        remove_columns="messages",  # renamed to "text"
    )
    example = next(iter(dataset))
    print("example:", example)

    dataset = dataset.map(
        partial(tokenize_fn, tokenizer),
        batched=False,
        num_proc=128, 
    )
    dataset = build_optimized_chunks(
        dataset=dataset, 
        chunk_size=training_args.training_max_length,
    )

    print(dataset)

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

