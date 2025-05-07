import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
# from vllm import LLM, SamplingParams
import re
import importlib.util
import os
import argparse
# import vllm.envs as envs
import random
import time
from datetime import datetime
from tqdm import tqdm
from Utils.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from Utils.parser import *
from Utils.data_loader import load_data
from Utils.math_normalization import *
from Utils.grader import *
import pickle
from math import comb
from seer_attn import SeerDecodingQwen2ForCausalLM
from generation_utils import batch_exist_generate
from typing import Optional, Tuple

def calculate_overall_sparsity(
    all_batch_sparsitys_info: List[List[Optional[Tuple[Tuple[int, int], ...]]]]
) -> Tuple[int, int, float]:
    """
    Calculates the overall sparsity based on activation counts across batches and sequences.

    Sparsity here is defined as 1 - the ratio of total activated blocks to the total original blocks.

    Args:
        all_batch_sparsitys_info: A nested list structure.
            - Outer list represents the batch dimension.
            - Inner list represents the sequence dimension for that batch.
            - Each element in the inner list contains Optional sparsity info for a sequence step.
            - Sparsity info, if present, is a tuple of tuples: ((act1, org1), (act2, org2), ...),
              where 'act' is the activated block count and 'org' is the original block count
              (potentially summed across heads as described in the context code).

    Returns:
        The overall sparsity ratio 1-(total activated blocks / total original blocks) as a float.
        Returns 0.0 if the total original block count is zero.
    """
    total_activate_count = 0
    total_original_count = 0
    # Iterate through each batch in the input list
    # print(all_batch_sparsitys_info)
    for each_batch_sequence_info in all_batch_sparsitys_info:
        for each_step_sparsitys_info in each_batch_sequence_info:
            #print(each_step_sparsitys_info)
            for each_layer_sparsitys_info in each_step_sparsitys_info:
                total_activate_count += each_layer_sparsitys_info[0]
                total_original_count += each_layer_sparsitys_info[1]

    # Calculate the overall ratio, handling division by zero
    if total_original_count == 0:
        # If there were no original blocks, the ratio is undefined or could be considered 0.
        overall_sparsity_ratio = 0.0
    else:
        # Calculate the overall ratio
        overall_sparsity_ratio = 1 - total_activate_count / total_original_count

    # Return all three calculated values
    return total_activate_count, total_original_count, overall_sparsity_ratio


def parse_list(arg):
    return arg.split(',')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="./", help="model dir")
    parser.add_argument('--batch_size', type=int, default=16, help="batch_size")
    parser.add_argument('--limit', type=int, default=-1, help="limit")
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument('--data_name', type=str, default="math", help='identify how to extract answer')
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--max_tokens", default=32768, type=int)
    parser.add_argument("--prompt_type", default="qwen-instruct", type=str)
    parser.add_argument("--prompt_file_path", default="./prompts", type=str)
    parser.add_argument("--surround_with_messages", action="store_true")
    parser.add_argument("--use_few_shot", action="store_true")
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument("--sparsity_method", default='threshold', choices=["token_budget", "threshold"], type=str)
    parser.add_argument("--sliding_window_size", default=0, type=int)
    parser.add_argument("--threshold", default=0, type=float)
    parser.add_argument("--token_budget", default=2048, type=int)
    parser.add_argument("--block_size", default=64, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--attention_implementation", default="seer_sparse", choices=["seer_sparse", "seer_dense", "oracle_sparse", "fa2", "sdpa"], type=str)
    parser.add_argument("--use_batch_exist", action="store_true")
    parser.add_argument("--use_fused_kernel", action="store_true")
    parser.add_argument("--profile_sparsity", action="store_true")
    parser.add_argument("--run_id", default=0, type=int)
    args = parser.parse_args()
    
    return args

def get_conversation_prompt_by_messages(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def get_three_prompt(prompt_type, data_name):
    file_path = os.path.join(".", "prompts", prompt_type, f"{data_name}.py")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if hasattr(module, 'system_prompt'):
        system_prompt = module.system_prompt
    else:
        raise AttributeError(f"'system_prompt' not found in {file_path}")
    
    if hasattr(module, 'few_shot_prompt'):
        few_shot_prompt = module.few_shot_prompt
    else:
        raise AttributeError(f"'few_shot_prompt' not found in {file_path}")
    
    if hasattr(module, 'question_format'):
        question_format = module.question_format
    else:
        raise AttributeError(f"'question_format' not found in {file_path}")

    return system_prompt, few_shot_prompt, question_format


def infer(args):
    print(args)
    model_name_or_path = args.model_name_or_path
    print(f"current eval model: {model_name_or_path}")
    device = f"cuda:{args.rank}"

    generate_lens = []
    
    examples = load_data(args.data_name, args.split, args.data_dir)

    limit = args.limit
    if limit > 0:
        examples = examples[:limit]
    
    if args.profile_sparsity:
        assert args.attention_implementation in ["seer_sparse", "oracle_sparse"], "profile_sparsity only support seer_sparse and oracle_sparse"

    if args.attention_implementation == "seer_sparse":
        config = AutoConfig.from_pretrained(model_name_or_path)
        base_model = config.base_model
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
            padding_side="left",
            use_fast=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            trust_remote_code=True, 
            padding_side="left",
            use_fast=True,
        )
    prompt_batch = []
    for example in tqdm(examples, total=len(examples)):
        # parse question and answer
        question = parse_question(example, args.data_name)
        system_prompt, few_shot_prompt, question_format = get_three_prompt(args.prompt_type, args.data_name)
        
        if args.use_few_shot:
            cur_prompt = few_shot_prompt + question_format.format(question=question)
        else:
            cur_prompt = question_format.format(question=question)
        if args.surround_with_messages:
            if args.data_name in ["aime", "math", "olympiadbench"]:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": cur_prompt}
                ]
            else:
                # for gpqa
                messages = [
                    {"role": "user", "content": cur_prompt}
                ]
            cur_prompt = get_conversation_prompt_by_messages(tokenizer=tokenizer, messages=messages)
        prompt_batch.append(cur_prompt)

    
        
    output_runnum_subdir = os.path.join(args.output_dir, f"run_{args.run_id}")
    os.makedirs(output_runnum_subdir, exist_ok=True)

    completion_filepath = os.path.join(output_runnum_subdir, "completions.json")
    sparsity_info_filepath = os.path.join(output_runnum_subdir, "sparsity_info.json")
    other_info_filepath = os.path.join(output_runnum_subdir, "other_info.json")


    start_i = 0
    completions = []
    generate_lens = []
    all_batch_sparsitys_info = []
    total_time = 0

    if os.path.exists(completion_filepath):
        print(f"Loading checkpoint from {completion_filepath}")
        with open(completion_filepath, 'r') as f:
            completions = json.load(f)
            start_i = len(completions)
            print(f"Resuming from {start_i}...")
        if start_i == len(examples):
            print(f"Run {args.run_id} already completed. Skipping...")
            return
    
    if os.path.exists(sparsity_info_filepath):
        with open(sparsity_info_filepath, 'r') as f:
            all_batch_sparsitys_info = json.load(f)

    if os.path.exists(other_info_filepath):
        with open(other_info_filepath, 'r') as f:
            other_info = json.load(f)

        generate_lens = other_info['generate_lens']
        total_time = other_info['total_time']



    if args.attention_implementation == "seer_sparse" or args.attention_implementation == "oracle_sparse" or args.attention_implementation == "seer_dense":
        model = SeerDecodingQwen2ForCausalLM.from_pretrained(model_name_or_path,
                                                torch_dtype=torch.bfloat16,
                                                device_map=device,
                                                load_gate = args.attention_implementation == "seer_sparse",
                                                use_cache=True,
                                                seerattn_sparsity_method=args.sparsity_method,
                                                seerattn_threshold=args.threshold,
                                                seerattn_sliding_window_size=args.sliding_window_size,
                                                seerattn_token_budget=args.token_budget,
                                                seerattn_gate_block_size=args.block_size,
                                                seerattn_implementation = args.attention_implementation,
                                                use_flash_rope=args.use_fused_kernel,
                                                fused_norm=args.use_fused_kernel,
                                                seerattn_output_sparsity=args.profile_sparsity,
        )

    elif args.attention_implementation == "fa2":
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                    torch_dtype=torch.bfloat16,
                                                    device_map=device,
                                                    use_cache=True,
                                                    attn_implementation="flash_attention_2",
                                                    trust_remote_code=True)
    elif args.attention_implementation == "sdpa":
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                    torch_dtype=torch.bfloat16,
                                                    device_map=f"cuda:{args.rank}",
                                                    use_cache=True,
                                                    trust_remote_code=True)
    else:
        raise ValueError(f"Unknown attention implementation: {args.attention_implementation}")
    
    model.eval()

    batch_size = args.batch_size

    

    for i in range(start_i, len(prompt_batch), batch_size):
        # Tokenize the prompt batch
        begin = time.time()
        print("start batch: ", i, flush=True)
        batch_prompts = prompt_batch[i:i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True).to(device)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask


        if args.use_batch_exist:
            if args.attention_implementation == "seer_sparse" or args.attention_implementation == "oracle_sparse":
                outputs, batch_sparsitys_info = model.batch_exist_generate(
                    input_ids=batch_input_ids,
                    attention_mask=attention_mask,
                    max_length = args.max_tokens,
                    do_sample=True,
                )
            else:
                outputs = batch_exist_generate(
                    model,
                    input_ids=batch_input_ids,
                    attention_mask=attention_mask,
                    max_length = args.max_tokens,
                    do_sample=True,
                )

        else:
            outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                max_length = args.max_tokens,
                do_sample=True,
                num_return_sequences=1
            )

        end = time.time()
        batch_time = (end - begin) / 60
        total_time = total_time + batch_time
        print("get output in batch: ", i, flush=True)
        
        
        if args.profile_sparsity:
            all_batch_sparsitys_info.append(batch_sparsitys_info)

        for j in range(len(outputs)):
            output_seq = outputs[j]
            output_tokens = (output_seq != tokenizer.pad_token_id).sum().item()
            prompt_tokens = (batch_input_ids[j] != tokenizer.pad_token_id).sum().item()
            generate_lens.append(output_tokens - prompt_tokens)

        batch_results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        completions.extend(batch_results)
        print("finish batch: ", i, flush=True)
        
    
        with open(completion_filepath, 'w') as f:
            json.dump(completions, f, indent=4)
            
        if args.profile_sparsity:
            with open(sparsity_info_filepath, 'w') as f:
                json.dump(all_batch_sparsitys_info, f, indent=4)

        other_info = {
            "generate_lens": generate_lens,
            "total_time": total_time,
        }
            
        with open(other_info_filepath, 'w') as f:
            json.dump(other_info, f)
        
        

    print("llm generate done")


    if args.profile_sparsity:
        total_activate_count, total_original_count, overall_sparsity_ratio = calculate_overall_sparsity(all_batch_sparsitys_info)
        print("Overall_sparsity: ", overall_sparsity_ratio)
        
    
    with open(completion_filepath, 'w') as f:
        json.dump(completions, f, indent=4)
        

    if args.profile_sparsity:
        sparsity_info = {"sparsity_info": all_batch_sparsitys_info}
        with open(sparsity_info_filepath, 'w') as f:
            json.dump(sparsity_info, f, indent=4)

    if args.profile_sparsity:
        other_info = {
            "generate_lens": generate_lens,
            "total_time": total_time,
            "overall_sparsity": overall_sparsity_ratio,
        }
    else:
        other_info = {
            "generate_lens": generate_lens,
            "total_time": total_time,
        }
        
    with open(other_info_filepath, 'w') as f:
        json.dump(other_info, f)
    
    print(f"Successfully saved run{args.run_id}!")

    

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    infer(args)