import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers
import torch
from vllm import LLM, SamplingParams
import re
import importlib.util
import os
import argparse
import vllm.envs as envs
import random
import time
from datetime import datetime
from tqdm import tqdm
from Utils.parser import *
from Utils.data_loader import load_data
from Utils.math_normalization import *
from Utils.grader import *
from math import comb


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
    parser.add_argument('--start_idx', type=int, default=0, help="data[start:end]")
    parser.add_argument('--end_idx', type=int, default=-1, help="data[start:end], if -1, data[start:]")
    parser.add_argument("--max_tokens", default=32768, type=int)
    parser.add_argument("--prompt_type", default="qwen-instruct", type=str)
    parser.add_argument("--prompt_file_path", default="./prompts", type=str)
    parser.add_argument("--surround_with_messages", action="store_true")
    parser.add_argument("--use_few_shot", action="store_true")
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--run_id", default=0, type=int)
    args = parser.parse_args()
    
    return args

def set_random_seed():

    high_precision_time = time.time_ns()  
    pid = os.getpid()                     
    system_random = random.SystemRandom() 
    
    dynamic_seed = high_precision_time ^ (pid << 32) ^ system_random.randint(0, 2**64)
    transformers.set_seed(dynamic_seed % (2**32))  

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

    model_name_or_path = args.model_name_or_path
    print(f"current eval model: {model_name_or_path}")

    available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(available_gpus) == 1:
        envs.VLLM_HOST_IP="0.0.0.0" or "127.0.0.1"
    print(f"available_gpus: {available_gpus}")

    
    examples = load_data(args.data_name, args.split, args.data_dir)
    if args.end_idx == -1:
        args.end_idx = len(examples)
    examples = examples[args.start_idx:args.end_idx]

    limit = args.limit
    if limit > 0:
        examples = examples[:limit]

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
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
                    {"role": "user", "content": cur_prompt + "\nPlease reason step by step, and put your final answer within \\boxed{}."}
                ]
            else:
                # for gpqa
                messages = [
                    {"role": "user", "content": cur_prompt}
                ]
            cur_prompt = get_conversation_prompt_by_messages(tokenizer=tokenizer, messages=messages)
        prompt_batch.append(cur_prompt)


    sampling_params = SamplingParams(temperature=0.6, 
                                     max_tokens=args.max_tokens, 
                                     top_p=0.95,
                                     )
    llm = LLM(model=model_name_or_path, 
              tensor_parallel_size=len(available_gpus), 
              trust_remote_code=True, 
            #   swap_space=60,
            #   gpu_memory_utilization=0.96,
              )

    generate_lens = []
    
    output_runnum_subdir = os.path.join(args.output_dir, f"run_{args.run_id}")
    os.makedirs(output_runnum_subdir, exist_ok=True)

    completion_filepath = os.path.join(output_runnum_subdir, "completions.json")
    other_info_filepath = os.path.join(output_runnum_subdir, "other_info.json")

    completions = []

    set_random_seed()
    start = time.time()
 
    completion = llm.generate(prompt_batch, sampling_params)

    end = time.time()
    total_time = (end - start) / 60
    print("llm generate done")
    
    for i in range(len(examples)):
        for j in range(len(completion[i].outputs)):
            completions.append(completion[i].outputs[j].text)
            generate_lens.append(len(completion[i].outputs[j].token_ids))


    with open(completion_filepath, 'w') as f:
        json.dump(completions, f, indent=4)
        

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