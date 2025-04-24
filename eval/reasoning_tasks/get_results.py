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

def calculate_average_percentage(sparsitys):
    activate_block = 0
    original_block = 0


    for activate_block_count, original_block_count in sparsitys:
        activate_block += activate_block_count
        original_block += original_block_count
            
    # average_percentage_weighted = weighted_sum / total_weight
    return activate_block, original_block


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="./", help="model dir")
    parser.add_argument('--n_sampling', type=int, default=1, help="n for sampling")
    parser.add_argument('--batch_size', type=int, default=16, help="batch_size")
    parser.add_argument('--limit', type=int, default=-1, help="limit")
    parser.add_argument("--k", type=int, default=1, help="Value of k for pass@k calculation")
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument('--data_name', type=str, default="math", help='identify how to extract answer')
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument('--start_idx', type=int, default=0, help="data[start:end]")
    parser.add_argument('--end_idx', type=int, default=-1, help="data[start:end], if -1, data[start:]")
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--max_tokens", default=32768, type=int)
    parser.add_argument("--prompt_type", default="qwen-instruct", type=str)
    parser.add_argument("--prompt_file_path", default="./prompts", type=str)
    parser.add_argument("--surround_with_messages", action="store_true")
    parser.add_argument("--use_few_shot", action="store_true")
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--dtype", default='auto', type=str)
    parser.add_argument("--threshold", default=0, type=float)
    parser.add_argument("--rank", default=0, type=int)
    # parser.add_argument("--attention_implementation", default="oracle_sparse", choices=["seer_sparse", "oracle_sparse", "fa2", "sdpa"], type=str)
    # parser.add_argument("--use_batch_exist", default=1, type=int)
    # parser.add_argument("--use_sparse_kernel", default=0, type=int)
    parser.add_argument('--repeat', type=int, default=1, help="repeat")
    # parser.add_argument("--gate_hidden_size", default=128, type=int)
    # parser.add_argument("--q_head_pooling_type", default="Qproj", type=str)
    # parser.add_argument("--block_size", default=64, type=int)
    parser.add_argument("--profile_sparsity", action="store_true")
    args = parser.parse_args()
    
    args.top_p = 1 if args.temperature == 0 else args.top_p # top_p must be 1 when using greedy 
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
    generate_lens = []
    prompt_lens = []

    n_sampling = args.n_sampling
    
    examples = load_data(args.data_name, args.split, args.data_dir)
    if args.end_idx == -1:
        args.end_idx = len(examples)
    examples = examples[args.start_idx:args.end_idx]

    limit = args.limit
    if limit > 0:
        examples = examples[:limit]
    examples = examples * args.repeat


    with open(f"./completions_{args.rank}.json", 'r') as f:
        completions = json.load(f)
    

    with open(f"./others_{args.rank}.json", 'r') as f:
        checkpoint_data = json.load(f)

    output_path_txt = checkpoint_data['output_path_txt']
    generate_lens = checkpoint_data['generate_lens']
    total_time = checkpoint_data['total_time']
    if args.profile_sparsity:
        overall_sparsity = checkpoint_data['overall_sparsity']

    if os.path.exists(f"./completions_{args.rank}.json"):
        os.remove(f"./completions_{args.rank}.json")

    if os.path.exists(f"./others_{args.rank}.json"):
        os.remove(f"./others_{args.rank}.json")
    
    print("Successfully loaded!")

    # check all the correct
    correct_cnt = 0
    for i in range(len(completions)):
        d = examples[i]
        gt_cot, gt_ans = parse_ground_truth(d, args.data_name)
        generated_responses = [completions[i]]
        generated_answers = [extract_answer(generated_response, args.data_name) for generated_response in generated_responses]
        is_correct_list = [check_is_correct(generated_answer, gt_ans) for generated_answer in generated_answers]
        is_correct = any(is_correct_list)
        if is_correct:
            correct_cnt += 1

    print("generate_lens: ", generate_lens)
    
    print(f"correct cnt / total cnt: {correct_cnt}/{len(examples)}")
    print(f"Acc: {correct_cnt / len(examples):.4f}")


    # generate_len
    average_generate_len = sum(generate_lens) / len(generate_lens)
    max_generate_len = max(generate_lens)
    print(f"Max generate length: {max_generate_len}")
    print(f"Average generate length: {average_generate_len}")

    
    average_time_per_token = total_time / sum(generate_lens)
    print(f"Total time: {total_time}s")
    print(f"Average time per token: {average_time_per_token}")

    # sparsity
    print("Overall_sparsity: ", overall_sparsity_ratio)
    
    with open(output_path_txt, "a") as f:
        f.write(f"Acc: {correct_cnt / len(examples):.4f}\n")
        f.write(f"Average generate length: {average_generate_len}\n")
        f.write(f"Max generate length: {max_generate_len}\n")
        f.write(f"Total time: {total_time/60:.2f}\n")
        f.write(f"Average time per token: {average_time_per_token}\n")
        if args.profile_sparsity:
            f.write(f"Overall sparsity: {overall_sparsity_ratio}\n")
        f.write("\n")

    print("Results saved to ", output_path_txt)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    infer(args)