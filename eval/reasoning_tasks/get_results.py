import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
# from vllm import LLM, SamplingParams
import re
import importlib.util
import os
import argparse
# import vllm.envs as envs
from tqdm import tqdm
from Utils.parser import *
from Utils.data_loader import load_data
from Utils.math_normalization import *
from Utils.grader import *
import pickle
from math import comb
import glob

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
    parser.add_argument('--limit', type=int, default=-1, help="limit")
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument('--data_name', type=str, default="math", help='identify how to extract answer')
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument('--start_idx', type=int, default=0, help="data[start:end]")
    parser.add_argument('--end_idx', type=int, default=-1, help="data[start:end], if -1, data[start:]")
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument('--repeat', type=int, default=1, help="repeat")
    parser.add_argument("--profile_sparsity", action="store_true")
    args = parser.parse_args()
    
    return args


def infer(args):
    print(args)
    generate_lens = []
    examples = load_data(args.data_name, args.split, args.data_dir)
    if args.end_idx == -1:
        args.end_idx = len(examples)
    examples = examples[args.start_idx:args.end_idx]

    limit = args.limit
    if limit > 0:
        examples = examples[:limit]
    examples = examples * args.repeat

    completion_file = glob.glob(os.path.join(args.output_dir, f"*completions.json"))[0]
    print("completion_file: ", completion_file)

    with open(completion_file, 'r') as f:
        completions = json.load(f)
    
    ckpt_file = glob.glob(os.path.join(args.output_dir, f"*ckpt.json"))[0]
    print("ckpt_file: ", ckpt_file)

    with open(ckpt_file, 'r') as f:
        checkpoint_data = json.load(f)

    output_path_txt = checkpoint_data['output_path_txt']
    generate_lens = checkpoint_data['generate_lens']
    total_time = checkpoint_data['total_time']
    if args.profile_sparsity:
        overall_sparsity = checkpoint_data['overall_sparsity']
    
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
    if args.profile_sparsity:
        print("Overall_sparsity: ", overall_sparsity)
    
    with open(output_path_txt, "a") as f:
        f.write(f"Acc: {correct_cnt / len(examples):.4f}\n")
        f.write(f"Average generate length: {average_generate_len}\n")
        f.write(f"Max generate length: {max_generate_len}\n")
        f.write(f"Total time: {total_time/60:.2f}\n")
        f.write(f"Average time per token: {average_time_per_token}\n")
        if args.profile_sparsity:
            f.write(f"Overall sparsity: {overall_sparsity}\n")
        f.write("\n")

    print("Results saved to ", output_path_txt)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    infer(args)