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

def parse_list(arg):
    return arg.split(',')

def str_to_bool(s):
    s = s.lower() 
    if s in ['true', '1', 'yes']:
        return True
    elif s in ['false', '0', 'no']:
        return False
    else:
        raise ValueError(f"Invalid boolean value: {s}")

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
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument("--threshold", default=0, type=float)
    parser.add_argument("--block_size", default=64, type=int)
    parser.add_argument("--repeat", default=1, type=int)
    parser.add_argument("--attention_implementation", default="seer_sparse", choices=["seer_sparse", "seer_dense", "oracle_sparse", "fa2", "sdpa"], type=str)
    parser.add_argument("--use_batch_exist", default=True, type=str_to_bool)
    parser.add_argument("--use_fused_kernel", action="store_true")
    parser.add_argument("--profile_sparsity", action="store_true")
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--num_gpus", default=0, type=int)
    args = parser.parse_args()
    
    return args


def get_results(args):
    print(args)
    generate_lens = []
    examples = load_data(args.data_name, args.split, args.data_dir)
    if args.end_idx == -1:
        args.end_idx = len(examples)
    examples = examples[args.start_idx:args.end_idx]

    limit = args.limit
    if limit > 0:
        examples = examples[:limit]

    output_config_subdir = os.path.join(args.output_dir, f"{args.data_name}_bs{args.batch_size}_{args.attention_implementation}_T{args.threshold}_blocksize{args.block_size}_batchexist{args.use_batch_exist}")

    Acc_list = []
    generate_lens_list = []
    total_time_list = []
    overall_sparsity_list = []

    num_runs = args.repeat * args.num_gpus
    for i in range(num_runs):
        output_runnum_subdir = os.path.join(output_config_subdir, f"run_{i}")

        completion_filepath = os.path.join(output_runnum_subdir, "completions.json")
        
        with open(completion_filepath, 'r') as f:
            completions = json.load(f)
        
        other_info_filepath = os.path.join(output_runnum_subdir, "other_info.json")

        with open(other_info_filepath, 'r') as f:
            other_info = json.load(f)

        generate_lens = other_info['generate_lens']
        total_time = other_info['total_time']
        if args.profile_sparsity:
            overall_sparsity = other_info['overall_sparsity']
        
        print(f"Successfully loaded run{i}!")

        # check all the correct
        correct_cnt = 0
        print("len(completions)",len(completions))
        print("len(examples)",len(examples))
        for i in range(len(completions)):
            d = examples[i]
            gt_cot, gt_ans = parse_ground_truth(d, args.data_name)
            generated_responses = [completions[i]]
            generated_answers = [extract_answer(generated_response, args.data_name) for generated_response in generated_responses]
            is_correct_list = [check_is_correct(generated_answer, gt_ans) for generated_answer in generated_answers]
            is_correct = any(is_correct_list)
            if is_correct:
                correct_cnt += 1


        Acc = correct_cnt / len(examples)
        print("Acc:",Acc)

        average_generate_len = sum(generate_lens) / len(generate_lens)
        max_generate_len = max(generate_lens)

        average_time_per_token = total_time / sum(generate_lens)
        

        Acc_list.append(Acc)
        generate_lens_list += generate_lens
        total_time_list.append(total_time)
        if args.profile_sparsity:
            overall_sparsity_list.append(overall_sparsity)

        summary_filepath = os.path.join(output_runnum_subdir, "summary.txt")

        with open(summary_filepath, "a") as f:
            f.write(f"Acc: {Acc:.4f}\n")
            f.write(f"Average generate length: {average_generate_len}\n")
            f.write(f"Max generate length: {max_generate_len}\n")
            f.write(f"Total time: {total_time:.2f}\n")
            f.write(f"Average time per token: {average_time_per_token}\n")
            if args.profile_sparsity:
                f.write(f"Overall sparsity: {overall_sparsity}\n")
            f.write("\n")


    print("generate_lens: ", generate_lens_list)
    Acc = sum(Acc_list) / len(Acc_list)
    print(f"Acc: {Acc:.4f}")


    # generate_len
    average_generate_len = sum(generate_lens_list) / len(generate_lens_list)
    max_generate_len = max(generate_lens_list)
    print(f"Max generate length: {max_generate_len}")
    print(f"Average generate length: {average_generate_len}")

    total_time = sum(total_time_list) / len(total_time_list)
    average_time_per_token = sum(total_time_list) / sum(generate_lens_list)
    print(f"Total time: {total_time}min")
    print(f"Average time per token: {average_time_per_token}")

    # sparsity
    if args.profile_sparsity:
        overall_sparsity = sum(overall_sparsity_list) / len(overall_sparsity_list)
        if args.profile_sparsity:
            print("Overall_sparsity: ", overall_sparsity)

    overall_summary_filepath = os.path.join(output_config_subdir, "overall_summary.txt")
    with open(overall_summary_filepath, "a") as f:
        f.write(f"Acc: {Acc:.4f}\n")
        f.write(f"Average generate length: {average_generate_len}\n")
        f.write(f"Max generate length: {max_generate_len}\n")
        f.write(f"Total time: {total_time:.2f}min\n")
        f.write(f"Average time per token: {average_time_per_token}\n")
        if args.profile_sparsity:
            f.write(f"Overall sparsity: {overall_sparsity}\n")
        f.write("\n")


    print("Results saved to ", overall_summary_filepath)

    



    


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    get_results(args)