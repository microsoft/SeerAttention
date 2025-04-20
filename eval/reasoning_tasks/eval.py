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
    for batch_sequence_info in all_batch_sparsitys_info:
        for seq_sparsitys_info in batch_sequence_info:
            for activate_count, original_count in seq_sparsitys_info:
                total_activate_count += activate_count
                total_original_count += original_count

    # Calculate the overall ratio, handling division by zero
    if total_original_count == 0:
        # If there were no original blocks, the ratio is undefined or could be considered 0.
        overall_sparsity_ratio = 0.0
    else:
        # Calculate the overall ratio
        overall_sparsity_ratio = total_activate_count / total_original_count

    # Return all three calculated values
    return total_activate_count, total_original_count, overall_sparsity_ratio


def parse_list(arg):
    return arg.split(',')

def save_completions(completions, filepath):
    with open(filepath, "a") as f:
        for completion in completions:
            f.write(f"completion: {completion}\n")

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
    parser.add_argument('--stop', type=parse_list)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--dtype", default='auto', type=str)
    parser.add_argument("--threshold", default=0, type=float)
    parser.add_argument("--block_size", default=64, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--attention_implementation", default="seer_sparse", choices=["seer_sparse", "oracle_sparse", "fa2", "sdpa"], type=str)
    parser.add_argument("--use_batch_exist", action="store_true")
    parser.add_argument("--use_fused_kernel", action="store_true")
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
            if args.data_name in ["aime", "math"]:
                # messages = [
                #     {"role": "user", "content": cur_prompt + "\nPlease reason step by step, and put your final answer within \\boxed{}."}
                # ]
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


    if args.attention_implementation == "seer_sparse" or args.attention_implementation == "oracle_sparse":
        if args.use_fused_kernel:
            model = SeerDecodingQwen2ForCausalLM.from_pretrained(model_name_or_path,
                                                    torch_dtype=torch.bfloat16,
                                                    device_map="auto",
                                                    load_gate = args.attention_implementation == "seer_sparse",
                                                    use_cache=True,
                                                    fused_norm=True,
                                                    seerattn_threshold=args.threshold,
                                                    seerattn_gate_block_size=args.block_size,
                                                    seerattn_use_oracle_sparse = args.attention_implementation == "oracle_sparse",
                                                    use_flash_rope=True,
            )
        else:
            model = SeerDecodingQwen2ForCausalLM.from_pretrained(model_name_or_path,
                                                    torch_dtype=torch.bfloat16,
                                                    device_map="auto",
                                                    load_gate = args.attention_implementation == "seer_sparse",
                                                    seerattn_threshold=args.threshold,
                                                    seerattn_gate_block_size=args.block_size,
                                                    seerattn_use_oracle_sparse = args.attention_implementation == "oracle_sparse",
                                                    use_cache=True,)
    elif args.attention_implementation == "fa2":
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                    torch_dtype=torch.bfloat16,
                                                    device_map="auto",
                                                    use_cache=True,
                                                    attn_implementation="flash_attention_2",
                                                    trust_remote_code=True)
    elif args.attention_implementation == "sdpa":
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                    torch_dtype=torch.bfloat16,
                                                    device_map="auto",
                                                    use_cache=True,
                                                    trust_remote_code=True)
    else:
        raise ValueError(f"Unknown attention implementation: {args.attention_implementation}")
    
    model.eval()


    generate_lens = []
    correct_cnt = 0
    output_subdir = f"{args.data_name}_bs_{args.batch_size}_attn_{args.attention_implementation}_T{args.threshold}_blocksize{args.block_size}_batch_exist_{args.use_batch_exist}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_filename = f"ckpt.jsonl"
    args.output_dir = os.path.join(args.output_dir, output_subdir) 
    os.makedirs(args.output_dir, exist_ok=True)
    output_path_txt = os.path.join(args.output_dir, "summary.txt")
    output_completions_path = os.path.join(args.output_dir, "completions.json")
    checkpoint_filename_json = os.path.join(args.output_dir, checkpoint_filename)
    completions = []
    batch_size = args.batch_size

    all_batch_sparsitys_info = []

    begin = time.time()

    for i in range(0, len(prompt_batch), batch_size):
        # Tokenize the prompt batch
        print("start batch: ", i, flush=True)
        batch_prompts = prompt_batch[i:i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        print("start batch: ", i, flush=True)

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

        
        print("get output in batch: ", i, flush=True)
        
        all_batch_sparsitys_info += batch_sparsitys_info

        for j in range(len(outputs)):
            output_seq = outputs[j]
            num_tokens = (output_seq != tokenizer.pad_token_id).sum().item()
            generate_lens.append(num_tokens - len(batch_input_ids[j]))

        batch_results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        completions.extend(batch_results)
        print("finish batch: ", i, flush=True)


    total_activate_count, total_original_count, overall_sparsity_ratio = calculate_overall_sparsity(all_batch_sparsitys_info)
    print("total_activate_count: ", total_activate_count)
    print("total_original_count: ", total_original_count)
    print("overall_sparsity: ", overall_sparsity_ratio)
    
    # check all the correct
    for i in range(len(prompt_batch)):
        d = examples[i]
        gt_cot, gt_ans = parse_ground_truth(d, args.data_name)
        generated_responses = [completions[i]]
        generated_answers = [extract_answer(generated_response, args.data_name) for generated_response in generated_responses]
        is_correct_list = [check_is_correct(generated_answer, gt_ans) for generated_answer in generated_answers]
        is_correct = any(is_correct_list)
        if is_correct:
            correct_cnt += 1


    
    print("llm generate done")
    if os.path.exists(checkpoint_filename_json):
        os.remove(checkpoint_filename_json)

    print("generate_lens: ", generate_lens)
    
    print(f"correct cnt / total cnt: {correct_cnt}/{len(examples)}")
    print(f"Acc: {correct_cnt / len(examples):.4f}")


    # generate_len
    average_generate_len = sum(generate_lens) / len(generate_lens)
    max_generate_len = max(generate_lens)
    print(f"Max generate length: {max_generate_len}")
    print(f"Average generate length: {average_generate_len}")

    end = time.time()
    total_time = end - begin
    average_time_per_token = total_time / sum(generate_lens)
    print(f"Total time: {total_time}s")
    print(f"Average time per token: {average_time_per_token}")
    
    with open(output_path_txt, "a") as f:
        f.write(f"Acc: {correct_cnt / len(examples):.4f}\n")
        f.write(f"Average generate length: {average_generate_len}\n")
        f.write(f"Max generate length: {max_generate_len}\n")
        f.write(f"Total time: {total_time/60:.2f}min\n")
        f.write(f"Average time per token: {average_time_per_token}\n")
        f.write(f"Total activate count: {total_activate_count}\n")
        f.write(f"Total original count: {total_original_count}\n")
        f.write(f"Overall sparsity: {overall_sparsity_ratio}\n")
        f.write("\n")

    print("Results saved to ", output_path_txt)


    # Save completions to json
    with open(output_completions_path, "w") as f:
        json.dump(completions, f)
    

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    set_seed(args.seed)
    infer(args)