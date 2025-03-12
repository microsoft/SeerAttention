# Modified from https://github.com/GAIR-NLP/LIMO/blob/main/eval/eval.py by Shuming Guo

import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import importlib.util
import os
import argparse
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
from seer_attn.qwen_decode_sparse.configuration_qwen2_seerattn import SeerAttnQwen2Config
from seer_attn.qwen_decode_sparse.modeling_qwen2_seerattn_inference import SeerAttnQwen2ForCausalLMInf

def calculate_average_percentage(file_path):
    weighted_sum = 0
    total_weight = 0
    total_percentage = 0
    count = 0

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(' ')
            percentage = float(parts[0].rstrip('%'))
            weight = int(parts[1])
            # percentage = float(line.strip().strip('%'))
            total_percentage += percentage
            count += 1
            weighted_sum += percentage * weight
            total_weight += weight
            

    if count == 0:
        return 0.0

    average_percentage = total_percentage / count
    average_percentage_weighted = weighted_sum / total_weight
    return average_percentage, average_percentage_weighted


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
    parser.add_argument("--max_tokens", default=2048, type=int)
    parser.add_argument("--prompt_type", default="qwen-instruct", type=str)
    parser.add_argument("--prompt_file_path", default="./prompts", type=str)
    parser.add_argument("--surround_with_messages", action="store_true")
    parser.add_argument("--use_few_shot", action="store_true")
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument('--stop', type=parse_list)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--dtype", default='auto', type=str)
    parser.add_argument("--completions_save_dir", default='./completions', type=str)
    parser.add_argument("--threshold", default=0, type=float)
    args = parser.parse_args()
    
    args.top_p = 1 if args.temperature == 0 else args.top_p 
    print(f"current stop list: {args.stop}")
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
    

    if args.threshold == 0:
        model = SeerAttnQwen2ForCausalLMInf.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            use_prefill_seerattn=False,
            use_decode_seerattn=False,
            device_map='auto',
        )
    else:
        model = SeerAttnQwen2ForCausalLMInf.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            use_prefill_seerattn=False,
            use_decode_seerattn=True,
            seerattn_sparsity_method = "threshold",
            seerattn_threshold = args.threshold,
            device_map='auto',
        )
    model.eval()


    config = SeerAttnQwen2Config.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
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
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": cur_prompt}
            ]
            cur_prompt = get_conversation_prompt_by_messages(tokenizer=tokenizer, messages=messages)
        prompt_batch.append(cur_prompt)
    print(prompt_batch[0])


    progress = tqdm(total=len(prompt_batch), desc="Generating Completions")
    times = []

    file_outputs = []
    correct_cnt = 0
    completions_save_file = f'{args.output_dir}/{args.data_name}_with_{args.threshold}_completion.txt'
    batch_size = args.batch_size
    for i in range(0, len(prompt_batch), batch_size):
        # Tokenize the prompt batch
        batch_prompts = prompt_batch[i:i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask
        batch_input_ids = batch_input_ids.cuda()
        attention_mask = attention_mask.cuda()
        
        
        torch.cuda.synchronize()
        time1 = time.time()
        outputs = model.generate(
            input_ids=batch_input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_tokens,
            num_return_sequences=1,
        )
        torch.cuda.synchronize()
        time2 = time.time()
        times.append(time2-time1)
        model.refresh_PoolingCache()

        for j in range(len(outputs)):
            generate_lens.append(len(outputs[j]) - len(batch_input_ids[j]))
            prompt_lens.append(len(batch_input_ids[j]))

        completions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            
        
        for j in range(i,min(i+batch_size,len(prompt_batch))):
            d = examples[j]
            gt_cot, gt_ans = parse_ground_truth(d, args.data_name)
            question = parse_question(d, args.data_name)
            generated_responses = [completions[j-i]]
            generated_answers = [extract_answer(generated_response, args.data_name) for generated_response in generated_responses]
            is_correct_list = [check_is_correct(generated_answer, gt_ans) for generated_answer in generated_answers]
            is_correct = any(is_correct_list)
            if is_correct:
                correct_cnt += 1
        progress.update(len(batch_prompts))
    
    print("llm generate done")

    print("generate_lens: ", generate_lens)
    
    print(f"correct cnt / total cnt: {correct_cnt}/{len(examples)}")
    print(f"Acc: {correct_cnt / len(examples):.4f}")

    # generate_len
    average_generate_len = sum(generate_lens) / len(generate_lens)
    average_prompt_len = sum(prompt_lens) / len(prompt_lens)
    max_generate_len = max(generate_lens)
    max_generate_len_index = generate_lens.index(max_generate_len)
    corresponding_prompt_len = prompt_lens[max_generate_len_index]
    print(f"Max generate length: {max_generate_len}")
    print(f"Corresponding prompt length: {corresponding_prompt_len}")
    print(f"Average generate length: {average_generate_len}")
    print(f"Average prompt length: {average_prompt_len}")

    # time
    average_time = sum(times) / len(times)
    average_time_per_token = sum(times) / sum(generate_lens)
    print(f"Average time: {average_time}")
    print(f"Average time per token: {average_time_per_token}")

    # sparsity
    if args.threshold != 0:
        average_percentage_replaced, average_percentage_replaced_weighted = calculate_average_percentage("./percentage_replaced_log.txt")
        print(f"Decode Sparsity weighted: {average_percentage_replaced_weighted}")
        with open("./percentage_replaced_log.txt", "w") as f:
            f.truncate(0)


    

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    infer(args)