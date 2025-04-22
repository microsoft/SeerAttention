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
from seer_attn import SeerDecodingQwen2ForCausalLM, SeerDecodingQwen2ForCausalLM_Dense
from generation_utils import batch_exist_generate

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
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--attention_implementation", default="ours", choices=["ours", "fa2"], type=str)
    parser.add_argument("--use_batch_exist", default=1, type=int)
    parser.add_argument("--use_sparse_kernel", default=0, type=int)
    parser.add_argument('--repeat', type=int, default=1, help="repeat")
    parser.add_argument("--gate_hidden_size", default=128, type=int)
    parser.add_argument("--q_head_pooling_type", default="Qproj", type=str)
    parser.add_argument("--block_size", default=64, type=int)
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


    if args.use_sparse_kernel == 1:
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


    if args.attention_implementation == "ours":
        if args.use_sparse_kernel == 1:
            model = SeerDecodingQwen2ForCausalLM.from_pretrained(model_name_or_path,
                                                    torch_dtype=torch.bfloat16,
                                                    device_map="auto",
                                                    use_cache=True,
                                                    fused_norm=True,
                                                    seerattn_threshold=0.0,
                                                    use_flash_rope=True,
            )
        else:
            model = SeerDecodingQwen2ForCausalLM_Dense.from_pretrained(model_name_or_path,
                                                torch_dtype=torch.bfloat16,
                                                device_map="auto",
                                                use_cache=True,
                                                fused_norm=True,
                                                use_flash_rope=True,
            )
    elif args.attention_implementation == "fa2":
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                    torch_dtype=torch.bfloat16,
                                                    device_map="auto",
                                                    use_cache=True,
                                                    attn_implementation="flash_attention_2",
                                                    trust_remote_code=True)
    else:
        raise ValueError(f"Unknown attention implementation: {args.attention_implementation}")
    
    model.eval()

    resume_i = 0
    times = []
    generate_lens = []
    sparsitys_all = []
    correct_cnt = 0
    if args.use_sparse_kernel == 1:
        output_filename = f"{args.data_name}_bs{args.batch_size}_batchexist{args.use_batch_exist}_{args.attention_implementation}_sparsekernel.txt"
    else:
        output_filename = f"{args.data_name}_bs{args.batch_size}_batchexist{args.use_batch_exist}_{args.attention_implementation}_densekernel.txt"
    os.makedirs(args.output_dir, exist_ok=True)
    output_path_txt = os.path.join(args.output_dir, output_filename)
    # output_completions_path = os.path.join(args.output_dir, "completions.json")
    completions = []
    batch_size = args.batch_size

    # fixed_generator = torch.Generator(device="cuda").manual_seed(42)
    begin = time.time()

    for i in range(0, len(prompt_batch), batch_size):
        # Tokenize the prompt batch
        print("start iteration: ", i, flush=True)
        batch_prompts = prompt_batch[i:i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
        print("tokenize done: ", i, flush=True)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask
        # batch_input_ids = batch_input_ids.cuda()
        # attention_mask = attention_mask.cuda()

        print("start iteration: ", i, flush=True)

        # transformers.set_seed(42)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        print("args.use_batch_exist:",args.use_batch_exist)
        if args.use_batch_exist == 1:
            if args.use_sparse_kernel == 1:
                assert args.attention_implementation == "ours"
                outputs = model.batch_exist_generate(
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

        
        print("get output in iteration: ", i, flush=True)
        
        for j in range(len(outputs)):
            output_seq = outputs[j]
            num_tokens = (output_seq != tokenizer.pad_token_id).sum().item()
            generate_lens.append(num_tokens - len(batch_input_ids[j]))

        batch_results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(batch_results)
        # completions.extend(batch_results)

        for j in range(i,min(i+batch_size,len(prompt_batch))):
            d = examples[j]
            gt_cot, gt_ans = parse_ground_truth(d, args.data_name)
            generated_responses = [batch_results[j-i]]
            generated_answers = [extract_answer(generated_response, args.data_name) for generated_response in generated_responses]
            is_correct_list = [check_is_correct(generated_answer, gt_ans) for generated_answer in generated_answers]
            is_correct = any(is_correct_list)
            if is_correct:
                correct_cnt += 1

                
        print("finish iteration: ", i, flush=True)



    # # check all the correct
    # for i in range(len(prompt_batch)):
    #     d = examples[i]
    #     gt_cot, gt_ans = parse_ground_truth(d, args.data_name)
    #     generated_responses = [completions[i]]
    #     generated_answers = [extract_answer(generated_response, args.data_name) for generated_response in generated_responses]
    #     is_correct_list = [check_is_correct(generated_answer, gt_ans) for generated_answer in generated_answers]
    #     is_correct = any(is_correct_list)
    #     if is_correct:
    #         correct_cnt += 1


    
    print("llm generate done")
    # if os.path.exists(checkpoint_filename_json):
    #     os.remove(checkpoint_filename_json)

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

    # # sparsity
    # activate_block, original_block = calculate_average_percentage(sparsitys_all)
    # # average_percentage_replaced_weighted = (weighted_sum + past_weighted_sum) / (total_weight + past_total_weight)
    # average_percentage_replaced_weighted = (1 - activate_block / original_block) * 100
    # print(f"Average percentage replaced weighted: {average_percentage_replaced_weighted}%")
    
    with open(output_path_txt, "a") as f:
        f.write(f"Acc: {correct_cnt / len(examples):.4f}\n")
        f.write(f"Average generate length: {average_generate_len}\n")
        f.write(f"Max generate length: {max_generate_len}\n")
        f.write(f"Total time: {total_time/60:.2f}\n")
        f.write(f"Average time per token: {average_time_per_token}\n")
        # f.write(f"Average percentage replaced weighted: {average_percentage_replaced_weighted}\n")
        f.write("\n")

    print("Results saved to ", output_path_txt)


    # # Save completions to json
    # with open(output_completions_path, "w") as f:
    #     json.dump(completions, f)

# def set_rng_seed(seed):
#     random.seed(seed) #为python设置随机种子
#     np.random.seed(seed)  #为numpy设置随机种子
#     torch.manual_seed(seed)   #为CPU设置随机种子
#     torch.cuda.manual_seed(seed)   #为当前GPU设置随机种子
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     torch.cuda.manual_seed_all(seed)   #为所有GPU设置随机种子
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.enabled = True
#     # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
#     # torch.use_deterministic_algorithms(True)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    # set_rng_seed(0)
    infer(args)