import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
import importlib
from seer_attn import SeerAttnLlamaForCausalLM, SeerAttnQwen2ForCausalLM

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, required=True)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--output_path', type=str, default="./")
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    messages = [{"role": "user", "content": prompt}]
    prompt =  tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(rank, world_size, data, max_gen, prompt_format, dataset, device, model_name, out_path, threshold, lock):
    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(model_name, device, threshold)
    max_length = model.config.max_position_embeddings - 500

    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
                f.write('\n')
    dist.destroy_process_group()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, device, threshold):
    config = AutoConfig.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, padding_side="left")
    if "llama" in path.lower():
        model = SeerAttnLlamaForCausalLM.from_pretrained(
            path, 
            torch_dtype=torch.bfloat16,
            seerattn_sparsity_method='threshold',
            seerattn_last_block_dense=True,
            seerattn_threshold=threshold
        ).to(device)
    elif "qwen" in path.lower():
        model = SeerAttnQwen2ForCausalLM.from_pretrained(
            path, 
            torch_dtype=torch.bfloat16,
            seerattn_sparsity_method='threshold',
            seerattn_last_block_dense=True,
            seerattn_threshold=threshold
        ).to(device)
    else:
        raise ValueError("Model not supported")

    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    threshold = args.threshold

    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    

    lock = mp.Lock()

    output_dir = os.path.join(args.output_path, "pred") if not args.e else os.path.join(args.output_path, "pred_e")

    for dataset in datasets:
        print(f"Predicting on {dataset}...")
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            # model_name_for_path = model_name.replace("/", "_") + "_" + str(threshold)
            model_name_for_path = model_name.split("/")[-1] + "_" + str(threshold)
            if not os.path.exists(f"{output_dir}/{model_name_for_path}"):
                os.makedirs(f"{output_dir}/{model_name_for_path}")
            out_path = f"{output_dir}/{model_name_for_path}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            if not os.path.exists(f"{output_dir}/{model_name_for_path}"):
                os.makedirs(f"{output_dir}/{model_name_for_path}")
            out_path = f"{output_dir}/{model_name_for_path}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], \
                        max_gen, prompt_format, dataset, device, model_name, out_path, threshold, lock))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
