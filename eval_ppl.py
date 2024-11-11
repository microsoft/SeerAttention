import argparse
import datasets
import gc
import sys
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
from seer_attn import LlamaForCausalLMSeerAttnInf
import random
import os
import importlib



def compute_perplexity(
    encodings, 
    model, 
    tokenizer, 
    add_start_token: bool = True, 
    device=None, 
    max_length=None, 
    sliding_window=256, 
    truncate=False, 
    hide_progress=False,
):
    if device is not None:
        assert device in ["gpu", "cpu",
                          "cuda"], "device should be either gpu or cpu."
        if device == "gpu":
            device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if add_start_token:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    if max_length and truncate:
        encoded_texts = [x[0:max_tokenized_len] for x in encoded_texts]
        attn_masks = [x[0:max_tokenized_len] for x in attn_masks]
        sliding_window = max_tokenized_len

    pbar = tqdm(total=len(encoded_texts), disable=hide_progress)
    nlls = []
    for encoding_index in range(0, len(encoded_texts)):

        labels = torch.tensor(encoded_texts[encoding_index:encoding_index+1])
        seq_len = labels.size(1)

        prev_end_loc = 0
        for begin_loc in range(0, seq_len, sliding_window):

            end_loc = min(begin_loc + max_tokenized_len, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = labels[:, begin_loc:end_loc].to(device)

            if add_start_token:
                bos_tokens_tensor = torch.tensor(
                    [[tokenizer.bos_token_id]] * input_ids.size(dim=0)).to(device)
                input_ids = torch.cat(
                    [bos_tokens_tensor, input_ids], dim=1)

            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            ppl = float(torch.exp(torch.stack(nlls).mean()).float().cpu())
            pbar.set_postfix(ppl=ppl)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        pbar.update(1)

    ppl = float(torch.exp(torch.stack(nlls).mean()).float().cpu())
    return {"mean_perplexity": ppl}


def main(args):
    models = [x[0] for x in args.model]
    tokenizer = AutoTokenizer.from_pretrained(
        models[0], 
        trust_remote_code=True,
        cache_dir = args.cache_dir,
    )
    tokenizer.pad_token = tokenizer.eos_token

    if args.tokenized:
        try:
            input_texts = datasets.load_from_disk(args.tokenized)
        except:
            input_texts = datasets.load_dataset(
                args.tokenized, name=args.subset, split=args.split, cache_dir=args.cache_dir, trust_remote_code=True)
    else:
        input_texts = datasets.load_dataset(
            args.dataset, name=args.subset, split=args.split, cache_dir=args.cache_dir, trust_remote_code=True)

        def tokenize(example):
            tokenized = tokenizer(
                example[args.feature],
                add_special_tokens=False,
                truncation=False,
                max_length=sys.maxsize,
                return_attention_mask=True,
            )
            example["input_ids"] = tokenized["input_ids"]
            example["attention_mask"] = tokenized["attention_mask"]
            example["tokenized_len"] = len(tokenized["input_ids"])
            return example

        input_texts = input_texts.map(tokenize)
        if args.save_tokenized:
            input_texts.save_to_disk(args.save_tokenized)
            print(f"Saved tokenized dataset to {args.save_tokenized}")
            return

    if args.dataset_min_tokens:
        input_texts = input_texts.filter(
            lambda x: x["tokenized_len"] >= args.dataset_min_tokens)
    if args.samples:
        input_texts = input_texts[:args.samples]

    tokens = [args.min_tokens]
    while args.min_tokens < args.max_tokens:
        point = tokens[-1] * 2
        if point <= args.max_tokens:
            tokens.append(point)
        else:
            break


    nz_ratios = args.nz_ratios.split(",")
    nz_ratios = [float(x) for x in nz_ratios]

    results = []
    for model in tqdm(models, desc="Model", leave=False, disable=False):
        torch.cuda.empty_cache()

        if args.use_seer_attn:
            loaded = LlamaForCausalLMSeerAttnInf.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                nz_ratio=1.0,
                device_map='auto',
            )

        else:
            loaded = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                cache_dir = args.cache_dir,
                device_map='auto',
            )
        

        result = []
        for max_length in tokens:
            if args.use_seer_attn:
                for nz_ratio in nz_ratios:
                    loaded.config.nz_ratio = nz_ratio
                    ppl = compute_perplexity(
                        model=loaded, 
                        tokenizer=tokenizer, 
                        encodings=input_texts,
                        add_start_token=tokenizer.bos_token is not None, 
                        max_length=max_length,
                        sliding_window=args.sliding_window, 
                        truncate=args.truncate,
                    )['mean_perplexity']
                    print(f"{model}: {max_length}_{nz_ratio}={ppl}")
                    result.append(f"{model}: {max_length}_{nz_ratio}={ppl}")
            else:
                ppl = compute_perplexity(
                    model=loaded, 
                    tokenizer=tokenizer, 
                    encodings=input_texts,
                    add_start_token=tokenizer.bos_token is not None, 
                    max_length=max_length,
                    sliding_window=args.sliding_window, 
                    truncate=args.truncate,
                )['mean_perplexity']
                print(f"{model}: {max_length}={ppl}")
                result.append(f"{model}: {max_length}={ppl}")      

        results.append(result)


    print(tokens)
    print(results)

    if args.output_file:
        if not os.path.exists(os.path.dirname(args.output_file)):
            os.makedirs(os.path.dirname(args.output_file))
    
        with open(args.output_file, "a", encoding="utf-8") as f:
            for i, result in enumerate(results):
                result_str = '\n'.join([str(x) for x in result])
                f.write(f"{result_str}\n")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", action="append", nargs="+")
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-s", "--subset", type=str)
    parser.add_argument("-f", "--feature", type=str)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--min-tokens", type=int, default=8192)
    parser.add_argument("--dataset-min-tokens", type=int)
    parser.add_argument("--sliding-window", type=int, default=256)
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--save-tokenized", type=str)
    parser.add_argument("--tokenized", type=str)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--cache_dir", default='/dev/shm/cache')
    parser.add_argument("--use_seer_attn", action="store_true")
    parser.add_argument("--nz_ratios", type=str, default="0.5,0.2,0.1")
    parser.add_argument("--random_seed", type=int, default=47)
    args = parser.parse_args()


    main(args)