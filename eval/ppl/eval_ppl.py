import argparse
import datasets
import gc
import sys
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
from seer_attn import SeerAttnLlamaForCausalLM, SeerAttnQwen2ForCausalLM
import os


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
        assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
        if device == "gpu":
            device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if add_start_token:
        assert tokenizer.bos_token is not None, "Input model must have a BOS token"
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
    sparsity = []
    
    for encoding_index in range(len(encoded_texts)):
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
                input_ids = torch.cat([bos_tokens_tensor, input_ids], dim=1)

            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids, use_cache=False)
                neg_log_likelihood = outputs.loss

            # Sparsity calculation
            if model.config.seerattn_sparsity_method == 'threshold':
                sparsity.append(0.0) ## use PROFILE_FILE env variable to get the sparsity
            else:  # nz_ratio
                sparsity.append(model.config.seerattn_nz_ratio)

            outputs = None
            input_ids = None
            target_ids = None
            gc.collect()
            torch.cuda.empty_cache() 


            nlls.append(neg_log_likelihood)
            ppl = float(torch.exp(torch.stack(nlls).mean()).float().cpu())
            pbar.set_postfix(ppl=ppl)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        pbar.update(1)

    ppl = float(torch.exp(torch.stack(nlls).mean()).float().cpu())
    avg_sparsity = sum(sparsity)/len(sparsity) if sparsity else 0
    return {"mean_perplexity": ppl, "sparsity": avg_sparsity}


def main(args):
    config = AutoConfig.from_pretrained(args.model)
    base_dir = config.base_model
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_dir, 
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Dataset loading (same as original)
    if args.tokenized:
        try:
            input_texts = datasets.load_from_disk(args.tokenized)
        except:
            input_texts = datasets.load_dataset(
                args.tokenized, name=args.subset, split=args.split, 
                trust_remote_code=True)
    else:
        input_texts = datasets.load_dataset(
            args.dataset, name=args.subset, split=args.split, 
            trust_remote_code=True)

        def tokenize(example):
            tokenized = tokenizer(
                example[args.feature],
                add_special_tokens=False,
                truncation=False,
                max_length=sys.maxsize,
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

    results = []
    model_path = args.model
    torch.cuda.empty_cache()
    
    # Model loading with config parameters
    if args.use_seer_attn:
        if "llama" in base_dir.lower():
            model = SeerAttnLlamaForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map='auto',
                seerattn_sparsity_method=args.sparsity_method,
                seerattn_threshold=float(args.threshold.split(",")[0]),
                seerattn_nz_ratio=float(args.nz_ratios.split(",")[0]),
                seerattn_gate_type=args.gate_type,
                seerattn_last_block_dense=False,
            )
        elif "qwen" in base_dir.lower():
            model = SeerAttnQwen2ForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map='auto',
                seerattn_sparsity_method=args.sparsity_method,
                seerattn_threshold=float(args.threshold.split(",")[0]),
                seerattn_nz_ratio=float(args.nz_ratios.split(",")[0]),
                seerattn_gate_type=args.gate_type,
                seerattn_last_block_dense=False,
            )
        else:
            raise ValueError("Model unsupported for SeerAttn")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            attn_implementation="flash_attention_2"
        )

    result = []
    for max_length in tokens:
        if args.use_seer_attn:
            params = (args.threshold.split(",") if args.sparsity_method == 'threshold' 
                        else args.nz_ratios.split(","))
            for param_val in params:
                if args.sparsity_method == 'threshold':
                    model.config.seerattn_threshold = float(param_val)
                else:
                    model.config.seerattn_nz_ratio = float(param_val)
                
                output = compute_perplexity(
                    model=model, 
                    tokenizer=tokenizer, 
                    encodings=input_texts,
                    add_start_token=tokenizer.bos_token is not None, 
                    max_length=max_length,
                    sliding_window=args.sliding_window, 
                    truncate=args.truncate,
                )
                ppl = output['mean_perplexity']
                sparsity = output['sparsity']
                result_str = (f"{model_path}: {max_length}tokens | "
                                f"{args.sparsity_method}={param_val} | "
                                f"ppl={ppl:.2f} | density={sparsity:.2f}")
                print(result_str)
                result.append(result_str)
        else:
            output = compute_perplexity(
                model=model, 
                tokenizer=tokenizer, 
                encodings=input_texts,
                add_start_token=tokenizer.bos_token is not None, 
                max_length=max_length,
                sliding_window=args.sliding_window, 
                truncate=args.truncate,
            )
            ppl = output['mean_perplexity']
            result_str = f"{model_path}: {max_length}tokens | ppl={ppl:.2f}"
            print(result_str)
            result.append(result_str)

    results.append(result)

    # Save results (same as original)
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "a") as f:
            for result in results:
                f.write("\n".join(result) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Original arguments
    parser.add_argument("-m", "--model", required=True)
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
    parser.add_argument("--use_seer_attn", action="store_true")
    
    # New sparsity parameters
    parser.add_argument("--sparsity-method", choices=['threshold', 'nz_ratio'], default='threshold')
    parser.add_argument("--threshold", type=str, default="0.001")
    parser.add_argument("--nz-ratios", type=str, default="0.5")
    parser.add_argument("--gate-type", type=str, default="Qavg_Kmaxminavg")
    parser.add_argument("--random_seed", type=int, default=47)
    
    args = parser.parse_args()
    torch.manual_seed(args.random_seed)
    
    main(args)