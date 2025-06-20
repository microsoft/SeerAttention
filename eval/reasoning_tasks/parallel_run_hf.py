#!/usr/bin/env python3
import subprocess
import os
import sys
import argparse
import time
from collections import deque # Use deque for efficient pop/append


def choose_task_config(model_size, output_dir):
    output_dir = output_dir.lower()
    if model_size != "32B":
        task_config = {
            "aime24": {"bs": 15, "total_run": 64},
            "aime25": {"bs": 15, "total_run": 64},
            "math": {"bs": 75, "total_run": 8},
            "gpqa": {"bs": 30, "total_run": 16},
            "olympiadbench": {"bs": 15, "total_run": 8},
            "livecodebench": {"bs": 15, "total_run": 8},
        }
    else:
        raise ValueError(f"Not support model_size: {model_size}")
    
    return task_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tasks using subprocess.")
    parser.add_argument("--model_dir", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                        help="Model directory path")
    parser.add_argument("--model_size", type=str, default="14B", help="model_size")
    parser.add_argument("--tasks", type=str, default="aime",
                        help="Comma-separated list of tasks (e.g., aime,math,gpqa)")
    parser.add_argument("--output_dir", type=str, default="./results/aime",
                        help="Directory to store output results")
    parser.add_argument("--attention_implementation", type=str, default="seer_sparse",
                        help="attention implementations")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Limit for the number of samples to process")
    parser.add_argument("--num_gpus", default="8", type=int)
    parser.add_argument("--block_size", default="64", type=str)
    parser.add_argument("--sparsity_method", default='threshold', choices=["token_budget", "threshold"], type=str)
    parser.add_argument("--sliding_window_size", default="0", type=str)
    parser.add_argument("--threshold", default="0", type=str)
    parser.add_argument("--token_budget", default="2048", type=str)
    parser.add_argument("--max_tokens", default="32768", type=str)
    parser.add_argument("--start_layer", type=int, default=0, help="Start sparse layer, '0' means all layers")
    parser.add_argument("--profile_sparsity", action="store_true",
                        help="Flag to profile sparsity in eval.py")
    args = parser.parse_args()

    limit = args.limit
    num_gpus = args.num_gpus
    max_tokens = args.max_tokens

    model_dir = args.model_dir
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    sparsity_method = args.sparsity_method
    token_budgets = [t.strip() for t in args.token_budget.split(",") if t.strip()]
    sliding_window_sizes = [s.strip() for s in args.sliding_window_size.split(",") if s.strip()]
    thresholds = [t.strip() for t in args.threshold.split(",") if t.strip()]
    block_sizes = [b.strip() for b in args.block_size.split(",") if b.strip()]

    model_subfolder = os.path.basename(model_dir.rstrip('/'))
    output_dir = os.path.join(args.output_dir, model_subfolder)
    attention_implementation = args.attention_implementation
    if attention_implementation == "seer_sparse":
        assert len(block_sizes) == 1 and block_sizes[0] == "64", "seer_sparse only supports block_size=64"

    task_config = choose_task_config(args.model_size, output_dir)

    for task in tasks:
        if task not in task_config:
            print(f"Error: Unknown task '{task}'")
            sys.exit(1)

        bs = task_config[task]["bs"]
        total_run = task_config[task]["total_run"]

        print(f"\n{'='*40}")
        print(f"Starting task: {task}")
        print(f"Batch size: {bs} | total_run: {total_run}")

        for block_size in block_sizes:
            print(f"Block size: {block_size}")
            if sparsity_method == "token_budget":
                param_combinations = [
                    (sw, tb) 
                    for sw in sliding_window_sizes 
                    for tb in token_budgets
                ]
            elif sparsity_method == "threshold":
                param_combinations = [
                    (sw, th) 
                    for sw in sliding_window_sizes 
                    for th in thresholds
                ]

            for params in param_combinations:
                if sparsity_method == "token_budget":
                    sliding_window_size, token_budget = params
                    param_desc = f"window={sliding_window_size}, budget={token_budget}"
                    cli_params = [
                        "--token_budget", str(token_budget),
                        "--sliding_window_size", str(sliding_window_size),
                    ]
                elif sparsity_method == "threshold":
                    sliding_window_size, threshold = params
                    param_desc = f"window={sliding_window_size}, threshold={threshold}"
                    cli_params = [
                        "--threshold", str(threshold),
                        "--sliding_window_size", str(sliding_window_size),
                    ]

                print(f"\n{'─'*30}")
                print(f"Processing Task:{task} | Block_size:{block_size} | {sparsity_method}: {param_desc}")

                active_procs = {}
                available_gpus = deque(range(num_gpus))
                completed_runs = 0
                run_counter = 0

                if sparsity_method == "token_budget":
                    output_config_subdir = os.path.join(output_dir, f"{task}_bs{bs}_{sparsity_method}_B{token_budget}_start{args.start_layer}_blocksize{block_size}_{attention_implementation}")
                elif sparsity_method == "threshold":
                    output_config_subdir = os.path.join(output_dir, f"{task}_bs{bs}_{sparsity_method}_T{threshold}_start{args.start_layer}_blocksize{block_size}_{attention_implementation}")

                os.makedirs(output_config_subdir, exist_ok=True)

                # if overall_summary.txt exist in args.output_dir, continue
                overall_summary_filepath = os.path.join(output_config_subdir, "overall_summary.txt")
                if os.path.exists(overall_summary_filepath):
                    print(f"Skip {param_desc} because overall_summary.txt already exists.")
                    continue

                while run_counter < total_run or active_procs:
                    for proc, info in list(active_procs.items()):
                        if proc.poll() is not None:
                            print(f"Run {info['run_id']} on GPU {info['gpu_id']} finished.")
                            available_gpus.append(info['gpu_id'])
                            del active_procs[proc]
                            completed_runs += 1

                    while run_counter < total_run and available_gpus:
                        gpu_id = available_gpus.popleft()
                        current_run_id = run_counter
                        run_counter += 1

                        print(f"Launching run {current_run_id} on GPU {gpu_id}...")
                        
                        env = os.environ.copy()
                        # env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                        cmd = [
                            "python", "eval_hf.py",
                            "--model_name_or_path", model_dir,
                            "--data_name", task,
                            "--batch_size", str(bs),
                            "--limit", str(limit),
                            "--output_dir", output_config_subdir,
                            "--attention_implementation", attention_implementation,
                            "--use_batch_exist",
                            "--use_fused_kernel",
                            "--surround_with_messages",
                            "--rank", str(gpu_id),
                            "--sparsity_method", sparsity_method,
                            "--block_size", str(block_size),
                            "--run_id", str(current_run_id),
                            "--max_tokens", str(max_tokens),
                            "--start_layer", str(args.start_layer),
                        ] + cli_params

                        if args.profile_sparsity:
                            cmd.append("--profile_sparsity")

                        proc = subprocess.Popen(cmd, env=env)
                        active_procs[proc] = {"gpu_id": gpu_id, "run_id": current_run_id}

                    if (run_counter < total_run and not available_gpus) or (run_counter >= total_run and active_procs):
                        time.sleep(5)

                if task != "livecodebench": # remove this line at last
                    get_results_cmd = [
                        "python", "summary_results.py",
                        "--model_name_or_path", model_dir,
                        "--data_name", task,
                        "--limit", str(limit),
                        "--output_dir", output_config_subdir,
                        "--total_run", str(total_run),
                    ]

                    if args.profile_sparsity:
                        get_results_cmd.append("--profile_sparsity")

                    try:
                        subprocess.run(get_results_cmd, check=True)
                        print(f"Successfully generated results for {param_desc}")
                    except subprocess.CalledProcessError as e:
                        print(f"Error generating results: {e}")
                        
            print(f"\nCompleted: {block_size}")
        print(f"\nCompleted: {task}")

    print("\n All tasks and configurations completed!")