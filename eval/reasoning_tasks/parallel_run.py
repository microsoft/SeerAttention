#!/usr/bin/env python3
import subprocess
import os
import sys
import argparse
import time
from collections import deque # Use deque for efficient pop/append

limit = -1
num_gpus = 8

def Choose_task_config(model_size):
    if model_size == "7B" or model_size == "8B":
       task_config = {
            "aime": {"bs": 30, "total_run": 64},
            "math": {"bs": 500, "total_run": 8},
            "gpqa": {"bs": 200, "total_run": 16},
            "olympiadbench": {"bs": 120, "total_run": 8}
        }
    elif model_size == "14B":
        task_config = {
            "aime": {"bs": 30, "total_run": 64},
            "math": {"bs": 250, "total_run": 8},
            "gpqa": {"bs": 100, "total_run": 16},
            "olympiadbench": {"bs": 60, "total_run": 8}
        }
    elif model_size == "32B":
        task_config = {
            "aime": {"bs": 30, "total_run": 64},
            "math": {"bs": 100, "total_run": 8},
            "gpqa": {"bs": 50, "total_run": 16},
            "olympiadbench": {"bs": 30, "total_run": 8}
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
    parser.add_argument("--attention", type=str, default="seer_sparse",
                        help="attention implementations")
    parser.add_argument("--profile_sparsity", action="store_true",
                        help="Flag to profile sparsity in eval.py")
    parser.add_argument("--threshold", type=str, default="4e-3",
                        help="Comma-separated list of float thresholds")
    args = parser.parse_args()

    model_dir = args.model_dir
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    thresholds = [t.strip() for t in args.threshold.split(",") if t.strip()]

    model_subfolder = os.path.basename(model_dir.rstrip('/'))
    output_dir = os.path.join(args.output_dir, model_subfolder)
    attention_implementations = args.attention

    task_config = Choose_task_config(args.model_size)

    for task in tasks:
        if task not in task_config:
            print(f"Error: Unknown task '{task}'")
            sys.exit(1)

        bs = task_config[task]["bs"]
        total_run = task_config[task]["total_run"]

        print(f"Starting task: {task}  | attention: {attention_implementations}")
        print(f"Batch size: {bs} | total_run: {total_run}")

        for threshold in thresholds:
            print(f"--- Starting evaluation for threshold: {threshold} ---")
            
            # --- MODIFICATION START ---
            # Keep track of active processes and the GPU they are assigned to
            # Use a dictionary: {process: gpu_id}
            active_procs = {} 
            # Keep track of available GPU IDs. Initialize with all GPUs.
            available_gpus = deque(range(num_gpus)) 
            # --- MODIFICATION END ---

            # Use a single loop counter for the runs to launch
            run_counter = 0
            # Keep track of completed runs to ensure total_run are processed
            completed_runs = 0 

            # Continue as long as there are runs to launch OR runs still active
            while run_counter < total_run or active_procs:

                # --- MODIFICATION: Check for finished processes first ---
                # Use list(active_procs.items()) to avoid RuntimeError: dictionary changed size during iteration
                for proc, info in list(active_procs.items()):
                    if proc.poll() is not None:
                        print(f"Run {info['run_id']} on GPU {info['gpu_id']} finished.")
                        available_gpus.append(info['gpu_id'])
                        del active_procs[proc]
                        completed_runs += 1
                # --- MODIFICATION END ---


                # --- MODIFICATION: Launch new processes if possible ---
                # Check if there are runs left to launch AND there's an available GPU
                while run_counter < total_run and available_gpus:
                    # Get the next available GPU ID
                    gpu_id = available_gpus.popleft() # Take from the left (FIFO)
                    
                    # Assign the current run number
                    current_run_id = run_counter
                    run_counter += 1 # Increment the counter for the next potential run

                    print(f"Launching run {current_run_id} on GPU {gpu_id}...")

                    env = os.environ.copy()
                    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                    # Set CUDA_VISIBLE_DEVICES for isolation (optional but good practice)
                    # env["CUDA_VISIBLE_DEVICES"] = str(gpu_id) # Uncomment if eval.py doesn't handle rank/device selection well

                    cmd = [
                        "python", "eval.py",
                        "--model_name_or_path", model_dir,
                        "--data_name", task,
                        "--batch_size", str(bs),
                        "--limit", str(limit),
                        "--output_dir", output_dir,
                        "--attention_implementation", attention_implementations,
                        "--use_batch_exist",
                        "--use_fused_kernel",
                        "--surround_with_messages",
                        # Pass the assigned GPU ID. Ensure eval.py uses this to select the device.
                        "--rank", str(gpu_id), 
                        "--threshold", threshold,
                        "--run_id", str(current_run_id), # Pass the unique run ID
                    ]
                    if args.profile_sparsity:
                        cmd.append("--profile_sparsity")
                    
                    # Launch the process
                    proc = subprocess.Popen(cmd, env=env)
                    # Store the process and its assigned GPU and run ID
                    active_procs[proc] = {"gpu_id": gpu_id, "run_id": current_run_id} 

                # --- MODIFICATION END ---

                # If no GPUs are available or all runs launched, wait briefly before checking again
                if (run_counter < total_run and not available_gpus) or (run_counter >= total_run and active_procs):
                     time.sleep(5) # Shorter sleep time now as we check more actively


            # --- Original wait loop removed as the logic is integrated above ---

            print(f"All {total_run} runs for threshold {threshold} completed.")

            # --- Run get_results.py (unchanged) ---
            get_results_cmd = [
                "python", "get_results.py",
                "--model_name_or_path", model_dir,
                "--data_name", task,
                "--batch_size", str(bs),
                "--limit", str(limit),
                "--output_dir", output_dir,
                "--attention_implementation", attention_implementations,
                "--use_batch_exist",
                "--total_run", str(total_run),
                "--threshold", threshold
            ]
            if args.profile_sparsity:
                get_results_cmd.append("--profile_sparsity")

            try:
                subprocess.run(get_results_cmd, check=True)
                print(f"--- Successfully generated results for threshold: {threshold} ---")
            except subprocess.CalledProcessError as e:
                print(f"--- Error running get_results.py for threshold {threshold}: {e} ---")
                # Decide if you want to exit or continue with the next threshold/task
                # sys.exit(1) 

        print(f"Completed: {task}-{attention_implementations}")

    print("All tasks and configurations completed!")