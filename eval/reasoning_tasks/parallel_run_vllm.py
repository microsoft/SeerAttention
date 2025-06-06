#!/usr/bin/env python3
import subprocess
import os
import sys
import argparse
import time
from collections import deque # Use deque for efficient pop/append


task_config = {
    "aime24": {"total_run": 64},
    "aime25": {"total_run": 64},
    "math": {"total_run": 8},
    "gpqa": {"total_run": 16},
    "olympiadbench": {"total_run": 8},
    "livecodebench": {"total_run": 8},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tasks using subprocess.")
    parser.add_argument("--model_dir", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                        help="Model directory path")
    parser.add_argument("--tasks", type=str, default="aime",
                        help="Comma-separated list of tasks (e.g., aime,math,gpqa)")
    parser.add_argument("--output_dir", type=str, default="./results/aime",
                        help="Directory to store output results")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Limit for the number of samples to process")
    parser.add_argument("--num_gpus", default="8", type=int)
    parser.add_argument("--max_tokens", default=32768, type=int,
                        help="Maximum number of tokens to generate")
    args = parser.parse_args()
    limit = args.limit
    num_gpus = args.num_gpus
    max_tokens = args.max_tokens

    model_dir = args.model_dir
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    model_subfolder = os.path.basename(model_dir.rstrip('/'))
    output_dir = os.path.join(args.output_dir, model_subfolder)


    for task in tasks:
        if task not in task_config:
            print(f"Error: Unknown task '{task}'")
            sys.exit(1)

        total_run = task_config[task]["total_run"]

        print(f"Starting task: {task}")
            
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

        output_config_subdir = os.path.join(output_dir, f"{task}_vllm_{max_tokens}")
        os.makedirs(output_config_subdir, exist_ok=True)

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
                # env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                # Set CUDA_VISIBLE_DEVICES for isolation (optional but good practice)
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id) # Uncomment if eval.py doesn't handle rank/device selection well

                cmd = [
                    "python", "eval_vllm.py",
                    "--model_name_or_path", model_dir,
                    "--data_name", task,
                    "--output_dir", output_config_subdir,
                    "--surround_with_messages",
                    "--limit", str(limit),
                    "--rank", str(gpu_id), 
                    "--run_id", str(current_run_id), # Pass the unique run ID
                    "--max_tokens", str(max_tokens),
                ]
                
                # Launch the process
                proc = subprocess.Popen(cmd, env=env)
                # Store the process and its assigned GPU and run ID
                active_procs[proc] = {"gpu_id": gpu_id, "run_id": current_run_id} 

            # --- MODIFICATION END ---

            # If no GPUs are available or all runs launched, wait briefly before checking again
            if (run_counter < total_run and not available_gpus) or (run_counter >= total_run and active_procs):
                    time.sleep(5) # Shorter sleep time now as we check more actively


        # --- Original wait loop removed as the logic is integrated above ---

        print(f"All {total_run} runs completed.")

        # --- Run get_results.py (unchanged) ---
        get_results_cmd = [
            "python", "summary_results.py",
            "--model_name_or_path", model_dir,
            "--data_name", task,
            "--output_dir", output_config_subdir,
            "--limit", str(limit),
            "--total_run", str(total_run),
        ]

        try:
            subprocess.run(get_results_cmd, check=True)
            print(f"--- Successfully generated results  ---")
        except subprocess.CalledProcessError as e:
            print(f"--- Error running get_results.py: {e} ---")
            # Decide if you want to exit or continue with the next threshold/task
            # sys.exit(1) 

        print(f"Completed: {task}")

    print("All tasks and configurations completed!")