#!/bin/bash

# --- Configuration ---
task="aime"
bs=30
model_dir="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
base_output_dir="./results_oracle_aime_grid_search_multi_run" # Base directory for results
num_runs=8 # Number of times to run each block size configuration

# Parameters to iterate over
#thresholds=(0.0001 0.0005 0.001 0.005)
thresholds=(0.0001)
block_sizes=(16 32 64 128)
gpus=(4 5 6 7) # Corresponding GPU IDs (ensure length matches block_sizes)

# --- Ensure base output directory exists ---
mkdir -p "$base_output_dir"

# --- Check if GPU array length matches block_sizes array length ---
if [ ${#block_sizes[@]} -ne ${#gpus[@]} ]; then
    echo "Error: The number of block sizes (${#block_sizes[@]}) must match the number of GPUs (${#gpus[@]})."
    exit 1
fi

# --- Main Loop ---
echo "Starting experiment grid search with multiple runs..."
echo "Model: $model_dir"
echo "Task: $task"
echo "Batch Size: $bs"
echo "Number of runs per configuration: $num_runs"

# Iterate through thresholds sequentially
for threshold in "${thresholds[@]}"; do
    echo "====================================================="
    echo "Processing Threshold: $threshold"
    echo "====================================================="

    pids=() # Array to store process IDs of background jobs (one per block_size)

    # Launch block sizes in parallel for the current threshold
    for i in "${!block_sizes[@]}"; do
        # Assign block_size and gpu_id *before* the subshell
        current_block_size=${block_sizes[$i]}
        current_gpu_id=${gpus[$i]}

        # Use a subshell to run the repetitions sequentially for this block_size
        # Run this subshell in the background (&)
        (
            echo "--- Starting runs for BlockSize=$current_block_size on GPU $current_gpu_id (Threshold=$threshold) ---"

            # Inner loop for repetitions
            for run_num in $(seq 1 $num_runs); do
                # Create a unique output directory for this specific run
                output_dir="${base_output_dir}/thresh_${threshold}_block_${current_block_size}_run_${run_num}"
                mkdir -p "$output_dir"

                echo "Launching Run $run_num/$num_runs: Threshold=$threshold, BlockSize=$current_block_size on GPU $current_gpu_id"
                echo "Output Dir: $output_dir"

                # Run the command *synchronously* within this subshell for the current repetition
                # The entire subshell runs in the background relative to the main script
                CUDA_VISIBLE_DEVICES=$current_gpu_id python eval.py \
                    --model_name_or_path "$model_dir" \
                    --data_name "$task" \
                    --batch_size "$bs" \
                    --limit -1 \
                    --output_dir "$output_dir" \
                    --attention_implementation oracle_sparse \
                    --threshold "$threshold" \
                    --block_size "$current_block_size" \
                    --use_batch_exist \
                    --surround_with_messages > "$output_dir/run.log" 2>&1 # Redirect stdout/stderr

                # Optional: Check exit status of the python command
                if [ $? -ne 0 ]; then
                    echo "Warning: Run $run_num for BlockSize=$current_block_size on GPU $current_gpu_id failed. Check log: $output_dir/run.log"
                else
                    echo "Finished Run $run_num/$num_runs for BlockSize=$current_block_size on GPU $current_gpu_id"
                fi
                # Add a small delay if needed, though usually not necessary if I/O isn't overwhelming
                # sleep 1
            done

            echo "--- Completed all $num_runs runs for BlockSize=$current_block_size on GPU $current_gpu_id ---"

        ) & # End of subshell, run it in the background

        pids+=($!) # Store the process ID of the background subshell

    done # End of block_sizes loop

    # Wait for all background jobs (subshells for each block_size) to complete for the current threshold
    echo "Waiting for all parallel block size jobs (each with $num_runs sequential runs) for threshold $threshold to complete... PIDs: ${pids[*]}"
    wait # Waits for all background jobs (subshells) started in this loop iteration
    echo "All jobs for threshold $threshold finished."
    echo "====================================================="

done # End of thresholds loop

echo "-----------------------------------------------------"
echo "All experiments completed."
echo "-----------------------------------------------------"