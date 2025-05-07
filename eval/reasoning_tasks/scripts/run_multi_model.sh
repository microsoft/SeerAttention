#!/bin/bash
# Set the folder that contains your model directories.
models_dir=${MODELS_DIR:-"./models"}
# Define the subfolder that will store all eval results.
results_dir="${models_dir}/results"
echo "results_dir: ${results_dir}"
threshold=${THRESHOLD:-"5e-3"}

# Example tasks to evaluate on (modify as necessary)
# TASKS="aime,math,gpqa,olympiadbench"
tasks=${TASKS:-"aime"}
# Any extra arguments you need to pass to the evaluation script, e.g., the attention method.
attention_implementation="seer_sparse"
sparsity_method="threshold"
model_size="14B"
num_gpus=8
limit=-1

# Create the results folder if it doesn't exist
mkdir -p "$results_dir"

echo "Starting evaluation for all models in: ${models_dir}"
for model in "$models_dir"/*/ ; do
    # Ensure the file is indeed a folder

    model_basename=$(basename "${model%/}")
    # Skip the "results" subfolder.
    if [ "$model_basename" = "results" ]; then
         echo "Skipping directory: ${model_basename}"
         continue
    fi

    if [ -d "$model" ]; then
        # Remove trailing slash and get the base directory name.
        model_dir="${model%/}"
        output_dir=${results_dir}
        # output_dir="${results_dir}/${model_basename}"
        
        echo "Evaluating model: ${model_basename}"
        echo "Results will be saved to: ${output_dir}"
        
        # Run the evaluation script for the model.
        python parallel_run_hf.py \
            --model_dir "$model_dir" \
            --model_size "$model_size" \
            --tasks "$tasks" \
            --output_dir "$output_dir" \
            --attention_implementation "$attention_implementation" \
            --sparsity_method "$sparsity_method" \
            --profile_sparsity \
            --threshold "$threshold" \
            --num_gpus "$num_gpus" \
            --limit "$limit" \
        
        echo "Completed evaluation for model: ${model_basename}"
    fi
done

echo "All model evaluations completed!"