import os
import json
import argparse
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

def calculate_quantile_sparsity(
    all_batch_sparsitys_info: List[List[Optional[Tuple[Tuple[int, int], ...]]]],
    group_size: int = 1000
) -> List[float]:
    """
    Calculates sparsity for each quantile group of sequence steps.

    Each group aggregates results over `group_size` sequence steps from the flattened list.
    The sparsity is computed as 1 - (total activated blocks / total original blocks) for that group.

    Args:
        all_batch_sparsitys_info: Nested list structure from each batch.
        group_size: Number of sequence steps per quantile group.

    Returns:
        A list of sparsity values for each quantile.
    """
    # Flatten the sparsity info by summing over layers per sequence step.
    flattened = []
    for each_batch_sequence_info in all_batch_sparsitys_info:
        for each_step_sparsitys_info in each_batch_sequence_info:
            act = 0
            orig = 0
            for each_layer_sparsitys_info in each_step_sparsitys_info:
                act += each_layer_sparsitys_info[0]
                orig += each_layer_sparsitys_info[1]
            flattened.append((act, orig))
    
    quantile_results = []
    num_groups = (len(flattened) + group_size - 1) // group_size  # ceiling division
    for i in range(num_groups):
        start = i * group_size
        end = min((i + 1) * group_size, len(flattened))
        group = flattened[start:end]
        total_act = sum(item[0] for item in group)
        total_orig = sum(item[1] for item in group)
        if total_orig == 0:
            sparsity = 0.0
        else:
            sparsity = 1 - total_act / total_orig
        quantile_results.append(sparsity)
    return quantile_results

def plot_sparsity_from_json(json_filepath):
    # Read the JSON file.
    with open(json_filepath, 'r') as f:
        data = json.load(f)
    
    all_batch_sparsitys_info = data.get("sparsity_info")
    if all_batch_sparsitys_info is None:
        raise ValueError("JSON file does not contain 'quantile_sparsities' or 'all_batch_sparsitys_info' key.")
    quantile_sparsities = calculate_quantile_sparsity(all_batch_sparsitys_info, group_size=1000)
    print("len:", len(quantile_sparsities))
    print("Calculated quantile sparsities from all_batch_sparsitys_info:")
    print(quantile_sparsities[15])
    print(quantile_sparsities[-2])
    print(quantile_sparsities)

    
    # Plot the sparsity distribution.
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(quantile_sparsities)), quantile_sparsities, marker='o')
    plt.xlabel("Quantile group (each group = 1k sequence steps)")
    plt.ylabel("Sparsity")
    plt.title("Quantile Sparsity Distribution")
    plt.grid(True)
    
    # Save the plot as a PNG file with the same base name as the JSON file.
    base_name = os.path.splitext(os.path.basename(json_filepath))[0]
    output_png = os.path.join(os.path.dirname(json_filepath), base_name + ".png")
    plt.savefig(output_png)
    print("Saved sparsity plot to", output_png)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot sparsity data from a JSON file.")
    parser.add_argument("--file", type=str, help="Path to the JSON file containing sparsity information")
    args = parser.parse_args()
    
    plot_sparsity_from_json(args.file)