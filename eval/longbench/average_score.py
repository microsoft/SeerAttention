import json
import argparse

# Load JSON data from file
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Calculate the average score across all datasets for each range
def calculate_averages_by_range(data):
    range_totals = {"0-4k": 0, "4-8k": 0, "8k+": 0}
    dataset_count = {"0-4k": 0, "4-8k": 0, "8k+": 0}

    for dataset, values in data.items():
        for key in range_totals.keys():
            if key in values:
                range_totals[key] += values[key]
                dataset_count[key] += 1

    # Calculate the averages
    averages_by_range = {key: range_totals[key] / dataset_count[key] for key in range_totals}

    return averages_by_range

# Main function
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Parse JSON file and calculate average scores across datasets.")
    parser.add_argument('file_path', type=str, help='Path to the JSON file')

    # Parse command line arguments
    args = parser.parse_args()

    # Load data from JSON file
    data = load_json(args.file_path)
    
    # Calculate averages by range
    averages_by_range = calculate_averages_by_range(data)
    
    # Print the averages for each range
    for key, avg_score in averages_by_range.items():
        print(f"Range: {key}, Average Score: {avg_score:.2f}")

if __name__ == "__main__":
    main()
