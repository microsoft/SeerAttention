import argparse

def average_groups(file_path):
    groups = {'0-4k': [], '4k-8k': [], '8k+': []}
    with open(file_path, 'r') as file:
        for line in file:
            try:
                key, value = line.strip().split(':')
                key = float(key)
                value = float(value)
                if key < 4000:
                    groups['0-4k'].append(value)
                elif key < 8000:
                    groups['4k-8k'].append(value)
                else:
                    groups['8k+'].append(value)
            except ValueError:
                continue
    averages = {k: (sum(v)/len(v) if v else 0) for k, v in groups.items()}
    return averages

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate average sparsity for different groups.')
    parser.add_argument('--file', type=str, help='Path to the input file')
    args = parser.parse_args()

    file_path = args.file
    averages = average_groups(file_path)
    for group, avg in averages.items():
        print(f"{group}: {avg}")