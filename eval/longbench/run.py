import subprocess
import argparse



def run_commands(output_dir, model_checkpoint, thresholds):
    # Loop through the nz_ratios and run both commands
    model_dir = model_checkpoint.split("/")[-1]
    for t in thresholds:
        # Construct the pred_sparse.py command
        pred_command = [
            "python", "pred_threshold.py",
            "--model", model_checkpoint,
            "--e",
            "--output_path", output_dir,
            "--threshold", str(t)
        ]
        
        # Run the pred_sparse.py command
        print(f"Running pred_sparse.py for nz_ratio={t}")
        subprocess.run(pred_command)

        # Construct the eval.py command
        eval_command = [
            "python", "eval.py",
            "--model", f"{output_dir}/pred_e/{model_dir}_{t}",
            "--e"
        ]
        
        # Run the eval.py command
        print(f"Running eval.py for nz_ratio={t}")
        subprocess.run(eval_command)


if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser(description="Run pred_sparse.py and eval.py with different nz_ratio values.")
    
    # Add arguments for output directory, model checkpoint, and module name
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the commands")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--thresholds", nargs='+', type=float, required=True, help="List of nz_ratio values")

    # Parse the arguments
    args = parser.parse_args()

    # Run the commands with the parsed arguments
    run_commands(args.output_dir, args.model_checkpoint, args.thresholds)
