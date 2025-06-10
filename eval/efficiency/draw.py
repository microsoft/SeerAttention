import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse  


parser = argparse.ArgumentParser(description="Plot speedup results from CSV file.")
parser.add_argument("--data_path", type=str, default="./kernel_test.csv",
                    help="Path to the input CSV file (e.g., 'results_new/kernel_test_gqa64_8.csv')")

args = parser.parse_args()

df = pd.read_csv(args.data_path, delimiter=",", header=None, names=[
    "batch", "max_cache_seqlen", "sparse_ratio", 
    "fa3_dense_time", "triton_sparse_time", "tilelang_sparse_time"
])
for col in ["batch", "max_cache_seqlen", "sparse_ratio", "fa3_dense_time", "triton_sparse_time", "tilelang_sparse_time"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')


df["triton_speedup"] = df["fa3_dense_time"] / df["triton_sparse_time"]
df["tilelang_speedup"] = df["fa3_dense_time"] / df["tilelang_sparse_time"]


config_order = [
    "16*8192", "16*16384", "16*32768", "16*65536", "16*131072",
    "8*8192", "8*16384", "8*32768", "8*65536", "8*131072",
    "4*8192", "4*16384", "4*32768", "4*65536", "4*131072",
    "2*8192", "2*16384", "2*32768", "2*65536", "2*131072",
    "1*8192", "1*16384", "1*32768", "1*65536", "1*131072",
]


fig, axes = plt.subplots(5, 5, figsize=(25, 20))
axes = axes.flatten()  


for idx, config in enumerate(config_order):
    
    batch, max_cache_seqlen = map(int, config.split("*"))
    subset = df[(df["batch"] == batch) & (df["max_cache_seqlen"] == max_cache_seqlen)]
    
    if not subset.empty:
        ax = axes[idx]
        sparse_ratios = subset["sparse_ratio"].values
        triton_speedup = subset["triton_speedup"].values
        tilelang_speedup = subset["tilelang_speedup"].values
        
        
        ax.plot(sparse_ratios, triton_speedup, marker='o', label="Triton Sparse")
        ax.plot(sparse_ratios, tilelang_speedup, marker='s', label="TileLang Sparse")
        ax.plot(sparse_ratios, np.ones_like(sparse_ratios), linestyle="--", color="gray", label="FA3 Baseline")
        
        
        ax.set_title(f"batch={batch}, max_cache_seqlen={max_cache_seqlen}", fontsize=10)
        ax.set_xlabel("Sparse Ratio", fontsize=8)
        ax.set_ylabel("Speedup", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(loc="upper left", fontsize=6)
    else:
        axes[idx].axis("off")  


plt.tight_layout()
fig_path = args.data_path.replace(".csv", "_speedup.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.show()