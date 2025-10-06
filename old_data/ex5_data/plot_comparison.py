import pandas as pd
import matplotlib.pyplot as plt
import os


df = pd.read_csv("ex5_data/compare_ACO_MMAS.csv")


output_dir = "./plots_ex5"
os.makedirs(output_dir, exist_ok=True)


for func in df["Function"].unique():
    subset = df[df["Function"] == func]

    plt.figure(figsize=(6, 4))
    plt.bar(subset["Alg"], subset["mean"], yerr=subset["std"], capsize=6)
    plt.xlabel("Algorithm", fontsize=12)
    plt.ylabel("Mean Fitness", fontsize=12)
    plt.title(f"Performance Comparison on {func}", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    
    output_path = os.path.join(output_dir, f"{func}_comparison_bar.png")
    plt.savefig(output_path)
    plt.close()
    print(f" Saved: {output_path}")

print("\n All plots saved in:", os.path.abspath(output_dir))
