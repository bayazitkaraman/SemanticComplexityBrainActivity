from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIG_DIR = "results/revision_figures"
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

def plot_pc1_baseline_correlations():
    path = "results/diagnostics/pc1_vs_baselines_correlations.csv"
    if not os.path.exists(path):
        print(f"Skipping PC1 baseline figure; missing {path}")
        return
    df = pd.read_csv(path)
    pairs = [
        ("r_PC1_wordcount", "Word count"),
        ("r_PC1_mean_charlen", "Mean word length"),
        ("r_PC1_total_chars", "Total chars"),
        ("r_PC1_time", "Time"),
    ]
    pairs = [(c, l) for c, l in pairs if c in df.columns]
    if not pairs:
        return
    x = np.arange(len(df))
    width = 0.8 / len(pairs)
    plt.figure(figsize=(10, 5))
    for i, (col, label) in enumerate(pairs):
        offset = (i - (len(pairs)-1)/2) * width
        plt.bar(x + offset, df[col], width=width, label=label)
    plt.axhline(0, linewidth=1)
    plt.xticks(x, df["story"], rotation=30, ha="right")
    plt.ylabel("Pearson r with PC1")
    plt.title("PC1 correlation with simple lexical/time baselines")
    plt.legend()
    savefig("figR1_pc1_vs_lexical_baselines.png")

def plot_shared_pca_variance():
    path = "results/regressors/shared_pca_explained_variance.csv"
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    plt.figure(figsize=(8, 5))
    plt.bar(df["component"], df["explained_variance_ratio"])
    plt.ylabel("Explained variance ratio")
    plt.xlabel("Shared PCA component")
    plt.title("Shared cross-story GPT-2 PCA explained variance")
    plt.xticks(rotation=45, ha="right")
    savefig("figR2_shared_pca_explained_variance.png")

def plot_layer_summary():
    path = "results/layer_analysis/layer_overall_summary.csv"
    if not os.path.exists(path):
        return
    df = pd.read_csv(path).sort_values("layer")
    plt.figure(figsize=(8, 5))
    plt.plot(df["layer"], df["mean_abs_r"], marker="o")
    plt.xlabel("GPT-2 layer")
    plt.ylabel("Mean absolute ROI correlation")
    plt.title("Layer comparison: shared-PC1 alignment with ROI activity")
    plt.xticks(df["layer"])
    savefig("figR3_gpt2_layer_overall_summary.png")

def plot_encoding_model_summary():
    path = "results/model_comparison/encoding_model_comparison.csv"
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    metric = "cv_r" if "cv_r" in df.columns else "cv_r2"
    summary = df.groupby("model", as_index=False)[metric].mean().sort_values(metric, ascending=False)
    plt.figure(figsize=(9, 5))
    plt.bar(summary["model"], summary[metric])
    plt.ylabel("Mean cross-validated prediction r" if metric == "cv_r" else "Mean cross-validated R²")
    plt.xlabel("Model")
    plt.title("Encoding model comparison")
    plt.xticks(rotation=30, ha="right")
    savefig("figR4_encoding_model_mean_cv_r.png")

def plot_encoding_delta_by_roi():
    path = "results/model_comparison/encoding_model_delta_summary_by_roi.csv"
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    delta_cols = [c for c in df.columns if c.startswith("delta_cv_r_")]
    if not delta_cols:
        delta_cols = [c for c in df.columns if c.startswith("delta_")]
    if not delta_cols:
        print("Skipping delta figure; no delta columns found.")
        return
    col = delta_cols[-1]
    df = df.sort_values(col, ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    plt.barh(df["roi"], df[col])
    plt.xlabel("Δ cross-validated prediction r over baseline-only")
    plt.ylabel("ROI")
    plt.title("Added value of GPT-2 PCs beyond lexical/time baselines")
    plt.gca().invert_yaxis()
    savefig("figR5_encoding_delta_by_roi.png")

def main():
    plot_pc1_baseline_correlations()
    plot_shared_pca_variance()
    plot_layer_summary()
    plot_encoding_model_summary()
    plot_encoding_delta_by_roi()

if __name__ == "__main__":
    main()
