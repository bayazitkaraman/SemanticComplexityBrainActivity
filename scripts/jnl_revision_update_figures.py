from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REV_DIR = ROOT / "Submission (Journal of Neurolinguistics)" / "Revision_JNL-D-26-00113_2026-06-29"


def bootstrap_ci(values: np.ndarray, seed: int = 7, n_boot: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = values[np.isfinite(values)]
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fig2_model_comparison_with_delta_distribution() -> None:
    long_path = ROOT / "results" / "model_comparison_gentle_motionQC_only" / "encoding_model_comparison.csv"
    scan_path = ROOT / "results" / "jnl_revision_major_review" / "primary_scan_level_model_values.csv"
    inference_path = ROOT / "results" / "jnl_revision_major_review" / "primary_inference_summary.csv"
    long = pd.read_csv(long_path)
    scan = pd.read_csv(scan_path)
    inference = pd.read_csv(inference_path).iloc[0]

    model_order = [
        "combined_pc1_to_pc5",
        "combined_pc1",
        "baseline_only",
        "gpt2_pc1_to_pc5",
        "gpt2_pc1",
    ]
    labels = {
        "combined_pc1_to_pc5": "Baseline + GPT-2 PC1--PC5",
        "combined_pc1": "Baseline + GPT-2 PC1",
        "baseline_only": "Lexical-plus-time baseline",
        "gpt2_pc1_to_pc5": "GPT-2 PC1--PC5 only",
        "gpt2_pc1": "GPT-2 PC1 only",
    }
    summary = (
        long.groupby("model", as_index=False)
        .agg(mean=("cv_r", "mean"), sem=("cv_r", lambda x: x.std(ddof=1) / np.sqrt(len(x))))
        .set_index("model")
        .loc[model_order]
        .reset_index()
    )

    deltas = scan["mean_roi_delta_cv_r"].to_numpy(float)
    mean_delta = float(np.nanmean(deltas))
    median_delta = float(np.nanmedian(deltas))
    ci_low = float(inference["scan_ci95_low"])
    ci_high = float(inference["scan_ci95_high"])
    frac_positive = float(np.mean(deltas > 0))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), gridspec_kw={"width_ratios": [1.0, 1.15]})

    ax = axes[0]
    y = np.arange(len(summary))
    colors = ["#2C7FB8", "#7FCDBB", "#D9D9D9", "#A1D99B", "#FCAE91"]
    ax.barh(y, summary["mean"], xerr=1.96 * summary["sem"], color=colors, edgecolor="black", linewidth=0.45)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[m] for m in summary["model"]])
    ax.invert_yaxis()
    ax.set_xlabel("Mean held-out prediction ($cv_r$)")
    ax.set_title("A. Encoding model means")
    ax.grid(axis="x", alpha=0.22)

    ax = axes[1]
    bins = np.linspace(-0.16, 0.16, 33)
    ax.hist(deltas, bins=bins, color="#6BAED6", edgecolor="white", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=1.0, label="No improvement")
    ax.axvline(mean_delta, color="#08519C", linewidth=2.0, label="Mean")
    ax.axvline(median_delta, color="#CB181D", linestyle="--", linewidth=1.8, label="Median")
    ax.set_xlabel("Subject-story scan mean $\\Delta cv_r$")
    ax.set_ylabel("Number of scans")
    ax.set_title("B. Baseline + GPT-2 PC1--PC5 improvement")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    text = (
        f"Mean = {mean_delta:.4f}\n"
        f"95% CI [{ci_low:.4f}, {ci_high:.4f}]\n"
        f"Positive scans = {frac_positive:.1%}"
    )
    ax.text(
        0.03,
        0.95,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#BDBDBD", alpha=0.95),
    )

    fig.tight_layout()
    fig.savefig(REV_DIR / "fig2_model_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(REV_DIR / "fig2_model_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def fig4_hemisphere_sensitivity() -> None:
    hemi_path = ROOT / "results" / "jnl_revision_major_review" / "hemisphere_sensitivity" / "hemisphere_scan_level_wide.csv"
    stats_path = ROOT / "results" / "jnl_revision_major_review" / "hemisphere_sensitivity" / "hemisphere_scan_stats.csv"
    paired_path = ROOT / "results" / "jnl_revision_major_review" / "hemisphere_sensitivity" / "hemisphere_left_right_paired_summary.csv"
    wide = pd.read_csv(hemi_path)
    stats = pd.read_csv(stats_path).set_index("hemisphere")
    paired = pd.read_csv(paired_path).iloc[0]

    left = wide["left"].to_numpy(float)
    right = wide["right"].to_numpy(float)
    diff = wide["left_minus_right_delta_cv_r"].to_numpy(float)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.0), gridspec_kw={"width_ratios": [0.85, 1.15]})

    ax = axes[0]
    means = [stats.loc["left", "mean"], stats.loc["right", "mean"]]
    ci_low = [stats.loc["left", "ci95_low"], stats.loc["right", "ci95_low"]]
    ci_high = [stats.loc["left", "ci95_high"], stats.loc["right", "ci95_high"]]
    yerr = np.array([[means[0] - ci_low[0], means[1] - ci_low[1]], [ci_high[0] - means[0], ci_high[1] - means[1]]])
    ax.bar([0, 1], means, yerr=yerr, color=["#3182BD", "#31A354"], edgecolor="black", linewidth=0.5, capsize=4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Left", "Right"])
    ax.set_ylabel("Mean $\\Delta cv_r$")
    ax.set_title("A. Hemisphere means")
    ax.grid(axis="y", alpha=0.22)

    ax = axes[1]
    positions = np.array([0, 1])
    jitter = np.linspace(-0.06, 0.06, len(left))
    rng = np.random.default_rng(11)
    jitter = rng.normal(0, 0.035, len(left))
    ax.scatter(np.zeros_like(left) + jitter, left, s=12, alpha=0.22, color="#3182BD", edgecolors="none")
    ax.scatter(np.ones_like(right) + jitter, right, s=12, alpha=0.22, color="#31A354", edgecolors="none")
    for l, r in zip(left[::8], right[::8]):
        ax.plot([0, 1], [l, r], color="#9E9E9E", alpha=0.20, linewidth=0.55)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Left", "Right"])
    ax.set_ylabel("Subject-story scan mean $\\Delta cv_r$")
    ax.set_title("B. Scan-level paired values")
    ax.grid(axis="y", alpha=0.22)
    text = (
        f"Left - right mean = {paired['mean']:.4f}\n"
        f"95% CI [{paired['ci95_low']:.4f}, {paired['ci95_high']:.4f}]\n"
        f"p = {paired['p_ttest_two_sided']:.3f}"
    )
    ax.text(
        0.03,
        0.95,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#BDBDBD", alpha=0.95),
    )

    fig.suptitle("Hemisphere sensitivity for GPT-2 PC1--PC5 model improvement", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(REV_DIR / "fig4_roi_delta_stat_map.png", dpi=300, bbox_inches="tight")
    fig.savefig(REV_DIR / "fig4_roi_delta_stat_map.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig2_model_comparison_with_delta_distribution()
    fig4_hemisphere_sensitivity()


if __name__ == "__main__":
    main()
