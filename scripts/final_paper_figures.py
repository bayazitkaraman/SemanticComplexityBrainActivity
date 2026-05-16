from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def fig1_pipeline(output_dir: Path):
    steps = [
        ("Narratives dataset", "8 stories, 408 scans"),
        ("fMRIPrep MNI BOLD", "ROI time series"),
        ("Gentle word timing", "word onset/offset to TR bins"),
        ("GPT-2 embeddings", "contextual vectors"),
        ("Shared PCA", "PC1--PC5 semantic regressors"),
        ("HRF convolution", "Glover HRF"),
        ("Encoding models", "baseline vs GPT-2"),
        ("Final analyses", "ROIs, layers, sensitivity"),
    ]

    fig, ax = plt.subplots(figsize=(6.4, 8.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y_positions = np.linspace(0.92, 0.08, len(steps))
    box_w, box_h = 0.70, 0.065
    x = 0.15

    for i, ((title, subtitle), y) in enumerate(zip(steps, y_positions)):
        rect = Rectangle((x, y - box_h / 2), box_w, box_h, fill=False, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y + 0.010, title, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(x + box_w / 2, y - 0.015, subtitle, ha="center", va="center", fontsize=8.5)

        if i < len(steps) - 1:
            y_next = y_positions[i + 1]
            arrow = FancyArrowPatch(
                (0.5, y - box_h / 2 - 0.004),
                (0.5, y_next + box_h / 2 + 0.004),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
            )
            ax.add_patch(arrow)

    ax.set_title("Final analysis pipeline", fontsize=14, fontweight="bold", pad=8)
    savefig(output_dir / "fig1_updated_pipeline.png")


def fig2_model_comparison(final_tables_dir: Path, output_dir: Path):
    df = pd.read_csv(final_tables_dir / "table2_primary_model_comparison.csv")
    df = df.sort_values("cv_r_mean", ascending=True)

    label_map = {
        "combined_pc1_to_pc5": "Baseline + GPT-2 PC1--PC5",
        "combined_pc1": "Baseline + GPT-2 PC1",
        "baseline_only": "Lexical/time baseline",
        "gpt2_pc1_to_pc5": "GPT-2 PC1--PC5 only",
        "gpt2_pc1": "GPT-2 PC1 only",
    }
    labels = [label_map.get(x, x) for x in df["model"]]

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.barh(labels, df["cv_r_mean"], xerr=df["cv_r_sem"] * 1.96, capsize=3)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Mean cross-validated prediction correlation ($cv\\_r$)")
    ax.set_ylabel("")
    ax.set_title("Encoding model comparison")
    fig.tight_layout()
    savefig(output_dir / "fig2_model_comparison.png")


def fig3_roi_delta(final_tables_dir: Path, output_dir: Path):
    df = pd.read_csv(final_tables_dir / "table4_roi_delta_stats_combined_pc1_to_pc5_vs_baseline.csv")
    df = df.sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(8.0, 7.4))
    ax.barh(df["roi"], df["mean"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Mean improvement over baseline ($\\Delta cv\\_r$)")
    ax.set_ylabel("")
    ax.set_title("ROI-level benefit of GPT-2 PC1--PC5 over baseline")
    fig.tight_layout()
    savefig(output_dir / "fig3_roi_delta.png")


def fig4_layer_summary(final_tables_dir: Path, output_dir: Path):
    df = pd.read_csv(final_tables_dir / "table5_layer_overall_summary.csv")
    df = df.sort_values("layer")

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(df["layer"], df["mean_abs_r"], marker="o")
    ax.set_xlabel("GPT-2 layer")
    ax.set_ylabel("Mean absolute ROI correlation")
    ax.set_title("Layer-wise GPT-2/brain alignment")
    ax.set_xticks(df["layer"])
    fig.tight_layout()
    savefig(output_dir / "fig4_layer_summary.png")


def fig5_sensitivity(final_tables_dir: Path, output_dir: Path):
    df = pd.read_csv(final_tables_dir / "table7_sensitivity_model_comparison_by_strategy.csv")
    model = "combined_pc1_to_pc5"
    sub = df[df["model"] == model].copy()
    if sub.empty:
        return

    order = ["gentle_motionQC_only", "gentle_motion6_nohpf", "gentle_motion24_nohpf"]
    pretty = {
        "gentle_motionQC_only": "Motion QC only",
        "gentle_motion6_nohpf": "Motion6",
        "gentle_motion24_nohpf": "Motion24",
    }
    sub["order"] = sub["run"].apply(lambda x: order.index(x) if x in order else 999)
    sub = sub.sort_values("order")
    labels = [pretty.get(x, x) for x in sub["run"]]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(labels, sub["cv_r"])
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Mean $cv\\_r$")
    ax.set_title("Sensitivity to motion regression")
    fig.tight_layout()
    savefig(output_dir / "fig5_sensitivity_motion_regression.png")


def fig6_story_delta(final_tables_dir: Path, output_dir: Path):
    values_path = final_tables_dir / "table3b_primary_paired_model_values.csv"
    if not values_path.exists():
        return

    df = pd.read_csv(values_path)
    dcol = "delta_combined_pc1_to_pc5_minus_baseline_only"
    if dcol not in df.columns:
        return

    story = df.groupby(["story", "subject"], as_index=False)[dcol].mean()
    summary = story.groupby("story")[dcol].agg(["mean", "sem", "count"]).reset_index()
    summary = summary.sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.barh(summary["story"], summary["mean"], xerr=summary["sem"] * 1.96, capsize=3)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Mean scan-level improvement ($\\Delta cv\\_r$)")
    ax.set_ylabel("")
    ax.set_title("Story-level variability")
    fig.tight_layout()
    savefig(output_dir / "fig6_story_level_delta.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-tables-dir", default="results/final_tables")
    parser.add_argument("--output-dir", default="results/final_figures")
    args = parser.parse_args()

    final_tables_dir = Path(args.final_tables_dir)
    output_dir = ensure_dir(args.output_dir)

    fig1_pipeline(output_dir)
    fig2_model_comparison(final_tables_dir, output_dir)
    fig3_roi_delta(final_tables_dir, output_dir)
    fig4_layer_summary(final_tables_dir, output_dir)
    fig5_sensitivity(final_tables_dir, output_dir)
    fig6_story_delta(final_tables_dir, output_dir)

    print(f"Saved final figures to: {output_dir}")
    for p in sorted(output_dir.glob("fig*.png")):
        print(" -", p)
    for p in sorted(output_dir.glob("fig*.pdf")):
        print(" -", p)


if __name__ == "__main__":
    main()
