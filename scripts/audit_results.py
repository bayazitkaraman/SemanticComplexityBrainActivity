"""
Audit generated result files for the Brain and Language submission.

This script does not rerun the analysis. It checks whether the final tables and
figures needed for the manuscript/supplement exist and whether key counts are
internally consistent.

Run:
    python scripts/audit_results.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXPECTED_FINAL_TABLES = [
    "table1_dataset_story_counts.csv",
    "table1b_motion_qc_summary.csv",
    "table1c_motion_qc_counts_by_story.csv",
    "table2_primary_model_comparison.csv",
    "table3_primary_paired_model_delta_stats.csv",
    "table3b_primary_paired_model_values.csv",
    "table4_roi_delta_stats_combined_pc1_to_pc5_vs_baseline.csv",
    "table5_layer_overall_summary.csv",
    "table6_top_rois_best_layer.csv",
    "table7_sensitivity_model_comparison_by_strategy.csv",
    "table7b_sensitivity_best_model_by_strategy.csv",
    "table8_motion_correlation_with_gpt2_delta.csv",
]

EXPECTED_FINAL_FIGURES = [
    "fig1_updated_pipeline.png",
    "fig2_model_comparison.png",
    "fig3_roi_delta.png",
    "fig4_layer_summary.png",
    "fig4_roi_delta_stat_map.png",
    "fig5_sensitivity_motion_regression.png",
    "fig6_story_level_delta.png",
]


def check_exists(folder: Path, names: list[str]) -> list[str]:
    missing = []
    for name in names:
        path = folder / name
        status = "OK" if path.exists() else "MISSING"
        print(f"{status:8s} {path}")
        if not path.exists():
            missing.append(str(path))
    return missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-tables-dir", default="results/final_tables")
    parser.add_argument("--final-figures-dir", default="results/final_figures")
    args = parser.parse_args()

    final_tables = Path(args.final_tables_dir)
    final_figures = Path(args.final_figures_dir)

    print("\nFinal table files")
    missing = check_exists(final_tables, EXPECTED_FINAL_TABLES)
    print("\nFinal figure files")
    missing += check_exists(final_figures, EXPECTED_FINAL_FIGURES)

    print("\nConsistency checks")
    try:
        dataset = pd.read_csv(final_tables / "table1_dataset_story_counts.csv")
        paired = pd.read_csv(final_tables / "table3b_primary_paired_model_values.csv")
        roi = pd.read_csv(final_tables / "table4_roi_delta_stats_combined_pc1_to_pc5_vs_baseline.csv")
        model = pd.read_csv(final_tables / "table2_primary_model_comparison.csv")

        total_row = dataset[dataset["story"].astype(str).str.upper() == "TOTAL"]
        if not total_row.empty:
            scans = int(total_row.iloc[0]["n_subject_story_rows"])
            subjects = int(total_row.iloc[0]["n_unique_subjects"])
            print(f"OK       dataset total: {scans} subject-story scans, {subjects} unique subjects")
        else:
            scans = paired.groupby(["story", "subject"]).ngroups
            print("WARNING  no TOTAL row in dataset table")

        n_scans = paired.groupby(["story", "subject"]).ngroups
        n_rois = paired["roi"].nunique()
        expected_rows = n_scans * n_rois
        print(f"OK       paired table: {paired.shape[0]} rows = {n_scans} scans x {n_rois} ROIs")
        if paired.shape[0] != expected_rows:
            print(f"WARNING  paired rows do not equal scans x ROIs ({expected_rows})")

        model_counts = sorted(model["cv_r_count"].dropna().unique().tolist()) if "cv_r_count" in model else []
        print(f"OK       model comparison cv_r counts: {model_counts}")

        if "q_wilcoxon_greater_fdr" in roi:
            max_q = float(roi["q_wilcoxon_greater_fdr"].max())
            print(f"OK       ROI FDR q-values present; max q = {max_q:.6g}")
        else:
            print("WARNING  ROI table lacks q_wilcoxon_greater_fdr")

    except Exception as e:
        print(f"WARNING  consistency checks could not be completed: {e}")

    if missing:
        raise SystemExit(f"\nAudit finished with {len(missing)} missing files.")

    print("\nAudit finished: all expected final files were found.")


if __name__ == "__main__":
    main()
