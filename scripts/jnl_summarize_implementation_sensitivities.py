from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "jnl_revision_major_review"


def pc5_row(label: str, directory: str) -> dict:
    path = ROOT / directory / "pc_count_sensitivity_scan_stats.csv"
    df = pd.read_csv(path)
    row = df.loc[df["n_pcs"] == 5].iloc[0].to_dict()
    return {
        "analysis": label,
        "source": str(path.relative_to(ROOT)),
        "n": int(row["n"]),
        "mean_delta_cv_r": row["mean"],
        "median_delta_cv_r": row["median"],
        "ci95_low": row["ci95_low"],
        "ci95_high": row["ci95_high"],
        "t_stat": row["t_stat"],
        "p_ttest_two_sided": row["p_ttest_two_sided"],
        "p_wilcoxon_greater": row["p_wilcoxon_greater"],
    }


def circular_rows() -> list[dict]:
    path = ROOT / "results/journal_neurolinguistics_robustness/circular_shift_null/circular_shift_summary.csv"
    df = pd.read_csv(path)
    rows = []
    for measure in ["real_scan_level_delta", "paired_scan_real_minus_mean_null_delta"]:
        row = df.loc[df["measure"] == measure].iloc[0].to_dict()
        rows.append(
            {
                "analysis": f"circular_shift_{measure}",
                "source": str(path.relative_to(ROOT)),
                "n": int(row["n"]),
                "mean_delta_cv_r": row["mean"],
                "median_delta_cv_r": row["median"],
                "ci95_low": row["ci95_low"],
                "ci95_high": row["ci95_high"],
                "t_stat": row["t_stat"],
                "p_ttest_two_sided": row["p_ttest_two_sided"],
                "p_wilcoxon_greater": row["p_wilcoxon_greater"],
            }
        )
    return rows


def main() -> None:
    rows = [
        pc5_row(
            "corrected_primary_pc1_to_pc5",
            "results/journal_neurolinguistics_robustness/pc_count_full_corrected",
        ),
        pc5_row(
            "corrected_cv_gap_5_pc1_to_pc5",
            "results/journal_neurolinguistics_robustness/cv_gap_5_corrected",
        ),
        pc5_row(
            "corrected_cv_gap_10_pc1_to_pc5",
            "results/journal_neurolinguistics_robustness/cv_gap_10_corrected",
        ),
        pc5_row(
            "corrected_atlas_threshold_25_pc1_to_pc5",
            "results/journal_neurolinguistics_robustness/atlas_threshold_25_corrected",
        ),
        pc5_row(
            "corrected_atlas_threshold_50_pc1_to_pc5",
            "results/journal_neurolinguistics_robustness/atlas_threshold_50_corrected",
        ),
        pc5_row(
            "corrected_zero_empty_tr_plus_silence_pc1_to_pc5",
            "results/journal_neurolinguistics_robustness/empty_tr_zero_silence_corrected",
        ),
    ]
    rows.extend(circular_rows())

    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "implementation_sensitivity_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved: {OUT / 'implementation_sensitivity_summary.csv'}")


if __name__ == "__main__":
    main()
