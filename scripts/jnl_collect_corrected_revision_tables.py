from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final_tables"

DELTA = "delta_cv_r_combined_pc1_to_pc5_minus_baseline"
BASELINE = "baseline_only"
COMBINED = "combined_pc1_to_pc5"


def summarize(values: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if x.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "sem": np.nan,
            "t": np.nan,
            "t_p_two_sided": np.nan,
            "wilcoxon_p_greater": np.nan,
        }
    t = stats.ttest_1samp(x, 0.0)
    try:
        w = stats.wilcoxon(x, alternative="greater", zero_method="wilcox")
        p_w = float(w.pvalue)
    except ValueError:
        p_w = np.nan
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else np.nan,
        "sem": float(stats.sem(x)) if x.size > 1 else np.nan,
        "t": float(t.statistic),
        "t_p_two_sided": float(t.pvalue),
        "wilcoxon_p_greater": p_w,
    }


def scan_delta_table(model_dir: Path, label: str) -> pd.DataFrame:
    delta = pd.read_csv(model_dir / "encoding_model_deltas.csv")
    required = {"story", "subject", BASELINE, COMBINED, DELTA}
    missing = required.difference(delta.columns)
    if missing:
        raise ValueError(f"{model_dir} is missing columns: {sorted(missing)}")
    scan = (
        delta.groupby(["story", "subject"], as_index=False)
        .agg(
            baseline_mean_cv_r=(BASELINE, "mean"),
            combined_pc1_to_pc5_mean_cv_r=(COMBINED, "mean"),
            scan_delta_cv_r=(DELTA, "mean"),
        )
        .sort_values(["story", "subject"])
    )
    scan["model"] = label
    return scan


def write_embedding_tables() -> None:
    models = [
        (
            "GPT-2 final-layer contextual PC1-PC5",
            ROOT / "results" / "model_comparison_gentle_motionQC_only",
        ),
        (
            "BERT-base static input-embedding PC1-PC5",
            ROOT / "results" / "model_comparison_static_bert_base_uncased_gentle_motionQC_only",
        ),
        (
            "BERT-base contextual final-layer PC1-PC5",
            ROOT / "results" / "model_comparison_contextual_bert_base_uncased_gentle_motionQC_only",
        ),
    ]

    scans = [scan_delta_table(path, label) for label, path in models]
    rows = []
    for scan in scans:
        label = str(scan["model"].iloc[0])
        participant = (
            scan.groupby("subject", as_index=False)["scan_delta_cv_r"]
            .mean()
            .rename(columns={"scan_delta_cv_r": "participant_delta_cv_r"})
        )
        scan_stats = summarize(scan["scan_delta_cv_r"])
        participant_stats = summarize(participant["participant_delta_cv_r"])
        rows.append(
            {
                "model": label,
                "n_scans": int(scan.shape[0]),
                "n_subjects": int(scan["subject"].nunique()),
                "baseline_mean_cv_r": float(scan["baseline_mean_cv_r"].mean()),
                "combined_pc1_to_pc5_mean_cv_r": float(scan["combined_pc1_to_pc5_mean_cv_r"].mean()),
                "scan_mean_delta_cv_r": scan_stats["mean"],
                "scan_median_delta_cv_r": scan_stats["median"],
                "scan_sd_delta_cv_r": scan_stats["std"],
                "scan_positive_n": int((scan["scan_delta_cv_r"] > 0).sum()),
                "scan_t": scan_stats["t"],
                "scan_t_p_two_sided": scan_stats["t_p_two_sided"],
                "scan_wilcoxon_p_greater": scan_stats["wilcoxon_p_greater"],
                "participant_mean_delta_cv_r": participant_stats["mean"],
                "participant_median_delta_cv_r": participant_stats["median"],
                "participant_t_p_two_sided": participant_stats["t_p_two_sided"],
                "participant_wilcoxon_p_greater": participant_stats["wilcoxon_p_greater"],
            }
        )

    pd.DataFrame(rows).to_csv(
        FINAL / "table13_embedding_model_comparison_summary.csv", index=False
    )

    wide = scans[0][["story", "subject", "scan_delta_cv_r"]].rename(
        columns={"scan_delta_cv_r": "gpt2"}
    )
    wide = wide.merge(
        scans[1][["story", "subject", "scan_delta_cv_r"]].rename(
            columns={"scan_delta_cv_r": "bert_static"}
        ),
        on=["story", "subject"],
        how="inner",
    )
    wide = wide.merge(
        scans[2][["story", "subject", "scan_delta_cv_r"]].rename(
            columns={"scan_delta_cv_r": "bert_contextual"}
        ),
        on=["story", "subject"],
        how="inner",
    )

    comparisons = [
        ("bert_contextual minus gpt2", "bert_contextual", "gpt2"),
        ("bert_contextual minus bert_static", "bert_contextual", "bert_static"),
        ("bert_static minus gpt2", "bert_static", "gpt2"),
    ]
    comp_rows = []
    for label, first, second in comparisons:
        diff = wide[first] - wide[second]
        t = stats.ttest_1samp(diff, 0.0)
        try:
            w_first = stats.wilcoxon(diff, alternative="greater", zero_method="wilcox")
            w_second = stats.wilcoxon(-diff, alternative="greater", zero_method="wilcox")
            p_first = float(w_first.pvalue)
            p_second = float(w_second.pvalue)
        except ValueError:
            p_first = p_second = np.nan
        comp_rows.append(
            {
                "comparison": label,
                "n_scans": int(diff.notna().sum()),
                "mean_difference": float(diff.mean()),
                "median_difference": float(diff.median()),
                "t": float(t.statistic),
                "t_p_two_sided": float(t.pvalue),
                "wilcoxon_p_first_greater": p_first,
                "wilcoxon_p_second_greater": p_second,
                "first_greater_n": int((diff > 0).sum()),
            }
        )
    pd.DataFrame(comp_rows).to_csv(
        FINAL / "table13b_embedding_model_paired_scan_comparisons.csv", index=False
    )


def copy_updated_tables() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    copies = [
        (
            ROOT / "results" / "journal_neurolinguistics_robustness" / "pc_count_full_corrected" / "pc_count_sensitivity_scan_stats.csv",
            FINAL / "table9_pc_count_sensitivity_scan_stats.csv",
        ),
        (
            ROOT / "results" / "journal_neurolinguistics_robustness" / "pc_count_full_corrected" / "pc_count_sensitivity_participant_stats.csv",
            FINAL / "table9b_pc_count_sensitivity_participant_stats.csv",
        ),
        (
            ROOT / "results" / "journal_neurolinguistics_robustness" / "motion_strategy_scan_level_delta_tests.csv",
            FINAL / "table10_motion_strategy_scan_level_delta_tests.csv",
        ),
        (
            ROOT / "results" / "journal_neurolinguistics_robustness" / "leave_one_story_out_primary_delta.csv",
            FINAL / "table11_leave_one_story_out_primary_delta.csv",
        ),
        (
            ROOT / "results" / "journal_neurolinguistics_robustness" / "circular_shift_null" / "circular_shift_summary.csv",
            FINAL / "table12_circular_shift_null_summary.csv",
        ),
    ]
    for src, dst in copies:
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copyfile(src, dst)


def main() -> None:
    copy_updated_tables()
    write_embedding_tables()
    print(f"Updated corrected JNL supplementary tables in {FINAL}")


if __name__ == "__main__":
    main()
