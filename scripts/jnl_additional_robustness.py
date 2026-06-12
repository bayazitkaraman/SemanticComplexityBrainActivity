from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "journal_neurolinguistics_robustness"
TARGET = "delta_cv_r_combined_pc1_to_pc5_minus_baseline"


def ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def summarize_delta(values: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    t_stat, p_t = stats.ttest_1samp(values, 0.0)
    try:
        w = stats.wilcoxon(values, alternative="greater", zero_method="wilcox")
        w_stat, p_w = float(w.statistic), float(w.pvalue)
    except ValueError:
        w_stat, p_w = np.nan, np.nan
    ci = stats.bootstrap(
        (values.to_numpy(),),
        np.mean,
        confidence_level=0.95,
        n_resamples=10000,
        random_state=42,
        method="basic",
    ).confidence_interval
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(ddof=1)),
        "sem": float(stats.sem(values)),
        "ci95_low": float(ci.low),
        "ci95_high": float(ci.high),
        "t_stat": float(t_stat),
        "p_ttest_two_sided": float(p_t),
        "wilcoxon_stat": w_stat,
        "p_wilcoxon_greater": p_w,
    }


def load_scan_level_delta(path: Path, target: str = TARGET) -> pd.DataFrame:
    df = pd.read_csv(path)
    if target not in df.columns:
        raise KeyError(f"{target} not in {path}")
    return (
        df.groupby(["story", "subject"], as_index=False)[target]
        .mean()
        .rename(columns={target: "mean_roi_delta_cv_r"})
    )


def mixed_effects_primary() -> pd.DataFrame:
    scan = load_scan_level_delta(
        RESULTS / "model_comparison_gentle_motionQC_only" / "encoding_model_deltas.csv"
    )
    scan.to_csv(OUT / "primary_scan_level_delta_for_mixed_effects.csv", index=False)

    rows = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        crossed = scan.copy()
        crossed["all_scans"] = "all"
        model = smf.mixedlm(
            "mean_roi_delta_cv_r ~ 1",
            crossed,
            groups=crossed["all_scans"],
            vc_formula={
                "subject": "0 + C(subject)",
                "story": "0 + C(story)",
            },
        )
        result = model.fit(reml=True, method="lbfgs", maxiter=1000, disp=False)
    ci = result.conf_int().loc["Intercept"]
    rows.append(
        {
            "model": "delta_cv_r ~ 1 + (1 | subject) + (1 | story)",
            "n_subject_story_scans": int(scan.shape[0]),
            "n_subjects": int(scan["subject"].nunique()),
            "n_stories": int(scan["story"].nunique()),
            "intercept": float(result.params["Intercept"]),
            "se": float(result.bse["Intercept"]),
            "z": float(result.tvalues["Intercept"]),
            "p": float(result.pvalues["Intercept"]),
            "ci95_low": float(ci[0]),
            "ci95_high": float(ci[1]),
            "converged": bool(result.converged),
            "warnings": " | ".join(str(w.message) for w in caught),
        }
    )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "mixed_effects_primary_delta.csv", index=False)
    return out


def motion_delta_tests() -> pd.DataFrame:
    run_dirs = {
        "Motion-QC only": RESULTS / "model_comparison_gentle_motionQC_only",
        "Motion6, no HPF": RESULTS / "model_comparison_gentle_motion6_nohpf",
        "Motion24, no HPF": RESULTS / "model_comparison_gentle_motion24_nohpf",
    }
    rows = []
    for label, directory in run_dirs.items():
        scan = load_scan_level_delta(directory / "encoding_model_deltas.csv")
        stats_row = summarize_delta(scan["mean_roi_delta_cv_r"])
        stats_row["strategy"] = label
        rows.append(stats_row)
    out = pd.DataFrame(rows)
    cols = ["strategy"] + [c for c in out.columns if c != "strategy"]
    out = out[cols]
    out.to_csv(OUT / "motion_strategy_scan_level_delta_tests.csv", index=False)
    return out


def story_robustness() -> tuple[pd.DataFrame, pd.DataFrame]:
    scan = load_scan_level_delta(
        RESULTS / "model_comparison_gentle_motionQC_only" / "encoding_model_deltas.csv"
    )

    story = (
        scan.groupby("story")["mean_roi_delta_cv_r"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    story.to_csv(OUT / "story_level_primary_delta.csv", index=False)

    rows = []
    for held_out_story in sorted(scan["story"].unique()):
        kept = scan.loc[scan["story"] != held_out_story, "mean_roi_delta_cv_r"]
        stats_row = summarize_delta(kept)
        stats_row["held_out_story"] = held_out_story
        rows.append(stats_row)
    leave_one = pd.DataFrame(rows)
    leave_one = leave_one[
        ["held_out_story"] + [c for c in leave_one.columns if c != "held_out_story"]
    ]
    leave_one.to_csv(OUT / "leave_one_story_out_primary_delta.csv", index=False)
    return story, leave_one


def main() -> None:
    ensure_out()
    mixed = mixed_effects_primary()
    motion = motion_delta_tests()
    story, leave_one = story_robustness()
    print("Mixed effects:")
    print(mixed.to_string(index=False))
    print("\nMotion delta tests:")
    print(motion.to_string(index=False))
    print("\nStory-level primary deltas:")
    print(story.to_string(index=False))
    print("\nLeave-one-story-out primary deltas:")
    print(leave_one.to_string(index=False))
    print(f"\nSaved outputs in {OUT}")


if __name__ == "__main__":
    main()
