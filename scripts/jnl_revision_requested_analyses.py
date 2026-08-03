from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from jnl_pc_count_sensitivity import (  # noqa: E402
    LEXICAL_BASELINE_COLS,
    safe_name,
    summarize,
    vectorized_cv_r,
)
from roi_encoding_model_comparison import load_regressors  # noqa: E402
from semantic_fmri.utils import load_story_list  # noqa: E402


TARGET_DELTA = "delta_cv_r_combined_pc1_to_pc5_minus_baseline"
PRIMARY_MODEL = "combined_pc1_to_pc5"
BASELINE_MODEL = "baseline_only"


def scan_level_from_delta_file(path: Path, target: str = TARGET_DELTA) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"story", "subject", "roi", BASELINE_MODEL, PRIMARY_MODEL, target}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return (
        df.groupby(["story", "subject"], as_index=False)
        .agg(
            mean_roi_baseline_cv_r=(BASELINE_MODEL, "mean"),
            mean_roi_combined_cv_r=(PRIMARY_MODEL, "mean"),
            mean_roi_delta_cv_r=(target, "mean"),
            n_rois=("roi", "count"),
        )
        .sort_values(["story", "subject"])
    )


def mixed_effect_intercept(scan: pd.DataFrame, value_col: str, label: str) -> dict:
    data = scan[["story", "subject", value_col]].dropna().copy()
    data = data.rename(columns={value_col: "y"})
    data["all_scans"] = "all"
    model = smf.mixedlm(
        "y ~ 1",
        data,
        groups=data["all_scans"],
        vc_formula={"subject": "0 + C(subject)", "story": "0 + C(story)"},
    )
    caught_messages = []
    result = None
    for method in ["lbfgs", "powell", "cg", "bfgs", "nm"]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            candidate = model.fit(reml=True, method=method, maxiter=2000, disp=False)
        caught_messages.extend(f"{method}: {w.message}" for w in caught)
        result = candidate
        if bool(candidate.converged):
            break
    if result is None:
        raise RuntimeError("Mixed-effects fit failed before returning a result.")
    ci = result.conf_int().loc["Intercept"]
    estimate = float(result.params["Intercept"])
    z_value = float(result.tvalues["Intercept"])
    p_two_sided = float(result.pvalues["Intercept"])
    return {
        "analysis": label,
        "model": "y ~ 1 + (1 | subject) + (1 | story)",
        "n_subject_story_scans": int(data.shape[0]),
        "n_subjects": int(data["subject"].nunique()),
        "n_stories": int(data["story"].nunique()),
        "estimate": estimate,
        "se": float(result.bse["Intercept"]),
        "z": z_value,
        "p": p_two_sided,
        "p_two_sided": p_two_sided,
        "p_one_sided_positive": float(stats.norm.sf(z_value)),
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
        "converged": bool(result.converged),
        "warnings": " | ".join(caught_messages),
    }


def add_summary_prefix(row: dict, prefix: str) -> dict:
    return {f"{prefix}_{key}": value for key, value in row.items()}


def primary_inference_tables(primary_dir: Path, motion_dirs: dict[str, Path], out_dir: Path) -> None:
    primary_scan = scan_level_from_delta_file(primary_dir / "encoding_model_deltas.csv")
    primary_scan.to_csv(out_dir / "primary_scan_level_model_values.csv", index=False)

    participant = (
        primary_scan.groupby("subject", as_index=False)
        .agg(mean_delta_cv_r=("mean_roi_delta_cv_r", "mean"))
        .sort_values("subject")
    )
    participant.to_csv(out_dir / "primary_participant_level_delta_values.csv", index=False)

    rows = []
    scan_stats = summarize(primary_scan["mean_roi_delta_cv_r"])
    participant_stats = summarize(participant["mean_delta_cv_r"])
    mixed = mixed_effect_intercept(primary_scan, "mean_roi_delta_cv_r", "primary_delta")
    rows.append(
        {
            "comparison": "combined PC1-PC5 minus lexical/time baseline",
            **add_summary_prefix(scan_stats, "scan"),
            **add_summary_prefix(participant_stats, "participant"),
            **{f"mixed_{k}": v for k, v in mixed.items() if k not in {"analysis", "model"}},
            "mixed_model": mixed["model"],
        }
    )
    pd.DataFrame(rows).to_csv(out_dir / "primary_inference_summary.csv", index=False)
    pd.DataFrame([mixed]).to_csv(out_dir / "primary_mixed_effects_crossed.csv", index=False)

    distribution_rows = []
    all_scan_rows = []
    for strategy, directory in motion_dirs.items():
        scan = scan_level_from_delta_file(directory / "encoding_model_deltas.csv")
        scan["strategy"] = strategy
        all_scan_rows.append(scan)
        values = scan["mean_roi_delta_cv_r"].dropna()
        distribution_rows.append(
            {
                "strategy": strategy,
                "n_scans": int(scan.shape[0]),
                "mean_delta_cv_r": float(values.mean()),
                "median_delta_cv_r": float(values.median()),
                "sd_delta_cv_r": float(values.std(ddof=1)),
                "min_delta_cv_r": float(values.min()),
                "q25_delta_cv_r": float(values.quantile(0.25)),
                "q75_delta_cv_r": float(values.quantile(0.75)),
                "max_delta_cv_r": float(values.max()),
                "fraction_delta_positive": float((scan["mean_roi_delta_cv_r"] > 0).mean()),
                "fraction_baseline_cv_r_positive": float((scan["mean_roi_baseline_cv_r"] > 0).mean()),
                "fraction_combined_cv_r_positive": float((scan["mean_roi_combined_cv_r"] > 0).mean()),
                "fraction_both_absolute_cv_r_negative": float(
                    ((scan["mean_roi_baseline_cv_r"] < 0) & (scan["mean_roi_combined_cv_r"] < 0)).mean()
                ),
            }
        )
    pd.concat(all_scan_rows, ignore_index=True).to_csv(
        out_dir / "motion_strategy_scan_level_model_values.csv", index=False
    )
    pd.DataFrame(distribution_rows).to_csv(
        out_dir / "motion_strategy_delta_distribution_summary.csv", index=False
    )


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text.lower())


def story_descriptor_tables(
    subject_list: Path,
    regressor_dir: Path,
    primary_scan_csv: Path,
    out_dir: Path,
) -> None:
    stories = load_story_list(str(subject_list))
    selected = {}
    for story in stories:
        selected.setdefault(story["name"], story)

    scan = pd.read_csv(primary_scan_csv)
    story_effect = (
        scan.groupby("story", as_index=False)
        .agg(
            n_subject_story_scans=("subject", "nunique"),
            mean_delta_cv_r=("mean_roi_delta_cv_r", "mean"),
            median_delta_cv_r=("mean_roi_delta_cv_r", "median"),
            sd_delta_cv_r=("mean_roi_delta_cv_r", "std"),
            fraction_delta_positive=("mean_roi_delta_cv_r", lambda x: float((x > 0).mean())),
        )
    )

    rows = []
    for story_name, story in sorted(selected.items()):
        reg = load_regressors(regressor_dir, story_name)
        transcript_path = ROOT / story["transcript_file"]
        words = tokenize(transcript_path.read_text(encoding="utf-8")) if transcript_path.exists() else []
        duration = float(reg["duration_sec"].iloc[0]) if "duration_sec" in reg.columns else np.nan
        n_trs = int(reg.shape[0])
        total_words = int(reg["word_count"].sum()) if "word_count" in reg.columns else len(words)
        nonempty = int((reg["word_count"] > 0).sum()) if "word_count" in reg.columns else np.nan
        speech_duration = float(reg["speech_duration_sec"].sum()) if "speech_duration_sec" in reg.columns else np.nan
        rows.append(
            {
                "story": story_name,
                "duration_sec": duration,
                "TR_sec": float(reg["TR"].iloc[0]) if "TR" in reg.columns else 1.5,
                "n_TRs": n_trs,
                "n_empty_TRs": int(n_trs - nonempty) if not pd.isna(nonempty) else np.nan,
                "fraction_empty_TRs": float(1.0 - nonempty / n_trs) if n_trs and not pd.isna(nonempty) else np.nan,
                "total_words_aligned_to_TRs": total_words,
                "word_rate_per_sec": float(total_words / duration) if duration and duration > 0 else np.nan,
                "mean_words_per_TR": float(total_words / n_trs) if n_trs else np.nan,
                "mean_words_per_nonempty_TR": float(total_words / nonempty) if nonempty else np.nan,
                "speech_duration_sec": speech_duration,
                "mean_pause_fraction": float(reg["pause_fraction"].mean()) if "pause_fraction" in reg.columns else np.nan,
                "mean_articulation_rate": float(reg["articulation_rate"].replace([np.inf, -np.inf], np.nan).mean())
                if "articulation_rate" in reg.columns
                else np.nan,
                "transcript_total_words": int(len(words)),
                "transcript_unique_words": int(len(set(words))),
                "transcript_type_token_ratio": float(len(set(words)) / len(words)) if words else np.nan,
            }
        )

    descriptors = pd.DataFrame(rows).merge(story_effect, on="story", how="left")
    descriptors = descriptors.sort_values("mean_delta_cv_r", ascending=False)
    descriptors.to_csv(out_dir / "story_descriptors_and_primary_delta.csv", index=False)

    corr_rows = []
    descriptor_cols = [
        "duration_sec",
        "n_TRs",
        "fraction_empty_TRs",
        "word_rate_per_sec",
        "mean_words_per_TR",
        "mean_words_per_nonempty_TR",
        "mean_pause_fraction",
        "mean_articulation_rate",
        "transcript_type_token_ratio",
    ]
    for col in descriptor_cols:
        sub = descriptors[[col, "mean_delta_cv_r"]].replace([np.inf, -np.inf], np.nan).dropna()
        if sub.shape[0] >= 4 and sub[col].std(ddof=0) > 0:
            r, p = stats.pearsonr(sub[col], sub["mean_delta_cv_r"])
            rho, p_s = stats.spearmanr(sub[col], sub["mean_delta_cv_r"])
        else:
            r = p = rho = p_s = np.nan
        corr_rows.append(
            {
                "descriptor": col,
                "n_stories": int(sub.shape[0]),
                "pearson_r": float(r) if not pd.isna(r) else np.nan,
                "pearson_p": float(p) if not pd.isna(p) else np.nan,
                "spearman_rho": float(rho) if not pd.isna(rho) else np.nan,
                "spearman_p": float(p_s) if not pd.isna(p_s) else np.nan,
            }
        )
    pd.DataFrame(corr_rows).to_csv(out_dir / "story_descriptor_correlations.csv", index=False)


def alpha_sensitivity(
    subject_list: Path,
    regressor_dir: Path,
    roi_cache_dir: Path,
    out_dir: Path,
    alphas: list[float],
    n_splits: int,
) -> None:
    stories = load_story_list(str(subject_list))
    rows = []
    for i, story in enumerate(stories, start=1):
        print(f"[alpha {i}/{len(stories)}] {story['name']} {story['subject']}", flush=True)
        cache_path = roi_cache_dir / f"{safe_name(story['name'])}_{safe_name(story['subject'])}.npz"
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing ROI cache: {cache_path}")
        cached = np.load(cache_path, allow_pickle=False)
        roi_names = [str(v) for v in cached["roi_names"].tolist()]
        y_matrix = cached["y_matrix"]
        reg = load_regressors(regressor_dir, story["name"])
        pc_cols = [f"PC{j}_hrf" for j in range(1, 6)]
        baseline_x = reg[LEXICAL_BASELINE_COLS].to_numpy()
        combined_x = reg[LEXICAL_BASELINE_COLS + pc_cols].to_numpy()
        for alpha in alphas:
            baseline_cv_r, n_used = vectorized_cv_r(
                baseline_x, y_matrix, n_splits=n_splits, alpha=alpha
            )
            combined_cv_r, _ = vectorized_cv_r(
                combined_x, y_matrix, n_splits=n_splits, alpha=alpha
            )
            for roi_idx, roi in enumerate(roi_names):
                rows.append(
                    {
                        "alpha": alpha,
                        "story": story["name"],
                        "subject": story["subject"],
                        "roi": roi,
                        "baseline_cv_r": baseline_cv_r[roi_idx],
                        "combined_pc1_to_pc5_cv_r": combined_cv_r[roi_idx],
                        "delta_cv_r": combined_cv_r[roi_idx] - baseline_cv_r[roi_idx],
                        "n_timepoints": int(n_used),
                    }
                )
    long = pd.DataFrame(rows)
    long.to_csv(out_dir / "alpha_sensitivity_long.csv", index=False)
    scan = (
        long.groupby(["alpha", "story", "subject"], as_index=False)
        .agg(
            mean_roi_baseline_cv_r=("baseline_cv_r", "mean"),
            mean_roi_combined_cv_r=("combined_pc1_to_pc5_cv_r", "mean"),
            mean_roi_delta_cv_r=("delta_cv_r", "mean"),
        )
        .sort_values(["alpha", "story", "subject"])
    )
    scan.to_csv(out_dir / "alpha_sensitivity_scan_level.csv", index=False)

    stats_rows = []
    mixed_rows = []
    for alpha, group in scan.groupby("alpha"):
        row = summarize(group["mean_roi_delta_cv_r"])
        row["alpha"] = alpha
        row["fraction_delta_positive"] = float((group["mean_roi_delta_cv_r"] > 0).mean())
        row["fraction_combined_cv_r_positive"] = float((group["mean_roi_combined_cv_r"] > 0).mean())
        stats_rows.append(row)
        mixed_rows.append(
            mixed_effect_intercept(
                group,
                "mean_roi_delta_cv_r",
                f"alpha_{alpha:g}_combined_pc1_to_pc5_minus_baseline",
            )
        )
    stats_df = pd.DataFrame(stats_rows)
    stats_df = stats_df[["alpha"] + [c for c in stats_df.columns if c != "alpha"]]
    stats_df.to_csv(out_dir / "alpha_sensitivity_scan_stats.csv", index=False)
    pd.DataFrame(mixed_rows).to_csv(out_dir / "alpha_sensitivity_mixed_effects.csv", index=False)


def pc_count_mixed_effects(pc_scan_csv: Path, out_dir: Path) -> None:
    scan = pd.read_csv(pc_scan_csv)
    mixed_rows = []
    for n_pcs, group in scan.groupby("n_pcs"):
        mixed_rows.append(
            mixed_effect_intercept(
                group.rename(columns={"mean_roi_delta_cv_r": "delta"}),
                "delta",
                f"pc1_to_pc{int(n_pcs)}_minus_baseline",
            )
        )
    pd.DataFrame(mixed_rows).to_csv(out_dir / "pc_count_sensitivity_mixed_effects.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject-list",
        default="configs/stories_all_fmriprep_confounds_motionfiltered.py",
    )
    parser.add_argument("--regressor-dir", default="results/regressors")
    parser.add_argument("--primary-dir", default="results/model_comparison_gentle_motionQC_only")
    parser.add_argument("--motion6-dir", default="results/model_comparison_gentle_motion6_nohpf")
    parser.add_argument("--motion24-dir", default="results/model_comparison_gentle_motion24_nohpf")
    parser.add_argument(
        "--roi-cache-dir",
        default="results/journal_neurolinguistics_robustness/pc_count_full_corrected/roi_time_series_cache",
    )
    parser.add_argument(
        "--pc-scan-csv",
        default="results/journal_neurolinguistics_robustness/pc_count_full_corrected/pc_count_sensitivity_scan_level.csv",
    )
    parser.add_argument("--output-dir", default="results/jnl_revision_major_review")
    parser.add_argument("--alphas", default="0.1,1,10,100,1000")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--skip-alpha", action="store_true")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    subject_list = ROOT / args.subject_list
    regressor_dir = ROOT / args.regressor_dir
    primary_dir = ROOT / args.primary_dir
    motion_dirs = {
        "Motion-QC only": primary_dir,
        "Motion6, no high-pass": ROOT / args.motion6_dir,
        "Motion24, no high-pass": ROOT / args.motion24_dir,
    }

    primary_inference_tables(primary_dir, motion_dirs, out_dir)
    story_descriptor_tables(
        subject_list,
        regressor_dir,
        out_dir / "primary_scan_level_model_values.csv",
        out_dir,
    )
    pc_count_mixed_effects(ROOT / args.pc_scan_csv, out_dir)

    if not args.skip_alpha:
        alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
        alpha_sensitivity(
            subject_list,
            regressor_dir,
            ROOT / args.roi_cache_dir,
            out_dir,
            alphas=alphas,
            n_splits=args.n_splits,
        )

    print(f"Saved revision analysis tables in {out_dir}")


if __name__ == "__main__":
    main()
