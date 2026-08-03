from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from roi_encoding_model_comparison import load_regressors  # noqa: E402
from semantic_fmri.utils import (  # noqa: E402
    ROI_LABELS_TO_TEST,
    extract_roi_timeseries,
    fetch_harvard_oxford_atlas,
    load_story_list,
)


LEXICAL_BASELINE_COLS = [
    "word_count_hrf",
    "mean_char_len_hrf",
    "total_chars_hrf",
    "time_TR",
]


def limit_subjects_per_story(stories: list[dict], max_subjects_per_story: int) -> list[dict]:
    if max_subjects_per_story <= 0:
        return stories
    counts: defaultdict[str, int] = defaultdict(int)
    selected = []
    for story in stories:
        if counts[story["name"]] < max_subjects_per_story:
            selected.append(story)
            counts[story["name"]] += 1
    return selected


def bootstrap_mean_ci(values: np.ndarray) -> tuple[float, float]:
    if values.size < 2:
        return np.nan, np.nan
    ci = stats.bootstrap(
        (values,),
        np.mean,
        confidence_level=0.95,
        n_resamples=10000,
        random_state=42,
        method="basic",
    ).confidence_interval
    return float(ci.low), float(ci.high)


def summarize(values: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "sem": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "t_stat": np.nan,
            "p_ttest_two_sided": np.nan,
            "wilcoxon_stat": np.nan,
            "p_wilcoxon_greater": np.nan,
        }
    t_stat, p_t = stats.ttest_1samp(values, 0.0)
    try:
        w = stats.wilcoxon(values, alternative="greater", zero_method="wilcox")
        w_stat, p_w = float(w.statistic), float(w.pvalue)
    except ValueError:
        w_stat, p_w = np.nan, np.nan
    ci_low, ci_high = bootstrap_mean_ci(values)
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
        "sem": float(stats.sem(values)) if values.size > 1 else np.nan,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "t_stat": float(t_stat),
        "p_ttest_two_sided": float(p_t),
        "wilcoxon_stat": w_stat,
        "p_wilcoxon_greater": p_w,
    }


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def load_or_extract_roi_matrix(
    story: dict,
    atlas_img,
    atlas_labels,
    cache_dir: Path,
    atlas_threshold: float = 0.0,
) -> tuple[list[str], np.ndarray, float, float, float]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{safe_name(story['name'])}_{safe_name(story['subject'])}.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        return (
            [str(v) for v in cached["roi_names"].tolist()],
            cached["y_matrix"],
            float(cached["bold_tr"]),
            float(cached["onset"]),
            float(cached["duration"]),
        )

    roi_ts, bold_tr, onset, duration, _ = extract_roi_timeseries(
        story,
        roi_labels=ROI_LABELS_TO_TEST,
        atlas_img=atlas_img,
        atlas_labels=atlas_labels,
        fwhm=6.0,
        atlas_threshold=atlas_threshold,
    )
    roi_names = list(roi_ts.keys())
    y_matrix = np.column_stack([roi_ts[roi] for roi in roi_names])
    np.savez_compressed(
        cache_path,
        roi_names=np.asarray(roi_names),
        y_matrix=y_matrix,
        bold_tr=float(bold_tr),
        onset=float(onset),
        duration=float(duration),
    )
    return roi_names, y_matrix, bold_tr, onset, duration


def foldwise_target_matrix(
    y_matrix: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    do_detrend: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Detrend and scale target columns using training-fold statistics only."""
    y_train = y_matrix[train_idx, :].astype(float, copy=True)
    y_test = y_matrix[test_idx, :].astype(float, copy=True)

    if do_detrend and len(train_idx) >= 2:
        x_train = train_idx.astype(float)
        x_test = test_idx.astype(float)
        design_train = np.column_stack([x_train, np.ones_like(x_train)])
        beta = np.linalg.lstsq(design_train, y_train, rcond=None)[0]
        y_train = y_train - design_train @ beta
        design_test = np.column_stack([x_test, np.ones_like(x_test)])
        y_test = y_test - design_test @ beta

    mu = np.mean(y_train, axis=0, keepdims=True)
    sd = np.std(y_train, axis=0, keepdims=True)
    safe_sd = np.where((sd > 0) & np.isfinite(sd), sd, 1.0)
    y_train = (y_train - mu) / safe_sd
    y_test = (y_test - mu) / safe_sd
    return y_train, y_test


def vectorized_cv_r(
    x_matrix: np.ndarray,
    y_matrix: np.ndarray,
    n_splits: int = 5,
    alpha: float = 10.0,
    do_detrend: bool = True,
    cv_gap: int = 0,
) -> tuple[np.ndarray, int]:
    x_matrix = np.asarray(x_matrix, dtype=float)
    y_matrix = np.asarray(y_matrix, dtype=float)

    n = min(len(x_matrix), len(y_matrix))
    x_matrix, y_matrix = x_matrix[:n], y_matrix[:n]
    valid = np.all(np.isfinite(x_matrix), axis=1) & np.all(np.isfinite(y_matrix), axis=1)
    x_matrix, y_matrix = x_matrix[valid], y_matrix[valid]

    if len(y_matrix) < 30:
        return np.full(y_matrix.shape[1], np.nan), len(y_matrix)

    n_splits = min(n_splits, max(2, len(y_matrix) // 40))
    if n_splits < 2:
        return np.full(y_matrix.shape[1], np.nan), len(y_matrix)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, fit_intercept=True)),
        ]
    )
    true_parts = []
    pred_parts = []
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=max(0, int(cv_gap)))
    for train_idx, test_idx in splitter.split(x_matrix):
        if len(test_idx) < 5:
            continue
        y_train, y_test = foldwise_target_matrix(
            y_matrix, train_idx, test_idx, do_detrend=do_detrend
        )
        model.fit(x_matrix[train_idx], y_train)
        true_parts.append(y_test)
        pred_parts.append(model.predict(x_matrix[test_idx]))

    if not true_parts:
        return np.full(y_matrix.shape[1], np.nan), len(y_matrix)

    y_true = np.vstack(true_parts)
    y_pred = np.vstack(pred_parts)
    true_centered = y_true - np.nanmean(y_true, axis=0, keepdims=True)
    pred_centered = y_pred - np.nanmean(y_pred, axis=0, keepdims=True)
    numerator = np.nansum(true_centered * pred_centered, axis=0)
    denominator = np.sqrt(
        np.nansum(true_centered**2, axis=0) * np.nansum(pred_centered**2, axis=0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        cv_r = numerator / denominator
    cv_r[denominator == 0] = np.nan
    return cv_r, len(y_matrix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject-list",
        default="configs/stories_all_fmriprep_confounds_motionfiltered.py",
    )
    parser.add_argument("--regressor-dir", default="results/regressors")
    parser.add_argument(
        "--output-dir",
        default="results/journal_neurolinguistics_robustness",
    )
    parser.add_argument(
        "--roi-cache-dir",
        default="",
        help="Optional existing ROI cache. Leave blank to use output-dir/roi_time_series_cache.",
    )
    parser.add_argument("--max-pcs", type=int, default=10)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--max-subjects-per-story", type=int, default=0)
    parser.add_argument("--no-detrend", action="store_true")
    parser.add_argument("--cv-gap", type=int, default=0)
    parser.add_argument("--atlas-threshold", type=float, default=0.0)
    parser.add_argument(
        "--include-silence-indicator",
        action="store_true",
        help="Add an HRF-convolved empty-TR indicator to the lexical/time baseline if present.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = ROOT / args.roi_cache_dir if args.roi_cache_dir else output_dir / "roi_time_series_cache"
    long_path = output_dir / "pc_count_sensitivity_long.csv"

    stories = load_story_list(str(ROOT / args.subject_list))
    stories = limit_subjects_per_story(stories, args.max_subjects_per_story)

    rows = []
    completed_pairs: set[tuple[str, str]] = set()
    expected_rows_per_pair = len(ROI_LABELS_TO_TEST) * args.max_pcs
    if args.resume and long_path.exists():
        existing = pd.read_csv(long_path)
        rows = existing.to_dict("records")
        counts = existing.groupby(["story", "subject"]).size()
        completed_pairs = {
            (story, subject)
            for (story, subject), count in counts.items()
            if int(count) >= expected_rows_per_pair
        }
        print(f"Resuming with {len(completed_pairs)} completed subject-story rows")

    print(f"Subject-story rows: {len(stories)}")
    print(f"Testing combined baseline + PC1..PCn for n=1..{args.max_pcs}")
    print(f"Output directory: {output_dir}")

    atlas_img, atlas_labels = fetch_harvard_oxford_atlas()
    baseline_cols = list(LEXICAL_BASELINE_COLS)
    if args.include_silence_indicator:
        baseline_cols.append("silence_indicator_hrf")

    for i, story in enumerate(stories, start=1):
        pair = (story["name"], story["subject"])
        if pair in completed_pairs:
            print(f"[{i}/{len(stories)}] {story['name']} {story['subject']} (cached result)", flush=True)
            continue
        print(f"[{i}/{len(stories)}] {story['name']} {story['subject']}", flush=True)
        reg_df = load_regressors(ROOT / args.regressor_dir, story["name"])
        pc_cols = [f"PC{j}_hrf" for j in range(1, args.max_pcs + 1)]
        missing = [c for c in baseline_cols + pc_cols if c not in reg_df.columns]
        if missing:
            raise ValueError(f"Missing columns in {story['name']} regressors: {missing}")

        roi_names, y_matrix, bold_tr, onset, duration = load_or_extract_roi_matrix(
            story, atlas_img, atlas_labels, cache_dir, atlas_threshold=args.atlas_threshold
        )
        baseline_x = reg_df[baseline_cols].to_numpy()
        combined_x_by_n = {
            n: reg_df[baseline_cols + pc_cols[:n]].to_numpy()
            for n in range(1, args.max_pcs + 1)
        }

        baseline_cv_r, n_used = vectorized_cv_r(
            baseline_x,
            y_matrix,
            n_splits=args.n_splits,
            alpha=args.ridge_alpha,
            do_detrend=not args.no_detrend,
            cv_gap=args.cv_gap,
        )
        combined_cv_r_by_n = {
            n_pcs: vectorized_cv_r(
                combined_x,
                y_matrix,
                n_splits=args.n_splits,
                alpha=args.ridge_alpha,
                do_detrend=not args.no_detrend,
                cv_gap=args.cv_gap,
            )[0]
            for n_pcs, combined_x in combined_x_by_n.items()
        }
        for roi_idx, roi in enumerate(roi_names):
            for n_pcs, combined_x in combined_x_by_n.items():
                combined_cv_r = combined_cv_r_by_n[n_pcs][roi_idx]
                rows.append(
                    {
                        "story": story["name"],
                        "subject": story["subject"],
                        "roi": roi,
                        "n_pcs": n_pcs,
                        "baseline_cv_r": baseline_cv_r[roi_idx],
                        "combined_cv_r": combined_cv_r,
                        "delta_cv_r": combined_cv_r - baseline_cv_r[roi_idx],
                        "n_timepoints": n_used,
                        "TR": bold_tr,
                        "duration": duration,
                        "onset": onset,
                        "atlas_threshold": args.atlas_threshold,
                        "cv_gap": args.cv_gap,
                        "include_silence_indicator": bool(args.include_silence_indicator),
                    }
                )
        if args.checkpoint_every > 0 and i % args.checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(long_path, index=False)
            print(f"Checkpoint saved: {long_path}", flush=True)

    long = pd.DataFrame(rows)
    long.to_csv(long_path, index=False)
    print(f"Saved: {long_path}")

    scan = (
        long.groupby(["n_pcs", "story", "subject"], as_index=False)
        .agg(mean_roi_delta_cv_r=("delta_cv_r", "mean"))
        .sort_values(["n_pcs", "story", "subject"])
    )
    scan_path = output_dir / "pc_count_sensitivity_scan_level.csv"
    scan.to_csv(scan_path, index=False)
    print(f"Saved: {scan_path}")

    scan_stats = []
    for n_pcs, group in scan.groupby("n_pcs"):
        row = summarize(group["mean_roi_delta_cv_r"])
        row["n_pcs"] = int(n_pcs)
        scan_stats.append(row)
    scan_stats_df = pd.DataFrame(scan_stats)
    scan_stats_df = scan_stats_df[["n_pcs"] + [c for c in scan_stats_df.columns if c != "n_pcs"]]
    scan_stats_path = output_dir / "pc_count_sensitivity_scan_stats.csv"
    scan_stats_df.to_csv(scan_stats_path, index=False)
    print(f"Saved: {scan_stats_path}")

    participant = (
        long.groupby(["n_pcs", "subject"], as_index=False)
        .agg(mean_delta_cv_r=("delta_cv_r", "mean"))
        .sort_values(["n_pcs", "subject"])
    )
    participant_path = output_dir / "pc_count_sensitivity_participant_level.csv"
    participant.to_csv(participant_path, index=False)
    print(f"Saved: {participant_path}")

    participant_stats = []
    for n_pcs, group in participant.groupby("n_pcs"):
        row = summarize(group["mean_delta_cv_r"])
        row["n_pcs"] = int(n_pcs)
        participant_stats.append(row)
    participant_stats_df = pd.DataFrame(participant_stats)
    participant_stats_df = participant_stats_df[
        ["n_pcs"] + [c for c in participant_stats_df.columns if c != "n_pcs"]
    ]
    participant_stats_path = output_dir / "pc_count_sensitivity_participant_stats.csv"
    participant_stats_df.to_csv(participant_stats_path, index=False)
    print(f"Saved: {participant_stats_path}")

    roi_stats = []
    for (n_pcs, roi), group in long.groupby(["n_pcs", "roi"]):
        row = summarize(group["delta_cv_r"])
        row["n_pcs"] = int(n_pcs)
        row["roi"] = roi
        roi_stats.append(row)
    roi_stats_df = pd.DataFrame(roi_stats)
    roi_stats_df = roi_stats_df[
        ["n_pcs", "roi"] + [c for c in roi_stats_df.columns if c not in {"n_pcs", "roi"}]
    ]
    roi_stats_path = output_dir / "pc_count_sensitivity_roi_stats.csv"
    roi_stats_df.to_csv(roi_stats_path, index=False)
    print(f"Saved: {roi_stats_path}")

    print("\nScan-level summary:")
    print(scan_stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
