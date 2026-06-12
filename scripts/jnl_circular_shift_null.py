from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from jnl_pc_count_sensitivity import (
    LEXICAL_BASELINE_COLS,
    load_or_extract_roi_matrix,
    summarize,
    vectorized_cv_r,
)
from roi_encoding_model_comparison import load_regressors
from semantic_fmri.utils import fetch_harvard_oxford_atlas, load_story_list


ROOT = Path(__file__).resolve().parents[1]
PC_COLS = [f"PC{i}_hrf" for i in range(1, 6)]


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


def shift_bounds(n_timepoints: int, min_shift_trs: int) -> tuple[int, int]:
    if n_timepoints <= 2 * min_shift_trs + 1:
        lower = max(1, n_timepoints // 5)
        upper = max(lower + 1, n_timepoints - lower)
        return lower, upper
    return min_shift_trs, n_timepoints - min_shift_trs


def empirical_p_greater(observed: float, null_values: pd.Series) -> float:
    null = pd.to_numeric(null_values, errors="coerce").dropna().to_numpy(dtype=float)
    if null.size == 0:
        return np.nan
    return float((1 + np.sum(null >= observed)) / (null.size + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject-list",
        default="configs/stories_all_fmriprep_confounds_motionfiltered.py",
    )
    parser.add_argument("--regressor-dir", default="results/regressors")
    parser.add_argument(
        "--roi-cache-dir",
        default="results/journal_neurolinguistics_robustness/pc_count_full/roi_time_series_cache",
    )
    parser.add_argument(
        "--output-dir",
        default="results/journal_neurolinguistics_robustness/circular_shift_null",
    )
    parser.add_argument("--n-shifts", type=int, default=100)
    parser.add_argument("--min-shift-trs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--max-subjects-per-story", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    scan_path = output_dir / "circular_shift_scan_level.csv"
    null_path = output_dir / "circular_shift_null_draws.csv"
    global_path = output_dir / "circular_shift_global_null.csv"
    paired_path = output_dir / "circular_shift_scan_level_with_mean_null.csv"
    summary_path = output_dir / "circular_shift_summary.csv"

    stories = load_story_list(str(ROOT / args.subject_list))
    stories = limit_subjects_per_story(stories, args.max_subjects_per_story)
    roi_cache_dir = ROOT / args.roi_cache_dir

    scan_rows: list[dict] = []
    null_rows: list[dict] = []
    completed_pairs: set[tuple[str, str]] = set()
    if args.resume and scan_path.exists() and null_path.exists():
        existing_scan = pd.read_csv(scan_path)
        existing_null = pd.read_csv(null_path)
        scan_rows = existing_scan.to_dict("records")
        null_rows = existing_null.to_dict("records")
        null_counts = existing_null.groupby(["story", "subject"]).size()
        completed_pairs = {
            (story, subject)
            for (story, subject), count in null_counts.items()
            if int(count) >= args.n_shifts
        }
        print(f"Resuming with {len(completed_pairs)} completed subject-story rows")

    print(f"Subject-story rows: {len(stories)}")
    print(f"Circular shifts per scan: {args.n_shifts}")
    print(f"Minimum circular shift: {args.min_shift_trs} TRs")
    print(f"Output directory: {output_dir}")

    atlas_img, atlas_labels = fetch_harvard_oxford_atlas()
    rng = np.random.default_rng(args.seed)

    for i, story in enumerate(stories, start=1):
        pair = (story["name"], story["subject"])
        if pair in completed_pairs:
            print(f"[{i}/{len(stories)}] {story['name']} {story['subject']} (cached result)", flush=True)
            continue

        print(f"[{i}/{len(stories)}] {story['name']} {story['subject']}", flush=True)
        reg_df = load_regressors(ROOT / args.regressor_dir, story["name"])
        missing = [c for c in LEXICAL_BASELINE_COLS + PC_COLS if c not in reg_df.columns]
        if missing:
            raise ValueError(f"Missing columns in {story['name']} regressors: {missing}")

        roi_names, y_matrix, bold_tr, onset, duration = load_or_extract_roi_matrix(
            story, atlas_img, atlas_labels, roi_cache_dir
        )
        baseline_x = reg_df[LEXICAL_BASELINE_COLS].to_numpy()
        pc_x = reg_df[PC_COLS].to_numpy()
        combined_x = np.column_stack([baseline_x, pc_x])

        baseline_cv_r, n_used = vectorized_cv_r(
            baseline_x,
            y_matrix,
            n_splits=args.n_splits,
            alpha=args.ridge_alpha,
        )
        real_cv_r, _ = vectorized_cv_r(
            combined_x,
            y_matrix,
            n_splits=args.n_splits,
            alpha=args.ridge_alpha,
        )
        real_delta_by_roi = real_cv_r - baseline_cv_r
        real_mean_delta = float(np.nanmean(real_delta_by_roi))

        scan_rows.append(
            {
                "story": story["name"],
                "subject": story["subject"],
                "n_rois": int(len(roi_names)),
                "n_timepoints": int(n_used),
                "TR": bold_tr,
                "onset": onset,
                "duration": duration,
                "real_mean_roi_delta_cv_r": real_mean_delta,
                "real_median_roi_delta_cv_r": float(np.nanmedian(real_delta_by_roi)),
                "baseline_mean_cv_r": float(np.nanmean(baseline_cv_r)),
                "real_combined_mean_cv_r": float(np.nanmean(real_cv_r)),
            }
        )

        lower, upper = shift_bounds(len(pc_x), args.min_shift_trs)
        shifts = rng.integers(lower, upper, size=args.n_shifts)
        for shift_id, shift_trs in enumerate(shifts, start=1):
            shifted_pc_x = np.roll(pc_x, int(shift_trs), axis=0)
            shifted_x = np.column_stack([baseline_x, shifted_pc_x])
            shifted_cv_r, _ = vectorized_cv_r(
                shifted_x,
                y_matrix,
                n_splits=args.n_splits,
                alpha=args.ridge_alpha,
            )
            null_delta = shifted_cv_r - baseline_cv_r
            null_rows.append(
                {
                    "story": story["name"],
                    "subject": story["subject"],
                    "shift_id": int(shift_id),
                    "shift_trs": int(shift_trs),
                    "shift_seconds": float(shift_trs * bold_tr),
                    "null_mean_roi_delta_cv_r": float(np.nanmean(null_delta)),
                    "null_median_roi_delta_cv_r": float(np.nanmedian(null_delta)),
                    "real_mean_roi_delta_cv_r": real_mean_delta,
                    "real_minus_null_mean_roi_delta_cv_r": float(
                        real_mean_delta - np.nanmean(null_delta)
                    ),
                }
            )

        if args.checkpoint_every > 0 and i % args.checkpoint_every == 0:
            pd.DataFrame(scan_rows).to_csv(scan_path, index=False)
            pd.DataFrame(null_rows).to_csv(null_path, index=False)
            print(f"Checkpoint saved after {i} rows", flush=True)

    scan = pd.DataFrame(scan_rows)
    null = pd.DataFrame(null_rows)
    scan.to_csv(scan_path, index=False)
    null.to_csv(null_path, index=False)

    global_null = (
        null.groupby("shift_id", as_index=False)
        .agg(
            null_mean_delta_cv_r=("null_mean_roi_delta_cv_r", "mean"),
            null_median_delta_cv_r=("null_mean_roi_delta_cv_r", "median"),
            mean_real_minus_null_delta_cv_r=("real_minus_null_mean_roi_delta_cv_r", "mean"),
        )
        .sort_values("shift_id")
    )
    global_null.to_csv(global_path, index=False)

    scan_null = (
        null.groupby(["story", "subject"], as_index=False)
        .agg(
            mean_null_delta_cv_r=("null_mean_roi_delta_cv_r", "mean"),
            median_null_delta_cv_r=("null_mean_roi_delta_cv_r", "median"),
        )
    )
    paired = scan.merge(scan_null, on=["story", "subject"], how="left")
    paired["real_minus_mean_null_delta_cv_r"] = (
        paired["real_mean_roi_delta_cv_r"] - paired["mean_null_delta_cv_r"]
    )
    paired.to_csv(paired_path, index=False)

    real_stats = summarize(scan["real_mean_roi_delta_cv_r"])
    null_global_stats = summarize(global_null["null_mean_delta_cv_r"])
    diff_global_stats = summarize(global_null["mean_real_minus_null_delta_cv_r"])
    paired_stats = summarize(paired["real_minus_mean_null_delta_cv_r"])
    observed = float(scan["real_mean_roi_delta_cv_r"].mean())
    summary = pd.DataFrame(
        [
            {"measure": "real_scan_level_delta", **real_stats},
            {"measure": "global_null_mean_delta", **null_global_stats},
            {"measure": "real_minus_global_null_delta", **diff_global_stats},
            {"measure": "paired_scan_real_minus_mean_null_delta", **paired_stats},
            {
                "measure": "empirical_global_null_p_greater",
                "n": int(global_null.shape[0]),
                "mean": empirical_p_greater(observed, global_null["null_mean_delta_cv_r"]),
            },
        ]
    )
    summary.to_csv(summary_path, index=False)

    print(f"Saved: {scan_path}")
    print(f"Saved: {null_path}")
    print(f"Saved: {global_path}")
    print(f"Saved: {paired_path}")
    print(f"Saved: {summary_path}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
