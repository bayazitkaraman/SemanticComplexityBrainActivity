from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img, smooth_img
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
from semantic_fmri.utils import (  # noqa: E402
    ROI_LABELS_TO_TEST,
    clean_roi_matrix,
    fetch_harvard_oxford_atlas,
    get_confounds_file_for_story,
    get_env_cleaning_strategy,
    get_env_high_pass,
    load_fmriprep_confounds,
    load_story_list,
    normalize_path,
    read_story_event,
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
    return {
        "analysis": label,
        "model": "y ~ 1 + (1 | subject) + (1 | story)",
        "n_subject_story_scans": int(data.shape[0]),
        "n_subjects": int(data["subject"].nunique()),
        "n_stories": int(data["story"].nunique()),
        "estimate": float(result.params["Intercept"]),
        "se": float(result.bse["Intercept"]),
        "z": float(result.tvalues["Intercept"]),
        "p": float(result.pvalues["Intercept"]),
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
        "converged": bool(result.converged),
        "warnings": " | ".join(caught_messages),
    }


def extract_hemisphere_roi_timeseries(
    story: dict,
    atlas_img,
    atlas_labels: list[str],
    atlas_threshold: float,
    fwhm: float,
) -> tuple[list[str], np.ndarray, float, float, float]:
    bold_file = normalize_path(story["bold_file"])
    fmri_img = smooth_img(nib.load(bold_file), fwhm=fwhm)
    fmri_data = fmri_img.get_fdata(dtype=np.float32)
    tr = float(fmri_img.header.get_zooms()[3])
    n_volumes = int(fmri_data.shape[3])
    onset, duration = read_story_event(story["events_file"])
    n_trs = int(duration // tr)
    bold_start = int(onset // tr)

    resampled_atlas = resample_to_img(
        atlas_img,
        fmri_img,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    atlas_data = resampled_atlas.get_fdata(dtype=np.float32)
    ijk = np.indices(atlas_data.shape[:3], dtype=float).reshape(3, -1).T
    x_coords = nib.affines.apply_affine(resampled_atlas.affine, ijk)[:, 0].reshape(atlas_data.shape[:3])
    hemi_masks = {"left": x_coords < 0, "right": x_coords > 0}

    roi_names = []
    roi_vectors = []
    for roi_label in ROI_LABELS_TO_TEST:
        if roi_label not in atlas_labels:
            continue
        roi_index = atlas_labels.index(roi_label)
        roi_mask = atlas_data[..., roi_index] > atlas_threshold
        for hemi, hemi_mask in hemi_masks.items():
            mask = roi_mask & hemi_mask
            if not np.any(mask):
                continue
            roi_names.append(f"{hemi}::{roi_label}")
            roi_vectors.append(np.nanmean(fmri_data[mask, :], axis=0))

    if not roi_vectors:
        return [], np.empty((0, 0)), tr, onset, duration

    roi_matrix = np.column_stack(roi_vectors)
    confounds = None
    strategy = get_env_cleaning_strategy()
    high_pass = get_env_high_pass(default=None)
    if strategy not in ["none", "no", "off"]:
        confounds = load_fmriprep_confounds(
            get_confounds_file_for_story(story),
            n_volumes=n_volumes,
            strategy=strategy,
        )
    roi_matrix = clean_roi_matrix(
        roi_matrix,
        tr=tr,
        confounds=confounds,
        detrend=True,
        standardize="zscore_sample",
        high_pass=high_pass,
    )

    end = min(bold_start + n_trs, roi_matrix.shape[0])
    segment = roi_matrix[bold_start:end, :]
    return roi_names, segment.astype(np.float32), tr, onset, duration


def load_or_extract_cache(
    story: dict,
    atlas_img,
    atlas_labels: list[str],
    cache_dir: Path,
    atlas_threshold: float,
    fwhm: float,
) -> tuple[list[str], np.ndarray, float, float, float]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{safe_name(story['name'])}_{safe_name(story['subject'])}.npz"
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        return (
            [str(v) for v in cached["roi_names"].tolist()],
            cached["y_matrix"],
            float(cached["bold_tr"]),
            float(cached["onset"]),
            float(cached["duration"]),
        )
    roi_names, y_matrix, tr, onset, duration = extract_hemisphere_roi_timeseries(
        story,
        atlas_img,
        atlas_labels,
        atlas_threshold=atlas_threshold,
        fwhm=fwhm,
    )
    np.savez_compressed(
        path,
        roi_names=np.asarray(roi_names),
        y_matrix=y_matrix,
        bold_tr=float(tr),
        onset=float(onset),
        duration=float(duration),
    )
    return roi_names, y_matrix, tr, onset, duration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject-list",
        default="configs/stories_all_fmriprep_confounds_motionfiltered.py",
    )
    parser.add_argument("--regressor-dir", default="results/regressors")
    parser.add_argument("--output-dir", default="results/jnl_revision_major_review/hemisphere_sensitivity")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--atlas-threshold", type=float, default=0.0)
    parser.add_argument("--fwhm", type=float, default=6.0)
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "roi_time_series_cache"
    stories = load_story_list(str(ROOT / args.subject_list))
    atlas_img, atlas_labels = fetch_harvard_oxford_atlas()
    regressor_dir = ROOT / args.regressor_dir
    pc_cols = [f"PC{i}_hrf" for i in range(1, 6)]

    rows = []
    long_path = out_dir / "hemisphere_encoding_long.csv"
    done_pairs = set()
    if long_path.exists():
        existing = pd.read_csv(long_path)
        rows = existing.to_dict("records")
        counts = existing.groupby(["story", "subject"]).size()
        done_pairs = {(story, subject) for (story, subject), count in counts.items() if int(count) >= 34}
        print(f"Resuming with {len(done_pairs)} completed subject-story rows", flush=True)

    for idx, story in enumerate(stories, start=1):
        pair = (story["name"], story["subject"])
        if pair in done_pairs:
            print(f"[{idx}/{len(stories)}] {story['name']} {story['subject']} cached", flush=True)
            continue
        print(f"[{idx}/{len(stories)}] {story['name']} {story['subject']}", flush=True)
        roi_names, y_matrix, tr, onset, duration = load_or_extract_cache(
            story,
            atlas_img,
            atlas_labels,
            cache_dir=cache_dir,
            atlas_threshold=args.atlas_threshold,
            fwhm=args.fwhm,
        )
        reg = load_regressors(regressor_dir, story["name"])
        baseline_x = reg[LEXICAL_BASELINE_COLS].to_numpy()
        combined_x = reg[LEXICAL_BASELINE_COLS + pc_cols].to_numpy()
        baseline_cv_r, n_used = vectorized_cv_r(
            baseline_x,
            y_matrix,
            n_splits=args.n_splits,
            alpha=args.ridge_alpha,
        )
        combined_cv_r, _ = vectorized_cv_r(
            combined_x,
            y_matrix,
            n_splits=args.n_splits,
            alpha=args.ridge_alpha,
        )
        for roi_idx, roi_name in enumerate(roi_names):
            hemi, roi = roi_name.split("::", 1)
            rows.append(
                {
                    "story": story["name"],
                    "subject": story["subject"],
                    "hemisphere": hemi,
                    "roi": roi,
                    "baseline_cv_r": baseline_cv_r[roi_idx],
                    "combined_pc1_to_pc5_cv_r": combined_cv_r[roi_idx],
                    "delta_cv_r": combined_cv_r[roi_idx] - baseline_cv_r[roi_idx],
                    "n_timepoints": int(n_used),
                    "TR": tr,
                    "duration": duration,
                    "onset": onset,
                    "atlas_threshold": args.atlas_threshold,
                }
            )
        if idx % 10 == 0:
            pd.DataFrame(rows).to_csv(long_path, index=False)
            print(f"Checkpoint saved: {long_path}", flush=True)

    long = pd.DataFrame(rows)
    long.to_csv(long_path, index=False)

    scan_hemi = (
        long.groupby(["hemisphere", "story", "subject"], as_index=False)
        .agg(
            mean_roi_baseline_cv_r=("baseline_cv_r", "mean"),
            mean_roi_combined_cv_r=("combined_pc1_to_pc5_cv_r", "mean"),
            mean_roi_delta_cv_r=("delta_cv_r", "mean"),
        )
        .sort_values(["hemisphere", "story", "subject"])
    )
    scan_hemi.to_csv(out_dir / "hemisphere_scan_level.csv", index=False)

    stats_rows = []
    mixed_rows = []
    for hemisphere, group in scan_hemi.groupby("hemisphere"):
        row = summarize(group["mean_roi_delta_cv_r"])
        row["hemisphere"] = hemisphere
        row["fraction_delta_positive"] = float((group["mean_roi_delta_cv_r"] > 0).mean())
        row["fraction_combined_cv_r_positive"] = float((group["mean_roi_combined_cv_r"] > 0).mean())
        stats_rows.append(row)
        mixed_rows.append(
            mixed_effect_intercept(
                group,
                "mean_roi_delta_cv_r",
                f"{hemisphere}_hemisphere_combined_pc1_to_pc5_minus_baseline",
            )
        )
    stats_df = pd.DataFrame(stats_rows)
    stats_df = stats_df[["hemisphere"] + [c for c in stats_df.columns if c != "hemisphere"]]
    stats_df.to_csv(out_dir / "hemisphere_scan_stats.csv", index=False)
    pd.DataFrame(mixed_rows).to_csv(out_dir / "hemisphere_mixed_effects.csv", index=False)

    wide = scan_hemi.pivot_table(
        index=["story", "subject"],
        columns="hemisphere",
        values="mean_roi_delta_cv_r",
        aggfunc="mean",
    ).reset_index()
    if {"left", "right"}.issubset(wide.columns):
        wide["left_minus_right_delta_cv_r"] = wide["left"] - wide["right"]
        row = summarize(wide["left_minus_right_delta_cv_r"])
        row["comparison"] = "left minus right mean ROI delta"
        pd.DataFrame([row]).to_csv(out_dir / "hemisphere_left_right_paired_summary.csv", index=False)
    wide.to_csv(out_dir / "hemisphere_scan_level_wide.csv", index=False)

    roi_stats = []
    for (hemisphere, roi), group in long.groupby(["hemisphere", "roi"]):
        row = summarize(group["delta_cv_r"])
        row["hemisphere"] = hemisphere
        row["roi"] = roi
        row["fraction_delta_positive"] = float((group["delta_cv_r"] > 0).mean())
        roi_stats.append(row)
    roi_stats_df = pd.DataFrame(roi_stats)
    roi_stats_df = roi_stats_df[
        ["hemisphere", "roi"] + [c for c in roi_stats_df.columns if c not in {"hemisphere", "roi"}]
    ]
    roi_stats_df.to_csv(out_dir / "hemisphere_roi_stats.csv", index=False)
    print(f"Saved hemisphere sensitivity outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
