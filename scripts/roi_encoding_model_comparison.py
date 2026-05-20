from __future__ import annotations
import argparse, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy.signal import detrend
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from semantic_fmri.utils import (
    ROI_LABELS_TO_TEST, ACOUSTIC_CONTROL_COLUMNS, ensure_dir, extract_roi_timeseries,
    fetch_harvard_oxford_atlas, load_story_list, safe_pearson
)

def limit_subjects_per_story(stories, max_subjects_per_story):
    if max_subjects_per_story is None or max_subjects_per_story <= 0:
        return stories
    counts = defaultdict(int)
    selected = []
    for s in stories:
        if counts[s["name"]] < max_subjects_per_story:
            selected.append(s)
            counts[s["name"]] += 1
    return selected

def load_regressors(regressor_dir, story_name):
    path = Path(regressor_dir) / f"{story_name}_shared_pca_regressors.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run shared_pca_regressors.py first.")
    return pd.read_csv(path)

def clean_y(y, do_detrend=True):
    y = np.asarray(y, dtype=float)
    valid = ~np.isnan(y)
    if valid.sum() < 5:
        return y
    y2 = y.copy()
    if do_detrend:
        y2[valid] = detrend(y2[valid], type="linear")
    mu = np.nanmean(y2)
    sd = np.nanstd(y2)
    if sd > 0 and not np.isnan(sd):
        y2 = (y2 - mu) / sd
    return y2

def cross_validated_prediction(X, y, n_splits=5, alpha=10.0, do_detrend=True):
    X = np.asarray(X, dtype=float)
    y = clean_y(np.asarray(y, dtype=float), do_detrend=do_detrend)

    n = min(len(X), len(y))
    X, y = X[:n], y[:n]
    valid = ~np.isnan(y) & np.all(~np.isnan(X), axis=1)
    X, y = X[valid], y[valid]

    if len(y) < 30 or np.nanstd(y) == 0:
        return np.nan, np.nan, np.nan, np.nan, len(y)

    n_splits = min(n_splits, max(2, len(y) // 40))
    if n_splits < 2:
        return np.nan, np.nan, np.nan, np.nan, len(y)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, fit_intercept=True)),
    ])

    y_true_all, y_pred_all = [], []
    sse_train_mean = 0.0
    sst_train_mean = 0.0
    fold_r2_values = []

    for train_idx, test_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
        if len(test_idx) < 5:
            continue
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        y_test = y[test_idx]
        y_true_all.extend(y_test)
        y_pred_all.extend(pred)

        # Alternative CV R2 denominator that uses the training-fold mean as the
        # available no-signal predictor. This is often more interpretable for
        # forward-chaining time-series CV than a single concatenated test mean.
        train_mean = float(np.mean(y[train_idx]))
        sse_train_mean += float(np.sum((y_test - pred) ** 2))
        sst_train_mean += float(np.sum((y_test - train_mean) ** 2))
        if np.nanstd(y_test) > 0:
            fold_r2_values.append(float(r2_score(y_test, pred)))

    y_true_all = np.asarray(y_true_all, dtype=float)
    y_pred_all = np.asarray(y_pred_all, dtype=float)

    if len(y_true_all) < 10 or np.nanstd(y_true_all) == 0 or np.nanstd(y_pred_all) == 0:
        return np.nan, np.nan, np.nan, np.nan, len(y)

    cv_r, _ = safe_pearson(y_true_all, y_pred_all)
    cv_r2 = float(r2_score(y_true_all, y_pred_all))
    cv_r2_train_mean = float(1.0 - sse_train_mean / sst_train_mean) if sst_train_mean > 0 else np.nan
    cv_r2_fold_mean = float(np.mean(fold_r2_values)) if fold_r2_values else np.nan
    return cv_r2, cv_r, cv_r2_train_mean, cv_r2_fold_mean, len(y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-list", default="configs/stories_small.py")
    parser.add_argument("--regressor-dir", default="results/regressors")
    parser.add_argument("--output-dir", default="results/model_comparison")
    parser.add_argument("--n-pcs", type=int, default=5)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--max-subjects-per-story", type=int, default=0)
    parser.add_argument("--no-detrend", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--include-acoustic", action="store_true",
                        help="Add speech-timing/audio controls and acoustic-adjusted combined models.")
    args = parser.parse_args()

    if args.quick:
        args.max_subjects_per_story = 2
        args.n_pcs = min(args.n_pcs, 3)

    ensure_dir(args.output_dir)

    stories = load_story_list(args.subject_list)
    stories = limit_subjects_per_story(stories, args.max_subjects_per_story)

    print(f"Subject-story rows: {len(stories)}")
    print(f"Using PC1..PC{args.n_pcs}")
    print("Primary metric: cv_r; secondary metric: cv_r2")
    print(f"ROI detrending: {not args.no_detrend}")

    atlas_img, atlas_labels = fetch_harvard_oxford_atlas()

    lexical_baseline_cols = ["word_count_hrf", "mean_char_len_hrf", "total_chars_hrf", "time_TR_hrf"]
    pc_cols = [f"PC{i}_hrf" for i in range(1, args.n_pcs + 1)]

    rows = []
    for i, story in enumerate(stories, start=1):
        print(f"\n[{i}/{len(stories)}] {story['name']} {story['subject']}")
        reg_df = load_regressors(args.regressor_dir, story["name"])

        acoustic_cols = [f"{c}_hrf" for c in ACOUSTIC_CONTROL_COLUMNS if f"{c}_hrf" in reg_df.columns]
        baseline_cols = lexical_baseline_cols
        acoustic_baseline_cols = lexical_baseline_cols + acoustic_cols

        needed = lexical_baseline_cols + pc_cols
        missing = [c for c in needed if c not in reg_df.columns]
        if missing:
            raise ValueError(f"Missing columns in {story['name']} regressors: {missing}")
        if args.include_acoustic and not acoustic_cols:
            print(f"WARNING: no acoustic/speech-timing HRF columns found for {story['name']}; acoustic-adjusted models will match lexical/time baseline.")

        roi_ts, bold_tr, onset, duration, n_trs = extract_roi_timeseries(
            story, roi_labels=ROI_LABELS_TO_TEST, atlas_img=atlas_img,
            atlas_labels=atlas_labels, fwhm=6.0, atlas_threshold=0.0
        )

        model_mats = {
            "baseline_only": reg_df[baseline_cols].to_numpy(),
            "gpt2_pc1": reg_df[["PC1_hrf"]].to_numpy(),
            f"gpt2_pc1_to_pc{args.n_pcs}": reg_df[pc_cols].to_numpy(),
            "combined_pc1": reg_df[baseline_cols + ["PC1_hrf"]].to_numpy(),
            f"combined_pc1_to_pc{args.n_pcs}": reg_df[baseline_cols + pc_cols].to_numpy(),
        }

        if args.include_acoustic:
            model_mats.update({
                "baseline_acoustic_only": reg_df[acoustic_baseline_cols].to_numpy(),
                "combined_acoustic_pc1": reg_df[acoustic_baseline_cols + ["PC1_hrf"]].to_numpy(),
                f"combined_acoustic_pc1_to_pc{args.n_pcs}": reg_df[acoustic_baseline_cols + pc_cols].to_numpy(),
            })

        for roi, y in roi_ts.items():
            for model_name, X in model_mats.items():
                cv_r2, cv_r, cv_r2_train_mean, cv_r2_fold_mean, n_used = cross_validated_prediction(
                    X, y, n_splits=args.n_splits, alpha=args.ridge_alpha,
                    do_detrend=not args.no_detrend
                )
                rows.append({
                    "story": story["name"], "subject": story["subject"], "roi": roi,
                    "model": model_name, "cv_r2": cv_r2, "cv_r": cv_r,
                    "cv_r2_train_mean": cv_r2_train_mean,
                    "cv_r2_fold_mean": cv_r2_fold_mean,
                    "n_timepoints": n_used, "TR": bold_tr,
                    "duration": duration, "onset": onset,
                })

    df = pd.DataFrame(rows)
    out_csv = Path(args.output_dir) / "encoding_model_comparison.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    summary = (
        df.groupby(["model", "roi"], as_index=False)
        .agg(mean_cv_r=("cv_r", "mean"), sd_cv_r=("cv_r", "std"),
             mean_cv_r2=("cv_r2", "mean"), sd_cv_r2=("cv_r2", "std"),
             mean_cv_r2_train_mean=("cv_r2_train_mean", "mean"),
             mean_cv_r2_fold_mean=("cv_r2_fold_mean", "mean"),
             n=("cv_r", "count"))
        .sort_values(["model", "mean_cv_r"], ascending=[True, False])
    )
    summary_csv = Path(args.output_dir) / "encoding_model_group_summary_by_roi.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    wide_r = df.pivot_table(index=["story", "subject", "roi"], columns="model", values="cv_r", aggfunc="mean").reset_index()
    wide_r2 = df.pivot_table(index=["story", "subject", "roi"], columns="model", values="cv_r2", aggfunc="mean").reset_index()
    metric_cols = [c for c in wide_r2.columns if c not in ["story", "subject", "roi"]]
    wide_r2 = wide_r2.rename(columns={c: f"{c}_r2" for c in metric_cols})
    wide = wide_r.merge(wide_r2, on=["story", "subject", "roi"], how="left")

    if "baseline_only" in wide.columns and "combined_pc1" in wide.columns:
        wide["delta_cv_r_combined_pc1_minus_baseline"] = wide["combined_pc1"] - wide["baseline_only"]
    pcs_model = f"combined_pc1_to_pc{args.n_pcs}"
    if "baseline_only" in wide.columns and pcs_model in wide.columns:
        wide[f"delta_cv_r_combined_pc1_to_pc{args.n_pcs}_minus_baseline"] = wide[pcs_model] - wide["baseline_only"]

    acoustic_pcs_model = f"combined_acoustic_pc1_to_pc{args.n_pcs}"
    if "baseline_acoustic_only" in wide.columns and "combined_acoustic_pc1" in wide.columns:
        wide["delta_cv_r_combined_acoustic_pc1_minus_acoustic_baseline"] = (
            wide["combined_acoustic_pc1"] - wide["baseline_acoustic_only"]
        )
    if "baseline_acoustic_only" in wide.columns and acoustic_pcs_model in wide.columns:
        wide[f"delta_cv_r_combined_acoustic_pc1_to_pc{args.n_pcs}_minus_acoustic_baseline"] = (
            wide[acoustic_pcs_model] - wide["baseline_acoustic_only"]
        )

    if "baseline_only_r2" in wide.columns and "combined_pc1_r2" in wide.columns:
        wide["delta_cv_r2_combined_pc1_minus_baseline"] = wide["combined_pc1_r2"] - wide["baseline_only_r2"]
    pcs_model_r2 = f"combined_pc1_to_pc{args.n_pcs}_r2"
    if "baseline_only_r2" in wide.columns and pcs_model_r2 in wide.columns:
        wide[f"delta_cv_r2_combined_pc1_to_pc{args.n_pcs}_minus_baseline"] = wide[pcs_model_r2] - wide["baseline_only_r2"]
    acoustic_pcs_model_r2 = f"combined_acoustic_pc1_to_pc{args.n_pcs}_r2"
    if "baseline_acoustic_only_r2" in wide.columns and "combined_acoustic_pc1_r2" in wide.columns:
        wide["delta_cv_r2_combined_acoustic_pc1_minus_acoustic_baseline"] = (
            wide["combined_acoustic_pc1_r2"] - wide["baseline_acoustic_only_r2"]
        )
    if "baseline_acoustic_only_r2" in wide.columns and acoustic_pcs_model_r2 in wide.columns:
        wide[f"delta_cv_r2_combined_acoustic_pc1_to_pc{args.n_pcs}_minus_acoustic_baseline"] = (
            wide[acoustic_pcs_model_r2] - wide["baseline_acoustic_only_r2"]
        )

    delta_csv = Path(args.output_dir) / "encoding_model_deltas.csv"
    wide.to_csv(delta_csv, index=False)
    print(f"Saved: {delta_csv}")

    delta_summary = wide.groupby("roi", as_index=False).mean(numeric_only=True)
    delta_summary_csv = Path(args.output_dir) / "encoding_model_delta_summary_by_roi.csv"
    delta_summary.to_csv(delta_summary_csv, index=False)
    print(f"Saved: {delta_summary_csv}")

if __name__ == "__main__":
    main()
