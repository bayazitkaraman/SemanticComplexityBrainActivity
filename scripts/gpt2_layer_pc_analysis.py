"""
GPT-2 layer comparison analysis.

Reviewer concern:
- Why use only the final GPT-2 layer?

This script compares PC1 regressors extracted from GPT-2 layers 0..12.
It fits PCA in a shared cross-story space separately for each layer, then
correlates each layer-PC1 regressor with ROI BOLD activity.

Layer meaning:
- 0 = embedding output
- 1..12 = GPT-2 transformer layers
- 12 = final layer used in the original manuscript

Run quick test
--------------
python gpt2_layer_pc_analysis.py --quick --subject-list experiment_subject_list_all

Run full layer comparison
-------------------------
python gpt2_layer_pc_analysis.py --subject-list experiment_subject_list_all

Optional with empirical permutation p-values, slower
----------------------------------------------------
python gpt2_layer_pc_analysis.py --subject-list experiment_subject_list_all --n-perm 1000
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from semantic_fmri.utils import (
    ROI_LABELS_TO_TEST,
    best_lag_correlation,
    compute_gpt2_embeddings_by_layer,
    ensure_dir,
    extract_roi_timeseries,
    fetch_harvard_oxford_atlas,
    get_unique_stories,
    hrf_convolve,
    load_gpt2,
    load_story_list,
    permutation_p_for_best_lag,
    transcript_to_tr_chunks_uniform,
)


def parse_layers(text: str):
    if text.lower() == "all":
        return list(range(13))
    return [int(x.strip()) for x in text.split(",") if x.strip()]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-list", default="experiment_subject_list",
                        help="Python module or .py file containing `stories`.")
    parser.add_argument("--layers", default="all",
                        help='Comma-separated layers, e.g. "0,6,12", or "all".')
    parser.add_argument("--tr", type=float, default=1.5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lag-min", type=int, default=0,
                        help="Recommended 0 after HRF convolution.")
    parser.add_argument("--lag-max", type=int, default=8)
    parser.add_argument("--n-perm", type=int, default=0,
                        help="0 = no permutation p-values. Use 1000+ for final if needed.")
    parser.add_argument("--max-subjects-per-story", type=int, default=0)
    parser.add_argument("--quick", action="store_true",
                        help="Use layers 0,6,12 and max 2 subjects/story.")
    parser.add_argument("--output-dir", default="results/layer_analysis")
    args = parser.parse_args()

    if args.quick:
        args.layers = "0,6,12"
        args.max_subjects_per_story = 2
        args.n_perm = 0

    layers = parse_layers(args.layers)
    ensure_dir(args.output_dir)

    stories = load_story_list(args.subject_list)
    stories = limit_subjects_per_story(stories, args.max_subjects_per_story)
    unique_stories = get_unique_stories(stories)

    tokenizer, model, device = load_gpt2()
    print(f"Loaded GPT-2 on {device}")
    print(f"Layers: {layers}")
    print(f"Subject-story rows: {len(stories)}")

    # ---------------------------------------------------------------------
    # Step 1: compute embeddings per story and layer
    # ---------------------------------------------------------------------
    story_embeddings = {}
    story_meta = {}

    for story in unique_stories:
        print(f"\nExtracting GPT-2 layer embeddings for {story['name']}")
        chunks, onset, duration, n_trs = transcript_to_tr_chunks_uniform(story, tr=args.tr)
        embs_by_layer = compute_gpt2_embeddings_by_layer(
            chunks,
            tokenizer,
            model,
            layers=layers,
            max_length=args.max_length,
            device=device,
        )
        story_embeddings[story["name"]] = embs_by_layer
        story_meta[story["name"]] = {
            "onset": onset,
            "duration": duration,
            "n_trs": n_trs,
        }

    # ---------------------------------------------------------------------
    # Step 2: fit one shared PCA per layer across stories, create PC1_hrf
    # ---------------------------------------------------------------------
    layer_regressors = {layer: {} for layer in layers}
    evr_rows = []

    for layer in layers:
        print(f"Fitting shared PCA for layer {layer}")
        X_all = np.vstack([story_embeddings[s["name"]][layer] for s in unique_stories])

        scaler = StandardScaler()
        X_scaled_all = scaler.fit_transform(X_all)

        pca = PCA(n_components=1)
        pca.fit(X_scaled_all)

        evr_rows.append({
            "layer": layer,
            "PC1_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        })

        for story in unique_stories:
            X = story_embeddings[story["name"]][layer]
            pc1 = pca.transform(scaler.transform(X))[:, 0]
            pc1_hrf = hrf_convolve(pc1, tr=args.tr)
            layer_regressors[layer][story["name"]] = pc1_hrf

    evr_csv = os.path.join(args.output_dir, "layer_pc1_explained_variance.csv")
    pd.DataFrame(evr_rows).to_csv(evr_csv, index=False)
    print(f"Saved: {evr_csv}")

    # ---------------------------------------------------------------------
    # Step 3: extract ROI time series and correlate each layer regressor
    # ---------------------------------------------------------------------
    atlas_img, atlas_labels = fetch_harvard_oxford_atlas()
    rows = []

    for s_idx, story in enumerate(stories, start=1):
        print(f"\n[{s_idx}/{len(stories)}] Extracting ROIs: {story['name']} {story['subject']}")
        roi_ts, bold_tr, onset, duration, n_trs = extract_roi_timeseries(
            story,
            roi_labels=ROI_LABELS_TO_TEST,
            atlas_img=atlas_img,
            atlas_labels=atlas_labels,
            fwhm=6.0,
            atlas_threshold=0.0,
        )

        for layer in layers:
            reg = layer_regressors[layer][story["name"]]

            for roi, y in roi_ts.items():
                r, p, best_lag = best_lag_correlation(
                    reg,
                    y,
                    lag_min=args.lag_min,
                    lag_max=args.lag_max,
                )

                p_emp = permutation_p_for_best_lag(
                    reg,
                    y,
                    observed_r=r,
                    lag_min=args.lag_min,
                    lag_max=args.lag_max,
                    n_perm=args.n_perm,
                    seed=42 + layer,
                ) if args.n_perm > 0 else np.nan

                rows.append({
                    "story": story["name"],
                    "subject": story["subject"],
                    "roi": roi,
                    "layer": layer,
                    "r": r,
                    "abs_r": abs(r) if not np.isnan(r) else np.nan,
                    "p": p,
                    "p_emp": p_emp,
                    "best_lag": best_lag,
                    "TR": bold_tr,
                    "duration": duration,
                    "onset": onset,
                })

    df = pd.DataFrame(rows)

    # FDR within each story-subject-layer when p_emp exists
    if args.n_perm > 0 and not df.empty:
        df["q_roi"] = np.nan
        df["signif_roi_fdr"] = False

        for keys, sub_df in df.groupby(["story", "subject", "layer"]):
            idx = sub_df.index
            pvals = sub_df["p_emp"].fillna(1.0).to_numpy()
            reject, qvals, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
            df.loc[idx, "q_roi"] = qvals
            df.loc[idx, "signif_roi_fdr"] = reject

    out_csv = os.path.join(args.output_dir, "layer_roi_correlation_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    summary = (
        df.groupby(["layer", "roi"], as_index=False)
          .agg(mean_abs_r=("abs_r", "mean"),
               mean_r=("r", "mean"),
               n=("r", "count"))
          .sort_values(["mean_abs_r"], ascending=False)
    )
    summary_csv = os.path.join(args.output_dir, "layer_roi_correlation_group_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    layer_summary = (
        df.groupby("layer", as_index=False)
          .agg(mean_abs_r=("abs_r", "mean"),
               mean_r=("r", "mean"),
               n=("r", "count"))
          .sort_values("layer")
    )
    layer_summary_csv = os.path.join(args.output_dir, "layer_overall_summary.csv")
    layer_summary.to_csv(layer_summary_csv, index=False)
    print(f"Saved: {layer_summary_csv}")


if __name__ == "__main__":
    main()
