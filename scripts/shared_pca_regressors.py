"""
Create story-comparable GPT-2 PCA regressors using a shared PCA space.

Why this matters
----------------
The rejected review asked whether PC1 from different stories corresponds to the
same underlying dimension. This script addresses that by:

1. Extracting GPT-2 final-layer embeddings for each story.
2. Concatenating embeddings across all stories.
3. Fitting ONE StandardScaler + PCA across the combined data.
4. Projecting each story into the same PC space.
5. Saving PC1..PCN and HRF-convolved versions.

Run
---
python shared_pca_regressors.py
python shared_pca_regressors.py --subject-list experiment_subject_list_all --n-components 10
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from semantic_fmri.utils import (
    compute_gpt2_embeddings_by_layer,
    compute_text_baselines,
    ensure_dir,
    get_unique_stories,
    hrf_convolve,
    load_gpt2,
    load_story_list,
    safe_pearson,
    transcript_to_tr_chunks_uniform,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-list", default="experiment_subject_list",
                        help="Python module or .py file containing `stories`.")
    parser.add_argument("--tr", type=float, default=1.5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--output-dir", default="results/regressors")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    stories = load_story_list(args.subject_list)
    unique_stories = get_unique_stories(stories)

    tokenizer, model, device = load_gpt2()
    print(f"Loaded GPT-2 on {device}")

    story_data = {}
    all_X = []

    for story in unique_stories:
        print(f"Extracting final-layer embeddings for {story['name']}")
        word_chunks, onset, duration, n_trs = transcript_to_tr_chunks_uniform(story, tr=args.tr)
        X = compute_gpt2_embeddings_by_layer(
            word_chunks,
            tokenizer,
            model,
            layers=[12],
            max_length=args.max_length,
            device=device,
        )[12]

        baseline_df = compute_text_baselines(word_chunks)

        story_data[story["name"]] = {
            "story": story,
            "word_chunks": word_chunks,
            "X": X,
            "baseline_df": baseline_df,
            "onset": onset,
            "duration": duration,
            "n_trs": n_trs,
        }
        all_X.append(X)

    X_all = np.vstack(all_X)
    n_components = min(args.n_components, X_all.shape[0], X_all.shape[1])

    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)

    pca = PCA(n_components=n_components)
    pca.fit(X_all_scaled)

    evr_rows = []
    for i, evr in enumerate(pca.explained_variance_ratio_, start=1):
        evr_rows.append({"component": f"PC{i}", "explained_variance_ratio": float(evr)})
    evr_csv = os.path.join(args.output_dir, "shared_pca_explained_variance.csv")
    pd.DataFrame(evr_rows).to_csv(evr_csv, index=False)
    print(f"Saved: {evr_csv}")

    sanity_rows = []

    for story_name, data in story_data.items():
        print(f"Saving shared-PCA regressors for {story_name}")

        X_scaled = scaler.transform(data["X"])
        scores = pca.transform(X_scaled)

        out_df = data["baseline_df"].copy()
        out_df["story"] = story_name
        out_df["TR"] = args.tr
        out_df["onset_sec"] = data["onset"]
        out_df["duration_sec"] = data["duration"]

        # Raw and HRF-convolved lexical baselines
        for col in ["word_count", "mean_char_len", "total_chars", "time_TR"]:
            out_df[f"{col}_hrf"] = hrf_convolve(out_df[col].to_numpy(), tr=args.tr)

        # Raw and HRF-convolved shared PCs
        for i in range(scores.shape[1]):
            pc_col = f"PC{i+1}"
            out_df[pc_col] = scores[:, i]
            out_df[f"{pc_col}_hrf"] = hrf_convolve(scores[:, i], tr=args.tr)

        # PC1 sanity correlations
        pc1 = out_df["PC1"].to_numpy()
        row = {"story": story_name}
        for col in ["word_count", "mean_char_len", "total_chars", "time_TR"]:
            r, p = safe_pearson(pc1, out_df[col].to_numpy())
            row[f"r_PC1_{col}"] = r
            row[f"p_PC1_{col}"] = p
        sanity_rows.append(row)

        out_csv = os.path.join(args.output_dir, f"{story_name}_shared_pca_regressors.csv")
        out_df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")

    sanity_csv = os.path.join(args.output_dir, "shared_pc1_vs_baselines_correlations.csv")
    pd.DataFrame(sanity_rows).to_csv(sanity_csv, index=False)
    print(f"Saved: {sanity_csv}")


if __name__ == "__main__":
    main()
