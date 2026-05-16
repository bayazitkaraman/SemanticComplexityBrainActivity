"""
Diagnostic script: tests whether GPT-2 PC1 is reducible to simple lexical/time features.

This script answers one manuscript concern:
- Is PC1 just word count, word length, or a linear time trend?

It does NOT test whether GPT-2 explains brain activity beyond baselines.
For that, run roi_encoding_model_comparison.py after shared_pca_regressors.py.

Run
---
python semantic_baseline_regressors.py
python semantic_baseline_regressors.py --subject-list experiment_subject_list_all
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import os

import numpy as np
import pandas as pd

from semantic_fmri.utils import (
    compute_gpt2_embeddings_by_layer,
    compute_text_baselines,
    ensure_dir,
    fit_pca_scores,
    get_unique_stories,
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
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--output-dir", default="results/diagnostics")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    stories = load_story_list(args.subject_list)
    unique_stories = get_unique_stories(stories)

    tokenizer, model, device = load_gpt2()
    print(f"Loaded GPT-2 on {device}")

    rows = []

    for story in unique_stories:
        print(f"\nComputing PC1 baseline diagnostics for story: {story['name']}")

        word_chunks, onset, duration, n_trs = transcript_to_tr_chunks_uniform(story, tr=args.tr)
        layer_embs = compute_gpt2_embeddings_by_layer(
            word_chunks,
            tokenizer,
            model,
            layers=[12],
            max_length=args.max_length,
            device=device,
        )
        X = layer_embs[12]

        scores, pca, _ = fit_pca_scores(X, n_components=args.n_components)
        pc1 = scores[:, 0]

        baseline_df = compute_text_baselines(word_chunks)
        t = baseline_df["time_TR"].to_numpy()

        r_wc, p_wc = safe_pearson(pc1, baseline_df["word_count"])
        r_char, p_char = safe_pearson(pc1, baseline_df["mean_char_len"])
        r_total_chars, p_total_chars = safe_pearson(pc1, baseline_df["total_chars"])
        r_time, p_time = safe_pearson(pc1, t)

        row = {
            "story": story["name"],
            "n_TRs": n_trs,
            "duration_sec": duration,
            "onset_sec": onset,
            "n_words_total": int(sum(baseline_df["word_count"])),
            "r_PC1_wordcount": r_wc,
            "p_PC1_wordcount": p_wc,
            "r_PC1_mean_charlen": r_char,
            "p_PC1_mean_charlen": p_char,
            "r_PC1_total_chars": r_total_chars,
            "p_PC1_total_chars": p_total_chars,
            "r_PC1_time": r_time,
            "p_PC1_time": p_time,
        }

        for i, evr in enumerate(pca.explained_variance_ratio_, start=1):
            row[f"PC{i}_explained_variance"] = float(evr)

        rows.append(row)

        ts_df = baseline_df.copy()
        for i in range(scores.shape[1]):
            ts_df[f"PC{i+1}"] = scores[:, i]

        ts_df.to_csv(
            os.path.join(args.output_dir, f"{story['name']}_pc_scores_and_baselines.csv"),
            index=False,
        )

    out_csv = os.path.join(args.output_dir, "pc1_vs_baselines_correlations.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
