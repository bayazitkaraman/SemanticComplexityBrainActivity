"""
Create story-comparable token-embedding PCA regressors.

This mirrors scripts/shared_pca_regressors.py but replaces contextual GPT-2
hidden states with either non-contextual input token embeddings or contextual
hidden states from a cached Hugging Face model such as bert-base-uncased.

The output files use the same *_shared_pca_regressors.csv naming convention so
they can be passed directly to roi_encoding_model_comparison.py via
--regressor-dir.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

from semantic_fmri.utils import (
    compute_acoustic_control_baselines,
    compute_text_baselines,
    ensure_dir,
    get_unique_stories,
    hrf_convolve,
    load_story_list,
    safe_pearson,
    transcript_to_tr_chunks_uniform,
)


def load_embedding_model(model_name: str, device: str | None, local_files_only: bool):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        output_hidden_states=True,
    )
    model.eval()
    model.to(device)
    embedding_layer = model.get_input_embeddings()
    return tokenizer, model, embedding_layer, model.config.hidden_size, device


def compute_static_token_embeddings(
    word_chunks: Sequence[str],
    tokenizer,
    embedding_layer,
    hidden_size: int,
    max_length: int,
    device: str,
) -> np.ndarray:
    """Mean-pool model input-token embeddings for each TR text chunk."""
    rows = []
    with torch.no_grad():
        for chunk in word_chunks:
            text = chunk.strip() if isinstance(chunk, str) else str(chunk).strip()
            if not text:
                rows.append(np.zeros(hidden_size, dtype=np.float32))
                continue

            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            input_ids = encoded.get("input_ids")
            if input_ids is None or input_ids.numel() == 0:
                rows.append(np.zeros(hidden_size, dtype=np.float32))
                continue

            input_ids = input_ids.to(device)
            emb = embedding_layer(input_ids).squeeze(0).mean(dim=0)
            rows.append(emb.detach().cpu().numpy().astype(np.float32))

    return np.stack(rows, axis=0)


def compute_contextual_token_embeddings(
    word_chunks: Sequence[str],
    tokenizer,
    model,
    hidden_size: int,
    max_length: int,
    device: str,
    layer: int,
) -> np.ndarray:
    """Mean-pool contextual hidden states over non-special tokens in each TR."""
    rows = []
    with torch.no_grad():
        for chunk in word_chunks:
            text = chunk.strip() if isinstance(chunk, str) else str(chunk).strip()
            if not text:
                rows.append(np.zeros(hidden_size, dtype=np.float32))
                continue

            encoded = tokenizer(
                text,
                add_special_tokens=True,
                return_tensors="pt",
                return_special_tokens_mask=True,
                truncation=True,
                max_length=max_length,
            )
            special_tokens_mask = encoded.pop("special_tokens_mask").squeeze(0)
            encoded = {k: v.to(device) for k, v in encoded.items()}

            outputs = model(**encoded)
            hidden_states = outputs.hidden_states
            actual_layer = layer
            if actual_layer < 0:
                actual_layer = len(hidden_states) + actual_layer
            if actual_layer < 0 or actual_layer >= len(hidden_states):
                raise ValueError(f"Invalid layer {layer}. Valid range: 0..{len(hidden_states)-1}")

            token_hidden = hidden_states[actual_layer].squeeze(0)
            non_special = (special_tokens_mask == 0).to(token_hidden.device)
            if "attention_mask" in encoded:
                non_special = non_special & (encoded["attention_mask"].squeeze(0).bool())

            if int(non_special.sum().item()) == 0:
                rows.append(np.zeros(hidden_size, dtype=np.float32))
                continue

            emb = token_hidden[non_special].mean(dim=0)
            rows.append(emb.detach().cpu().numpy().astype(np.float32))

    return np.stack(rows, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-list", default="configs/stories_all.py")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--embedding-mode", choices=["static", "contextual"], default="static",
                        help="Use input-token embeddings or contextual hidden states.")
    parser.add_argument("--layer", type=int, default=-1,
                        help="Hidden-state layer for contextual mode. -1 is the final layer.")
    parser.add_argument("--tr", type=float, default=1.5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--output-dir", default="results/regressors_static_bert_base_uncased")
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-download", action="store_true",
                        help="Allow Hugging Face to download the model if it is not cached.")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    stories = load_story_list(args.subject_list)
    unique_stories = get_unique_stories(stories)

    tokenizer, model, embedding_layer, hidden_size, device = load_embedding_model(
        args.model_name,
        device=args.device,
        local_files_only=not args.allow_download,
    )
    print(f"Loaded {args.embedding_mode} embeddings from {args.model_name} on {device}")

    story_data = {}
    all_X = []

    for story in unique_stories:
        print(f"Extracting {args.embedding_mode} embeddings for {story['name']}")
        word_chunks, onset, duration, n_trs = transcript_to_tr_chunks_uniform(story, tr=args.tr)
        if args.embedding_mode == "static":
            X = compute_static_token_embeddings(
                word_chunks,
                tokenizer=tokenizer,
                embedding_layer=embedding_layer,
                hidden_size=hidden_size,
                max_length=args.max_length,
                device=device,
            )
        else:
            X = compute_contextual_token_embeddings(
                word_chunks,
                tokenizer=tokenizer,
                model=model,
                hidden_size=hidden_size,
                max_length=args.max_length,
                device=device,
                layer=args.layer,
            )

        baseline_df = compute_text_baselines(word_chunks)
        acoustic_df = compute_acoustic_control_baselines(story, tr=args.tr)
        baseline_df = baseline_df.merge(acoustic_df, on="time_TR", how="left")
        for col in acoustic_df.columns:
            if col != "time_TR":
                baseline_df[col] = baseline_df[col].fillna(0.0)

        story_data[story["name"]] = {
            "story": story,
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

    evr_rows = [
        {"component": f"PC{i}", "explained_variance_ratio": float(evr)}
        for i, evr in enumerate(pca.explained_variance_ratio_, start=1)
    ]
    evr_csv = Path(args.output_dir) / "shared_pca_explained_variance.csv"
    pd.DataFrame(evr_rows).to_csv(evr_csv, index=False)
    print(f"Saved: {evr_csv}")

    metadata = {
        "embedding_type": (
            "static_input_token_embeddings"
            if args.embedding_mode == "static"
            else "contextual_hidden_states"
        ),
        "embedding_mode": args.embedding_mode,
        "model_name": args.model_name,
        "hidden_size": hidden_size,
        "layer": args.layer if args.embedding_mode == "contextual" else None,
        "empty_tr_strategy": "zero_vector",
        "n_components": n_components,
        "timing_strategy": os.environ.get("SEMANTIC_FMRI_TIMING", "gentle"),
        "gentle_dir": os.environ.get("SEMANTIC_FMRI_GENTLE_DIR", ""),
    }
    metadata_path = Path(args.output_dir) / "static_embedding_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved: {metadata_path}")

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

        baseline_cols = [
            "word_count", "mean_char_len", "total_chars", "time_TR",
            "speech_duration_sec", "mean_word_duration_sec",
            "pause_fraction", "articulation_rate", "word_rate",
            "audio_rms", "audio_abs_mean",
        ]
        for col in baseline_cols:
            if col in out_df.columns:
                out_df[f"{col}_hrf"] = hrf_convolve(out_df[col].to_numpy(), tr=args.tr)

        for i in range(scores.shape[1]):
            pc_col = f"PC{i+1}"
            out_df[pc_col] = scores[:, i]
            out_df[f"{pc_col}_hrf"] = hrf_convolve(scores[:, i], tr=args.tr)

        pc1 = out_df["PC1"].to_numpy()
        row = {"story": story_name}
        for col in [
            "word_count", "mean_char_len", "total_chars", "time_TR",
            "speech_duration_sec", "pause_fraction", "audio_rms",
        ]:
            if col not in out_df.columns:
                continue
            r, p = safe_pearson(pc1, out_df[col].to_numpy())
            row[f"r_PC1_{col}"] = r
            row[f"p_PC1_{col}"] = p
        sanity_rows.append(row)

        out_csv = Path(args.output_dir) / f"{story_name}_shared_pca_regressors.csv"
        out_df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")

    sanity_csv = Path(args.output_dir) / "shared_pc1_vs_baselines_correlations.csv"
    pd.DataFrame(sanity_rows).to_csv(sanity_csv, index=False)
    print(f"Saved: {sanity_csv}")


if __name__ == "__main__":
    main()
