from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from jnl_13_story_individual_local import convolved_scores  # noqa: E402
from jnl_13_story_loso import (  # noqa: E402
    STORIES,
    load_data,
    project,
    story_balanced_pca,
    story_subjects,
    template_map,
)
from jnl_pc_count_sensitivity import vectorized_cv_r  # noqa: E402
from jnl_revision_context_layer_models import (  # noqa: E402
    independent_chunk_embeddings_batched,
    mixed_effect_intercept,
)
from jnl_shared_response_context_validation import one_sample_summary  # noqa: E402
from semantic_fmri.utils import load_gpt2, transcript_to_tr_chunks_uniform  # noqa: E402
from static_embedding_regressors import load_embedding_model  # noqa: E402


EXPECTED_SCANS = 567


def bert_embeddings_batched(
    chunks: list[str],
    tokenizer,
    model,
    embedding_layer,
    hidden_size: int,
    max_length: int,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    static = np.zeros((len(chunks), hidden_size), dtype=np.float32)
    contextual = np.zeros((len(chunks), hidden_size), dtype=np.float32)
    nonempty_indices = [
        index for index, chunk in enumerate(chunks) if str(chunk).strip()
    ]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    with torch.no_grad():
        for start in range(0, len(nonempty_indices), batch_size):
            indices = nonempty_indices[start : start + batch_size]
            texts = [str(chunks[index]).strip() for index in indices]

            static_encoded = tokenizer(
                texts,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            static_encoded = {key: value.to(device) for key, value in static_encoded.items()}
            static_hidden = embedding_layer(static_encoded["input_ids"])
            static_mask = static_encoded["attention_mask"].unsqueeze(-1).to(static_hidden.dtype)
            static_batch = (static_hidden * static_mask).sum(dim=1) / static_mask.sum(
                dim=1
            ).clamp_min(1.0)

            contextual_encoded = tokenizer(
                texts,
                add_special_tokens=True,
                return_tensors="pt",
                return_special_tokens_mask=True,
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            special_mask = contextual_encoded.pop("special_tokens_mask")
            contextual_encoded = {
                key: value.to(device) for key, value in contextual_encoded.items()
            }
            outputs = model(**contextual_encoded)
            contextual_hidden = outputs.hidden_states[-1]
            contextual_mask = (
                contextual_encoded["attention_mask"].bool()
                & ~special_mask.to(device).bool()
            ).unsqueeze(-1).to(contextual_hidden.dtype)
            contextual_batch = (
                contextual_hidden * contextual_mask
            ).sum(dim=1) / contextual_mask.sum(dim=1).clamp_min(1.0)

            static[indices] = static_batch.detach().cpu().numpy().astype(np.float32)
            contextual[indices] = (
                contextual_batch.detach().cpu().numpy().astype(np.float32)
            )
    return static, contextual


def load_cached(directory: Path) -> dict[str, np.ndarray]:
    missing = [story for story in STORIES if not (directory / f"{story}.npy").exists()]
    if missing:
        raise FileNotFoundError(f"Missing cached embeddings in {directory}: {missing}")
    values = {
        story: np.load(directory / f"{story}.npy", allow_pickle=False)
        for story in STORIES
    }
    if not all(np.all(np.isfinite(array)) for array in values.values()):
        raise ValueError(f"Non-finite embedding values found in {directory}.")
    return values


def ensure_bert_caches(args: argparse.Namespace) -> dict[str, dict[str, np.ndarray]]:
    root = ROOT / args.output_dir / "embeddings"
    static_dir = root / "bert_static_input"
    contextual_dir = root / "bert_contextual_final"
    static_dir.mkdir(parents=True, exist_ok=True)
    contextual_dir.mkdir(parents=True, exist_ok=True)
    missing = [
        story
        for story in STORIES
        if not (static_dir / f"{story}.npy").exists()
        or not (contextual_dir / f"{story}.npy").exists()
    ]
    if missing:
        tokenizer, model, embedding_layer, hidden_size, device = load_embedding_model(
            args.bert_model,
            device=args.device,
            local_files_only=not args.allow_download,
        )
        templates = template_map(args)
        for story in missing:
            chunks, _, _, _ = transcript_to_tr_chunks_uniform(templates[story], tr=1.5)
            print(f"Embedding BERT static/contextual: {story} ({len(chunks)} TRs)", flush=True)
            static, contextual = bert_embeddings_batched(
                chunks,
                tokenizer,
                model,
                embedding_layer,
                hidden_size,
                args.bert_max_length,
                args.embedding_batch_size,
                device,
            )
            np.save(static_dir / f"{story}.npy", static)
            np.save(contextual_dir / f"{story}.npy", contextual)
        del model, embedding_layer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "bert_contextual_final": load_cached(contextual_dir),
        "bert_static_input": load_cached(static_dir),
    }


def ensure_gpt2_layer4_cache(args: argparse.Namespace) -> dict[str, np.ndarray]:
    directory = ROOT / args.output_dir / "embeddings" / "gpt2_independent_layer4"
    directory.mkdir(parents=True, exist_ok=True)
    missing = [story for story in STORIES if not (directory / f"{story}.npy").exists()]
    if missing:
        tokenizer, model, device = load_gpt2()
        model.eval()
        templates = template_map(args)
        for story in missing:
            chunks, _, _, _ = transcript_to_tr_chunks_uniform(templates[story], tr=1.5)
            print(f"Embedding GPT-2 independent layer 4: {story} ({len(chunks)} TRs)", flush=True)
            values = independent_chunk_embeddings_batched(
                chunks,
                tokenizer,
                model,
                layer=4,
                batch_size=args.embedding_batch_size,
                max_length=args.gpt2_max_length,
                device=device,
            )
            np.save(directory / f"{story}.npy", values)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return load_cached(directory)


def loso_scores(
    embeddings: dict[str, np.ndarray], n_components: int
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    scores: dict[str, np.ndarray] = {}
    metadata_rows = []
    for omitted_story in STORIES:
        training = [story for story in STORIES if story != omitted_story]
        mean, scale, components, eigenvalues = story_balanced_pca(
            embeddings, training, n_components=n_components
        )
        scores[omitted_story] = project(
            embeddings[omitted_story], mean, scale, components
        )
        metadata_rows.append(
            {
                "omitted_story": omitted_story,
                "n_training_stories": len(training),
                "test_story_in_feature_fit": False,
                **{
                    f"pc{index + 1}_variance_ratio": float(
                        eigenvalues[index] / eigenvalues.sum()
                    )
                    for index in range(n_components)
                },
            }
        )
    return scores, pd.DataFrame(metadata_rows)


def evaluate_representations(
    args: argparse.Namespace,
    embeddings: dict[str, dict[str, np.ndarray]],
    baselines: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    roi_names: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_spaces = {}
    metadata_tables = []
    for representation, values in embeddings.items():
        scores, metadata = loso_scores(values, args.n_components)
        score_spaces[representation] = scores
        metadata.insert(0, "representation", representation)
        metadata_tables.append(metadata)

    scan_rows = []
    roi_rows = []
    for story in STORIES:
        subjects = story_subjects(args, story)
        if len(subjects) != len(targets[story]):
            raise RuntimeError(f"Subject count mismatch for {story}.")
        convolved = {
            representation: convolved_scores(scores[story], args.n_components)
            for representation, scores in score_spaces.items()
        }
        print(f"Evaluating auxiliary embeddings: {story} ({len(subjects)} scans)", flush=True)
        for subject_index, subject in enumerate(subjects):
            target = targets[story][subject_index]
            baseline = baselines[story]
            n = min(
                len(target),
                len(baseline),
                *(len(values) for values in convolved.values()),
            )
            baseline_design = baseline[:n]
            target = target[:n]
            baseline_r, n_used = vectorized_cv_r(
                baseline_design,
                target,
                n_splits=args.n_splits,
                alpha=args.alpha,
                do_detrend=True,
                cv_gap=args.cv_gap,
            )
            for representation, pcs in convolved.items():
                combined = np.column_stack([baseline_design, pcs[:n]])
                combined_r, _ = vectorized_cv_r(
                    combined,
                    target,
                    n_splits=args.n_splits,
                    alpha=args.alpha,
                    do_detrend=True,
                    cv_gap=args.cv_gap,
                )
                delta = combined_r - baseline_r
                scan_rows.append(
                    {
                        "representation": representation,
                        "story": story,
                        "subject": subject,
                        "n_timepoints": int(n_used),
                        "mean_baseline_cv_r": float(np.mean(baseline_r)),
                        "mean_combined_cv_r": float(np.mean(combined_r)),
                        "mean_delta_cv_r": float(np.mean(delta)),
                    }
                )
                for roi_index, roi in enumerate(roi_names[story]):
                    roi_rows.append(
                        {
                            "representation": representation,
                            "story": story,
                            "subject": subject,
                            "roi": roi,
                            "baseline_cv_r": float(baseline_r[roi_index]),
                            "combined_cv_r": float(combined_r[roi_index]),
                            "delta_cv_r": float(delta[roi_index]),
                        }
                    )
    scans = pd.DataFrame(scan_rows)
    rois = pd.DataFrame(roi_rows)
    for representation in embeddings:
        current = scans[scans["representation"].eq(representation)]
        if len(current) != EXPECTED_SCANS or current["story"].nunique() != len(STORIES):
            raise RuntimeError(
                f"{representation} has {len(current)} scans and "
                f"{current['story'].nunique()} stories; expected 567 and 13."
            )
    return scans, rois, pd.concat(metadata_tables, ignore_index=True)


def ci95(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    sem = float(stats.sem(values))
    low, high = stats.t.interval(0.95, len(values) - 1, loc=mean, scale=sem)
    return float(low), float(high)


def summarize(
    scans: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stories = (
        scans.groupby(["representation", "story"], as_index=False)
        .agg(
            n_scans=("subject", "size"),
            mean_baseline_cv_r=("mean_baseline_cv_r", "mean"),
            mean_combined_cv_r=("mean_combined_cv_r", "mean"),
            mean_delta_cv_r=("mean_delta_cv_r", "mean"),
        )
    )
    summary_rows = []
    mixed_rows = []
    for representation, group in stories.groupby("representation"):
        values = group["mean_delta_cv_r"].to_numpy(dtype=float)
        row = {"representation": representation, "scope": "all_13_intact_stories"}
        row.update(one_sample_summary(pd.Series(values)))
        row["ci95_low"], row["ci95_high"] = ci95(values)
        summary_rows.append(row)
        mixed_rows.append(
            mixed_effect_intercept(
                scans[scans["representation"].eq(representation)],
                "mean_delta_cv_r",
                representation,
            )
        )

    comparisons = [
        ("bert_contextual_minus_static", "bert_contextual_final", "bert_static_input"),
        (
            "gpt2_layer4_minus_final",
            "gpt2_independent_layer4",
            "gpt2_independent_final",
        ),
    ]
    paired_rows = []
    paired_mixed_rows = []
    for label, first, second in comparisons:
        first_scans = scans[scans["representation"].eq(first)][
            ["story", "subject", "mean_delta_cv_r"]
        ].rename(columns={"mean_delta_cv_r": "first"})
        second_scans = scans[scans["representation"].eq(second)][
            ["story", "subject", "mean_delta_cv_r"]
        ].rename(columns={"mean_delta_cv_r": "second"})
        paired = first_scans.merge(
            second_scans, on=["story", "subject"], validate="one_to_one"
        )
        paired["difference"] = paired["first"] - paired["second"]
        story_values = paired.groupby("story")["difference"].mean().to_numpy(dtype=float)
        row = {"comparison": label, "scope": "all_13_intact_stories"}
        row.update(one_sample_summary(pd.Series(story_values)))
        row["ci95_low"], row["ci95_high"] = ci95(story_values)
        paired_rows.append(row)
        paired_mixed_rows.append(
            mixed_effect_intercept(paired, "difference", label)
        )
    return (
        stories,
        pd.DataFrame(summary_rows),
        pd.DataFrame(mixed_rows),
        pd.DataFrame(paired_rows),
        pd.DataFrame(paired_mixed_rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="results/jnl_13_story_auxiliary_embeddings"
    )
    parser.add_argument(
        "--final-layer-embedding-dir",
        default="results/jnl_13_story_individual_local/independent_embeddings",
    )
    parser.add_argument(
        "--development-config",
        default="configs/stories_all_fmriprep_confounds_motionfiltered.py",
    )
    parser.add_argument(
        "--initial-evaluation-config",
        default="results/jnl_heldout_story_replication/heldout_motionfiltered_config.py",
    )
    parser.add_argument(
        "--new-evaluation-config",
        default=(
            "results/jnl_13_story_extension/new_story_fixed_pca/"
            "heldout_motionfiltered_config.py"
        ),
    )
    parser.add_argument(
        "--development-regressor-dir",
        default="results/jnl_revision_major_review/rich_audio_controls/rich_audio_regressors",
    )
    parser.add_argument(
        "--initial-evaluation-regressor-dir",
        default="results/jnl_heldout_story_replication/heldout_regressors",
    )
    parser.add_argument(
        "--new-evaluation-regressor-dir",
        default="results/jnl_13_story_extension/new_story_fixed_pca/heldout_regressors",
    )
    parser.add_argument(
        "--development-roi-cache",
        default=(
            "results/journal_neurolinguistics_robustness/pc_count_full_corrected/"
            "roi_time_series_cache"
        ),
    )
    parser.add_argument(
        "--initial-evaluation-roi-cache",
        default="results/jnl_heldout_story_replication/roi_cache",
    )
    parser.add_argument(
        "--new-evaluation-roi-cache",
        default="results/jnl_13_story_extension/new_story_fixed_pca/roi_cache",
    )
    parser.add_argument("--gentle-root", required=True)
    parser.add_argument("--bert-model", default="bert-base-uncased")
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--bert-max-length", type=int, default=128)
    parser.add_argument("--gpt2-max-length", type=int, default=128)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--cv-gap", type=int, default=10)
    parser.add_argument(
        "--stage", choices=["embeddings", "analysis", "all"], default="all"
    )
    args = parser.parse_args()

    os.environ["SEMANTIC_FMRI_TIMING"] = "gentle"
    os.environ["SEMANTIC_FMRI_GENTLE_DIR"] = str(Path(args.gentle_root))
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    representations = ensure_bert_caches(args)
    representations["gpt2_independent_layer4"] = ensure_gpt2_layer4_cache(args)
    representations["gpt2_independent_final"] = load_cached(
        ROOT / args.final_layer_embedding_dir
    )
    if args.stage == "embeddings":
        return

    baselines, targets, roi_names = load_data(args)
    scans, rois, metadata = evaluate_representations(
        args, representations, baselines, targets, roi_names
    )
    stories, summary, mixed, paired, paired_mixed = summarize(scans)

    scans.to_csv(output_dir / "scan_level.csv", index=False)
    rois.to_csv(output_dir / "roi_level.csv", index=False)
    metadata.to_csv(output_dir / "pca_metadata.csv", index=False)
    stories.to_csv(output_dir / "story_level.csv", index=False)
    summary.to_csv(output_dir / "representation_summary.csv", index=False)
    mixed.to_csv(output_dir / "representation_mixed_effects.csv", index=False)
    paired.to_csv(output_dir / "paired_comparison_summary.csv", index=False)
    paired_mixed.to_csv(output_dir / "paired_comparison_mixed_effects.csv", index=False)

    print("\nAll-13 auxiliary embedding results:\n", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nPaired comparisons:\n", flush=True)
    print(paired.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
