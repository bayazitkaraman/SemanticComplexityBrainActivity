from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
from jnl_story_grid import align_target_to_story_grid  # noqa: E402
from semantic_fmri.utils import (  # noqa: E402
    compute_text_baselines,
    get_unique_stories,
    hrf_convolve,
    load_gpt2,
    load_story_list,
    transcript_to_tr_chunks_uniform,
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


def tr_token_spans(chunks: list[str], tokenizer) -> tuple[list[int], list[tuple[int, int]]]:
    all_ids: list[int] = []
    spans: list[tuple[int, int]] = []
    for chunk in chunks:
        text = str(chunk).strip()
        if not text:
            spans.append((len(all_ids), len(all_ids)))
            continue
        prefix = " " if all_ids else ""
        ids = tokenizer.encode(prefix + text, add_special_tokens=False)
        start = len(all_ids)
        all_ids.extend(ids)
        spans.append((start, len(all_ids)))
    return all_ids, spans


def cumulative_current_token_embeddings(
    chunks: list[str],
    tokenizer,
    model,
    layer: int,
    max_length: int = 1024,
    output_stride: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    token_ids, spans = tr_token_spans(chunks, tokenizer)
    hidden_size = int(model.config.hidden_size)
    if not token_ids:
        return np.zeros((len(chunks), hidden_size), dtype=np.float32)

    n_tokens = len(token_ids)
    token_hidden = np.zeros((n_tokens, hidden_size), dtype=np.float32)
    filled = np.zeros(n_tokens, dtype=bool)

    with torch.no_grad():
        for seg_start in range(0, n_tokens, output_stride):
            seg_end = min(seg_start + output_stride, n_tokens)
            window_start = max(0, seg_end - max_length)
            window_ids = token_ids[window_start:seg_end]
            input_ids = torch.tensor([window_ids], dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids)
            hidden = outputs.hidden_states[layer].squeeze(0).detach().cpu().numpy()
            local_start = seg_start - window_start
            local_end = seg_end - window_start
            token_hidden[seg_start:seg_end] = hidden[local_start:local_end]
            filled[seg_start:seg_end] = True

    if not np.all(filled):
        raise RuntimeError("Not all story tokens received contextual hidden states.")

    tr_embeddings = np.zeros((len(chunks), hidden_size), dtype=np.float32)
    last_valid = token_hidden[0]
    eos_embedding = token_hidden[0]
    for i, (start, end) in enumerate(spans):
        if end > start:
            tr_embeddings[i] = token_hidden[start:end].mean(axis=0)
            last_valid = tr_embeddings[i]
        else:
            tr_embeddings[i] = last_valid if start > 0 else eos_embedding
    return tr_embeddings


def independent_chunk_embeddings_batched(
    chunks: list[str],
    tokenizer,
    model,
    layer: int,
    batch_size: int,
    max_length: int,
    device: str,
) -> np.ndarray:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = [str(chunk).strip() if str(chunk).strip() else tokenizer.eos_token for chunk in chunks]
    outputs_all = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            hidden = outputs.hidden_states[layer]
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            outputs_all.append(pooled.detach().cpu().numpy())
    return np.vstack(outputs_all).astype(np.float32)


def fit_shared_pca_regressors(
    story_embeddings: dict[str, np.ndarray],
    story_baselines: dict[str, pd.DataFrame],
    representation: str,
    out_dir: Path,
    n_components: int,
    tr: float,
) -> dict[str, pd.DataFrame]:
    rep_dir = out_dir / "regressors" / representation
    rep_dir.mkdir(parents=True, exist_ok=True)
    story_order = list(story_embeddings)
    x_all = np.vstack([story_embeddings[name] for name in story_order])
    n_components = min(n_components, x_all.shape[0], x_all.shape[1])
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_all)
    pca = PCA(n_components=n_components)
    pca.fit(x_scaled)

    evr = pd.DataFrame(
        {
            "representation": representation,
            "component": [f"PC{i}" for i in range(1, n_components + 1)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )
    evr.to_csv(rep_dir / "shared_pca_explained_variance.csv", index=False)

    out = {}
    for story_name, emb in story_embeddings.items():
        scores = pca.transform(scaler.transform(emb))
        df = story_baselines[story_name].copy()
        df["story"] = story_name
        for i in range(scores.shape[1]):
            col = f"PC{i + 1}"
            df[col] = scores[:, i]
            df[f"{col}_hrf"] = hrf_convolve(scores[:, i], tr=tr)
        df.to_csv(rep_dir / f"{story_name}_shared_pca_regressors.csv", index=False)
        out[story_name] = df
    return out


def evaluate_representation(
    representation: str,
    stories: list[dict],
    regressor_by_story: dict[str, pd.DataFrame],
    roi_cache_dir: Path,
    out_dir: Path,
    alpha: float,
    n_splits: int,
    n_pcs: int,
) -> pd.DataFrame:
    rows = []
    pc_cols = [f"PC{i}_hrf" for i in range(1, n_pcs + 1)]
    for idx, story in enumerate(stories, start=1):
        print(f"[{representation} {idx}/{len(stories)}] {story['name']} {story['subject']}", flush=True)
        cache_path = roi_cache_dir / f"{safe_name(story['name'])}_{safe_name(story['subject'])}.npz"
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing ROI cache: {cache_path}")
        cached = np.load(cache_path, allow_pickle=False)
        roi_names = [str(v) for v in cached["roi_names"].tolist()]
        reg = regressor_by_story[story["name"]]
        y_matrix, _ = align_target_to_story_grid(
            np.asarray(cached["y_matrix"], dtype=float),
            float(cached["bold_tr"]),
            float(cached["onset"]),
            max_timepoints=len(reg),
        )
        target_length = len(y_matrix)
        baseline_x = reg[LEXICAL_BASELINE_COLS].to_numpy()[:target_length]
        combined_x = reg[LEXICAL_BASELINE_COLS + pc_cols].to_numpy()[:target_length]
        baseline_cv_r, n_used = vectorized_cv_r(
            baseline_x, y_matrix, n_splits=n_splits, alpha=alpha
        )
        combined_cv_r, _ = vectorized_cv_r(
            combined_x, y_matrix, n_splits=n_splits, alpha=alpha
        )
        for roi_idx, roi in enumerate(roi_names):
            rows.append(
                {
                    "representation": representation,
                    "story": story["name"],
                    "subject": story["subject"],
                    "roi": roi,
                    "baseline_cv_r": baseline_cv_r[roi_idx],
                    "combined_cv_r": combined_cv_r[roi_idx],
                    "delta_cv_r": combined_cv_r[roi_idx] - baseline_cv_r[roi_idx],
                    "n_timepoints": int(n_used),
                    "alpha": alpha,
                    "n_pcs": n_pcs,
                }
            )
    long = pd.DataFrame(rows)
    rep_dir = out_dir / "encoding" / representation
    rep_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(rep_dir / "encoding_long.csv", index=False)
    scan = (
        long.groupby(["representation", "story", "subject"], as_index=False)
        .agg(
            mean_roi_baseline_cv_r=("baseline_cv_r", "mean"),
            mean_roi_combined_cv_r=("combined_cv_r", "mean"),
            mean_roi_delta_cv_r=("delta_cv_r", "mean"),
        )
        .sort_values(["story", "subject"])
    )
    scan.to_csv(rep_dir / "encoding_scan_level.csv", index=False)
    return scan


def representation_summaries(scan_tables: list[pd.DataFrame], primary_scan_csv: Path, out_dir: Path) -> None:
    primary = pd.read_csv(primary_scan_csv).copy()
    primary = primary[["story", "subject", "mean_roi_delta_cv_r"]].rename(
        columns={"mean_roi_delta_cv_r": "independent_final_layer12_existing"}
    )

    all_scan = []
    for table in scan_tables:
        all_scan.append(table)
    combined = pd.concat(all_scan, ignore_index=True)
    combined.to_csv(out_dir / "context_layer_encoding_scan_level_all.csv", index=False)

    stats_rows = []
    mixed_rows = []
    for representation, group in combined.groupby("representation"):
        row = summarize(group["mean_roi_delta_cv_r"])
        row["representation"] = representation
        row["fraction_delta_positive"] = float((group["mean_roi_delta_cv_r"] > 0).mean())
        row["fraction_combined_cv_r_positive"] = float((group["mean_roi_combined_cv_r"] > 0).mean())
        stats_rows.append(row)
        mixed_rows.append(
            mixed_effect_intercept(
                group,
                "mean_roi_delta_cv_r",
                f"{representation}_combined_pc1_to_pc5_minus_baseline",
            )
        )

    primary_stats = summarize(primary["independent_final_layer12_existing"])
    primary_stats["representation"] = "independent_final_layer12_existing"
    primary_stats["fraction_delta_positive"] = float((primary["independent_final_layer12_existing"] > 0).mean())
    stats_rows.append(primary_stats)
    mixed_rows.append(
        mixed_effect_intercept(
            primary.rename(columns={"independent_final_layer12_existing": "mean_roi_delta_cv_r"}),
            "mean_roi_delta_cv_r",
            "independent_final_layer12_existing_combined_pc1_to_pc5_minus_baseline",
        )
    )

    stats_df = pd.DataFrame(stats_rows)
    stats_df = stats_df[["representation"] + [c for c in stats_df.columns if c != "representation"]]
    stats_df.to_csv(out_dir / "context_layer_encoding_scan_stats.csv", index=False)
    pd.DataFrame(mixed_rows).to_csv(out_dir / "context_layer_encoding_mixed_effects.csv", index=False)

    wide = primary.copy()
    for representation, group in combined.groupby("representation"):
        col = representation
        sub = group[["story", "subject", "mean_roi_delta_cv_r"]].rename(columns={"mean_roi_delta_cv_r": col})
        wide = wide.merge(sub, on=["story", "subject"], how="left")
    wide.to_csv(out_dir / "context_layer_encoding_scan_level_wide.csv", index=False)

    comparison_rows = []
    base_col = "independent_final_layer12_existing"
    for col in [c for c in wide.columns if c not in {"story", "subject", base_col}]:
        diff = wide[col] - wide[base_col]
        row = summarize(diff)
        row["comparison"] = f"{col} minus {base_col}"
        mixed = mixed_effect_intercept(
            wide.assign(comparison_difference=diff),
            "comparison_difference",
            row["comparison"],
        )
        row["mixed_estimate"] = mixed["estimate"]
        row["mixed_ci95_low"] = mixed["ci95_low"]
        row["mixed_ci95_high"] = mixed["ci95_high"]
        row["mixed_p_two_sided"] = mixed["p"]
        row["mixed_converged"] = mixed["converged"]
        comparison_rows.append(row)
    comp = pd.DataFrame(comparison_rows)
    comp = comp[["comparison"] + [c for c in comp.columns if c != "comparison"]]
    comp.to_csv(out_dir / "context_layer_paired_comparisons_vs_final_layer12.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject-list",
        default="configs/stories_all_fmriprep_confounds_motionfiltered.py",
    )
    parser.add_argument(
        "--primary-scan-csv",
        default=(
            "results/jnl_individual_timing_corrected/gpt2_primary/"
            "primary_scan_level_model_values.csv"
        ),
    )
    parser.add_argument(
        "--roi-cache-dir",
        default="results/journal_neurolinguistics_robustness/pc_count_full_corrected/roi_time_series_cache",
    )
    parser.add_argument("--output-dir", default="results/jnl_revision_major_review/context_layer_models")
    parser.add_argument("--tr", type=float, default=1.5)
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--n-pcs", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--independent-max-length", type=int, default=128)
    parser.add_argument("--context-max-length", type=int, default=1024)
    parser.add_argument("--context-output-stride", type=int, default=256)
    parser.add_argument(
        "--reuse-regressor-root",
        default="",
        help="Reuse existing representation subdirectories instead of recomputing embeddings.",
    )
    parser.add_argument("--skip-context", action="store_true")
    parser.add_argument("--skip-layer4", action="store_true")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stories = load_story_list(str(ROOT / args.subject_list))
    unique_stories = get_unique_stories(stories)

    if args.reuse_regressor_root:
        source_root = ROOT / args.reuse_regressor_root
        scan_tables = []
        for representation, skip in [
            ("cumulative_context_layer12_current_tokens", args.skip_context),
            ("independent_chunk_layer4", args.skip_layer4),
        ]:
            if skip:
                continue
            representation_dir = source_root / representation
            regressors = {
                story["name"]: pd.read_csv(
                    representation_dir
                    / f"{story['name']}_shared_pca_regressors.csv"
                )
                for story in unique_stories
            }
            scan_tables.append(
                evaluate_representation(
                    representation,
                    stories,
                    regressors,
                    ROOT / args.roi_cache_dir,
                    out_dir,
                    alpha=args.alpha,
                    n_splits=args.n_splits,
                    n_pcs=args.n_pcs,
                )
            )
        if scan_tables:
            representation_summaries(scan_tables, ROOT / args.primary_scan_csv, out_dir)
        print(f"Saved context/layer model outputs in {out_dir}")
        return

    tokenizer, model, device = load_gpt2()
    print(f"Loaded GPT-2 on {device}")
    model.eval()

    story_chunks: dict[str, list[str]] = {}
    story_baselines: dict[str, pd.DataFrame] = {}
    for story in unique_stories:
        chunks, onset, duration, n_trs = transcript_to_tr_chunks_uniform(story, tr=args.tr)
        baseline = compute_text_baselines(chunks)
        baseline["TR"] = args.tr
        baseline["onset_sec"] = onset
        baseline["duration_sec"] = duration
        for col in ["word_count", "mean_char_len", "total_chars"]:
            baseline[f"{col}_hrf"] = hrf_convolve(baseline[col].to_numpy(), tr=args.tr)
        story_chunks[story["name"]] = chunks
        story_baselines[story["name"]] = baseline

    scan_tables = []

    if not args.skip_context:
        representation = "cumulative_context_layer12_current_tokens"
        embeddings = {}
        for story in unique_stories:
            print(f"Computing cumulative-context layer 12 embeddings: {story['name']}", flush=True)
            embeddings[story["name"]] = cumulative_current_token_embeddings(
                story_chunks[story["name"]],
                tokenizer,
                model,
                layer=12,
                max_length=args.context_max_length,
                output_stride=args.context_output_stride,
                device=device,
            )
        regs = fit_shared_pca_regressors(
            embeddings,
            story_baselines,
            representation,
            out_dir,
            n_components=args.n_components,
            tr=args.tr,
        )
        scan_tables.append(
            evaluate_representation(
                representation,
                stories,
                regs,
                ROOT / args.roi_cache_dir,
                out_dir,
                alpha=args.alpha,
                n_splits=args.n_splits,
                n_pcs=args.n_pcs,
            )
        )

    if not args.skip_layer4:
        representation = "independent_chunk_layer4"
        embeddings = {}
        for story in unique_stories:
            print(f"Computing independent chunk layer 4 embeddings: {story['name']}", flush=True)
            embeddings[story["name"]] = independent_chunk_embeddings_batched(
                story_chunks[story["name"]],
                tokenizer,
                model,
                layer=4,
                batch_size=args.batch_size,
                max_length=args.independent_max_length,
                device=device,
            )
        regs = fit_shared_pca_regressors(
            embeddings,
            story_baselines,
            representation,
            out_dir,
            n_components=args.n_components,
            tr=args.tr,
        )
        scan_tables.append(
            evaluate_representation(
                representation,
                stories,
                regs,
                ROOT / args.roi_cache_dir,
                out_dir,
                alpha=args.alpha,
                n_splits=args.n_splits,
                n_pcs=args.n_pcs,
            )
        )

    if scan_tables:
        representation_summaries(scan_tables, ROOT / args.primary_scan_csv, out_dir)
    print(f"Saved context/layer model outputs in {out_dir}")


if __name__ == "__main__":
    main()
