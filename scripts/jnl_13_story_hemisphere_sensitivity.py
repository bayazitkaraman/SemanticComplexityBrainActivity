from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from jnl_13_story_loso import (  # noqa: E402
    DEFAULT_REGRESSOR_DIR,
    STORIES,
    baseline_matrix,
    fold_scores,
    source_regressor_path,
    with_pcs,
)
from jnl_pc_count_sensitivity import vectorized_cv_r  # noqa: E402
from jnl_rescue_network_model_comparisons import bh_fdr  # noqa: E402
from jnl_shared_response_context_validation import (  # noqa: E402
    cross_group_cv_r,
    load_story_targets,
)
from jnl_shared_response_hemisphere_sensitivity import (  # noqa: E402
    add_roi_rows,
    extract_heldout_hemisphere_targets,
    inference_summary,
    validate_roi_names,
)


ANALYSES = ["full_group", "participant_split_half", "leave_one_participant_out"]


def load_embeddings(directory: Path) -> dict[str, np.ndarray]:
    missing = [story for story in STORIES if not (directory / f"{story}.npy").exists()]
    if missing:
        raise FileNotFoundError(f"Missing cumulative embeddings: {missing}")
    embeddings = {
        story: np.load(directory / f"{story}.npy", allow_pickle=False)
        for story in STORIES
    }
    if not all(np.all(np.isfinite(values)) for values in embeddings.values()):
        raise ValueError("Cumulative embedding cache contains non-finite values.")
    return embeddings


def hemisphere_cache(args: argparse.Namespace, story: str) -> Path:
    return ROOT / args.hemisphere_cache


def evaluate_story(
    args: argparse.Namespace,
    story: str,
    embeddings: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache = hemisphere_cache(args, story)
    targets, roi_names = load_story_targets(cache, story)
    validate_roi_names(roi_names, story)
    baseline = baseline_matrix(pd.read_csv(source_regressor_path(args, story)))
    scores, _ = fold_scores(embeddings, story)
    combined = with_pcs(baseline, scores[story])
    n = min(len(baseline), len(combined), targets.shape[1])
    baseline = baseline[:n]
    combined = combined[:n]
    targets = targets[:, :n]
    print(f"All-13 hemisphere sensitivity: {story} ({len(targets)} scans)", flush=True)

    full_rows = []
    group_target = targets.mean(axis=0)
    baseline_full, _ = vectorized_cv_r(
        baseline,
        group_target,
        args.n_splits,
        args.alpha,
        True,
        args.cv_gap,
    )
    combined_full, _ = vectorized_cv_r(
        combined,
        group_target,
        args.n_splits,
        args.alpha,
        True,
        args.cv_gap,
    )
    add_roi_rows(
        full_rows,
        analysis="full_group",
        story_set="all_13",
        story=story,
        roi_names=roi_names,
        baseline=baseline_full,
        combined=combined_full,
    )

    split_rows = []
    for seed in range(args.subject_splits):
        indices = np.random.default_rng(seed).permutation(len(targets))
        first = indices[: len(indices) // 2]
        second = indices[len(indices) // 2 :]
        baseline_split = (
            cross_group_cv_r(
                baseline, targets, first, second,
                args.n_splits, args.alpha, args.cv_gap,
            )
            + cross_group_cv_r(
                baseline, targets, second, first,
                args.n_splits, args.alpha, args.cv_gap,
            )
        ) / 2.0
        combined_split = (
            cross_group_cv_r(
                combined, targets, first, second,
                args.n_splits, args.alpha, args.cv_gap,
            )
            + cross_group_cv_r(
                combined, targets, second, first,
                args.n_splits, args.alpha, args.cv_gap,
            )
        ) / 2.0
        add_roi_rows(
            split_rows,
            analysis="participant_split_half",
            story_set="all_13",
            story=story,
            roi_names=roi_names,
            baseline=baseline_split,
            combined=combined_split,
            seed=seed,
        )

    files = sorted(cache.glob(f"{story}_*.npz"))
    subjects = [path.stem[len(story) + 1 :] for path in files]
    if len(subjects) != len(targets):
        raise RuntimeError(f"Hemisphere target/subject count mismatch for {story}.")
    leave_rows = []
    all_subjects = np.arange(len(targets))
    for heldout_index, subject in enumerate(subjects):
        training = all_subjects[all_subjects != heldout_index]
        testing = np.asarray([heldout_index])
        baseline_leave = cross_group_cv_r(
            baseline, targets, training, testing,
            args.n_splits, args.alpha, args.cv_gap,
        )
        combined_leave = cross_group_cv_r(
            combined, targets, training, testing,
            args.n_splits, args.alpha, args.cv_gap,
        )
        add_roi_rows(
            leave_rows,
            analysis="leave_one_participant_out",
            story_set="all_13",
            story=story,
            roi_names=roi_names,
            baseline=baseline_leave,
            combined=combined_leave,
            subject=subject,
        )
    return pd.DataFrame(full_rows), pd.DataFrame(split_rows), pd.DataFrame(leave_rows)


def summarize(
    full: pd.DataFrame,
    split: pd.DataFrame,
    leave: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    long = pd.concat([full, split, leave], ignore_index=True)
    story_means = (
        long.groupby(["analysis", "story", "hemisphere"], as_index=False)
        .agg(
            baseline_cv_r=("baseline_cv_r", "mean"),
            combined_cv_r=("combined_cv_r", "mean"),
            delta_cv_r=("delta_cv_r", "mean"),
        )
    )
    if story_means.groupby("analysis")["story"].nunique().ne(len(STORIES)).any():
        raise RuntimeError("A hemisphere analysis does not contain all 13 stories.")

    summary_rows = []
    paired_rows = []
    for analysis in ANALYSES:
        current = story_means[story_means["analysis"].eq(analysis)]
        for hemisphere in ["left", "right"]:
            row = {
                "scope": "all_13_intact_stories",
                "analysis": analysis,
                "hemisphere": hemisphere,
            }
            row.update(
                inference_summary(
                    current[current["hemisphere"].eq(hemisphere)]["delta_cv_r"]
                )
            )
            summary_rows.append(row)
        wide = current.pivot(index="story", columns="hemisphere", values="delta_cv_r")
        difference = wide["right"] - wide["left"]
        row = {
            "scope": "all_13_intact_stories",
            "analysis": analysis,
            "contrast": "right_minus_left",
        }
        row.update(inference_summary(difference))
        paired_rows.append(row)
    paired = pd.DataFrame(paired_rows)
    paired["p_fdr_across_three_tests"] = bh_fdr(paired["p_ttest_two_sided"])
    return story_means, pd.DataFrame(summary_rows), paired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="results/jnl_13_story_hemisphere_sensitivity"
    )
    parser.add_argument(
        "--embedding-dir", default="results/jnl_13_story_extension/loso/embeddings"
    )
    parser.add_argument(
        "--subject-list",
        default="configs/stories_all_fmriprep_confounds_motionfiltered.py",
    )
    parser.add_argument(
        "--regressor-dir",
        default=DEFAULT_REGRESSOR_DIR,
    )
    parser.add_argument(
        "--hemisphere-cache",
        default="results/jnl_13_story_resources/hemisphere_roi_cache",
    )
    parser.add_argument("--gentle-root", required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--cv-gap", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--subject-splits", type=int, default=50)
    parser.add_argument("--atlas-threshold", type=float, default=0.0)
    parser.add_argument("--fwhm", type=float, default=6.0)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    os.environ["SEMANTIC_FMRI_TIMING"] = "gentle"
    os.environ["SEMANTIC_FMRI_GENTLE_DIR"] = str(Path(args.gentle_root))
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = ROOT / args.hemisphere_cache
    cache.mkdir(parents=True, exist_ok=True)
    if args.rebuild_cache or not any(cache.glob("*.npz")):
        extract_heldout_hemisphere_targets(
            ROOT / args.subject_list,
            cache,
            atlas_threshold=args.atlas_threshold,
            fwhm=args.fwhm,
        )

    embeddings = load_embeddings(ROOT / args.embedding_dir)
    full_tables = []
    split_tables = []
    leave_tables = []
    for story in STORIES:
        full, split, leave = evaluate_story(args, story, embeddings)
        full_tables.append(full)
        split_tables.append(split)
        leave_tables.append(leave)

    full = pd.concat(full_tables, ignore_index=True)
    split = pd.concat(split_tables, ignore_index=True)
    leave = pd.concat(leave_tables, ignore_index=True)
    story, summary, paired = summarize(full, split, leave)
    full.to_csv(output_dir / "full_group_long.csv", index=False)
    split.to_csv(output_dir / "participant_split_half_long.csv", index=False)
    leave.to_csv(output_dir / "leave_one_participant_out_long.csv", index=False)
    story.to_csv(output_dir / "story_hemisphere_means.csv", index=False)
    summary.to_csv(output_dir / "hemisphere_inference_summary.csv", index=False)
    paired.to_csv(output_dir / "left_right_paired_summary.csv", index=False)

    print("\nAll-13 hemisphere effects:\n", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nAll-13 right-minus-left contrasts:\n", flush=True)
    print(paired.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
