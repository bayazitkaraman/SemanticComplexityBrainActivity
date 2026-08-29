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
    DEFAULT_ROI_CACHE,
    STORIES,
    baseline_matrix,
    cache_dir,
    fold_scores,
    source_regressor_path,
    story_subjects,
    with_pcs,
)
from jnl_fold_safe_subject_normalization import fold_safe_cv_r  # noqa: E402
from jnl_shared_response_context_validation import (  # noqa: E402
    load_story_targets,
    one_sample_summary,
)


EXPECTED_SCANS = 567


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


def evaluate_story(
    args: argparse.Namespace,
    story: str,
    embeddings: dict[str, np.ndarray],
) -> tuple[dict, list[dict], list[dict]]:
    targets, _ = load_story_targets(cache_dir(args, story), story, standardize=False)
    baseline = baseline_matrix(pd.read_csv(source_regressor_path(args, story)))
    scores, _ = fold_scores(embeddings, story)
    combined = with_pcs(baseline, scores[story])
    n = min(targets.shape[1], len(baseline), len(combined))
    targets = targets[:, :n]
    baseline = baseline[:n]
    combined = combined[:n]
    subjects = story_subjects(args, story)
    if len(subjects) != len(targets):
        raise RuntimeError(f"Target/subject count mismatch for {story}.")
    all_subjects = np.arange(len(targets), dtype=int)
    print(f"All-13 fold-safe normalization: {story} ({len(subjects)} scans)", flush=True)

    baseline_full = fold_safe_cv_r(
        baseline,
        targets,
        all_subjects,
        all_subjects,
        args.n_splits,
        args.alpha,
        args.cv_gap,
    )
    combined_full = fold_safe_cv_r(
        combined,
        targets,
        all_subjects,
        all_subjects,
        args.n_splits,
        args.alpha,
        args.cv_gap,
    )

    split_rows = []
    for seed in range(args.subject_splits):
        indices = np.random.default_rng(seed).permutation(len(targets))
        first = indices[: len(indices) // 2]
        second = indices[len(indices) // 2 :]
        baseline_split = (
            fold_safe_cv_r(
                baseline, targets, first, second,
                args.n_splits, args.alpha, args.cv_gap,
            )
            + fold_safe_cv_r(
                baseline, targets, second, first,
                args.n_splits, args.alpha, args.cv_gap,
            )
        ) / 2.0
        combined_split = (
            fold_safe_cv_r(
                combined, targets, first, second,
                args.n_splits, args.alpha, args.cv_gap,
            )
            + fold_safe_cv_r(
                combined, targets, second, first,
                args.n_splits, args.alpha, args.cv_gap,
            )
        ) / 2.0
        split_rows.append(
            {
                "story": story,
                "seed": seed,
                "mean_delta_cv_r": float(np.mean(combined_split - baseline_split)),
            }
        )

    leave_rows = []
    for heldout_index, subject in enumerate(subjects):
        training = all_subjects[all_subjects != heldout_index]
        testing = np.asarray([heldout_index], dtype=int)
        baseline_leave = fold_safe_cv_r(
            baseline,
            targets,
            training,
            testing,
            args.n_splits,
            args.alpha,
            args.cv_gap,
        )
        combined_leave = fold_safe_cv_r(
            combined,
            targets,
            training,
            testing,
            args.n_splits,
            args.alpha,
            args.cv_gap,
        )
        leave_rows.append(
            {
                "story": story,
                "subject": subject,
                "mean_delta_cv_r": float(np.mean(combined_leave - baseline_leave)),
            }
        )

    return (
        {
            "story": story,
            "n_subjects": len(subjects),
            "n_timepoints": n,
            "full_group": float(np.mean(combined_full - baseline_full)),
            "participant_split_half": float(
                pd.DataFrame(split_rows)["mean_delta_cv_r"].mean()
            ),
            "leave_one_participant_out": float(
                pd.DataFrame(leave_rows)["mean_delta_cv_r"].mean()
            ),
        },
        split_rows,
        leave_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="results/jnl_13_story_fold_safe_normalization"
    )
    parser.add_argument(
        "--embedding-dir", default="results/jnl_13_story_extension/loso/embeddings"
    )
    parser.add_argument(
        "--regressor-dir",
        default=DEFAULT_REGRESSOR_DIR,
    )
    parser.add_argument(
        "--roi-cache",
        default=DEFAULT_ROI_CACHE,
    )
    parser.add_argument("--gentle-root", required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--cv-gap", type=int, default=10)
    parser.add_argument("--subject-splits", type=int, default=50)
    args = parser.parse_args()

    os.environ["SEMANTIC_FMRI_TIMING"] = "gentle"
    os.environ["SEMANTIC_FMRI_GENTLE_DIR"] = str(Path(args.gentle_root))
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings = load_embeddings(ROOT / args.embedding_dir)

    story_rows = []
    split_rows = []
    leave_rows = []
    for story in STORIES:
        story_row, current_splits, current_leaves = evaluate_story(
            args, story, embeddings
        )
        story_rows.append(story_row)
        split_rows.extend(current_splits)
        leave_rows.extend(current_leaves)

    stories = pd.DataFrame(story_rows)
    if len(stories) != len(STORIES) or len(leave_rows) != EXPECTED_SCANS:
        raise RuntimeError(
            f"Fold-safe analysis has {len(stories)} stories and {len(leave_rows)} scans; "
            "expected 13 and 567."
        )
    summary_rows = []
    for analysis in [
        "full_group",
        "participant_split_half",
        "leave_one_participant_out",
    ]:
        row = {
            "scope": "all_13_intact_stories",
            "analysis": analysis,
        }
        row.update(one_sample_summary(stories[analysis]))
        summary_rows.append(row)

    stories.to_csv(output_dir / "fold_safe_story_means.csv", index=False)
    pd.DataFrame(split_rows).to_csv(
        output_dir / "fold_safe_split_half.csv", index=False
    )
    pd.DataFrame(leave_rows).to_csv(
        output_dir / "fold_safe_leave_one_participant_out.csv", index=False
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "fold_safe_summary.csv", index=False)
    print("\nAll-13 fold-safe participant-normalization sensitivity:\n", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
