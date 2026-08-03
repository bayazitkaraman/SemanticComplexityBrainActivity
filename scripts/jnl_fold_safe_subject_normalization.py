from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from jnl_heldout_story_replication import HELDOUT_STORIES, heldout_design  # noqa: E402
from jnl_shared_response_context_validation import (  # noqa: E402
    STORIES as DEVELOPMENT_STORIES,
    correlation_by_column,
    load_story_designs,
    load_story_targets,
    one_sample_summary,
)


def fold_safe_subject_average(
    raw_targets: np.ndarray,
    subject_indices: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    subject_targets = raw_targets[subject_indices]
    training_values = subject_targets[:, train_indices, :]
    mean = np.mean(training_values, axis=1, keepdims=True)
    scale = np.std(training_values, axis=1, keepdims=True)
    scale[(scale <= 0) | ~np.isfinite(scale)] = 1.0
    training_average = np.mean((training_values - mean) / scale, axis=0)
    test_average = np.mean(
        (subject_targets[:, test_indices, :] - mean) / scale,
        axis=0,
    )
    return training_average, test_average


def fold_safe_cv_r(
    x: np.ndarray,
    raw_targets: np.ndarray,
    training_subjects: np.ndarray,
    test_subjects: np.ndarray,
    n_splits: int,
    alpha: float,
    cv_gap: int,
) -> np.ndarray:
    true_parts = []
    prediction_parts = []
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=cv_gap)
    for train_indices, test_indices in splitter.split(x):
        training_target, _ = fold_safe_subject_average(
            raw_targets,
            training_subjects,
            train_indices,
            test_indices,
        )
        _, test_target = fold_safe_subject_average(
            raw_targets,
            test_subjects,
            train_indices,
            test_indices,
        )

        trend_design = np.column_stack(
            [train_indices.astype(float), np.ones(len(train_indices), dtype=float)]
        )
        trend_beta = np.linalg.lstsq(
            trend_design,
            training_target,
            rcond=None,
        )[0]
        y_train = training_target - trend_design @ trend_beta
        test_design = np.column_stack(
            [test_indices.astype(float), np.ones(len(test_indices), dtype=float)]
        )
        y_test = test_target - test_design @ trend_beta

        mean = np.mean(y_train, axis=0, keepdims=True)
        scale = np.std(y_train, axis=0, keepdims=True)
        scale[(scale <= 0) | ~np.isfinite(scale)] = 1.0
        y_train = (y_train - mean) / scale
        y_test = (y_test - mean) / scale

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha, fit_intercept=True)),
            ]
        )
        model.fit(x[train_indices], y_train)
        true_parts.append(y_test)
        prediction_parts.append(model.predict(x[test_indices]))
    return correlation_by_column(
        np.vstack(true_parts),
        np.vstack(prediction_parts),
    )


def load_design(story: str) -> tuple[np.ndarray, np.ndarray]:
    if story in DEVELOPMENT_STORIES:
        args = SimpleNamespace(
            rich_audio_regressor_dir=(
                "results/jnl_revision_major_review/rich_audio_controls/"
                "rich_audio_regressors"
            ),
            cumulative_context_regressor_dir=(
                "results/jnl_revision_major_review/context_layer_models/regressors/"
                "cumulative_context_layer12_current_tokens"
            ),
            independent_chunk_regressor_dir="results/regressors",
        )
        designs = load_story_designs(args, story)
        return designs["rich_audio"], designs["cumulative_context"]
    designs = heldout_design(
        ROOT / "results" / "jnl_heldout_story_replication" / "heldout_regressors",
        story,
    )
    return designs["baseline"], designs["combined"]


def cache_dir(story: str) -> Path:
    if story in DEVELOPMENT_STORIES:
        return (
            ROOT
            / "results"
            / "journal_neurolinguistics_robustness"
            / "pc_count_full_corrected"
            / "roi_time_series_cache"
        )
    return ROOT / "results" / "jnl_heldout_story_replication" / "roi_cache"


def evaluate_story(
    story: str,
    n_splits: int,
    alpha: float,
    cv_gap: int,
    subject_splits: int,
) -> tuple[dict, list[dict], list[dict]]:
    targets, _ = load_story_targets(cache_dir(story), story, standardize=False)
    baseline, combined = load_design(story)
    target_length = targets.shape[1]
    if len(baseline) < target_length or len(combined) < target_length:
        raise ValueError(
            f"Design is shorter than the aligned target for {story}: "
            f"baseline={len(baseline)}, combined={len(combined)}, target={target_length}"
        )
    baseline = baseline[:target_length]
    combined = combined[:target_length]
    all_subjects = np.arange(len(targets), dtype=int)

    baseline_r = fold_safe_cv_r(
        baseline, targets, all_subjects, all_subjects, n_splits, alpha, cv_gap
    )
    combined_r = fold_safe_cv_r(
        combined, targets, all_subjects, all_subjects, n_splits, alpha, cv_gap
    )
    full_value = float(np.mean(combined_r - baseline_r))

    split_rows = []
    for seed in range(subject_splits):
        shuffled = np.random.default_rng(seed).permutation(len(targets))
        first = shuffled[: len(shuffled) // 2]
        second = shuffled[len(shuffled) // 2 :]
        baseline_split = (
            fold_safe_cv_r(
                baseline, targets, first, second, n_splits, alpha, cv_gap
            )
            + fold_safe_cv_r(
                baseline, targets, second, first, n_splits, alpha, cv_gap
            )
        ) / 2.0
        combined_split = (
            fold_safe_cv_r(
                combined, targets, first, second, n_splits, alpha, cv_gap
            )
            + fold_safe_cv_r(
                combined, targets, second, first, n_splits, alpha, cv_gap
            )
        ) / 2.0
        split_rows.append(
            {
                "story": story,
                "seed": seed,
                "mean_delta_cv_r": float(np.mean(combined_split - baseline_split)),
            }
        )

    cache_files = sorted(cache_dir(story).glob(f"{story}_*.npz"))
    subjects = [path.stem[len(story) + 1 :] for path in cache_files]
    leave_rows = []
    for heldout_index, subject in enumerate(subjects):
        training = all_subjects[all_subjects != heldout_index]
        testing = np.asarray([heldout_index], dtype=int)
        baseline_leave = fold_safe_cv_r(
            baseline, targets, training, testing, n_splits, alpha, cv_gap
        )
        combined_leave = fold_safe_cv_r(
            combined, targets, training, testing, n_splits, alpha, cv_gap
        )
        leave_rows.append(
            {
                "story": story,
                "subject": subject,
                "mean_delta_cv_r": float(np.mean(combined_leave - baseline_leave)),
            }
        )

    split_value = float(pd.DataFrame(split_rows)["mean_delta_cv_r"].mean())
    leave_value = float(pd.DataFrame(leave_rows)["mean_delta_cv_r"].mean())
    story_row = {
        "story": story,
        "scope": "heldout" if story in HELDOUT_STORIES else "development",
        "n_subjects": len(targets),
        "n_timepoints": target_length,
        "full_group": full_value,
        "split_half": split_value,
        "leave_one_participant_out": leave_value,
    }
    return story_row, split_rows, leave_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="results/jnl_fold_safe_subject_normalization",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--cv-gap", type=int, default=10)
    parser.add_argument("--subject-splits", type=int, default=50)
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    story_rows = []
    split_rows = []
    leave_rows = []
    for story in DEVELOPMENT_STORIES + HELDOUT_STORIES:
        print(f"Fold-safe participant normalization: {story}", flush=True)
        story_row, story_splits, story_leaves = evaluate_story(
            story,
            args.n_splits,
            args.alpha,
            args.cv_gap,
            args.subject_splits,
        )
        story_rows.append(story_row)
        split_rows.extend(story_splits)
        leave_rows.extend(story_leaves)

    stories = pd.DataFrame(story_rows)
    summary_rows = []
    for scope, selected in [
        ("development", stories["scope"].eq("development")),
        ("heldout", stories["scope"].eq("heldout")),
        ("pooled", np.ones(len(stories), dtype=bool)),
    ]:
        for analysis in ["full_group", "split_half", "leave_one_participant_out"]:
            row = {"scope": scope, "analysis": analysis}
            row.update(one_sample_summary(stories.loc[selected, analysis]))
            summary_rows.append(row)

    stories.to_csv(output_dir / "fold_safe_story_means.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output_dir / "fold_safe_split_half.csv", index=False)
    pd.DataFrame(leave_rows).to_csv(
        output_dir / "fold_safe_leave_one_participant_out.csv",
        index=False,
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "fold_safe_summary.csv", index=False)
    print("\nFold-safe participant-normalization sensitivity:\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
