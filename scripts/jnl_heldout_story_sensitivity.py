from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from jnl_heldout_story_replication import HELDOUT_STORIES, heldout_design  # noqa: E402
from jnl_pc_count_sensitivity import vectorized_cv_r  # noqa: E402
from jnl_shared_response_context_validation import (  # noqa: E402
    cross_group_cv_r,
    load_story_targets,
    one_sample_summary,
)
from semantic_fmri.utils import hrf_convolve  # noqa: E402


def load_analysis_data(
    replication_dir: Path,
) -> dict[str, tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]]:
    cache_dir = replication_dir / "roi_cache"
    regressor_dir = replication_dir / "heldout_regressors"
    story_data = {}

    for story in HELDOUT_STORIES:
        targets, _ = load_story_targets(cache_dir, story)
        design = heldout_design(regressor_dir, story)
        reg = pd.read_csv(regressor_dir / f"{story}_shared_pca_regressors.csv")
        silence = reg["word_count"].eq(0).to_numpy(dtype=bool)
        silence_hrf = hrf_convolve(silence.astype(float), tr=1.5)

        raw_pcs = reg[[f"PC{i}" for i in range(1, 6)]].to_numpy(dtype=float)
        raw_pcs[silence] = 0.0
        zero_silence_pcs = np.column_stack(
            [hrf_convolve(raw_pcs[:, index], tr=1.5) for index in range(5)]
        )

        baseline_with_silence = np.column_stack([design["baseline"], silence_hrf])
        variants = {
            "locked_primary": (design["baseline"], design["combined"]),
            "explicit_silence_control": (
                baseline_with_silence,
                np.column_stack([baseline_with_silence, design["pcs"]]),
            ),
            "zero_semantics_during_silence": (
                baseline_with_silence,
                np.column_stack([baseline_with_silence, zero_silence_pcs]),
            ),
        }
        story_data[story] = (targets, variants)
    return story_data


def evaluate_setting(
    story_data: dict[str, tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]],
    variant: str,
    alpha: float,
    cv_gap: int,
    n_splits: int,
    subject_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_rows = []
    split_rows = []

    for story, (targets, variants) in story_data.items():
        baseline, combined = variants[variant]
        group_target = np.mean(targets, axis=0)
        baseline_r, _ = vectorized_cv_r(
            baseline, group_target, n_splits, alpha, True, cv_gap
        )
        combined_r, _ = vectorized_cv_r(
            combined, group_target, n_splits, alpha, True, cv_gap
        )
        full_rows.append(
            {
                "story": story,
                "variant": variant,
                "alpha": alpha,
                "cv_gap": cv_gap,
                "mean_delta_cv_r": float(np.mean(combined_r - baseline_r)),
            }
        )

        for seed in range(subject_splits):
            indices = np.random.default_rng(seed).permutation(len(targets))
            first = indices[: len(indices) // 2]
            second = indices[len(indices) // 2 :]
            baseline_r = (
                cross_group_cv_r(
                    baseline, targets, first, second, n_splits, alpha, cv_gap
                )
                + cross_group_cv_r(
                    baseline, targets, second, first, n_splits, alpha, cv_gap
                )
            ) / 2.0
            combined_r = (
                cross_group_cv_r(
                    combined, targets, first, second, n_splits, alpha, cv_gap
                )
                + cross_group_cv_r(
                    combined, targets, second, first, n_splits, alpha, cv_gap
                )
            ) / 2.0
            split_rows.append(
                {
                    "story": story,
                    "seed": seed,
                    "variant": variant,
                    "alpha": alpha,
                    "cv_gap": cv_gap,
                    "mean_delta_cv_r": float(np.mean(combined_r - baseline_r)),
                }
            )

    return pd.DataFrame(full_rows), pd.DataFrame(split_rows)


def summarize(
    full: pd.DataFrame,
    split: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    keys = ["variant", "alpha", "cv_gap"]
    for key, group in full.groupby(keys, sort=False):
        row = {
            "analysis": "full_group",
            "variant": key[0],
            "alpha": key[1],
            "cv_gap": key[2],
        }
        row.update(one_sample_summary(group["mean_delta_cv_r"]))
        rows.append(row)

    split_story = (
        split.groupby(keys + ["story"], as_index=False)["mean_delta_cv_r"].mean()
    )
    for key, group in split_story.groupby(keys, sort=False):
        row = {
            "analysis": "held_out_participant_split_half",
            "variant": key[0],
            "alpha": key[1],
            "cv_gap": key[2],
        }
        row.update(one_sample_summary(group["mean_delta_cv_r"]))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replication-dir",
        default="results/jnl_heldout_story_replication",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--subject-splits", type=int, default=50)
    args = parser.parse_args()

    replication_dir = ROOT / args.replication_dir
    story_data = load_analysis_data(replication_dir)
    settings = [
        ("locked_primary", 10.0, 10),
        ("explicit_silence_control", 10.0, 10),
        ("zero_semantics_during_silence", 10.0, 10),
        ("locked_primary", 1.0, 10),
        ("locked_primary", 100.0, 10),
        ("locked_primary", 1000.0, 10),
        ("locked_primary", 10.0, 5),
        ("locked_primary", 10.0, 15),
    ]

    full_parts = []
    split_parts = []
    for variant, alpha, cv_gap in settings:
        print(f"Evaluating {variant}, alpha={alpha:g}, gap={cv_gap}", flush=True)
        full, split = evaluate_setting(
            story_data,
            variant,
            alpha,
            cv_gap,
            args.n_splits,
            args.subject_splits,
        )
        full_parts.append(full)
        split_parts.append(split)

    full = pd.concat(full_parts, ignore_index=True)
    split = pd.concat(split_parts, ignore_index=True)
    summary = summarize(full, split)

    full.to_csv(replication_dir / "heldout_sensitivity_full_group.csv", index=False)
    split.to_csv(replication_dir / "heldout_sensitivity_split_half.csv", index=False)
    summary.to_csv(replication_dir / "heldout_sensitivity_summary.csv", index=False)

    seed_summary = (
        split.groupby(["variant", "alpha", "cv_gap", "seed"], as_index=False)[
            "mean_delta_cv_r"
        ]
        .mean()
        .groupby(["variant", "alpha", "cv_gap"], as_index=False)
        .agg(
            mean_across_seeds=("mean_delta_cv_r", "mean"),
            minimum_seed_mean=("mean_delta_cv_r", "min"),
            maximum_seed_mean=("mean_delta_cv_r", "max"),
            n_positive_seeds=("mean_delta_cv_r", lambda value: int((value > 0).sum())),
            n_seeds=("mean_delta_cv_r", "size"),
        )
    )
    seed_summary.to_csv(
        replication_dir / "heldout_sensitivity_seed_stability.csv", index=False
    )

    print("\nSensitivity summary:\n", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nSplit-seed stability:\n", flush=True)
    print(seed_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
