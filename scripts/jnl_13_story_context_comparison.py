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
    cross_story_transfer,
    load_data,
    save_within_results,
    within_story_loso,
)
from jnl_revision_context_layer_models import mixed_effect_intercept  # noqa: E402
from jnl_shared_response_context_validation import one_sample_summary  # noqa: E402


EXPECTED_SCANS = 567


def load_embeddings(directory: Path) -> dict[str, np.ndarray]:
    missing = [story for story in STORIES if not (directory / f"{story}.npy").exists()]
    if missing:
        raise FileNotFoundError(f"Missing embeddings in {directory}: {missing}")
    embeddings = {
        story: np.load(directory / f"{story}.npy", allow_pickle=False)
        for story in STORIES
    }
    if any(values.ndim != 2 for values in embeddings.values()):
        raise ValueError(f"Every embedding array in {directory} must be two-dimensional.")
    if not all(np.all(np.isfinite(values)) for values in embeddings.values()):
        raise ValueError(f"Non-finite embedding values found in {directory}.")
    return embeddings


def representation_story_values(
    full: pd.DataFrame,
    split: pd.DataFrame,
    leave: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return {
        "full_group": (
            full.groupby("story", as_index=False)["delta_cv_r"].mean()
        ),
        "participant_split_half": (
            split.groupby("story", as_index=False)["mean_delta_cv_r"].mean()
            .rename(columns={"mean_delta_cv_r": "delta_cv_r"})
        ),
        "leave_one_participant_out": (
            leave.groupby("story", as_index=False)["mean_delta_cv_r"].mean()
            .rename(columns={"mean_delta_cv_r": "delta_cv_r"})
        ),
    }


def paired_story_values(
    cumulative: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    independent: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cumulative_full, cumulative_split, cumulative_leave = cumulative
    independent_full, independent_split, independent_leave = independent

    rows: list[dict] = []
    paired_scan_rows: list[dict] = []

    full = cumulative_full.merge(
        independent_full,
        on=["story", "roi"],
        suffixes=("_cumulative", "_independent"),
        validate="one_to_one",
    )
    full["difference"] = full["delta_cv_r_cumulative"] - full["delta_cv_r_independent"]
    for story, group in full.groupby("story"):
        rows.append(
            {
                "analysis": "full_group",
                "story": story,
                "cumulative_delta_cv_r": float(group["delta_cv_r_cumulative"].mean()),
                "independent_delta_cv_r": float(group["delta_cv_r_independent"].mean()),
                "cumulative_minus_independent": float(group["difference"].mean()),
            }
        )

    split = cumulative_split.merge(
        independent_split,
        on=["story", "seed"],
        suffixes=("_cumulative", "_independent"),
        validate="one_to_one",
    )
    split["difference"] = (
        split["mean_delta_cv_r_cumulative"] - split["mean_delta_cv_r_independent"]
    )
    for story, group in split.groupby("story"):
        rows.append(
            {
                "analysis": "participant_split_half",
                "story": story,
                "cumulative_delta_cv_r": float(group["mean_delta_cv_r_cumulative"].mean()),
                "independent_delta_cv_r": float(group["mean_delta_cv_r_independent"].mean()),
                "cumulative_minus_independent": float(group["difference"].mean()),
            }
        )

    leave = cumulative_leave.merge(
        independent_leave,
        on=["story", "subject"],
        suffixes=("_cumulative", "_independent"),
        validate="one_to_one",
    )
    leave["difference"] = (
        leave["mean_delta_cv_r_cumulative"] - leave["mean_delta_cv_r_independent"]
    )
    for story, group in leave.groupby("story"):
        rows.append(
            {
                "analysis": "leave_one_participant_out",
                "story": story,
                "cumulative_delta_cv_r": float(group["mean_delta_cv_r_cumulative"].mean()),
                "independent_delta_cv_r": float(group["mean_delta_cv_r_independent"].mean()),
                "cumulative_minus_independent": float(group["difference"].mean()),
            }
        )
    paired_scan_rows = leave[
        [
            "story",
            "subject",
            "mean_delta_cv_r_cumulative",
            "mean_delta_cv_r_independent",
            "difference",
        ]
    ].rename(columns={"difference": "cumulative_minus_independent"}).to_dict("records")

    story = pd.DataFrame(rows)
    paired_scan = pd.DataFrame(paired_scan_rows)
    if story.groupby("analysis")["story"].nunique().ne(len(STORIES)).any():
        raise RuntimeError("A direct comparison does not contain all 13 stories.")
    if len(paired_scan) != EXPECTED_SCANS:
        raise RuntimeError(
            f"Expected {EXPECTED_SCANS} paired leave-one-participant-out scans, "
            f"found {len(paired_scan)}."
        )
    return story, paired_scan


def inference_table(story: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for analysis, group in story.groupby("analysis", sort=False):
        for representation, column in [
            ("cumulative_added", "cumulative_delta_cv_r"),
            ("independent_added", "independent_delta_cv_r"),
            ("cumulative_minus_independent", "cumulative_minus_independent"),
        ]:
            values = pd.to_numeric(group[column], errors="coerce")
            row = {
                "analysis": analysis,
                "representation": representation,
                "scope": "all_13_intact_stories",
            }
            row.update(one_sample_summary(values))
            rows.append(row)
    return pd.DataFrame(rows)


def transfer_comparison(
    cumulative: pd.DataFrame,
    independent: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cumulative_story = cumulative.groupby("story", as_index=False)["delta_r"].mean().rename(
        columns={"delta_r": "cumulative_delta_r"}
    )
    independent_story = independent.groupby("story", as_index=False)["delta_r"].mean().rename(
        columns={"delta_r": "independent_delta_r"}
    )
    story = cumulative_story.merge(
        independent_story, on="story", validate="one_to_one"
    )
    story["cumulative_minus_independent"] = (
        story["cumulative_delta_r"] - story["independent_delta_r"]
    )
    if len(story) != len(STORIES):
        raise RuntimeError("Cross-story comparison does not contain all 13 stories.")
    rows = []
    for representation, column in [
        ("cumulative_added", "cumulative_delta_r"),
        ("independent_added", "independent_delta_r"),
        ("cumulative_minus_independent", "cumulative_minus_independent"),
    ]:
        row = {
            "analysis": "train_12_test_1_full_story",
            "representation": representation,
            "scope": "all_13_intact_stories",
        }
        row.update(one_sample_summary(story[column]))
        rows.append(row)
    return story, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="results/jnl_13_story_context_comparison"
    )
    parser.add_argument(
        "--cumulative-embedding-dir",
        default="results/jnl_13_story_extension/loso/embeddings",
    )
    parser.add_argument(
        "--independent-embedding-dir",
        default="results/jnl_13_story_individual_local/independent_embeddings",
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
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--cv-gap", type=int, default=10)
    parser.add_argument("--subject-splits", type=int, default=50)
    args = parser.parse_args()

    os.environ["SEMANTIC_FMRI_TIMING"] = "gentle"
    os.environ["SEMANTIC_FMRI_GENTLE_DIR"] = str(Path(args.gentle_root))
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cumulative_embeddings = load_embeddings(ROOT / args.cumulative_embedding_dir)
    independent_embeddings = load_embeddings(ROOT / args.independent_embedding_dir)
    baselines, targets, roi_names = load_data(args)
    for story in STORIES:
        expected = min(len(baselines[story]), targets[story].shape[1])
        if len(cumulative_embeddings[story]) < expected or len(independent_embeddings[story]) < expected:
            raise RuntimeError(f"Embedding/design length mismatch for {story}.")

    representation_results = {}
    for name, embeddings in [
        ("cumulative", cumulative_embeddings),
        ("independent", independent_embeddings),
    ]:
        print(f"Running all-13 within-story analysis: {name}", flush=True)
        full, split, leave, metadata = within_story_loso(
            args, embeddings, baselines, targets, roi_names
        )
        save_within_results(output_dir / name, full, split, leave, metadata)
        representation_results[name] = (full, split, leave)

    story, paired_scan = paired_story_values(
        representation_results["cumulative"],
        representation_results["independent"],
    )
    inference = inference_table(story)
    story.to_csv(output_dir / "within_story_direct_comparison_story.csv", index=False)
    paired_scan.to_csv(
        output_dir / "within_story_direct_comparison_scan.csv", index=False
    )
    inference.to_csv(output_dir / "within_story_direct_comparison_inference.csv", index=False)

    mixed_rows = []
    for label, column in [
        ("cumulative_added", "mean_delta_cv_r_cumulative"),
        ("independent_added", "mean_delta_cv_r_independent"),
        ("cumulative_minus_independent", "cumulative_minus_independent"),
    ]:
        mixed_rows.append(mixed_effect_intercept(paired_scan, column, label))
    pd.DataFrame(mixed_rows).to_csv(
        output_dir / "within_story_leave_one_participant_out_mixed_effects.csv",
        index=False,
    )

    print("Running all-13 cross-story transfer: cumulative", flush=True)
    cumulative_transfer = cross_story_transfer(
        args, cumulative_embeddings, baselines, targets, roi_names
    )
    print("Running all-13 cross-story transfer: independent", flush=True)
    independent_transfer = cross_story_transfer(
        args, independent_embeddings, baselines, targets, roi_names
    )
    cumulative_transfer.to_csv(
        output_dir / "cross_story_cumulative_long.csv", index=False
    )
    independent_transfer.to_csv(
        output_dir / "cross_story_independent_long.csv", index=False
    )
    transfer_story, transfer_inference = transfer_comparison(
        cumulative_transfer, independent_transfer
    )
    transfer_story.to_csv(
        output_dir / "cross_story_direct_comparison_story.csv", index=False
    )
    transfer_inference.to_csv(
        output_dir / "cross_story_direct_comparison_inference.csv", index=False
    )

    print("\nWithin-story direct comparison:\n", flush=True)
    print(inference.to_string(index=False), flush=True)
    print("\nCross-story direct comparison:\n", flush=True)
    print(transfer_inference.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
