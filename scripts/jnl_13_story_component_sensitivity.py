from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import jnl_13_story_loso as loso  # noqa: E402
from jnl_shared_response_context_validation import one_sample_summary  # noqa: E402
from semantic_fmri.utils import hrf_convolve  # noqa: E402


def default_args() -> argparse.Namespace:
    return argparse.Namespace(
        regressor_dir=loso.DEFAULT_REGRESSOR_DIR,
        roi_cache=loso.DEFAULT_ROI_CACHE,
        alpha=10.0,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_design(
    baseline: np.ndarray, scores: np.ndarray, n_components: int
) -> np.ndarray:
    convolved = np.column_stack(
        [hrf_convolve(scores[:, index], tr=1.5) for index in range(n_components)]
    )
    n_timepoints = min(len(baseline), len(convolved))
    return np.column_stack([baseline[:n_timepoints], convolved[:n_timepoints]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="results/jnl_13_story_extension/component_count_sensitivity",
    )
    parser.add_argument("--maximum-components", type=int, default=10)
    args = parser.parse_args()

    model_args = default_args()
    embeddings = {
        story: np.load(
            ROOT
            / "results"
            / "jnl_13_story_extension"
            / "loso"
            / "embeddings"
            / f"{story}.npy",
            allow_pickle=False,
        )
        for story in loso.STORIES
    }
    baselines, targets, roi_names = loso.load_data(model_args)
    rows: list[dict] = []
    metadata: list[dict] = []

    for omitted_story in loso.STORIES:
        training_stories = [story for story in loso.STORIES if story != omitted_story]
        mean, scale, components, eigenvalues = loso.story_balanced_pca(
            embeddings,
            training_stories,
            n_components=args.maximum_components,
        )
        projected = {
            story: loso.project(embeddings[story], mean, scale, components)
            for story in loso.STORIES
        }
        metadata.append(
            {
                "omitted_story": omitted_story,
                **{
                    f"pc{index + 1}_variance_ratio": float(
                        eigenvalues[index] / eigenvalues.sum()
                    )
                    for index in range(args.maximum_components)
                },
            }
        )

        for n_components in range(1, args.maximum_components + 1):
            designs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for story in loso.STORIES:
                target = targets[story].mean(axis=0)
                baseline = baselines[story]
                combined = combined_design(
                    baseline, projected[story], n_components
                )
                n_timepoints = min(len(target), len(baseline), len(combined))
                designs[story] = (
                    baseline[:n_timepoints],
                    combined[:n_timepoints],
                    target[:n_timepoints],
                )

            baseline_r = loso.transfer_model(
                [designs[story][0] for story in training_stories],
                [designs[story][2] for story in training_stories],
                designs[omitted_story][0],
                designs[omitted_story][2],
                model_args.alpha,
            )
            combined_r = loso.transfer_model(
                [designs[story][1] for story in training_stories],
                [designs[story][2] for story in training_stories],
                designs[omitted_story][1],
                designs[omitted_story][2],
                model_args.alpha,
            )
            delta = combined_r - baseline_r
            for roi, baseline_value, combined_value, delta_value in zip(
                roi_names[omitted_story], baseline_r, combined_r, delta
            ):
                rows.append(
                    {
                        "story": omitted_story,
                        "n_components": n_components,
                        "roi": roi,
                        "baseline_r": float(baseline_value),
                        "combined_r": float(combined_value),
                        "delta_r": float(delta_value),
                    }
                )

    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    long_frame = pd.DataFrame(rows)
    long_frame.to_csv(output / "component_count_long.csv", index=False)
    story_frame = (
        long_frame.groupby(["n_components", "story"], as_index=False)
        .agg(
            baseline_r=("baseline_r", "mean"),
            combined_r=("combined_r", "mean"),
            delta_r=("delta_r", "mean"),
            positive_rois=("delta_r", lambda values: int((values > 0).sum())),
        )
    )
    story_frame.to_csv(output / "component_count_story_means.csv", index=False)

    summary_rows = []
    for n_components, frame in story_frame.groupby("n_components"):
        summary = {"n_components": int(n_components)}
        summary.update(one_sample_summary(frame["delta_r"]))
        summary_rows.append(summary)
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(output / "component_count_summary.csv", index=False)
    pd.DataFrame(metadata).to_csv(output / "pca_fold_metadata.csv", index=False)

    primary = pd.read_csv(
        ROOT
        / "results"
        / "jnl_13_story_extension"
        / "loso"
        / "cross_story_transfer_story_means.csv"
    ).sort_values("story")
    reproduced = story_frame.query("n_components == 5").sort_values("story")
    maximum_error = float(
        np.max(
            np.abs(
                primary["delta_r"].to_numpy(dtype=float)
                - reproduced["delta_r"].to_numpy(dtype=float)
            )
        )
    )
    if maximum_error > 1e-12:
        raise RuntimeError(f"Five-component result was not reproduced: {maximum_error}")

    audit = {
        "analysis": "13-story cross-narrative component-count sensitivity",
        "component_counts": list(range(1, args.maximum_components + 1)),
        "primary_component_count": 5,
        "status": "exploratory sensitivity; primary model unchanged",
        "maximum_primary_reproduction_error": maximum_error,
        "script_sha256": sha256(Path(__file__)),
    }
    (output / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(summary_frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
