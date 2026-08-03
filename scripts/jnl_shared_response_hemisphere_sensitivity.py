from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from jnl_heldout_story_replication import (  # noqa: E402
    HELDOUT_STORIES,
    heldout_design,
)
from jnl_pc_count_sensitivity import vectorized_cv_r  # noqa: E402
from jnl_rescue_network_model_comparisons import bh_fdr  # noqa: E402
from jnl_revision_hemisphere_sensitivity import load_or_extract_cache  # noqa: E402
from jnl_shared_response_context_validation import (  # noqa: E402
    STORIES as DEVELOPMENT_STORIES,
    cross_group_cv_r,
    load_story_designs,
    load_story_targets,
)
from semantic_fmri.utils import (  # noqa: E402
    fetch_harvard_oxford_atlas,
    load_story_list,
)


ANALYSES = [
    "full_group",
    "participant_split_half",
    "leave_one_participant_out",
]


def inference_summary(values: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(clean)
    if n < 2:
        raise ValueError("At least two story values are required for inference")
    mean = float(np.mean(clean))
    sem = float(stats.sem(clean))
    ci_low, ci_high = stats.t.interval(0.95, n - 1, loc=mean, scale=sem)
    t_result = stats.ttest_1samp(clean, 0.0)
    try:
        wilcoxon_p = float(stats.wilcoxon(clean, alternative="greater").pvalue)
        wilcoxon_two_sided_p = float(
            stats.wilcoxon(clean, alternative="two-sided").pvalue
        )
    except ValueError:
        wilcoxon_p = np.nan
        wilcoxon_two_sided_p = np.nan
    positives = int(np.sum(clean > 0))
    return {
        "n_stories": n,
        "mean": mean,
        "median": float(np.median(clean)),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "t": float(t_result.statistic),
        "p_ttest_two_sided": float(t_result.pvalue),
        "p_wilcoxon_greater": wilcoxon_p,
        "p_wilcoxon_two_sided": wilcoxon_two_sided_p,
        "n_positive_stories": positives,
        "p_sign_greater": float(
            stats.binomtest(positives, n, 0.5, alternative="greater").pvalue
        ),
    }


def extract_heldout_hemisphere_targets(
    config_path: Path,
    cache_dir: Path,
    atlas_threshold: float,
    fwhm: float,
) -> None:
    rows = load_story_list(str(config_path))
    atlas_img, atlas_labels = fetch_harvard_oxford_atlas()
    for index, story in enumerate(rows, start=1):
        print(
            f"Hemisphere ROI extraction [{index}/{len(rows)}] "
            f"{story['name']} {story['subject']}",
            flush=True,
        )
        load_or_extract_cache(
            story,
            atlas_img,
            atlas_labels,
            cache_dir=cache_dir,
            atlas_threshold=atlas_threshold,
            fwhm=fwhm,
        )


def load_design(story: str, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if story in DEVELOPMENT_STORIES:
        design_args = SimpleNamespace(
            rich_audio_regressor_dir=args.development_rich_audio_regressor_dir,
            cumulative_context_regressor_dir=args.development_cumulative_regressor_dir,
            independent_chunk_regressor_dir=args.development_independent_regressor_dir,
        )
        designs = load_story_designs(design_args, story)
        return designs["rich_audio"], designs["cumulative_context"]

    designs = heldout_design(ROOT / args.heldout_regressor_dir, story)
    return designs["baseline"], designs["combined"]


def validate_roi_names(roi_names: list[str], story: str) -> dict[str, list[int]]:
    indices: dict[str, list[int]] = {"left": [], "right": []}
    labels: dict[str, list[str]] = {"left": [], "right": []}
    for index, name in enumerate(roi_names):
        if "::" not in name:
            raise ValueError(f"Hemisphere prefix missing for {story}: {name}")
        hemisphere, label = name.split("::", 1)
        if hemisphere not in indices:
            raise ValueError(f"Unexpected hemisphere for {story}: {hemisphere}")
        indices[hemisphere].append(index)
        labels[hemisphere].append(label)
    if len(indices["left"]) != 17 or len(indices["right"]) != 17:
        raise ValueError(
            f"Expected 17 ROIs per hemisphere for {story}, got "
            f"left={len(indices['left'])}, right={len(indices['right'])}"
        )
    if labels["left"] != labels["right"]:
        raise ValueError(f"Left/right ROI order differs for {story}")
    return indices


def add_roi_rows(
    rows: list[dict],
    *,
    analysis: str,
    story_set: str,
    story: str,
    roi_names: list[str],
    baseline: np.ndarray,
    combined: np.ndarray,
    seed: int | None = None,
    subject: str | None = None,
) -> None:
    for index, name in enumerate(roi_names):
        hemisphere, roi = name.split("::", 1)
        row = {
            "analysis": analysis,
            "set": story_set,
            "story": story,
            "hemisphere": hemisphere,
            "roi": roi,
            "baseline_cv_r": float(baseline[index]),
            "combined_cv_r": float(combined[index]),
            "delta_cv_r": float(combined[index] - baseline[index]),
        }
        if seed is not None:
            row["seed"] = seed
        if subject is not None:
            row["subject"] = subject
        rows.append(row)


def evaluate_story(
    story: str,
    story_set: str,
    cache_dir: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    targets, roi_names = load_story_targets(cache_dir, story)
    validate_roi_names(roi_names, story)
    baseline_x, combined_x = load_design(story, args)

    n_timepoints = min(len(baseline_x), len(combined_x), targets.shape[1])
    baseline_x = baseline_x[:n_timepoints]
    combined_x = combined_x[:n_timepoints]
    targets = targets[:, :n_timepoints]
    print(
        f"Central hemisphere analysis: {story} ({story_set}), "
        f"subjects={len(targets)}, TRs={n_timepoints}",
        flush=True,
    )

    full_rows: list[dict] = []
    group_target = np.mean(targets, axis=0)
    baseline_full, _ = vectorized_cv_r(
        baseline_x,
        group_target,
        n_splits=args.n_splits,
        alpha=args.alpha,
        do_detrend=True,
        cv_gap=args.cv_gap,
    )
    combined_full, _ = vectorized_cv_r(
        combined_x,
        group_target,
        n_splits=args.n_splits,
        alpha=args.alpha,
        do_detrend=True,
        cv_gap=args.cv_gap,
    )
    add_roi_rows(
        full_rows,
        analysis="full_group",
        story_set=story_set,
        story=story,
        roi_names=roi_names,
        baseline=baseline_full,
        combined=combined_full,
    )

    split_rows: list[dict] = []
    for seed in range(args.subject_splits):
        indices = np.random.default_rng(seed).permutation(len(targets))
        first = indices[: len(indices) // 2]
        second = indices[len(indices) // 2 :]
        baseline_split = (
            cross_group_cv_r(
                baseline_x, targets, first, second,
                args.n_splits, args.alpha, args.cv_gap,
            )
            + cross_group_cv_r(
                baseline_x, targets, second, first,
                args.n_splits, args.alpha, args.cv_gap,
            )
        ) / 2.0
        combined_split = (
            cross_group_cv_r(
                combined_x, targets, first, second,
                args.n_splits, args.alpha, args.cv_gap,
            )
            + cross_group_cv_r(
                combined_x, targets, second, first,
                args.n_splits, args.alpha, args.cv_gap,
            )
        ) / 2.0
        add_roi_rows(
            split_rows,
            analysis="participant_split_half",
            story_set=story_set,
            story=story,
            roi_names=roi_names,
            baseline=baseline_split,
            combined=combined_split,
            seed=seed,
        )

    leave_rows: list[dict] = []
    cache_files = sorted(cache_dir.glob(f"{story}_*.npz"))
    subjects = [path.stem[len(story) + 1 :] for path in cache_files]
    if len(subjects) != len(targets):
        raise ValueError(f"Subject count differs from target count for {story}")
    all_subjects = np.arange(len(targets))
    for held_out_index, subject in enumerate(subjects):
        training = all_subjects[all_subjects != held_out_index]
        testing = np.asarray([held_out_index])
        baseline_leave = cross_group_cv_r(
            baseline_x, targets, training, testing,
            args.n_splits, args.alpha, args.cv_gap,
        )
        combined_leave = cross_group_cv_r(
            combined_x, targets, training, testing,
            args.n_splits, args.alpha, args.cv_gap,
        )
        add_roi_rows(
            leave_rows,
            analysis="leave_one_participant_out",
            story_set=story_set,
            story=story,
            roi_names=roi_names,
            baseline=baseline_leave,
            combined=combined_leave,
            subject=subject,
        )

    return pd.DataFrame(full_rows), pd.DataFrame(split_rows), pd.DataFrame(leave_rows)


def make_story_means(long_tables: list[pd.DataFrame]) -> pd.DataFrame:
    long = pd.concat(long_tables, ignore_index=True)
    means = (
        long.groupby(["analysis", "set", "story", "hemisphere"], as_index=False)
        .agg(
            baseline_cv_r=("baseline_cv_r", "mean"),
            combined_cv_r=("combined_cv_r", "mean"),
            delta_cv_r=("delta_cv_r", "mean"),
        )
        .sort_values(["analysis", "set", "story", "hemisphere"])
    )
    return means


def summarize_hemispheres(story_means: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    paired_rows = []
    scopes = {
        "development_eight_stories": story_means[story_means["set"].eq("development")],
        "heldout_three_stories": story_means[story_means["set"].eq("heldout")],
        "pooled_eleven_stories": story_means,
    }
    for scope, scoped in scopes.items():
        for analysis in ANALYSES:
            subset = scoped[scoped["analysis"].eq(analysis)]
            for hemisphere in ["left", "right"]:
                values = subset[subset["hemisphere"].eq(hemisphere)]["delta_cv_r"]
                row = {"scope": scope, "analysis": analysis, "hemisphere": hemisphere}
                row.update(inference_summary(values))
                summary_rows.append(row)

            wide = subset.pivot(index="story", columns="hemisphere", values="delta_cv_r")
            if not {"left", "right"}.issubset(wide.columns):
                raise ValueError(f"Missing paired hemisphere values for {scope}, {analysis}")
            difference = wide["right"] - wide["left"]
            row = {
                "scope": scope,
                "analysis": analysis,
                "contrast": "right_minus_left",
            }
            row.update(inference_summary(difference))
            paired_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    paired = pd.DataFrame(paired_rows)
    paired["p_fdr_across_nine_tests"] = bh_fdr(paired["p_ttest_two_sided"])
    return summary, paired


def write_summary_markdown(
    path: Path,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    lines = [
        "# Central shared-response hemisphere sensitivity",
        "",
        "The analysis used the final cumulative-context GPT-2 PC1-PC5 model, rich lexical/acoustic baseline, alpha=10, five forward-chaining folds, and a 10-TR gap. Each of the 17 predefined ROIs was extracted separately in left and right MNI hemispheres.",
        "",
        "## Hemisphere effects",
        "",
        summary.to_markdown(index=False),
        "",
        "## Paired right-minus-left contrasts",
        "",
        paired.to_markdown(index=False),
        "",
        (
            "Across all 11 stories, right-minus-left differences were nominally "
            "positive for full-group, participant split-half, and leave-one-participant-out "
            "analyses. None survived FDR correction across the nine scope-by-analysis "
            "paired tests (smallest q="
            f"{paired['p_fdr_across_nine_tests'].min():.6f}). The development and "
            "held-out contrasts were also not significant when evaluated separately, "
            "so this result does not establish lateralization."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--development-cache-dir",
        default=(
            "results/jnl_revision_major_review/hemisphere_sensitivity/"
            "roi_time_series_cache_corrected"
        ),
    )
    parser.add_argument(
        "--heldout-config",
        default="results/jnl_heldout_story_replication/heldout_motionfiltered_config.py",
    )
    parser.add_argument(
        "--heldout-cache-dir",
        default="results/jnl_shared_response_hemisphere_sensitivity/heldout_roi_cache",
    )
    parser.add_argument(
        "--heldout-regressor-dir",
        default="results/jnl_heldout_story_replication/heldout_regressors",
    )
    parser.add_argument(
        "--development-rich-audio-regressor-dir",
        default="results/jnl_revision_major_review/rich_audio_controls/rich_audio_regressors",
    )
    parser.add_argument(
        "--development-cumulative-regressor-dir",
        default=(
            "results/jnl_revision_major_review/context_layer_models/regressors/"
            "cumulative_context_layer12_current_tokens"
        ),
    )
    parser.add_argument(
        "--development-independent-regressor-dir",
        default="results/regressors",
    )
    parser.add_argument(
        "--output-dir",
        default="results/jnl_shared_response_hemisphere_sensitivity",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--cv-gap", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--subject-splits", type=int, default=50)
    parser.add_argument("--atlas-threshold", type=float, default=0.0)
    parser.add_argument("--fwhm", type=float, default=6.0)
    parser.add_argument("--skip-heldout-extraction", action="store_true")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    development_cache = ROOT / args.development_cache_dir
    heldout_cache = ROOT / args.heldout_cache_dir
    heldout_cache.mkdir(parents=True, exist_ok=True)

    if not args.skip_heldout_extraction:
        extract_heldout_hemisphere_targets(
            ROOT / args.heldout_config,
            heldout_cache,
            atlas_threshold=args.atlas_threshold,
            fwhm=args.fwhm,
        )

    all_full = []
    all_split = []
    all_leave = []
    for story in DEVELOPMENT_STORIES + HELDOUT_STORIES:
        story_set = "development" if story in DEVELOPMENT_STORIES else "heldout"
        cache_dir = development_cache if story_set == "development" else heldout_cache
        full, split, leave = evaluate_story(story, story_set, cache_dir, args)
        all_full.append(full)
        all_split.append(split)
        all_leave.append(leave)
        pd.concat(all_full, ignore_index=True).to_csv(
            output_dir / "full_group_long.csv", index=False
        )
        pd.concat(all_split, ignore_index=True).to_csv(
            output_dir / "participant_split_half_long.csv", index=False
        )
        pd.concat(all_leave, ignore_index=True).to_csv(
            output_dir / "leave_one_participant_out_long.csv", index=False
        )

    story_means = make_story_means(all_full + all_split + all_leave)
    summary, paired = summarize_hemispheres(story_means)
    story_means.to_csv(output_dir / "story_hemisphere_means.csv", index=False)
    summary.to_csv(output_dir / "hemisphere_inference_summary.csv", index=False)
    paired.to_csv(output_dir / "left_right_paired_summary.csv", index=False)
    write_summary_markdown(output_dir / "ANALYSIS_SUMMARY.md", summary, paired)

    print("\nHemisphere inference:\n", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nPaired right-minus-left inference:\n", flush=True)
    print(paired.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
