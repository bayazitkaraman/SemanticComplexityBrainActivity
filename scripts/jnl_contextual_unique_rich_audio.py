from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from jnl_pc_count_sensitivity import (  # noqa: E402
    LEXICAL_BASELINE_COLS,
    safe_name,
    vectorized_cv_r,
)
from jnl_inference_utils import (  # noqa: E402
    bh_fdr,
    fit_mixed_effect,
    simple_stats,
)
from jnl_rich_audio_controls import RICH_AUDIO_COLUMNS  # noqa: E402
from jnl_story_grid import align_target_to_story_grid  # noqa: E402
from roi_encoding_model_comparison import load_regressors  # noqa: E402
from semantic_fmri.utils import ACOUSTIC_CONTROL_COLUMNS, load_story_list  # noqa: E402


NETWORKS = {
    "all_17_language_related_rois": None,
    "temporal_core": [
        "Superior Temporal Gyrus, anterior division",
        "Superior Temporal Gyrus, posterior division",
        "Middle Temporal Gyrus, anterior division",
        "Middle Temporal Gyrus, posterior division",
    ],
    "frontotemporal": [
        "Inferior Frontal Gyrus, pars opercularis",
        "Inferior Frontal Gyrus, pars triangularis",
        "Superior Temporal Gyrus, anterior division",
        "Superior Temporal Gyrus, posterior division",
        "Middle Temporal Gyrus, anterior division",
        "Middle Temporal Gyrus, posterior division",
    ],
}


CONTRASTS = {
    "contextual_adds_to_rich_audio": ("rich_contextual", "rich_audio"),
    "static_adds_to_rich_audio": ("rich_static", "rich_audio"),
    "contextual_minus_static": ("rich_contextual", "rich_static"),
    "contextual_unique_beyond_static": ("rich_both", "rich_static"),
    "static_unique_beyond_contextual": ("rich_both", "rich_contextual"),
}


def merge_regressors(
    story_name: str,
    rich_audio_dir: Path,
    contextual_dir: Path,
    static_dir: Path,
) -> pd.DataFrame:
    rich = load_regressors(rich_audio_dir, story_name)
    contextual = load_regressors(contextual_dir, story_name)
    static = load_regressors(static_dir, story_name)

    if not (len(rich) == len(contextual) == len(static)):
        raise ValueError(
            f"Regressor lengths differ for {story_name}: "
            f"rich={len(rich)}, contextual={len(contextual)}, static={len(static)}"
        )
    for label, frame in [("contextual", contextual), ("static", static)]:
        if not np.allclose(rich["time_TR"], frame["time_TR"]):
            raise ValueError(f"time_TR differs between rich audio and {label} for {story_name}")

    merged = rich.copy()
    for i in range(1, 6):
        merged[f"contextual_PC{i}_hrf"] = contextual[f"PC{i}_hrf"].to_numpy()
        merged[f"static_PC{i}_hrf"] = static[f"PC{i}_hrf"].to_numpy()
    return merged


def model_columns(reg: pd.DataFrame) -> dict[str, list[str]]:
    existing_acoustic = [
        f"{column}_hrf"
        for column in ACOUSTIC_CONTROL_COLUMNS
        if f"{column}_hrf" in reg.columns
    ]
    rich_audio = [f"{column}_hrf" for column in RICH_AUDIO_COLUMNS]
    baseline = list(LEXICAL_BASELINE_COLS) + existing_acoustic + rich_audio
    contextual = [f"contextual_PC{i}_hrf" for i in range(1, 6)]
    static = [f"static_PC{i}_hrf" for i in range(1, 6)]
    return {
        "rich_audio": baseline,
        "rich_contextual": baseline + contextual,
        "rich_static": baseline + static,
        "rich_both": baseline + static + contextual,
    }


def evaluate(args: argparse.Namespace) -> pd.DataFrame:
    stories = load_story_list(str(ROOT / args.subject_list))
    frames: dict[str, pd.DataFrame] = {}
    rows = []

    for index, story in enumerate(stories, start=1):
        story_name = story["name"]
        if story_name not in frames:
            frames[story_name] = merge_regressors(
                story_name,
                ROOT / args.rich_audio_regressor_dir,
                ROOT / args.contextual_regressor_dir,
                ROOT / args.static_regressor_dir,
            )
        reg = frames[story_name]
        columns_by_model = model_columns(reg)
        cache_path = ROOT / args.roi_cache_dir / (
            f"{safe_name(story_name)}_{safe_name(story['subject'])}.npz"
        )
        if not cache_path.exists():
            raise FileNotFoundError(cache_path)
        cached = np.load(cache_path, allow_pickle=False)
        roi_names = [str(value) for value in cached["roi_names"].tolist()]
        y_matrix, _ = align_target_to_story_grid(
            np.asarray(cached["y_matrix"], dtype=float),
            float(cached["bold_tr"]),
            float(cached["onset"]),
            max_timepoints=len(reg),
        )
        target_length = len(y_matrix)

        print(
            f"[{index}/{len(stories)}] {story_name} {story['subject']}",
            flush=True,
        )
        cv_by_model = {}
        for model_name, columns in columns_by_model.items():
            missing = [column for column in columns if column not in reg.columns]
            if missing:
                raise ValueError(f"Missing columns for {story_name}/{model_name}: {missing}")
            cv_by_model[model_name], n_used = vectorized_cv_r(
                reg[columns].to_numpy(dtype=float)[:target_length],
                y_matrix,
                n_splits=args.n_splits,
                alpha=args.alpha,
                do_detrend=True,
                cv_gap=args.cv_gap,
            )

        for roi_index, roi in enumerate(roi_names):
            row = {
                "story": story_name,
                "subject": story["subject"],
                "roi": roi,
                "n_timepoints": int(n_used),
                "alpha": float(args.alpha),
                "cv_gap": int(args.cv_gap),
            }
            for model_name, values in cv_by_model.items():
                row[model_name] = values[roi_index]
            for contrast, (model_a, model_b) in CONTRASTS.items():
                row[contrast] = row[model_a] - row[model_b]
            rows.append(row)
    return pd.DataFrame(rows)


def network_scan_values(long: pd.DataFrame, network: str, contrast: str) -> pd.DataFrame:
    rois = NETWORKS[network]
    subset = long if rois is None else long[long["roi"].isin(rois)]
    return (
        subset.groupby(["story", "subject"], as_index=False)
        .agg(value=(contrast, "mean"), n_rois=("roi", "nunique"))
        .sort_values(["story", "subject"])
    )


def summarize(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    story_rows = []
    leave_rows = []
    for contrast in CONTRASTS:
        for network in NETWORKS:
            scan = network_scan_values(long, network, contrast)
            subject_values = scan.groupby("subject")["value"].mean()
            story_values = scan.groupby("story")["value"].mean()
            row = {
                "contrast": contrast,
                "network": network,
                "n_scans": int(scan.shape[0]),
                "n_subjects": int(scan["subject"].nunique()),
                "n_stories": int(scan["story"].nunique()),
                **{f"scan_{key}": value for key, value in simple_stats(scan["value"]).items()},
                **{
                    f"subject_{key}": value
                    for key, value in simple_stats(subject_values).items()
                },
                **{f"story_{key}": value for key, value in simple_stats(story_values).items()},
                **fit_mixed_effect(scan, "value"),
            }
            summary_rows.append(row)

            for story, value in story_values.items():
                story_rows.append(
                    {
                        "contrast": contrast,
                        "network": network,
                        "story": story,
                        "mean": float(value),
                        "n_scans": int(scan["story"].eq(story).sum()),
                    }
                )
                subset = scan[~scan["story"].eq(story)]
                leave_rows.append(
                    {
                        "contrast": contrast,
                        "network": network,
                        "left_out_story": story,
                        **{
                            f"scan_{key}": result
                            for key, result in simple_stats(subset["value"]).items()
                        },
                        **{
                            f"subject_{key}": result
                            for key, result in simple_stats(
                                subset.groupby("subject")["value"].mean()
                            ).items()
                        },
                    }
                )

    summary = pd.DataFrame(summary_rows)
    summary["scan_p_fdr_all_tests"] = bh_fdr(summary["scan_p_ttest_two_sided"])
    summary["mixed_p_fdr_all_tests"] = bh_fdr(summary["mixed_p_two_sided"])
    return summary, pd.DataFrame(story_rows), pd.DataFrame(leave_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject-list",
        default="configs/stories_all_fmriprep_confounds_motionfiltered.py",
    )
    parser.add_argument(
        "--roi-cache-dir",
        default=(
            "results/journal_neurolinguistics_robustness/pc_count_full_corrected/"
            "roi_time_series_cache"
        ),
    )
    parser.add_argument(
        "--rich-audio-regressor-dir",
        default="results/jnl_revision_major_review/rich_audio_controls/rich_audio_regressors",
    )
    parser.add_argument(
        "--contextual-regressor-dir",
        default="results/regressors_contextual_bert_base_uncased_gentle_motionfiltered",
    )
    parser.add_argument(
        "--static-regressor-dir",
        default="results/regressors_static_bert_base_uncased_gentle_motionfiltered",
    )
    parser.add_argument(
        "--output-dir",
        default="results/jnl_contextual_unique_rich_audio_gap5",
    )
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--cv-gap", type=int, default=5)
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    long = evaluate(args)
    summary, story, leave = summarize(long)

    long.to_csv(output_dir / "contextual_unique_rich_audio_long.csv", index=False)
    summary.to_csv(output_dir / "contextual_unique_rich_audio_summary.csv", index=False)
    story.to_csv(output_dir / "contextual_unique_rich_audio_story_means.csv", index=False)
    leave.to_csv(output_dir / "contextual_unique_rich_audio_leave_one_story.csv", index=False)

    columns = [
        "contrast",
        "network",
        "scan_mean",
        "scan_ci95_low",
        "scan_ci95_high",
        "scan_p_ttest_two_sided",
        "subject_p_ttest_two_sided",
        "story_p_ttest_two_sided",
        "mixed_estimate",
        "mixed_p_two_sided",
        "scan_p_fdr_all_tests",
        "mixed_p_fdr_all_tests",
    ]
    print("\nSummary:\n")
    print(summary[columns].to_string(index=False))
    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()
