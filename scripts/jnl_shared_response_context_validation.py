from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from jnl_pc_count_sensitivity import (  # noqa: E402
    LEXICAL_BASELINE_COLS,
    vectorized_cv_r,
)
from jnl_inference_utils import (  # noqa: E402
    bh_fdr,
    fit_mixed_effect,
    simple_stats,
)
from jnl_rich_audio_controls import RICH_AUDIO_COLUMNS  # noqa: E402
from jnl_story_grid import align_targets_to_common_story_grid  # noqa: E402
from roi_encoding_model_comparison import load_regressors  # noqa: E402
from semantic_fmri.utils import ACOUSTIC_CONTROL_COLUMNS, hrf_convolve  # noqa: E402


STORIES = [
    "black",
    "bronx",
    "forgot",
    "milkyway",
    "notthefallintact",
    "piemanpni",
    "shapesphysical",
    "shapessocial",
]


CONTRASTS = {
    "cumulative_context_additive": ("cumulative_context", "rich_audio"),
    "independent_chunk_additive": ("independent_chunk", "rich_audio"),
    "cumulative_minus_independent": ("cumulative_context", "independent_chunk"),
}


def standardize_subject_timeseries(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    mean = np.mean(y, axis=0, keepdims=True)
    sd = np.std(y, axis=0, keepdims=True)
    sd[(sd <= 0) | ~np.isfinite(sd)] = 1.0
    return (y - mean) / sd


def load_story_targets(
    cache_dir: Path,
    story: str,
    standardize: bool = True,
) -> tuple[np.ndarray, list[str]]:
    files = sorted(cache_dir.glob(f"{story}_*.npz"))
    if not files:
        raise FileNotFoundError(f"No ROI cache files found for {story} in {cache_dir}")

    subject_targets = []
    subject_trs = []
    subject_onsets = []
    roi_names: list[str] | None = None
    for path in files:
        cached = np.load(path, allow_pickle=False)
        current_names = [str(value) for value in cached["roi_names"].tolist()]
        if roi_names is None:
            roi_names = current_names
        elif current_names != roi_names:
            raise ValueError(f"ROI order differs in {path}")
        subject_targets.append(np.asarray(cached["y_matrix"], dtype=float))
        subject_trs.append(float(cached["bold_tr"]))
        subject_onsets.append(float(cached["onset"]))

    aligned_targets, _ = align_targets_to_common_story_grid(
        subject_targets,
        subject_trs,
        subject_onsets,
    )
    if standardize:
        aligned_targets = [standardize_subject_timeseries(target) for target in aligned_targets]

    targets = np.stack(aligned_targets, axis=0)
    return targets, roi_names or []


def load_story_designs(args: argparse.Namespace, story: str) -> dict[str, np.ndarray]:
    rich = load_regressors(ROOT / args.rich_audio_regressor_dir, story)
    cumulative = load_regressors(ROOT / args.cumulative_context_regressor_dir, story)
    independent = load_regressors(ROOT / args.independent_chunk_regressor_dir, story)

    if not (len(rich) == len(cumulative) == len(independent)):
        raise ValueError(
            f"Regressor lengths differ for {story}: rich={len(rich)}, "
            f"cumulative={len(cumulative)}, independent={len(independent)}"
        )
    for label, frame in [("cumulative", cumulative), ("independent", independent)]:
        if not np.allclose(rich["time_TR"], frame["time_TR"]):
            raise ValueError(f"time_TR differs between rich audio and {label} for {story}")

    existing_acoustic = [
        f"{column}_hrf"
        for column in ACOUSTIC_CONTROL_COLUMNS
        if f"{column}_hrf" in rich.columns
    ]
    rich_audio = [f"{column}_hrf" for column in RICH_AUDIO_COLUMNS]
    baseline_columns = list(LEXICAL_BASELINE_COLS) + existing_acoustic + rich_audio
    pc_columns = [f"PC{i}_hrf" for i in range(1, 6)]

    for label, frame, columns in [
        ("rich", rich, baseline_columns),
        ("cumulative", cumulative, pc_columns),
        ("independent", independent, pc_columns),
    ]:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing {label} columns for {story}: {missing}")

    baseline = rich[baseline_columns].to_numpy(dtype=float)
    cumulative_pcs = cumulative[pc_columns].to_numpy(dtype=float)
    independent_pcs = independent[pc_columns].to_numpy(dtype=float)
    silence = cumulative["word_count"].eq(0).to_numpy(dtype=bool)
    silence_hrf = hrf_convolve(silence.astype(float), tr=1.5)
    baseline_with_silence = np.column_stack([baseline, silence_hrf])
    cumulative_raw = cumulative[[f"PC{i}" for i in range(1, 6)]].to_numpy(dtype=float)
    cumulative_raw[silence] = 0.0
    cumulative_zero_silence = np.column_stack(
        [hrf_convolve(cumulative_raw[:, i], tr=1.5) for i in range(5)]
    )
    return {
        "rich_audio": baseline,
        "cumulative_context": np.column_stack([baseline, cumulative_pcs]),
        "independent_chunk": np.column_stack([baseline, independent_pcs]),
        "cumulative_pcs": cumulative_pcs,
        "rich_audio_with_silence": baseline_with_silence,
        "cumulative_with_silence": np.column_stack(
            [baseline_with_silence, cumulative_pcs]
        ),
        "cumulative_zero_during_silence": np.column_stack(
            [baseline_with_silence, cumulative_zero_silence]
        ),
    }


def correlation_by_column(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    true_centered = y_true - np.mean(y_true, axis=0, keepdims=True)
    pred_centered = y_pred - np.mean(y_pred, axis=0, keepdims=True)
    numerator = np.sum(true_centered * pred_centered, axis=0)
    denominator = np.sqrt(
        np.sum(true_centered**2, axis=0) * np.sum(pred_centered**2, axis=0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    result[denominator == 0] = np.nan
    return result


def cross_group_cv_r(
    x: np.ndarray,
    targets: np.ndarray,
    train_subjects: np.ndarray,
    test_subjects: np.ndarray,
    n_splits: int,
    alpha: float,
    cv_gap: int,
) -> np.ndarray:
    train_target = np.mean(targets[train_subjects], axis=0)
    test_target = np.mean(targets[test_subjects], axis=0)
    n_timepoints = min(len(x), len(train_target), len(test_target))
    x = x[:n_timepoints]
    train_target = train_target[:n_timepoints]
    test_target = test_target[:n_timepoints]

    true_parts = []
    prediction_parts = []
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=cv_gap)
    for train_index, test_index in splitter.split(x):
        trend_design = np.column_stack(
            [train_index.astype(float), np.ones(len(train_index), dtype=float)]
        )
        trend_beta = np.linalg.lstsq(
            trend_design, train_target[train_index], rcond=None
        )[0]
        y_train = train_target[train_index] - trend_design @ trend_beta
        test_design = np.column_stack(
            [test_index.astype(float), np.ones(len(test_index), dtype=float)]
        )
        y_test = test_target[test_index] - test_design @ trend_beta

        mean = np.mean(y_train, axis=0, keepdims=True)
        sd = np.std(y_train, axis=0, keepdims=True)
        sd[(sd <= 0) | ~np.isfinite(sd)] = 1.0
        y_train = (y_train - mean) / sd
        y_test = (y_test - mean) / sd

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha, fit_intercept=True)),
            ]
        )
        model.fit(x[train_index], y_train)
        true_parts.append(y_test)
        prediction_parts.append(model.predict(x[test_index]))

    return correlation_by_column(np.vstack(true_parts), np.vstack(prediction_parts))


def add_contrasts(row: dict[str, float]) -> None:
    for contrast, (model_a, model_b) in CONTRASTS.items():
        row[contrast] = row[model_a] - row[model_b]


def full_group_analysis(
    story_data: dict[str, tuple[np.ndarray, list[str], dict[str, np.ndarray]]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    for story, (targets, roi_names, designs) in story_data.items():
        group_target = np.mean(targets, axis=0)
        model_values = {}
        for model_name in ["rich_audio", "cumulative_context", "independent_chunk"]:
            model_values[model_name], _ = vectorized_cv_r(
                designs[model_name],
                group_target,
                n_splits=args.n_splits,
                alpha=args.alpha,
                do_detrend=True,
                cv_gap=args.cv_gap,
            )
        for roi_index, roi in enumerate(roi_names):
            row = {"story": story, "roi": roi}
            for model_name, values in model_values.items():
                row[model_name] = float(values[roi_index])
            add_contrasts(row)
            rows.append(row)
    return pd.DataFrame(rows)


def split_half_analysis(
    story_data: dict[str, tuple[np.ndarray, list[str], dict[str, np.ndarray]]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    for story, (targets, roi_names, designs) in story_data.items():
        print(f"Held-out participant validation: {story}", flush=True)
        for seed in range(args.subject_splits):
            indices = np.random.default_rng(seed).permutation(len(targets))
            first = indices[: len(indices) // 2]
            second = indices[len(indices) // 2 :]
            model_values = {}
            for model_name in ["rich_audio", "cumulative_context", "independent_chunk"]:
                forward = cross_group_cv_r(
                    designs[model_name],
                    targets,
                    first,
                    second,
                    args.n_splits,
                    args.alpha,
                    args.cv_gap,
                )
                reverse = cross_group_cv_r(
                    designs[model_name],
                    targets,
                    second,
                    first,
                    args.n_splits,
                    args.alpha,
                    args.cv_gap,
                )
                model_values[model_name] = (forward + reverse) / 2.0

            for roi_index, roi in enumerate(roi_names):
                row = {"story": story, "seed": seed, "roi": roi}
                for model_name, values in model_values.items():
                    row[model_name] = float(values[roi_index])
                add_contrasts(row)
                rows.append(row)
    return pd.DataFrame(rows)


def leave_one_subject_out_analysis(
    story_data: dict[str, tuple[np.ndarray, list[str], dict[str, np.ndarray]]],
    cache_dir: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    for story, (targets, roi_names, designs) in story_data.items():
        cache_files = sorted(cache_dir.glob(f"{story}_*.npz"))
        subjects = [path.stem[len(story) + 1 :] for path in cache_files]
        if len(subjects) != len(targets):
            raise ValueError(f"Subject count does not match target count for {story}")
        all_subjects = np.arange(len(targets))
        print(f"Leave-one-participant-out validation: {story}", flush=True)

        for held_out_index, subject in enumerate(subjects):
            train_subjects = all_subjects[all_subjects != held_out_index]
            test_subject = np.asarray([held_out_index])
            model_values = {}
            for model_name in ["rich_audio", "cumulative_context", "independent_chunk"]:
                model_values[model_name] = cross_group_cv_r(
                    designs[model_name],
                    targets,
                    train_subjects,
                    test_subject,
                    args.n_splits,
                    args.alpha,
                    args.cv_gap,
                )
            for roi_index, roi in enumerate(roi_names):
                row = {"story": story, "subject": subject, "roi": roi}
                for model_name, values in model_values.items():
                    row[model_name] = float(values[roi_index])
                add_contrasts(row)
                rows.append(row)
    return pd.DataFrame(rows)


def one_sample_summary(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    t_result = stats.ttest_1samp(clean, 0.0)
    try:
        wilcoxon_p = float(stats.wilcoxon(clean, alternative="greater").pvalue)
    except ValueError:
        wilcoxon_p = np.nan
    positives = int(np.sum(clean > 0))
    sign_p = float(stats.binomtest(positives, len(clean), 0.5, alternative="greater").pvalue)
    return {
        "n_stories": int(len(clean)),
        "mean": float(np.mean(clean)),
        "median": float(np.median(clean)),
        "std": float(np.std(clean, ddof=1)),
        "t": float(t_result.statistic),
        "p_ttest_two_sided": float(t_result.pvalue),
        "p_wilcoxon_greater": wilcoxon_p,
        "n_positive_stories": positives,
        "p_sign_greater": sign_p,
    }


def summarize_story_inference(
    long: pd.DataFrame,
    analysis: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    averaging_columns = ["story", "roi"]
    by_story_roi = long.groupby(averaging_columns, as_index=False)[list(CONTRASTS)].mean()
    by_story = by_story_roi.groupby("story", as_index=False)[list(CONTRASTS)].mean()

    summary_rows = []
    for contrast in CONTRASTS:
        row = {"analysis": analysis, "contrast": contrast}
        row.update(one_sample_summary(by_story[contrast]))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    roi_rows = []
    for contrast in CONTRASTS:
        for roi, group in by_story_roi.groupby("roi"):
            row = {"analysis": analysis, "contrast": contrast, "roi": roi}
            row.update(one_sample_summary(group[contrast]))
            roi_rows.append(row)
    roi_summary = pd.DataFrame(roi_rows)
    roi_summary["p_fdr_within_contrast"] = roi_summary.groupby("contrast")[
        "p_ttest_two_sided"
    ].transform(bh_fdr)
    return summary, roi_summary, by_story


def summarize_scan_inference(long: pd.DataFrame, analysis: str) -> pd.DataFrame:
    scan = long.groupby(["story", "subject"], as_index=False)[list(CONTRASTS)].mean()
    rows = []
    for contrast in CONTRASTS:
        subject_values = scan.groupby("subject")[contrast].mean()
        story_values = scan.groupby("story")[contrast].mean()
        row = {
            "analysis": analysis,
            "contrast": contrast,
            "n_scans": int(scan.shape[0]),
            "n_subjects": int(scan["subject"].nunique()),
            "n_stories": int(scan["story"].nunique()),
            **{f"scan_{key}": value for key, value in simple_stats(scan[contrast]).items()},
            **{
                f"subject_{key}": value
                for key, value in simple_stats(subject_values).items()
            },
            **{f"story_{key}": value for key, value in simple_stats(story_values).items()},
            **fit_mixed_effect(scan, contrast),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def alpha_gap_sensitivity(
    story_data: dict[str, tuple[np.ndarray, list[str], dict[str, np.ndarray]]]
) -> pd.DataFrame:
    rows = []
    for cv_gap in [5, 10, 15]:
        for alpha in [1.0, 10.0, 100.0, 1000.0]:
            story_values = []
            for story, (targets, _, designs) in story_data.items():
                group_target = np.mean(targets, axis=0)
                baseline, _ = vectorized_cv_r(
                    designs["rich_audio"], group_target, 5, alpha, True, cv_gap
                )
                combined, _ = vectorized_cv_r(
                    designs["cumulative_context"], group_target, 5, alpha, True, cv_gap
                )
                story_values.append(float(np.mean(combined - baseline)))
            row = {"cv_gap": cv_gap, "alpha": alpha}
            row.update(one_sample_summary(pd.Series(story_values)))
            rows.append(row)
    return pd.DataFrame(rows)


def silence_sensitivity(
    story_data: dict[str, tuple[np.ndarray, list[str], dict[str, np.ndarray]]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants = {
        "explicit_silence_control": "cumulative_with_silence",
        "zero_semantic_pcs_during_silence": "cumulative_zero_during_silence",
    }
    story_rows = []

    for variant, combined_name in variants.items():
        full_values = {}
        split_values: dict[str, list[float]] = {story: [] for story in story_data}
        for story, (targets, _, designs) in story_data.items():
            group_target = np.mean(targets, axis=0)
            baseline_r, _ = vectorized_cv_r(
                designs["rich_audio_with_silence"],
                group_target,
                args.n_splits,
                args.alpha,
                True,
                args.cv_gap,
            )
            combined_r, _ = vectorized_cv_r(
                designs[combined_name],
                group_target,
                args.n_splits,
                args.alpha,
                True,
                args.cv_gap,
            )
            full_values[story] = float(np.mean(combined_r - baseline_r))

            for seed in range(args.subject_splits):
                indices = np.random.default_rng(seed).permutation(len(targets))
                first = indices[: len(indices) // 2]
                second = indices[len(indices) // 2 :]
                baseline_split = (
                    cross_group_cv_r(
                        designs["rich_audio_with_silence"],
                        targets,
                        first,
                        second,
                        args.n_splits,
                        args.alpha,
                        args.cv_gap,
                    )
                    + cross_group_cv_r(
                        designs["rich_audio_with_silence"],
                        targets,
                        second,
                        first,
                        args.n_splits,
                        args.alpha,
                        args.cv_gap,
                    )
                ) / 2.0
                combined_split = (
                    cross_group_cv_r(
                        designs[combined_name],
                        targets,
                        first,
                        second,
                        args.n_splits,
                        args.alpha,
                        args.cv_gap,
                    )
                    + cross_group_cv_r(
                        designs[combined_name],
                        targets,
                        second,
                        first,
                        args.n_splits,
                        args.alpha,
                        args.cv_gap,
                    )
                ) / 2.0
                split_values[story].append(float(np.mean(combined_split - baseline_split)))

        for story, value in full_values.items():
            story_rows.append(
                {
                    "analysis": "full_group",
                    "variant": variant,
                    "story": story,
                    "mean_delta": value,
                }
            )
        for story, values in split_values.items():
            story_rows.append(
                {
                    "analysis": "held_out_participant_split_half",
                    "variant": variant,
                    "story": story,
                    "mean_delta": float(np.mean(values)),
                }
            )

    story_table = pd.DataFrame(story_rows)
    summary_rows = []
    for (analysis, variant), group in story_table.groupby(["analysis", "variant"]):
        row = {"analysis": analysis, "variant": variant}
        row.update(one_sample_summary(group["mean_delta"]))
        summary_rows.append(row)
    return pd.DataFrame(summary_rows), story_table


def circular_shift_null(
    story_data: dict[str, tuple[np.ndarray, list[str], dict[str, np.ndarray]]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = {}
    observed_story_rows = []
    for story, (targets, _, designs) in story_data.items():
        group_target = np.mean(targets, axis=0)
        baseline_r, _ = vectorized_cv_r(
            designs["rich_audio"],
            group_target,
            args.n_splits,
            args.alpha,
            True,
            args.cv_gap,
        )
        combined_r, _ = vectorized_cv_r(
            designs["cumulative_context"],
            group_target,
            args.n_splits,
            args.alpha,
            True,
            args.cv_gap,
        )
        observed_story_rows.append(
            {"story": story, "observed_delta": float(np.mean(combined_r - baseline_r))}
        )
        prepared[story] = (group_target, designs, baseline_r)

    observed = float(
        np.mean([row["observed_delta"] for row in observed_story_rows])
    )
    rng = np.random.default_rng(args.random_seed)
    null_rows = []
    for permutation in range(args.circular_shifts):
        story_values = []
        shifts = []
        for story, (group_target, designs, baseline_r) in prepared.items():
            pcs = designs["cumulative_pcs"]
            minimum_shift = min(args.minimum_shift_trs, max(1, len(pcs) // 4))
            shift = int(rng.integers(minimum_shift, len(pcs) - minimum_shift))
            shifted_pcs = np.roll(pcs, shift, axis=0)
            shifted_design = np.column_stack([designs["rich_audio"], shifted_pcs])
            shifted_r, _ = vectorized_cv_r(
                shifted_design,
                group_target,
                args.n_splits,
                args.alpha,
                True,
                args.cv_gap,
            )
            story_values.append(float(np.mean(shifted_r - baseline_r)))
            shifts.append(f"{story}:{shift}")
        null_rows.append(
            {
                "permutation": permutation,
                "null_mean_delta": float(np.mean(story_values)),
                "shifts": ";".join(shifts),
            }
        )
        if (permutation + 1) % 100 == 0:
            print(f"Circular shifts completed: {permutation + 1}", flush=True)

    null = pd.DataFrame(null_rows)
    p_value = float(
        (1 + np.sum(null["null_mean_delta"].to_numpy() >= observed))
        / (len(null) + 1)
    )
    summary = pd.DataFrame(
        [
            {
                "observed_mean_delta": observed,
                "n_permutations": int(len(null)),
                "null_mean": float(null["null_mean_delta"].mean()),
                "null_std": float(null["null_mean_delta"].std(ddof=1)),
                "null_95th_percentile": float(null["null_mean_delta"].quantile(0.95)),
                "p_circular_shift_greater": p_value,
            }
        ]
    )
    return summary, null


def main() -> None:
    parser = argparse.ArgumentParser()
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
        "--cumulative-context-regressor-dir",
        default=(
            "results/jnl_revision_major_review/context_layer_models/regressors/"
            "cumulative_context_layer12_current_tokens"
        ),
    )
    parser.add_argument(
        "--independent-chunk-regressor-dir",
        default="results/regressors",
    )
    parser.add_argument(
        "--output-dir",
        default="results/jnl_shared_response_context_validation",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--cv-gap", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--subject-splits", type=int, default=50)
    parser.add_argument("--circular-shifts", type=int, default=1000)
    parser.add_argument("--minimum-shift-trs", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260802)
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = ROOT / args.roi_cache_dir

    story_data = {}
    for story in STORIES:
        targets, roi_names = load_story_targets(cache_dir, story)
        designs = load_story_designs(args, story)
        target_length = targets.shape[1]
        for name, design in designs.items():
            if len(design) < target_length:
                raise ValueError(
                    f"Design {name} is shorter than the aligned target for {story}: "
                    f"{len(design)} < {target_length}"
                )
            designs[name] = design[:target_length]
        story_data[story] = (targets, roi_names, designs)
        print(
            f"Loaded {story}: {targets.shape[0]} subjects, "
            f"{targets.shape[1]} time points, {targets.shape[2]} ROIs",
            flush=True,
        )

    full = full_group_analysis(story_data, args)
    split = split_half_analysis(story_data, args)
    leave_one = leave_one_subject_out_analysis(story_data, cache_dir, args)
    full_summary, full_roi, full_story = summarize_story_inference(full, "full_group")
    split_summary, split_roi, split_story = summarize_story_inference(
        split, "held_out_participant_split_half"
    )
    leave_summary, leave_roi, leave_story = summarize_story_inference(
        leave_one, "leave_one_participant_out"
    )
    leave_scan_summary = summarize_scan_inference(
        leave_one, "leave_one_participant_out"
    )
    sensitivity = alpha_gap_sensitivity(story_data)
    silence_summary, silence_story = silence_sensitivity(story_data, args)
    shift_summary, shift_null = circular_shift_null(story_data, args)

    full.to_csv(output_dir / "full_group_long.csv", index=False)
    split.to_csv(output_dir / "split_half_long.csv", index=False)
    leave_one.to_csv(output_dir / "leave_one_participant_out_long.csv", index=False)
    all_story_summaries = pd.concat(
        [full_summary, split_summary, leave_summary], ignore_index=True
    )
    all_story_summaries.to_csv(
        output_dir / "story_level_summary.csv", index=False
    )
    pd.concat([full_roi, split_roi, leave_roi], ignore_index=True).to_csv(
        output_dir / "roi_story_level_summary.csv", index=False
    )
    full_story.to_csv(output_dir / "full_group_story_means.csv", index=False)
    split_story.to_csv(output_dir / "split_half_story_means.csv", index=False)
    leave_story.to_csv(output_dir / "leave_one_participant_out_story_means.csv", index=False)
    leave_scan_summary.to_csv(
        output_dir / "leave_one_participant_out_scan_summary.csv", index=False
    )
    sensitivity.to_csv(output_dir / "alpha_gap_sensitivity.csv", index=False)
    silence_summary.to_csv(output_dir / "silence_sensitivity_summary.csv", index=False)
    silence_story.to_csv(output_dir / "silence_sensitivity_story_means.csv", index=False)
    shift_summary.to_csv(output_dir / "circular_shift_summary.csv", index=False)
    shift_null.to_csv(output_dir / "circular_shift_null.csv", index=False)

    print("\nStory-level inference:\n")
    print(all_story_summaries.to_string(index=False))
    print("\nLeave-one-participant-out scan inference:\n")
    print(
        leave_scan_summary[
            [
                "contrast",
                "scan_mean",
                "scan_p_ttest_two_sided",
                "subject_p_ttest_two_sided",
                "story_p_ttest_two_sided",
                "mixed_estimate",
                "mixed_p_two_sided",
            ]
        ].to_string(index=False)
    )
    print("\nCircular-shift inference:\n")
    print(shift_summary.to_string(index=False))
    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()
