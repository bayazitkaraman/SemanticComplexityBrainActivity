from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from jnl_pc_count_sensitivity import LEXICAL_BASELINE_COLS, vectorized_cv_r  # noqa: E402
from jnl_revision_context_layer_models import cumulative_current_token_embeddings  # noqa: E402
from jnl_rich_audio_controls import RICH_AUDIO_COLUMNS  # noqa: E402
from jnl_shared_response_context_validation import (  # noqa: E402
    correlation_by_column,
    cross_group_cv_r,
    load_story_targets,
    one_sample_summary,
)
from semantic_fmri.utils import (  # noqa: E402
    ACOUSTIC_CONTROL_COLUMNS,
    get_unique_stories,
    hrf_convolve,
    load_gpt2,
    load_story_list,
    transcript_to_tr_chunks_uniform,
)


STORIES = [
    "shapessocial",
    "shapesphysical",
    "notthefallintact",
    "milkyway",
    "bronx",
    "forgot",
    "black",
    "piemanpni",
    "prettymouth",
    "sherlock",
    "merlin",
    "21styear",
    "tunnel",
]

DEVELOPMENT_STORIES = STORIES[:8]
INITIAL_EVALUATION_STORIES = STORIES[8:11]
NEW_EVALUATION_STORIES = STORIES[11:]


def template_map(args: argparse.Namespace) -> dict[str, dict]:
    sources = [
        ROOT / args.development_config,
        ROOT / args.initial_evaluation_config,
        ROOT / args.new_evaluation_config,
    ]
    templates: dict[str, dict] = {}
    for source in sources:
        rows = load_story_list(str(source))
        for row in get_unique_stories(rows):
            templates.setdefault(row["name"], row)
    missing = [story for story in STORIES if story not in templates]
    if missing:
        raise ValueError(f"Missing story templates: {missing}")
    return templates


def embedding_cache(args: argparse.Namespace) -> dict[str, np.ndarray]:
    output_dir = ROOT / args.output_dir / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    templates = template_map(args)
    missing = [story for story in STORIES if not (output_dir / f"{story}.npy").exists()]
    if missing:
        tokenizer, model, device = load_gpt2()
        model.eval()
        for story in missing:
            chunks, _, _, _ = transcript_to_tr_chunks_uniform(templates[story], tr=1.5)
            print(f"Embedding {story}: {len(chunks)} TRs", flush=True)
            values = cumulative_current_token_embeddings(
                chunks,
                tokenizer,
                model,
                layer=12,
                max_length=1024,
                output_stride=256,
                device=device,
            )
            np.save(output_dir / f"{story}.npy", values.astype(np.float32))
    embeddings = {
        story: np.load(output_dir / f"{story}.npy", allow_pickle=False)
        for story in STORIES
    }
    pd.DataFrame(
        [
            {
                "story": story,
                "n_trs": len(values),
                "n_features": values.shape[1],
            }
            for story, values in embeddings.items()
        ]
    ).to_csv(ROOT / args.output_dir / "embedding_inventory.csv", index=False)
    return embeddings


def story_balanced_pca(
    embeddings: dict[str, np.ndarray], training_stories: list[str], n_components: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = np.vstack([embeddings[story].mean(axis=0) for story in training_stories])
    seconds = np.vstack(
        [(embeddings[story].astype(np.float64) ** 2).mean(axis=0) for story in training_stories]
    )
    mean = means.mean(axis=0)
    variance = seconds.mean(axis=0) - mean**2
    scale = np.sqrt(np.maximum(variance, 1e-12))

    covariance = np.zeros((len(mean), len(mean)), dtype=np.float64)
    for story in training_stories:
        standardized = (embeddings[story].astype(np.float64) - mean) / scale
        covariance += standardized.T @ standardized / len(standardized)
    covariance /= len(training_stories)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    components = eigenvectors[:, order[:n_components]].T
    return mean, scale, components, eigenvalues


def project(
    values: np.ndarray, mean: np.ndarray, scale: np.ndarray, components: np.ndarray
) -> np.ndarray:
    return ((values.astype(np.float64) - mean) / scale) @ components.T


def source_regressor_path(args: argparse.Namespace, story: str) -> Path:
    if story in DEVELOPMENT_STORIES:
        return ROOT / args.development_regressor_dir / f"{story}_shared_pca_regressors.csv"
    if story in INITIAL_EVALUATION_STORIES:
        return ROOT / args.initial_evaluation_regressor_dir / f"{story}_shared_pca_regressors.csv"
    return ROOT / args.new_evaluation_regressor_dir / f"{story}_shared_pca_regressors.csv"


def baseline_matrix(frame: pd.DataFrame) -> np.ndarray:
    existing_acoustic = [
        f"{column}_hrf"
        for column in ACOUSTIC_CONTROL_COLUMNS
        if f"{column}_hrf" in frame.columns
    ]
    rich_audio = [f"{column}_hrf" for column in RICH_AUDIO_COLUMNS]
    columns = list(LEXICAL_BASELINE_COLS) + existing_acoustic + rich_audio
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing baseline columns: {missing}")
    return frame[columns].to_numpy(dtype=float)


def cache_dir(args: argparse.Namespace, story: str) -> Path:
    if story in DEVELOPMENT_STORIES:
        return ROOT / args.development_roi_cache
    if story in INITIAL_EVALUATION_STORIES:
        return ROOT / args.initial_evaluation_roi_cache
    return ROOT / args.new_evaluation_roi_cache


def story_subjects(args: argparse.Namespace, story: str) -> list[str]:
    files = sorted(cache_dir(args, story).glob(f"{story}_*.npz"))
    return [path.stem[len(story) + 1 :] for path in files]


def load_data(
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, list[str]]]:
    baselines: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    roi_names: dict[str, list[str]] = {}
    for story in STORIES:
        frame = pd.read_csv(source_regressor_path(args, story))
        baselines[story] = baseline_matrix(frame)
        targets[story], roi_names[story] = load_story_targets(cache_dir(args, story), story)
        if story != STORIES[0] and roi_names[story] != roi_names[STORIES[0]]:
            raise ValueError(f"ROI order differs for {story}")
        if len(story_subjects(args, story)) != len(targets[story]):
            raise ValueError(f"Subject and target counts differ for {story}")
    return baselines, targets, roi_names


def fold_scores(
    embeddings: dict[str, np.ndarray], omitted_story: str
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    training = [story for story in STORIES if story != omitted_story]
    mean, scale, components, eigenvalues = story_balanced_pca(embeddings, training)
    scores = {
        story: project(embeddings[story], mean, scale, components) for story in STORIES
    }
    metadata = {
        "omitted_story": omitted_story,
        "n_training_stories": len(training),
        **{
            f"pc{i + 1}_variance_ratio": float(eigenvalues[i] / eigenvalues.sum())
            for i in range(5)
        },
    }
    return scores, metadata


def with_pcs(baseline: np.ndarray, scores: np.ndarray) -> np.ndarray:
    pcs_hrf = np.column_stack([hrf_convolve(scores[:, i], tr=1.5) for i in range(5)])
    n = min(len(baseline), len(pcs_hrf))
    return np.column_stack([baseline[:n], pcs_hrf[:n]])


def within_story_loso(
    args: argparse.Namespace,
    embeddings: dict[str, np.ndarray],
    baselines: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    roi_names: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_rows: list[dict] = []
    split_rows: list[dict] = []
    leave_rows: list[dict] = []
    metadata_rows: list[dict] = []
    for story in STORIES:
        scores, metadata = fold_scores(embeddings, story)
        metadata_rows.append(metadata)
        target = targets[story]
        baseline = baselines[story]
        combined = with_pcs(baseline, scores[story])
        n = min(target.shape[1], len(baseline), len(combined))
        target = target[:, :n]
        baseline = baseline[:n]
        combined = combined[:n]
        group_target = target.mean(axis=0)
        baseline_r, _ = vectorized_cv_r(
            baseline, group_target, args.n_splits, args.alpha, True, args.cv_gap
        )
        combined_r, _ = vectorized_cv_r(
            combined, group_target, args.n_splits, args.alpha, True, args.cv_gap
        )
        for roi_index, roi in enumerate(roi_names[story]):
            full_rows.append(
                {
                    "story": story,
                    "roi": roi,
                    "baseline_cv_r": baseline_r[roi_index],
                    "combined_cv_r": combined_r[roi_index],
                    "delta_cv_r": combined_r[roi_index] - baseline_r[roi_index],
                }
            )

        for seed in range(args.subject_splits):
            indices = np.random.default_rng(seed).permutation(len(target))
            first = indices[: len(indices) // 2]
            second = indices[len(indices) // 2 :]
            baseline_split = (
                cross_group_cv_r(
                    baseline, target, first, second, args.n_splits, args.alpha, args.cv_gap
                )
                + cross_group_cv_r(
                    baseline, target, second, first, args.n_splits, args.alpha, args.cv_gap
                )
            ) / 2.0
            combined_split = (
                cross_group_cv_r(
                    combined, target, first, second, args.n_splits, args.alpha, args.cv_gap
                )
                + cross_group_cv_r(
                    combined, target, second, first, args.n_splits, args.alpha, args.cv_gap
                )
            ) / 2.0
            split_rows.append(
                {
                    "story": story,
                    "seed": seed,
                    "mean_baseline_cv_r": float(np.mean(baseline_split)),
                    "mean_combined_cv_r": float(np.mean(combined_split)),
                    "mean_delta_cv_r": float(np.mean(combined_split - baseline_split)),
                }
            )

        all_subjects = np.arange(len(target))
        subjects = story_subjects(args, story)
        for heldout_index, subject in enumerate(subjects):
            training = all_subjects[all_subjects != heldout_index]
            testing = np.asarray([heldout_index])
            baseline_leave = cross_group_cv_r(
                baseline, target, training, testing, args.n_splits, args.alpha, args.cv_gap
            )
            combined_leave = cross_group_cv_r(
                combined, target, training, testing, args.n_splits, args.alpha, args.cv_gap
            )
            leave_rows.append(
                {
                    "story": story,
                    "subject": subject,
                    "mean_baseline_cv_r": float(np.mean(baseline_leave)),
                    "mean_combined_cv_r": float(np.mean(combined_leave)),
                    "mean_delta_cv_r": float(np.mean(combined_leave - baseline_leave)),
                }
            )
        print(f"Completed within-story LOSO PCA: {story}", flush=True)
    return (
        pd.DataFrame(full_rows),
        pd.DataFrame(split_rows),
        pd.DataFrame(leave_rows),
        pd.DataFrame(metadata_rows),
    )


def detrend_and_standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    time = np.arange(len(values), dtype=float)
    design = np.column_stack([time, np.ones(len(time), dtype=float)])
    residual = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
    mean = residual.mean(axis=0, keepdims=True)
    scale = residual.std(axis=0, keepdims=True)
    scale[(scale <= 0) | ~np.isfinite(scale)] = 1.0
    return (residual - mean) / scale


def balanced_moments(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.vstack([values.mean(axis=0) for values in arrays]).mean(axis=0)
    second = np.vstack([(values**2).mean(axis=0) for values in arrays]).mean(axis=0)
    scale = np.sqrt(np.maximum(second - mean**2, 1e-12))
    return mean, scale


def transfer_model(
    train_x: list[np.ndarray],
    train_y: list[np.ndarray],
    test_x: np.ndarray,
    test_y: np.ndarray,
    alpha: float,
) -> np.ndarray:
    model, mean, scale = fit_transfer_model(train_x, train_y, alpha)
    prediction = model.predict((test_x - mean) / scale)
    return correlation_by_column(detrend_and_standardize(test_y), prediction)


def fit_transfer_model(
    train_x: list[np.ndarray],
    train_y: list[np.ndarray],
    alpha: float,
) -> tuple[Ridge, np.ndarray, np.ndarray]:
    mean, scale = balanced_moments(train_x)
    x_train = np.vstack([(values - mean) / scale for values in train_x])
    y_train = np.vstack([detrend_and_standardize(values) for values in train_y])
    weights = np.concatenate(
        [
            np.full(len(values), sum(map(len, train_x)) / (len(train_x) * len(values)))
            for values in train_x
        ]
    )
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(x_train, y_train, sample_weight=weights)
    return model, mean, scale


def cross_story_transfer(
    args: argparse.Namespace,
    embeddings: dict[str, np.ndarray],
    baselines: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    roi_names: dict[str, list[str]],
) -> pd.DataFrame:
    rows: list[dict] = []
    for omitted_story in STORIES:
        scores, _ = fold_scores(embeddings, omitted_story)
        training_stories = [story for story in STORIES if story != omitted_story]
        story_designs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for story in STORIES:
            target = targets[story].mean(axis=0)
            baseline = baselines[story]
            combined = with_pcs(baseline, scores[story])
            n = min(len(target), len(baseline), len(combined))
            story_designs[story] = (baseline[:n], combined[:n], target[:n])
        for model_name, design_index in [("rich_audio", 0), ("combined", 1)]:
            heldout_r = transfer_model(
                [story_designs[story][design_index] for story in training_stories],
                [story_designs[story][2] for story in training_stories],
                story_designs[omitted_story][design_index],
                story_designs[omitted_story][2],
                args.alpha,
            )
            for roi_index, roi in enumerate(roi_names[omitted_story]):
                rows.append(
                    {
                        "story": omitted_story,
                        "model": model_name,
                        "roi": roi,
                        "heldout_story_r": heldout_r[roi_index],
                    }
                )
        print(f"Completed cross-story transfer: {omitted_story}", flush=True)
    long = pd.DataFrame(rows)
    wide = long.pivot(index=["story", "roi"], columns="model", values="heldout_story_r").reset_index()
    wide["delta_r"] = wide["combined"] - wide["rich_audio"]
    return wide


def cross_story_participant_transfer(
    args: argparse.Namespace,
    embeddings: dict[str, np.ndarray],
    baselines: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    roi_names: dict[str, list[str]],
) -> pd.DataFrame:
    """Hold out both a complete story and one listener from neural-model fitting."""
    rows: list[dict] = []
    subjects = {story: story_subjects(args, story) for story in STORIES}
    for omitted_story in STORIES:
        scores, _ = fold_scores(embeddings, omitted_story)
        training_stories = [story for story in STORIES if story != omitted_story]
        story_designs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for story in STORIES:
            baseline = baselines[story]
            combined = with_pcs(baseline, scores[story])
            n = min(targets[story].shape[1], len(baseline), len(combined))
            story_designs[story] = (
                baseline[:n],
                combined[:n],
                targets[story][:, :n],
            )

        for heldout_index, heldout_subject in enumerate(subjects[omitted_story]):
            training_targets: list[np.ndarray] = []
            for story in training_stories:
                keep = np.asarray(
                    [subject != heldout_subject for subject in subjects[story]], dtype=bool
                )
                if not np.any(keep):
                    raise ValueError(
                        f"No training participants remain for {story} after excluding "
                        f"{heldout_subject}"
                    )
                training_targets.append(story_designs[story][2][keep].mean(axis=0))

            for model_name, design_index in [("rich_audio", 0), ("combined", 1)]:
                heldout_r = transfer_model(
                    [story_designs[story][design_index] for story in training_stories],
                    training_targets,
                    story_designs[omitted_story][design_index],
                    story_designs[omitted_story][2][heldout_index],
                    args.alpha,
                )
                for roi_index, roi in enumerate(roi_names[omitted_story]):
                    rows.append(
                        {
                            "story": omitted_story,
                            "subject": heldout_subject,
                            "model": model_name,
                            "roi": roi,
                            "heldout_story_participant_r": heldout_r[roi_index],
                        }
                    )
        print(
            f"Completed story-and-participant transfer: {omitted_story}", flush=True
        )

    long = pd.DataFrame(rows)
    wide = long.pivot(
        index=["story", "subject", "roi"],
        columns="model",
        values="heldout_story_participant_r",
    ).reset_index()
    wide["delta_r"] = wide["combined"] - wide["rich_audio"]
    return wide


def cross_story_temporal_shift_null(
    args: argparse.Namespace,
    embeddings: dict[str, np.ndarray],
    baselines: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(args.shift_seed)
    null_by_story: dict[str, np.ndarray] = {}
    observed_by_story: dict[str, float] = {}
    for omitted_story in STORIES:
        scores, _ = fold_scores(embeddings, omitted_story)
        training_stories = [story for story in STORIES if story != omitted_story]
        story_designs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for story in STORIES:
            target = targets[story].mean(axis=0)
            baseline = baselines[story]
            combined = with_pcs(baseline, scores[story])
            n = min(len(target), len(baseline), len(combined))
            story_designs[story] = (baseline[:n], combined[:n], target[:n])

        train_targets = [story_designs[story][2] for story in training_stories]
        baseline_model, baseline_mean, baseline_scale = fit_transfer_model(
            [story_designs[story][0] for story in training_stories],
            train_targets,
            args.alpha,
        )
        combined_model, combined_mean, combined_scale = fit_transfer_model(
            [story_designs[story][1] for story in training_stories],
            train_targets,
            args.alpha,
        )
        baseline_test, combined_test, target_test = story_designs[omitted_story]
        target_test = detrend_and_standardize(target_test)
        baseline_prediction = baseline_model.predict(
            (baseline_test - baseline_mean) / baseline_scale
        )
        baseline_r = correlation_by_column(target_test, baseline_prediction)
        observed_prediction = combined_model.predict(
            (combined_test - combined_mean) / combined_scale
        )
        observed_r = correlation_by_column(target_test, observed_prediction)
        observed_by_story[omitted_story] = float(np.mean(observed_r - baseline_r))

        n_timepoints = len(combined_test)
        allowed_offsets = np.arange(
            args.min_shift_trs, n_timepoints - args.min_shift_trs + 1
        )
        if not len(allowed_offsets):
            raise ValueError(f"No valid circular shifts for {omitted_story}")
        null_values = np.empty(args.n_shifts, dtype=float)
        for iteration in range(args.n_shifts):
            shifted = combined_test.copy()
            offset = int(rng.choice(allowed_offsets))
            shifted[:, -5:] = np.roll(shifted[:, -5:], offset, axis=0)
            prediction = combined_model.predict(
                (shifted - combined_mean) / combined_scale
            )
            shifted_r = correlation_by_column(target_test, prediction)
            null_values[iteration] = float(np.mean(shifted_r - baseline_r))
        null_by_story[omitted_story] = null_values
        print(f"Completed cross-story temporal null: {omitted_story}", flush=True)

    null_frame = pd.DataFrame(null_by_story)
    null_frame.insert(0, "iteration", np.arange(args.n_shifts))
    null_mean = null_frame[STORIES].mean(axis=1).to_numpy(dtype=float)
    observed_mean = float(np.mean(list(observed_by_story.values())))
    summary = pd.DataFrame(
        [
            {
                "analysis": "train_12_test_1_heldout_story_temporal_shift_null",
                "status": "post_outcome_exploratory_sensitivity",
                "n_stories": len(STORIES),
                "n_shifts": args.n_shifts,
                "minimum_shift_trs": args.min_shift_trs,
                "observed_mean_delta_r": observed_mean,
                "null_mean": float(np.mean(null_mean)),
                "null_std": float(np.std(null_mean, ddof=1)),
                "null_95th_percentile": float(np.quantile(null_mean, 0.95)),
                "empirical_p_greater": float(
                    (1 + np.sum(null_mean >= observed_mean)) / (1 + args.n_shifts)
                ),
            }
        ]
    )
    return null_frame, summary


def save_within_results(
    output_dir: Path,
    full: pd.DataFrame,
    split: pd.DataFrame,
    leave: pd.DataFrame,
    metadata: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    full.to_csv(output_dir / "full_group_long.csv", index=False)
    split.to_csv(output_dir / "split_half_long.csv", index=False)
    leave.to_csv(output_dir / "leave_one_participant_out_long.csv", index=False)
    metadata.to_csv(output_dir / "pca_fold_metadata.csv", index=False)
    full_story = full.groupby("story", as_index=False).delta_cv_r.mean()
    split_story = split.groupby("story", as_index=False).mean_delta_cv_r.mean()
    leave_story = leave.groupby("story", as_index=False).mean_delta_cv_r.mean()
    full_story.to_csv(output_dir / "full_group_story_means.csv", index=False)
    split_story.to_csv(output_dir / "split_half_story_means.csv", index=False)
    leave_story.to_csv(output_dir / "leave_one_participant_out_story_means.csv", index=False)
    summary = []
    for analysis, frame, column in [
        ("full_group", full_story, "delta_cv_r"),
        ("participant_split_half", split_story, "mean_delta_cv_r"),
        ("leave_one_participant_out", leave_story, "mean_delta_cv_r"),
    ]:
        row = {"analysis": analysis, "scope": "all_13_intact_stories"}
        row.update(one_sample_summary(frame[column]))
        summary.append(row)
    pd.DataFrame(summary).to_csv(output_dir / "summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/jnl_13_story_extension/loso")
    parser.add_argument(
        "--development-config",
        default="configs/stories_all_fmriprep_confounds_motionfiltered.py",
    )
    parser.add_argument(
        "--initial-evaluation-config",
        default="results/jnl_heldout_story_replication/heldout_motionfiltered_config.py",
    )
    parser.add_argument(
        "--new-evaluation-config",
        default=(
            "results/jnl_13_story_extension/new_story_fixed_pca/"
            "heldout_motionfiltered_config.py"
        ),
    )
    parser.add_argument(
        "--development-regressor-dir",
        default="results/jnl_revision_major_review/rich_audio_controls/rich_audio_regressors",
    )
    parser.add_argument(
        "--initial-evaluation-regressor-dir",
        default="results/jnl_heldout_story_replication/heldout_regressors",
    )
    parser.add_argument(
        "--new-evaluation-regressor-dir",
        default="results/jnl_13_story_extension/new_story_fixed_pca/heldout_regressors",
    )
    parser.add_argument(
        "--development-roi-cache",
        default=(
            "results/journal_neurolinguistics_robustness/pc_count_full_corrected/"
            "roi_time_series_cache"
        ),
    )
    parser.add_argument(
        "--initial-evaluation-roi-cache",
        default="results/jnl_heldout_story_replication/roi_cache",
    )
    parser.add_argument(
        "--new-evaluation-roi-cache",
        default="results/jnl_13_story_extension/new_story_fixed_pca/roi_cache",
    )
    parser.add_argument("--gentle-root", required=True)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--cv-gap", type=int, default=10)
    parser.add_argument("--subject-splits", type=int, default=50)
    parser.add_argument("--n-shifts", type=int, default=1000)
    parser.add_argument("--min-shift-trs", type=int, default=20)
    parser.add_argument("--shift-seed", type=int, default=20260805)
    parser.add_argument(
        "--stage",
        choices=[
            "embeddings",
            "within",
            "transfer",
            "participant-transfer",
            "shift-null",
            "all",
        ],
        default="all",
    )
    args = parser.parse_args()

    os.environ["SEMANTIC_FMRI_TIMING"] = "gentle"
    os.environ["SEMANTIC_FMRI_GENTLE_DIR"] = str(Path(args.gentle_root))
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings = embedding_cache(args)
    if args.stage == "embeddings":
        return
    baselines, targets, roi_names = load_data(args)
    if args.stage in {"within", "all"}:
        full, split, leave, metadata = within_story_loso(
            args, embeddings, baselines, targets, roi_names
        )
        save_within_results(output_dir / "within_story", full, split, leave, metadata)
    if args.stage in {"transfer", "all"}:
        transfer = cross_story_transfer(args, embeddings, baselines, targets, roi_names)
        transfer.to_csv(output_dir / "cross_story_transfer_long.csv", index=False)
        story = transfer.groupby("story", as_index=False).delta_r.mean()
        story.to_csv(output_dir / "cross_story_transfer_story_means.csv", index=False)
        summary = {"analysis": "train_12_test_1_full_story"}
        summary.update(one_sample_summary(story["delta_r"]))
        pd.DataFrame([summary]).to_csv(
            output_dir / "cross_story_transfer_summary.csv", index=False
        )
    if args.stage in {"participant-transfer", "all"}:
        participant_transfer = cross_story_participant_transfer(
            args, embeddings, baselines, targets, roi_names
        )
        participant_transfer.to_csv(
            output_dir / "cross_story_participant_transfer_long.csv", index=False
        )
        participant_story = participant_transfer.groupby(
            "story", as_index=False
        ).delta_r.mean()
        participant_story.to_csv(
            output_dir / "cross_story_participant_transfer_story_means.csv",
            index=False,
        )
        participant_summary = {
            "analysis": "train_12_test_1_story_and_participant_held_out",
            "status": "post_outcome_exploratory_sensitivity",
        }
        participant_summary.update(one_sample_summary(participant_story["delta_r"]))
        pd.DataFrame([participant_summary]).to_csv(
            output_dir / "cross_story_participant_transfer_summary.csv", index=False
        )
    if args.stage in {"shift-null", "all"}:
        null_frame, null_summary = cross_story_temporal_shift_null(
            args, embeddings, baselines, targets
        )
        null_frame.to_csv(
            output_dir / "cross_story_temporal_shift_null.csv", index=False
        )
        null_summary.to_csv(
            output_dir / "cross_story_temporal_shift_summary.csv", index=False
        )


if __name__ == "__main__":
    main()
