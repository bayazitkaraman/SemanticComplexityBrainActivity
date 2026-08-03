from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from jnl_heldout_story_replication import (  # noqa: E402
    DISCOVERY_STORIES,
    HELDOUT_STORIES,
)
from jnl_pc_count_sensitivity import LEXICAL_BASELINE_COLS, vectorized_cv_r  # noqa: E402
from jnl_revision_context_layer_models import (  # noqa: E402
    cumulative_current_token_embeddings,
)
from jnl_rich_audio_controls import RICH_AUDIO_COLUMNS  # noqa: E402
from jnl_shared_response_context_validation import (  # noqa: E402
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


ALL_STORIES = DISCOVERY_STORIES + HELDOUT_STORIES


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def story_templates(discovery_config: Path, heldout_config: Path) -> dict[str, dict]:
    rows = load_story_list(str(discovery_config)) + load_story_list(str(heldout_config))
    templates = {row["name"]: row for row in get_unique_stories(rows)}
    missing = [story for story in ALL_STORIES if story not in templates]
    if missing:
        raise ValueError(f"Missing story templates: {missing}")
    return templates


def existing_regressor_path(args: argparse.Namespace, story: str) -> Path:
    if story in DISCOVERY_STORIES:
        directory = resolve(args.discovery_locked_regressor_dir)
    else:
        directory = resolve(args.heldout_locked_regressor_dir)
    return directory / f"{story}_shared_pca_regressors.csv"


def fit_extended_regressors(args: argparse.Namespace, output_dir: Path) -> Path:
    regressor_dir = output_dir / "extended_pc_regressors"
    regressor_dir.mkdir(parents=True, exist_ok=True)
    expected = [regressor_dir / f"{story}_shared_pca_pc1_to_pc10.csv" for story in ALL_STORIES]
    if args.resume and all(path.exists() for path in expected):
        print("Using cached PC1-PC10 regressors", flush=True)
        return regressor_dir

    templates = story_templates(
        resolve(args.discovery_subject_list), resolve(args.heldout_subject_list)
    )
    tokenizer, model, device = load_gpt2()
    model.eval()
    embeddings: dict[str, np.ndarray] = {}
    for story in ALL_STORIES:
        chunks, _, _, _ = transcript_to_tr_chunks_uniform(templates[story], tr=1.5)
        print(f"Computing cumulative GPT-2 embeddings: {story}", flush=True)
        embeddings[story] = cumulative_current_token_embeddings(
            chunks,
            tokenizer,
            model,
            layer=12,
            max_length=1024,
            output_stride=256,
            device=device,
        )

    discovery_matrix = np.vstack([embeddings[story] for story in DISCOVERY_STORIES])
    scaler = StandardScaler().fit(discovery_matrix)
    pca = PCA(n_components=10, svd_solver="randomized", random_state=args.random_seed)
    pca.fit(scaler.transform(discovery_matrix))
    projected = {
        story: pca.transform(scaler.transform(embeddings[story])) for story in ALL_STORIES
    }

    projected_discovery = np.vstack([projected[story] for story in DISCOVERY_STORIES])
    locked_discovery = np.vstack(
        [
            pd.read_csv(existing_regressor_path(args, story))[
                [f"PC{i}" for i in range(1, 6)]
            ].to_numpy(dtype=float)
            for story in DISCOVERY_STORIES
        ]
    )
    component_signs = np.ones(10, dtype=float)
    audit_rows = []
    for component in range(5):
        correlation = float(
            stats.pearsonr(
                projected_discovery[:, component], locked_discovery[:, component]
            ).statistic
        )
        component_signs[component] = 1.0 if correlation >= 0 else -1.0
    for component in range(5, 10):
        loading = pca.components_[component]
        component_signs[component] = 1.0 if loading[np.argmax(np.abs(loading))] >= 0 else -1.0
    for story in projected:
        projected[story] = projected[story] * component_signs

    for story in ALL_STORIES:
        locked = pd.read_csv(existing_regressor_path(args, story))
        locked_scores = locked[[f"PC{i}" for i in range(1, 6)]].to_numpy(dtype=float)
        if len(locked_scores) != len(projected[story]):
            raise ValueError(
                f"Regressor length differs for {story}: "
                f"locked={len(locked_scores)}, projected={len(projected[story])}"
            )
        for component in range(5):
            audit_rows.append(
                {
                    "story": story,
                    "component": component + 1,
                    "projected_locked_correlation": float(
                        stats.pearsonr(
                            projected[story][:, component], locked_scores[:, component]
                        ).statistic
                    ),
                    "projected_locked_max_absolute_error": float(
                        np.max(
                            np.abs(
                                projected[story][:, component]
                                - locked_scores[:, component]
                            )
                        )
                    ),
                }
            )

        # The primary PC1-PC5 regressors remain byte-for-byte numerical inputs to the
        # sensitivity analysis. Only PC6-PC10 come from the extended decomposition.
        scores = projected[story].copy()
        scores[:, :5] = locked_scores
        frame = pd.DataFrame({"time_TR": locked["time_TR"].to_numpy(dtype=float)})
        for component in range(10):
            name = f"PC{component + 1}"
            frame[name] = scores[:, component]
            if component < 5 and f"{name}_hrf" in locked:
                frame[f"{name}_hrf"] = locked[f"{name}_hrf"].to_numpy(dtype=float)
            else:
                frame[f"{name}_hrf"] = hrf_convolve(scores[:, component], tr=1.5)
        frame.to_csv(
            regressor_dir / f"{story}_shared_pca_pc1_to_pc10.csv", index=False
        )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output_dir / "extended_pca_projection_audit.csv", index=False)
    if float(audit["projected_locked_correlation"].min()) < 0.995:
        raise RuntimeError("Extended PCA did not recover the locked component space closely.")
    pd.DataFrame(
        {
            "component": [f"PC{i}" for i in range(1, 11)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "alignment_sign": component_signs,
            "source_in_models": ["locked_primary"] * 5 + ["extended_pca"] * 5,
        }
    ).to_csv(output_dir / "extended_pca_metadata.csv", index=False)
    return regressor_dir


def baseline_columns(frame: pd.DataFrame) -> list[str]:
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
    return columns


def load_designs(
    args: argparse.Namespace, regressor_dir: Path, story: str
) -> tuple[np.ndarray, np.ndarray]:
    if story in DISCOVERY_STORIES:
        baseline_path = (
            resolve(args.discovery_baseline_regressor_dir)
            / f"{story}_shared_pca_regressors.csv"
        )
    else:
        baseline_path = existing_regressor_path(args, story)
    baseline_frame = pd.read_csv(baseline_path)
    baseline = baseline_frame[baseline_columns(baseline_frame)].to_numpy(dtype=float)
    extended = pd.read_csv(
        regressor_dir / f"{story}_shared_pca_pc1_to_pc10.csv"
    )
    pcs = extended[[f"PC{i}_hrf" for i in range(1, 11)]].to_numpy(dtype=float)
    return baseline, pcs


def load_all_story_data(
    args: argparse.Namespace, regressor_dir: Path
) -> dict[str, tuple[np.ndarray, list[str], np.ndarray, np.ndarray, list[str]]]:
    story_data = {}
    for story in ALL_STORIES:
        cache_dir = (
            resolve(args.discovery_roi_cache_dir)
            if story in DISCOVERY_STORIES
            else resolve(args.heldout_roi_cache_dir)
        )
        targets, roi_names = load_story_targets(cache_dir, story)
        baseline, pcs = load_designs(args, regressor_dir, story)
        n_timepoints = targets.shape[1]
        if min(len(baseline), len(pcs)) < n_timepoints:
            raise ValueError(f"Design is shorter than target for {story}")
        baseline = baseline[:n_timepoints]
        pcs = pcs[:n_timepoints]
        cache_files = sorted(cache_dir.glob(f"{story}_*.npz"))
        subjects = [path.stem[len(story) + 1 :] for path in cache_files]
        if len(subjects) != len(targets):
            raise ValueError(f"Subject order audit failed for {story}")
        story_data[story] = (targets, roi_names, baseline, pcs, subjects)
        print(
            f"Loaded {story}: {targets.shape[0]} participants, "
            f"{n_timepoints} time points, {len(roi_names)} ROIs",
            flush=True,
        )
    return story_data


def evaluate(
    story_data: dict[str, tuple[np.ndarray, list[str], np.ndarray, np.ndarray, list[str]]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_rows = []
    split_rows = []
    leave_rows = []
    component_counts = list(range(1, 11))

    for story, (targets, roi_names, baseline, pcs, subjects) in story_data.items():
        print(f"Evaluating PC-count sensitivity: {story}", flush=True)
        group_target = np.mean(targets, axis=0)
        baseline_full, _ = vectorized_cv_r(
            baseline, group_target, args.n_splits, args.alpha, True, args.cv_gap
        )
        for count in component_counts:
            combined, _ = vectorized_cv_r(
                np.column_stack([baseline, pcs[:, :count]]),
                group_target,
                args.n_splits,
                args.alpha,
                True,
                args.cv_gap,
            )
            for roi_index, roi in enumerate(roi_names):
                full_rows.append(
                    {
                        "component_count": count,
                        "story": story,
                        "roi": roi,
                        "baseline_cv_r": baseline_full[roi_index],
                        "combined_cv_r": combined[roi_index],
                        "delta_cv_r": combined[roi_index] - baseline_full[roi_index],
                    }
                )

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
            for count in component_counts:
                design = np.column_stack([baseline, pcs[:, :count]])
                combined_split = (
                    cross_group_cv_r(
                        design, targets, first, second,
                        args.n_splits, args.alpha, args.cv_gap,
                    )
                    + cross_group_cv_r(
                        design, targets, second, first,
                        args.n_splits, args.alpha, args.cv_gap,
                    )
                ) / 2.0
                split_rows.append(
                    {
                        "component_count": count,
                        "story": story,
                        "seed": seed,
                        "mean_delta_cv_r": float(np.mean(combined_split - baseline_split)),
                    }
                )

        all_subjects = np.arange(len(targets))
        for heldout_index, subject in enumerate(subjects):
            training = all_subjects[all_subjects != heldout_index]
            testing = np.asarray([heldout_index])
            baseline_leave = cross_group_cv_r(
                baseline, targets, training, testing,
                args.n_splits, args.alpha, args.cv_gap,
            )
            for count in component_counts:
                combined_leave = cross_group_cv_r(
                    np.column_stack([baseline, pcs[:, :count]]),
                    targets,
                    training,
                    testing,
                    args.n_splits,
                    args.alpha,
                    args.cv_gap,
                )
                leave_rows.append(
                    {
                        "component_count": count,
                        "story": story,
                        "subject": subject,
                        "mean_delta_cv_r": float(
                            np.mean(combined_leave - baseline_leave)
                        ),
                    }
                )
    return pd.DataFrame(full_rows), pd.DataFrame(split_rows), pd.DataFrame(leave_rows)


def story_means(
    full: pd.DataFrame, split: pd.DataFrame, leave: pd.DataFrame
) -> pd.DataFrame:
    full_story = (
        full.groupby(["component_count", "story"], as_index=False)["delta_cv_r"]
        .mean()
        .rename(columns={"delta_cv_r": "mean_delta_cv_r"})
    )
    full_story["analysis"] = "full_group"
    split_story = split.groupby(
        ["component_count", "story"], as_index=False
    ).mean_delta_cv_r.mean()
    split_story["analysis"] = "held_out_participant_split_half"
    leave_story = leave.groupby(
        ["component_count", "story"], as_index=False
    ).mean_delta_cv_r.mean()
    leave_story["analysis"] = "leave_one_participant_out"
    return pd.concat([full_story, split_story, leave_story], ignore_index=True)


def scope_summary(stories: pd.DataFrame) -> pd.DataFrame:
    scopes = {
        "development_eight_stories": set(DISCOVERY_STORIES),
        "heldout_three_stories": set(HELDOUT_STORIES),
        "expanded_eleven_stories": set(ALL_STORIES),
    }
    rows = []
    for (analysis, count), group in stories.groupby(["analysis", "component_count"]):
        for scope, members in scopes.items():
            selected = group[group["story"].isin(members)]
            row = {"analysis": analysis, "component_count": count, "scope": scope}
            row.update(one_sample_summary(selected["mean_delta_cv_r"]))
            rows.append(row)
    return pd.DataFrame(rows)


def reproduction_audit(args: argparse.Namespace, stories: pd.DataFrame) -> pd.DataFrame:
    expected_files = {
        ("full_group", "development"): (
            resolve(args.original_validation_dir) / "full_group_story_means.csv",
            "cumulative_context_additive",
        ),
        ("held_out_participant_split_half", "development"): (
            resolve(args.original_validation_dir) / "split_half_story_means.csv",
            "cumulative_context_additive",
        ),
        ("leave_one_participant_out", "development"): (
            resolve(args.original_validation_dir) / "leave_one_participant_out_story_means.csv",
            "cumulative_context_additive",
        ),
        ("full_group", "heldout"): (
            resolve(args.heldout_validation_dir) / "heldout_full_group_story_means.csv",
            "delta_cv_r",
        ),
        ("held_out_participant_split_half", "heldout"): (
            resolve(args.heldout_validation_dir) / "heldout_split_half_story_means.csv",
            "mean_delta_cv_r",
        ),
        ("leave_one_participant_out", "heldout"): (
            resolve(args.heldout_validation_dir)
            / "heldout_leave_one_participant_out_story_means.csv",
            "mean_delta_cv_r",
        ),
    }
    current = stories[stories["component_count"].eq(5)]
    rows = []
    for (analysis, scope), (path, value_column) in expected_files.items():
        expected = pd.read_csv(path)[["story", value_column]].rename(
            columns={value_column: "expected"}
        )
        observed = current[current["analysis"].eq(analysis)][
            ["story", "mean_delta_cv_r"]
        ].rename(columns={"mean_delta_cv_r": "observed"})
        merged = expected.merge(observed, on="story", how="inner")
        for row in merged.itertuples(index=False):
            rows.append(
                {
                    "analysis": analysis,
                    "scope": scope,
                    "story": row.story,
                    "expected": row.expected,
                    "observed": row.observed,
                    "absolute_error": abs(row.observed - row.expected),
                }
            )
    audit = pd.DataFrame(rows)
    if len(audit) != 33 or float(audit["absolute_error"].max()) > 1e-10:
        raise RuntimeError(
            "PC5 did not exactly reproduce the locked analyses: "
            f"n={len(audit)}, max error={audit['absolute_error'].max()}"
        )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--discovery-subject-list",
        default="configs/stories_all_fmriprep_confounds_motionfiltered.py",
    )
    parser.add_argument(
        "--heldout-subject-list",
        default="results/jnl_heldout_story_replication/heldout_motionfiltered_config.py",
    )
    parser.add_argument(
        "--discovery-roi-cache-dir",
        default=(
            "results/journal_neurolinguistics_robustness/pc_count_full_corrected/"
            "roi_time_series_cache"
        ),
    )
    parser.add_argument(
        "--heldout-roi-cache-dir",
        default="results/jnl_heldout_story_replication/roi_cache",
    )
    parser.add_argument(
        "--discovery-locked-regressor-dir",
        default=(
            "results/jnl_revision_major_review/context_layer_models/regressors/"
            "cumulative_context_layer12_current_tokens"
        ),
    )
    parser.add_argument(
        "--discovery-baseline-regressor-dir",
        default=(
            "results/jnl_revision_major_review/rich_audio_controls/"
            "rich_audio_regressors"
        ),
    )
    parser.add_argument(
        "--heldout-locked-regressor-dir",
        default="results/jnl_heldout_story_replication/heldout_regressors",
    )
    parser.add_argument(
        "--original-validation-dir",
        default="results/jnl_shared_response_context_validation",
    )
    parser.add_argument(
        "--heldout-validation-dir",
        default="results/jnl_heldout_story_replication",
    )
    parser.add_argument(
        "--output-dir",
        default="results/jnl_shared_response_pc_count_sensitivity",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--cv-gap", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--subject-splits", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=20260803)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    regressor_dir = fit_extended_regressors(args, output_dir)
    story_data = load_all_story_data(args, regressor_dir)
    full, split, leave = evaluate(story_data, args)
    stories = story_means(full, split, leave)
    summary = scope_summary(stories)
    audit = reproduction_audit(args, stories)

    full.to_csv(output_dir / "pc_count_full_group_long.csv", index=False)
    split.to_csv(output_dir / "pc_count_split_half.csv", index=False)
    leave.to_csv(output_dir / "pc_count_leave_one_participant_out.csv", index=False)
    stories.to_csv(output_dir / "pc_count_story_means.csv", index=False)
    summary.to_csv(output_dir / "pc_count_scope_summary.csv", index=False)
    audit.to_csv(output_dir / "pc5_reproduction_audit.csv", index=False)
    (output_dir / "analysis_specification.json").write_text(
        json.dumps(
            {
                "status": "post-hoc dimensionality sensitivity",
                "primary_components": "locked PC1-PC5",
                "additional_components": "PC6-PC10 fit on development stories only",
                "heldout_projection": "no held-out story used to fit scaling or PCA",
                "n_splits": args.n_splits,
                "cv_gap": args.cv_gap,
                "ridge_alpha": args.alpha,
                "participant_split_seeds": args.subject_splits,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\nPC-count scope summary:\n", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(
        f"\nPC5 reproduction maximum absolute error: "
        f"{audit['absolute_error'].max():.3g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
