from __future__ import annotations

import argparse
import hashlib
import json
import os
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

from jnl_13_story_loso import (  # noqa: E402
    DEFAULT_REGRESSOR_DIR,
    DEFAULT_ROI_CACHE,
    STORIES,
    load_data,
    project,
    source_regressor_path,
    story_balanced_pca,
    story_subjects,
    template_map,
)
from jnl_pc_count_sensitivity import LEXICAL_BASELINE_COLS, vectorized_cv_r  # noqa: E402
from jnl_revision_context_layer_models import (  # noqa: E402
    independent_chunk_embeddings_batched,
    mixed_effect_intercept,
)
from semantic_fmri.utils import hrf_convolve, load_gpt2, transcript_to_tr_chunks_uniform  # noqa: E402


VARIANTS = {
    "primary_rich_loso_gap10": {
        "baseline": "rich",
        "pca": "loso_story_balanced",
        "cv_gap": 10,
    },
    "lexical_loso_gap10": {
        "baseline": "lexical",
        "pca": "loso_story_balanced",
        "cv_gap": 10,
    },
    "rich_loso_gap0": {
        "baseline": "rich",
        "pca": "loso_story_balanced",
        "cv_gap": 0,
    },
    "rich_pooled_gap10": {
        "baseline": "rich",
        "pca": "pooled_tr",
        "cv_gap": 10,
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate_specification(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        specification = json.load(stream)
    specified_stories = specification["dataset"]["stories"]
    if specified_stories != STORIES:
        raise ValueError("The locked story order does not match the analysis script.")
    if specification["dataset"]["expected_motion_filtered_scans"] != 567:
        raise ValueError("Unexpected locked scan count.")
    return specification


def local_embedding_cache(args: argparse.Namespace) -> dict[str, np.ndarray]:
    output_dir = ROOT / args.output_dir / "independent_embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    templates = template_map(args)
    missing = [story for story in STORIES if not (output_dir / f"{story}.npy").exists()]
    if missing:
        tokenizer, model, device = load_gpt2()
        model.eval()
        for story in missing:
            chunks, _, _, _ = transcript_to_tr_chunks_uniform(templates[story], tr=1.5)
            print(f"Embedding independent TR chunks for {story}: {len(chunks)} TRs", flush=True)
            values = independent_chunk_embeddings_batched(
                chunks,
                tokenizer,
                model,
                layer=12,
                batch_size=args.embedding_batch_size,
                max_length=args.embedding_max_length,
                device=device,
            )
            np.save(output_dir / f"{story}.npy", values.astype(np.float32))

    embeddings = {
        story: np.load(output_dir / f"{story}.npy", allow_pickle=False)
        for story in STORIES
    }
    inventory = pd.DataFrame(
        [
            {
                "story": story,
                "n_trs": int(len(values)),
                "n_features": int(values.shape[1]),
                "finite": bool(np.all(np.isfinite(values))),
            }
            for story, values in embeddings.items()
        ]
    )
    inventory.to_csv(ROOT / args.output_dir / "embedding_inventory.csv", index=False)
    return embeddings


def feature_scores(
    embeddings: dict[str, np.ndarray], n_components: int
) -> tuple[dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    loso: dict[str, np.ndarray] = {}
    metadata_rows: list[dict] = []
    for omitted_story in STORIES:
        training = [story for story in STORIES if story != omitted_story]
        mean, scale, components, eigenvalues = story_balanced_pca(
            embeddings, training, n_components=n_components
        )
        loso[omitted_story] = project(
            embeddings[omitted_story], mean, scale, components
        )
        metadata_rows.append(
            {
                "pca": "loso_story_balanced",
                "omitted_story": omitted_story,
                "n_training_stories": len(training),
                "test_story_in_feature_fit": False,
                **{
                    f"pc{i + 1}_variance_ratio": float(eigenvalues[i] / eigenvalues.sum())
                    for i in range(n_components)
                },
            }
        )

    pooled_values = np.vstack([embeddings[story] for story in STORIES])
    scaler = StandardScaler()
    pooled_scaled = scaler.fit_transform(pooled_values)
    pca = PCA(n_components=n_components)
    pca.fit(pooled_scaled)
    pooled = {
        story: pca.transform(scaler.transform(embeddings[story]))
        for story in STORIES
    }
    metadata_rows.append(
        {
            "pca": "pooled_tr",
            "omitted_story": "none",
            "n_training_stories": len(STORIES),
            "test_story_in_feature_fit": True,
            **{
                f"pc{i + 1}_variance_ratio": float(pca.explained_variance_ratio_[i])
                for i in range(n_components)
            },
        }
    )
    return {"loso_story_balanced": loso, "pooled_tr": pooled}, pd.DataFrame(metadata_rows)


def lexical_baselines(args: argparse.Namespace) -> dict[str, np.ndarray]:
    baselines: dict[str, np.ndarray] = {}
    for story in STORIES:
        frame = pd.read_csv(source_regressor_path(args, story))
        missing = [column for column in LEXICAL_BASELINE_COLS if column not in frame]
        if missing:
            raise ValueError(f"Missing lexical baseline columns for {story}: {missing}")
        baselines[story] = frame[LEXICAL_BASELINE_COLS].to_numpy(dtype=float)
    return baselines


def convolved_scores(scores: np.ndarray, n_components: int) -> np.ndarray:
    return np.column_stack(
        [hrf_convolve(scores[:, index], tr=1.5) for index in range(n_components)]
    )


def scan_scores(
    args: argparse.Namespace,
    score_spaces: dict[str, dict[str, np.ndarray]],
    rich_baselines: dict[str, np.ndarray],
    lexical: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    roi_names: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scan_rows: list[dict] = []
    roi_rows: list[dict] = []
    baseline_cache: dict[tuple[str, int, str, int], np.ndarray] = {}

    for story in STORIES:
        subjects = story_subjects(args, story)
        if len(subjects) != len(targets[story]):
            raise ValueError(f"Subject and target counts differ for {story}.")
        pcs_by_space = {
            name: convolved_scores(scores[story], args.n_components)
            for name, scores in score_spaces.items()
        }
        print(f"Evaluating {story}: {len(subjects)} scans", flush=True)

        for subject_index, subject in enumerate(subjects):
            y_matrix = targets[story][subject_index]
            for variant, settings in VARIANTS.items():
                baseline_source = rich_baselines if settings["baseline"] == "rich" else lexical
                baseline = baseline_source[story]
                pcs = pcs_by_space[settings["pca"]]
                n = min(len(baseline), len(pcs), len(y_matrix))
                baseline_design = baseline[:n]
                combined_design = np.column_stack([baseline_design, pcs[:n]])
                y = y_matrix[:n]

                baseline_key = (
                    story,
                    subject_index,
                    settings["baseline"],
                    settings["cv_gap"],
                )
                if baseline_key not in baseline_cache:
                    baseline_cache[baseline_key], _ = vectorized_cv_r(
                        baseline_design,
                        y,
                        n_splits=args.n_splits,
                        alpha=args.alpha,
                        do_detrend=True,
                        cv_gap=settings["cv_gap"],
                    )
                baseline_r = baseline_cache[baseline_key]
                combined_r, n_used = vectorized_cv_r(
                    combined_design,
                    y,
                    n_splits=args.n_splits,
                    alpha=args.alpha,
                    do_detrend=True,
                    cv_gap=settings["cv_gap"],
                )
                delta = combined_r - baseline_r
                finite = np.isfinite(delta)
                if not np.any(finite):
                    raise ValueError(f"No finite ROI effects for {story} {subject} {variant}.")

                scan_rows.append(
                    {
                        "variant": variant,
                        "story": story,
                        "subject": subject,
                        "n_timepoints": int(n_used),
                        "n_rois": int(np.sum(finite)),
                        "mean_baseline_cv_r": float(np.mean(baseline_r[finite])),
                        "mean_combined_cv_r": float(np.mean(combined_r[finite])),
                        "mean_delta_cv_r": float(np.mean(delta[finite])),
                    }
                )
                for roi_index, roi in enumerate(roi_names[story]):
                    roi_rows.append(
                        {
                            "variant": variant,
                            "story": story,
                            "subject": subject,
                            "roi": roi,
                            "baseline_cv_r": float(baseline_r[roi_index]),
                            "combined_cv_r": float(combined_r[roi_index]),
                            "delta_cv_r": float(delta[roi_index]),
                        }
                    )

    return pd.DataFrame(scan_rows), pd.DataFrame(roi_rows)


def story_summary(scan: pd.DataFrame) -> pd.DataFrame:
    return (
        scan.groupby(["variant", "story"], as_index=False)
        .agg(
            n_scans=("subject", "size"),
            mean_baseline_cv_r=("mean_baseline_cv_r", "mean"),
            mean_combined_cv_r=("mean_combined_cv_r", "mean"),
            mean_delta_cv_r=("mean_delta_cv_r", "mean"),
            median_delta_cv_r=("mean_delta_cv_r", "median"),
            positive_scans=("mean_delta_cv_r", lambda values: int(np.sum(values > 0))),
        )
    )


def inferential_summary(stories: pd.DataFrame, scans: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    mixed_rows: list[dict] = []
    for variant in VARIANTS:
        values = stories.loc[
            stories["variant"].eq(variant), "mean_delta_cv_r"
        ].to_numpy(dtype=float)
        if len(values) != len(STORIES):
            raise ValueError(f"Expected 13 story means for {variant}, found {len(values)}.")
        t_result = stats.ttest_1samp(values, 0.0)
        wilcoxon = stats.wilcoxon(values, alternative="greater", zero_method="wilcox")
        positives = int(np.sum(values > 0))
        sign = stats.binomtest(positives, len(values), 0.5, alternative="greater")
        sem = float(stats.sem(values))
        ci = stats.t.interval(0.95, len(values) - 1, loc=float(np.mean(values)), scale=sem)
        rows.append(
            {
                "variant": variant,
                "n_stories": len(values),
                "n_scans": int(scans["variant"].eq(variant).sum()),
                "mean_delta_cv_r": float(np.mean(values)),
                "median_delta_cv_r": float(np.median(values)),
                "ci95_low": float(ci[0]),
                "ci95_high": float(ci[1]),
                "positive_stories": positives,
                "t_stat": float(t_result.statistic),
                "p_ttest_two_sided": float(t_result.pvalue),
                "wilcoxon_stat": float(wilcoxon.statistic),
                "p_wilcoxon_greater": float(wilcoxon.pvalue),
                "p_sign_greater": float(sign.pvalue),
            }
        )
        current_scans = scans.loc[scans["variant"].eq(variant)].copy()
        mixed_rows.append(
            mixed_effect_intercept(
                current_scans,
                value_col="mean_delta_cv_r",
                label=variant,
            )
        )
    return pd.DataFrame(rows), pd.DataFrame(mixed_rows)


def write_audit(
    args: argparse.Namespace,
    specification: dict,
    spec_checksum: str,
    embeddings: dict[str, np.ndarray],
    scans: pd.DataFrame,
    stories: pd.DataFrame,
) -> None:
    expected_scans = specification["dataset"]["expected_motion_filtered_scans"]
    primary = scans.loc[scans["variant"].eq("primary_rich_loso_gap10")]
    audit = {
        "specification_sha256": spec_checksum,
        "script_sha256": file_sha256(Path(__file__)),
        "stories": STORIES,
        "n_stories": int(primary["story"].nunique()),
        "n_scans": int(len(primary)),
        "n_unique_participants": int(primary["subject"].nunique()),
        "expected_unique_participants": specification["dataset"][
            "expected_unique_participants"
        ],
        "expected_scans": expected_scans,
        "scan_count_matches_specification": bool(len(primary) == expected_scans),
        "participant_count_matches_specification": bool(
            primary["subject"].nunique()
            == specification["dataset"]["expected_unique_participants"]
        ),
        "all_embeddings_finite": bool(
            all(np.all(np.isfinite(values)) for values in embeddings.values())
        ),
        "all_scan_effects_finite": bool(np.all(np.isfinite(scans["mean_delta_cv_r"]))),
        "all_story_effects_finite": bool(np.all(np.isfinite(stories["mean_delta_cv_r"]))),
        "all_variants_have_13_stories": bool(
            stories.groupby("variant")["story"].nunique().eq(13).all()
        ),
        "primary_test_story_excluded_from_pca": True,
        "outcome_based_exclusions": 0,
    }
    if not all(
        [
            audit["scan_count_matches_specification"],
            audit["participant_count_matches_specification"],
            audit["all_embeddings_finite"],
            audit["all_scan_effects_finite"],
            audit["all_story_effects_finite"],
            audit["all_variants_have_13_stories"],
        ]
    ):
        raise RuntimeError(f"Analysis audit failed: {audit}")
    with (ROOT / args.output_dir / "analysis_audit.json").open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/jnl_13_story_individual_local")
    parser.add_argument(
        "--specification",
        default="results/jnl_13_story_individual_local/locked_analysis_specification.json",
    )
    parser.add_argument(
        "--subject-list",
        default="configs/stories_all_fmriprep_confounds_motionfiltered.py",
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
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--embedding-max-length", type=int, default=128)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument(
        "--stage",
        choices=["embeddings", "analysis", "all"],
        default="all",
    )
    args = parser.parse_args()

    if args.n_components != 5 or args.n_splits != 5 or args.alpha != 10.0:
        raise ValueError("The locked primary settings require five PCs, five splits, and alpha=10.")

    os.environ["SEMANTIC_FMRI_TIMING"] = "gentle"
    os.environ["SEMANTIC_FMRI_GENTLE_DIR"] = str(Path(args.gentle_root))
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    specification_path = ROOT / args.specification
    specification = validate_specification(specification_path)
    spec_checksum = file_sha256(specification_path)
    (output_dir / "locked_analysis_specification.sha256").write_text(
        spec_checksum + "  " + specification_path.name + "\n",
        encoding="ascii",
    )
    print(f"Locked specification SHA-256: {spec_checksum}", flush=True)

    embeddings = local_embedding_cache(args)
    if args.stage == "embeddings":
        return

    rich_baselines, targets, roi_names = load_data(args)
    lexical = lexical_baselines(args)
    score_spaces, pca_metadata = feature_scores(embeddings, args.n_components)
    pca_metadata.to_csv(output_dir / "pca_metadata.csv", index=False)
    scans, roi_results = scan_scores(
        args,
        score_spaces,
        rich_baselines,
        lexical,
        targets,
        roi_names,
    )
    stories = story_summary(scans)
    summary, mixed = inferential_summary(stories, scans)

    scans.to_csv(output_dir / "scan_level.csv", index=False)
    roi_results.to_csv(output_dir / "roi_level.csv", index=False)
    stories.to_csv(output_dir / "story_level.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    mixed.to_csv(output_dir / "crossed_mixed_effects.csv", index=False)
    write_audit(args, specification, spec_checksum, embeddings, scans, stories)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
