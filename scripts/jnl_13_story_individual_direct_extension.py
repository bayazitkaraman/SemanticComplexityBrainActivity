from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import jnl_13_story_individual_local as analysis
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VARIANT = "direct_extension_lexical_pooled_gap0"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gentle-root", required=True)
    parser.add_argument(
        "--output-dir",
        default="results/jnl_13_story_individual_local/direct_extension",
    )
    parser.add_argument(
        "--embedding-dir",
        default="results/jnl_13_story_individual_local",
    )
    parser.add_argument(
        "--specification",
        default="results/jnl_13_story_individual_local/direct_extension_specification.json",
    )
    parser.add_argument(
        "--subject-list",
        default="configs/stories_all_fmriprep_confounds_motionfiltered.py",
    )
    parser.add_argument(
        "--regressor-dir",
        default=analysis.DEFAULT_REGRESSOR_DIR,
    )
    parser.add_argument(
        "--roi-cache",
        default=analysis.DEFAULT_ROI_CACHE,
    )
    args = parser.parse_args()
    args.n_components = 5
    args.n_splits = 5
    args.alpha = 10.0
    args.embedding_batch_size = 32
    args.embedding_max_length = 128

    specification_path = ROOT / args.specification
    specification = json.loads(specification_path.read_text(encoding="utf-8"))
    if "cannot replace" not in specification["interpretation"]:
        raise ValueError("The direct-extension interpretation constraint is missing.")
    checksum = sha256(specification_path)
    print(f"Direct-extension specification SHA-256: {checksum}", flush=True)

    os.environ["SEMANTIC_FMRI_TIMING"] = "gentle"
    os.environ["SEMANTIC_FMRI_GENTLE_DIR"] = str(Path(args.gentle_root))
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = {
        story: np.load(
            ROOT / args.embedding_dir / "independent_embeddings" / f"{story}.npy",
            allow_pickle=False,
        )
        for story in analysis.STORIES
    }
    rich_baselines, targets, roi_names = analysis.load_data(args)
    lexical = analysis.lexical_baselines(args)
    score_spaces, metadata = analysis.feature_scores(embeddings, args.n_components)
    analysis.VARIANTS.clear()
    analysis.VARIANTS[VARIANT] = {
        "baseline": "lexical",
        "pca": "pooled_tr",
        "cv_gap": 0,
    }
    scans, roi_results = analysis.scan_scores(
        args,
        score_spaces,
        rich_baselines,
        lexical,
        targets,
        roi_names,
    )
    stories = analysis.story_summary(scans)
    summary, mixed = analysis.inferential_summary(stories, scans)

    scans.to_csv(output_dir / "scan_level.csv", index=False)
    roi_results.to_csv(output_dir / "roi_level.csv", index=False)
    stories.to_csv(output_dir / "story_level.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    mixed.to_csv(output_dir / "crossed_mixed_effects.csv", index=False)
    metadata.loc[metadata["pca"].eq("pooled_tr")].to_csv(
        output_dir / "pca_metadata.csv", index=False
    )
    audit = {
        "specification_sha256": checksum,
        "script_sha256": sha256(Path(__file__)),
        "n_stories": int(scans["story"].nunique()),
        "n_scans": int(len(scans)),
        "n_unique_participants": int(scans["subject"].nunique()),
        "all_scan_effects_finite": bool(
            np.all(np.isfinite(scans["mean_delta_cv_r"]))
        ),
        "outcome_based_exclusions": 0,
        "replaces_locked_primary": False,
    }
    with (output_dir / "analysis_audit.json").open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2)
        stream.write("\n")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
