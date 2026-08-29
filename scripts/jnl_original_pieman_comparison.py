from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from argparse import Namespace
from difflib import SequenceMatcher
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import jnl_13_story_loso as loso  # noqa: E402
from jnl_heldout_story_replication import (  # noqa: E402
    load_or_extract_roi_matrix,
)
from jnl_revision_context_layer_models import (  # noqa: E402
    cumulative_current_token_embeddings,
)
from jnl_rich_audio_controls import (  # noqa: E402
    RICH_AUDIO_COLUMNS,
    compute_rich_audio_features,
)
from jnl_shared_response_context_validation import (  # noqa: E402
    correlation_by_column,
    load_story_targets,
)
from motion_qc_from_confounds import summarize_confounds  # noqa: E402
from semantic_fmri.utils import (  # noqa: E402
    ACOUSTIC_CONTROL_COLUMNS,
    compute_acoustic_control_baselines,
    compute_text_baselines,
    fetch_harvard_oxford_atlas,
    hrf_convolve,
    load_gpt2,
    transcript_to_tr_chunks_uniform,
)


STORY = "pieman"
REFERENCE_FOLD = "piemanpni"
REMOTE_ROOT = "https://s3.amazonaws.com/fcp-indi/data/Projects/narratives"
DEFAULT_OUTPUT = ROOT / "results" / "jnl_story_heterogeneity" / "original_pieman"
MAX_MEAN_FD = 0.30
MAX_PCT_FD_GT_0_5 = 0.20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_sort_key(path: Path) -> tuple[int, str]:
    if "_run-1_" in path.name:
        return 0, path.name
    if "_run-2_" in path.name:
        return 1, path.name
    return 0, path.name


def scan_stem(path: Path) -> str:
    return path.name.split("_space-")[0]


def subject_from_name(path: Path) -> str:
    match = re.search(r"(sub-\d+)", path.name)
    if match is None:
        raise ValueError(f"Cannot parse subject from {path}")
    return match.group(1)


def inventory(args: argparse.Namespace) -> pd.DataFrame:
    dataset = Path(args.dataset_dir)
    fmriprep = dataset / "derivatives" / "fmriprep"
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    exclusion_path = output / "scan_exclude.json"
    if not exclusion_path.exists():
        raise FileNotFoundError(
            f"Missing {exclusion_path}; retrieve the published Narratives exclusion file first."
        )
    exclusions = json.loads(exclusion_path.read_text(encoding="utf-8")).get(STORY, {})

    candidates: dict[str, list[Path]] = {}
    pattern = (
        "sub-*/func/*_task-pieman*_space-MNI152NLin2009cAsym_"
        "res-native_desc-preproc_bold.nii.gz"
    )
    for path in fmriprep.glob(pattern):
        if "piemanpni" in path.name:
            continue
        subject = subject_from_name(path)
        if any(value in path.name for value in exclusions.get(subject, [])):
            continue
        candidates.setdefault(subject, []).append(path)

    rows = []
    download_lines = ["create-dirs"]
    for subject in sorted(candidates):
        source_bold = sorted(candidates[subject], key=run_sort_key)[0]
        stem = scan_stem(source_bold)
        source_func = source_bold.parent
        confound_candidates = [
            source_func / f"{stem}_desc-confounds_timeseries.tsv",
            source_func / f"{stem}_desc-confounds_regressors.tsv",
        ]
        source_confounds = next((path for path in confound_candidates if path.exists()), None)
        if source_confounds is None:
            raise FileNotFoundError(f"No confounds pointer for {stem}")
        event_path = dataset / subject / "func" / f"{stem}_events.tsv"
        if not event_path.exists():
            raise FileNotFoundError(event_path)

        local_func = output / "fmriprep" / subject / "func"
        local_func.mkdir(parents=True, exist_ok=True)
        local_bold = local_func / source_bold.name
        local_confounds = local_func / source_confounds.name
        relative_root = f"derivatives/fmriprep/{subject}/func"
        bold_url = f"{REMOTE_ROOT}/{relative_root}/{source_bold.name}"
        confounds_url = f"{REMOTE_ROOT}/{relative_root}/{source_confounds.name}"

        rows.append(
            {
                "subject": subject,
                "run": "run-1" if "_run-1_" in stem else "run-2" if "_run-2_" in stem else "single",
                "events_file": str(event_path),
                "bold_file": str(local_bold),
                "confounds_file": str(local_confounds),
                "bold_url": bold_url,
                "confounds_url": confounds_url,
            }
        )
        for url, target in [(bold_url, local_bold), (confounds_url, local_confounds)]:
            if not target.exists() or target.stat().st_size < 1000:
                download_lines.extend([f'url = "{url}"', f'output = "{target.as_posix()}"'])

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "selected_scan_manifest.csv", index=False)
    (output / "downloads.curl").write_text("\n".join(download_lines) + "\n", encoding="utf-8")
    return frame


def motion_qc(args: argparse.Namespace) -> tuple[pd.DataFrame, list[dict]]:
    output = Path(args.output_dir)
    manifest = pd.read_csv(output / "selected_scan_manifest.csv")
    rows = []
    selected = []
    for record in manifest.to_dict(orient="records"):
        confounds = Path(record["confounds_file"])
        bold = Path(record["bold_file"])
        if not confounds.exists() or confounds.stat().st_size < 1000:
            raise FileNotFoundError(f"Confounds not downloaded: {confounds}")
        if not bold.exists() or bold.stat().st_size < 1000:
            raise FileNotFoundError(f"BOLD not downloaded: {bold}")
        qc = summarize_confounds(str(confounds))
        keep = bool(
            np.isfinite(qc["mean_fd"])
            and np.isfinite(qc["pct_fd_gt_0_5"])
            and qc["mean_fd"] <= MAX_MEAN_FD
            and qc["pct_fd_gt_0_5"] <= MAX_PCT_FD_GT_0_5
        )
        rows.append({"story": STORY, "keep_motion_qc": keep, **record, **qc})
        if keep:
            selected.append(
                {
                    "name": STORY,
                    "subject": record["subject"],
                    "bold_file": record["bold_file"],
                    "events_file": record["events_file"],
                    "transcript_file": str(ROOT / "data" / "transcripts" / "pieman_transcript.txt"),
                    "align_file": str(output / "gentle" / STORY / "align.json"),
                    "audio_file": str(output / "gentle" / STORY / "a.wav"),
                    "bold_source": "fmriprep_mni",
                    "confounds_file": record["confounds_file"],
                    "confounds_source": "fmriprep",
                }
            )
    qc_frame = pd.DataFrame(rows)
    qc_frame.to_csv(output / "motion_qc.csv", index=False)
    (output / "selected_rows.json").write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )
    if len(selected) < 20:
        raise RuntimeError(f"Only {len(selected)} original Pie Man scans survived motion QC")
    return qc_frame, selected


def central_args() -> Namespace:
    return Namespace(
        regressor_dir=loso.DEFAULT_REGRESSOR_DIR,
        roi_cache=loso.DEFAULT_ROI_CACHE,
        alpha=10.0,
    )


def load_central_embeddings() -> dict[str, np.ndarray]:
    directory = ROOT / "results" / "jnl_13_story_extension" / "loso" / "embeddings"
    return {
        story: np.load(directory / f"{story}.npy", allow_pickle=False)
        for story in loso.STORIES
    }


def create_regressors(args: argparse.Namespace, selected: list[dict]) -> Path:
    output = Path(args.output_dir)
    regressor_dir = output / "regressors"
    regressor_dir.mkdir(parents=True, exist_ok=True)
    template = selected[0]
    os.environ["SEMANTIC_FMRI_TIMING"] = "gentle"
    os.environ["SEMANTIC_FMRI_GENTLE_DIR"] = str(output / "gentle")
    chunks, onset, duration, _ = transcript_to_tr_chunks_uniform(template, tr=1.5)

    embedding_path = output / "pieman_embedding.npy"
    if embedding_path.exists() and not args.recompute_embeddings:
        original_embedding = np.load(embedding_path, allow_pickle=False)
    else:
        tokenizer, model, device = load_gpt2()
        model.eval()
        original_embedding = cumulative_current_token_embeddings(
            chunks,
            tokenizer,
            model,
            layer=12,
            max_length=1024,
            output_stride=256,
            device=device,
        )
        np.save(embedding_path, original_embedding.astype(np.float32))

    central = load_central_embeddings()
    training = [story for story in loso.STORIES if story != REFERENCE_FOLD]
    mean, scale, components, eigenvalues = loso.story_balanced_pca(
        central, training, n_components=10
    )
    scores = loso.project(original_embedding, mean, scale, components)

    baseline = compute_text_baselines(chunks)
    acoustic = compute_acoustic_control_baselines(template, tr=1.5)
    baseline = baseline.merge(acoustic, on="time_TR", how="left")
    for column in acoustic.columns:
        if column != "time_TR":
            baseline[column] = baseline[column].fillna(0.0)
    baseline["story"] = STORY
    baseline["TR"] = 1.5
    baseline["onset_sec"] = onset
    baseline["duration_sec"] = duration
    convolved = ["word_count", "mean_char_len", "total_chars"] + ACOUSTIC_CONTROL_COLUMNS
    for column in convolved:
        baseline[f"{column}_hrf"] = hrf_convolve(
            baseline[column].to_numpy(dtype=float), tr=1.5
        )
    for index in range(10):
        column = f"PC{index + 1}"
        baseline[column] = scores[:, index]
        baseline[f"{column}_hrf"] = hrf_convolve(scores[:, index], tr=1.5)

    rich = compute_rich_audio_features(template, output / "gentle", tr=1.5)
    for column in RICH_AUDIO_COLUMNS:
        rich[f"{column}_hrf"] = hrf_convolve(
            rich[column].to_numpy(dtype=float), tr=1.5
        )
    columns = ["time_TR"] + RICH_AUDIO_COLUMNS + [
        f"{column}_hrf" for column in RICH_AUDIO_COLUMNS
    ]
    result = baseline.merge(rich[columns], on="time_TR", how="left")
    for column in columns[1:]:
        result[column] = result[column].fillna(0.0)
    path = regressor_dir / "pieman_shared_pca_regressors.csv"
    result.to_csv(path, index=False)
    pd.DataFrame(
        {
            "component": np.arange(1, 11),
            "eigenvalue": eigenvalues[:10],
            "explained_variance_ratio": eigenvalues[:10] / eigenvalues.sum(),
        }
    ).to_csv(output / "reference_fold_pca_metadata.csv", index=False)
    return path


def extract_rois(args: argparse.Namespace, selected: list[dict]) -> Path:
    output = Path(args.output_dir)
    cache = output / "roi_cache"
    cache.mkdir(parents=True, exist_ok=True)
    atlas_img, atlas_labels = fetch_harvard_oxford_atlas()

    original_events = pd.read_csv(selected[0]["events_file"], sep="\t")
    millisecond_events = original_events.copy()
    millisecond_events["onset"] = millisecond_events["onset"] * 1000.0
    millisecond_events["duration"] = millisecond_events["duration"] * 1000.0
    millisecond_event_path = output / "events_for_millisecond_header.tsv"
    millisecond_events.to_csv(millisecond_event_path, sep="\t", index=False)

    header_rows = []
    for index, row in enumerate(selected, start=1):
        print(f"Original Pie Man ROI extraction [{index}/{len(selected)}] {row['subject']}", flush=True)
        raw_tr = float(nib.load(row["bold_file"]).header.get_zooms()[3])
        corrected = raw_tr > 100.0
        extraction_row = dict(row)
        if corrected:
            extraction_row["events_file"] = str(millisecond_event_path)
            existing_cache = cache / f"{STORY}_{row['subject']}.npz"
            if existing_cache.exists():
                existing_cache.unlink()
        names, matrix, returned_tr, returned_onset, returned_duration = load_or_extract_roi_matrix(
            extraction_row,
            atlas_img,
            atlas_labels,
            cache,
            atlas_threshold=0.0,
        )
        if corrected:
            cache_path = cache / f"{STORY}_{row['subject']}.npz"
            np.savez_compressed(
                cache_path,
                roi_names=np.asarray(names),
                y_matrix=matrix,
                bold_tr=1.5,
                onset=float(original_events.loc[original_events["trial_type"] == "story", "onset"].iloc[0]),
                duration=float(
                    original_events.loc[original_events["trial_type"] == "story", "duration"].iloc[0]
                ),
            )
        header_rows.append(
            {
                "subject": row["subject"],
                "nifti_header_tr": raw_tr,
                "bids_tr": 1.5,
                "unit_normalized": corrected,
                "extracted_timepoints": len(matrix),
                "extraction_returned_tr": returned_tr,
                "extraction_returned_onset": returned_onset,
                "extraction_returned_duration": returned_duration,
            }
        )
    pd.DataFrame(header_rows).to_csv(output / "header_tr_audit.csv", index=False)
    return cache


def fit_scores(
    train_baseline: list[np.ndarray],
    train_combined: list[np.ndarray],
    train_targets: list[np.ndarray],
    test_baseline: np.ndarray,
    test_combined: np.ndarray,
    test_target: np.ndarray,
    alpha: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline_r = loso.transfer_model(
        train_baseline, train_targets, test_baseline, test_target, alpha
    )
    combined_r = loso.transfer_model(
        train_combined, train_targets, test_combined, test_target, alpha
    )
    return baseline_r, combined_r, combined_r - baseline_r


def with_component_prefix(
    baseline: np.ndarray, scores: np.ndarray, n_components: int
) -> np.ndarray:
    pcs_hrf = np.column_stack(
        [hrf_convolve(scores[:, index], tr=1.5) for index in range(n_components)]
    )
    n = min(len(baseline), len(pcs_hrf))
    return np.column_stack([baseline[:n], pcs_hrf[:n]])


def transcript_delivery_comparison(
    args: argparse.Namespace,
    output: Path,
    regressor_frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, float]]:
    alignments = {
        STORY: output / "gentle" / STORY / "align.json",
        REFERENCE_FOLD: (
            Path(args.dataset_dir)
            / "stimuli"
            / "gentle"
            / REFERENCE_FOLD
            / "align.json"
        ),
    }
    fillers = {"uh", "um", "umm", "er", "erm", "ah", "oh", "hmm", "hm", "huh"}
    token_lists: dict[str, list[str]] = {}
    rows = []
    for story, path in alignments.items():
        alignment = json.loads(path.read_text(encoding="utf-8"))
        transcript = alignment["transcript"]
        tokens = [
            token.lower().replace("\u2019", "'")
            for token in re.findall(r"[A-Za-z]+(?:[\u2019'][A-Za-z]+)?", transcript)
        ]
        token_lists[story] = tokens
        aligned = [
            word
            for word in alignment["words"]
            if word.get("case") == "success" and "start" in word and "end" in word
        ]
        gaps = [
            float(second["start"] - first["end"])
            for first, second in zip(aligned, aligned[1:])
        ]
        aligned_span = float(
            max(word["end"] for word in aligned) - min(word["start"] for word in aligned)
        )
        frame = regressor_frames[story]
        nonempty_trs = int((frame["word_count"] > 0).sum())
        filler_count = int(sum(token in fillers for token in tokens))
        rows.append(
            {
                "story": story,
                "transcript_words": len(tokens),
                "unique_words": len(set(tokens)),
                "type_token_ratio": len(set(tokens)) / len(tokens),
                "aligned_success_words": len(aligned),
                "aligned_span_sec": aligned_span,
                "transcript_words_per_min": 60.0 * len(tokens) / aligned_span,
                "filler_count": filler_count,
                "fillers_per_1000_words": 1000.0 * filler_count / len(tokens),
                "question_mark_count": transcript.count("?"),
                "questions_per_1000_words": 1000.0 * transcript.count("?") / len(tokens),
                "immediate_token_repetitions": sum(
                    first == second for first, second in zip(tokens, tokens[1:])
                ),
                "pauses_over_0_5_sec": sum(gap > 0.5 for gap in gaps),
                "pauses_over_1_sec": sum(gap > 1.0 for gap in gaps),
                "model_TRs": len(frame),
                "nonempty_TRs": nonempty_trs,
                "silence_fraction": 1.0 - nonempty_trs / len(frame),
                "aligned_words_per_TR": float(frame["word_count"].sum() / len(frame)),
            }
        )

    first = token_lists[STORY]
    second = token_lists[REFERENCE_FOLD]
    first_set = set(first)
    second_set = set(second)
    first_bigrams = set(zip(first, first[1:]))
    second_bigrams = set(zip(second, second[1:]))
    similarity = {
        "token_sequence_match_ratio": float(
            SequenceMatcher(None, first, second, autojunk=False).ratio()
        ),
        "vocabulary_jaccard": float(
            len(first_set & second_set) / len(first_set | second_set)
        ),
        "bigram_jaccard": float(
            len(first_bigrams & second_bigrams) / len(first_bigrams | second_bigrams)
        ),
    }
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "transcript_delivery_comparison.csv", index=False)
    (output / "transcript_similarity.json").write_text(
        json.dumps(similarity, indent=2) + "\n", encoding="utf-8"
    )
    return frame, similarity


def evaluate(args: argparse.Namespace, selected: list[dict]) -> None:
    output = Path(args.output_dir)
    central_embeddings = load_central_embeddings()
    original_embedding = np.load(output / "pieman_embedding.npy", allow_pickle=False)
    embeddings = {**central_embeddings, STORY: original_embedding}
    baselines, targets, roi_names = loso.load_data(central_args())
    original_targets, original_rois = load_story_targets(output / "roi_cache", STORY)
    if original_rois != roi_names[REFERENCE_FOLD]:
        raise ValueError("Original Pie Man ROI order differs from the central analysis")

    original_frame = pd.read_csv(output / "regressors" / "pieman_shared_pca_regressors.csv")
    original_baseline = loso.baseline_matrix(original_frame)
    training = [story for story in loso.STORIES if story != REFERENCE_FOLD]
    mean, scale, components, _ = loso.story_balanced_pca(
        central_embeddings, training, n_components=10
    )
    projected = {
        story: loso.project(embeddings[story], mean, scale, components)
        for story in training + [REFERENCE_FOLD, STORY]
    }

    component_rows = []
    primary_rows = []
    for n_components in range(1, 11):
        train_y: list[np.ndarray] = []
        train_b: list[np.ndarray] = []
        train_c: list[np.ndarray] = []
        for story in training:
            y = targets[story].mean(axis=0)
            baseline = baselines[story]
            combined = with_component_prefix(baseline, projected[story], n_components)
            n = min(len(y), len(baseline), len(combined))
            train_y.append(y[:n])
            train_b.append(baseline[:n])
            train_c.append(combined[:n])

        for test_story, test_baseline, test_targets in [
            (REFERENCE_FOLD, baselines[REFERENCE_FOLD], targets[REFERENCE_FOLD]),
            (STORY, original_baseline, original_targets),
        ]:
            test_combined = with_component_prefix(
                test_baseline, projected[test_story], n_components
            )
            n = min(test_targets.shape[1], len(test_baseline), len(test_combined))
            baseline_r, combined_r, delta = fit_scores(
                train_b,
                train_c,
                train_y,
                test_baseline[:n],
                test_combined[:n],
                test_targets.mean(axis=0)[:n],
            )
            component_rows.append(
                {
                    "story": test_story,
                    "n_components": n_components,
                    "mean_baseline_r": float(np.mean(baseline_r)),
                    "mean_combined_r": float(np.mean(combined_r)),
                    "mean_delta_r": float(np.mean(delta)),
                    "positive_rois": int(np.sum(delta > 0)),
                }
            )
            if n_components == 5:
                for roi, baseline_value, combined_value, delta_value in zip(
                    original_rois, baseline_r, combined_r, delta
                ):
                    primary_rows.append(
                        {
                            "story": test_story,
                            "roi": roi,
                            "baseline_r": baseline_value,
                            "combined_r": combined_value,
                            "delta_r": delta_value,
                        }
                    )

    component_frame = pd.DataFrame(component_rows)
    component_frame.to_csv(output / "component_count_comparison.csv", index=False)
    primary_frame = pd.DataFrame(primary_rows)
    primary_frame.to_csv(output / "primary_roi_comparison.csv", index=False)

    stored = pd.read_csv(
        ROOT / "results" / "jnl_13_story_extension" / "loso" / "cross_story_transfer_story_means.csv"
    ).set_index("story")["delta_r"]
    reproduced = float(
        component_frame.query("story == @REFERENCE_FOLD and n_components == 5")[
            "mean_delta_r"
        ].iloc[0]
    )
    if abs(reproduced - float(stored[REFERENCE_FOLD])) > 5e-4:
        raise RuntimeError(
            f"Reference piemanpni fold was not reproduced: {reproduced} vs {stored[REFERENCE_FOLD]}"
        )

    subject_names = [row["subject"] for row in selected]
    central_subjects = {story: loso.story_subjects(central_args(), story) for story in training}
    participant_rows = []
    n_components = 5
    for heldout_index, heldout_subject in enumerate(subject_names):
        train_y = []
        train_b = []
        train_c = []
        for story in training:
            keep = np.asarray(
                [subject != heldout_subject for subject in central_subjects[story]], dtype=bool
            )
            y = targets[story][keep].mean(axis=0)
            baseline = baselines[story]
            combined = with_component_prefix(baseline, projected[story], n_components)
            n = min(len(y), len(baseline), len(combined))
            train_y.append(y[:n])
            train_b.append(baseline[:n])
            train_c.append(combined[:n])
        test_combined = with_component_prefix(
            original_baseline, projected[STORY], n_components
        )
        n = min(original_targets.shape[1], len(original_baseline), len(test_combined))
        baseline_r, combined_r, delta = fit_scores(
            train_b,
            train_c,
            train_y,
            original_baseline[:n],
            test_combined[:n],
            original_targets[heldout_index, :n],
        )
        for roi, baseline_value, combined_value, delta_value in zip(
            original_rois, baseline_r, combined_r, delta
        ):
            participant_rows.append(
                {
                    "story": STORY,
                    "subject": heldout_subject,
                    "roi": roi,
                    "baseline_r": baseline_value,
                    "combined_r": combined_value,
                    "delta_r": delta_value,
                }
            )
    participant_frame = pd.DataFrame(participant_rows)
    participant_frame.to_csv(output / "participant_holdout_long.csv", index=False)

    primary_train_y = []
    primary_train_b = []
    primary_train_c = []
    for story in training:
        y = targets[story].mean(axis=0)
        baseline = baselines[story]
        combined = with_component_prefix(baseline, projected[story], 5)
        n = min(len(y), len(baseline), len(combined))
        primary_train_y.append(y[:n])
        primary_train_b.append(baseline[:n])
        primary_train_c.append(combined[:n])
    test_combined = with_component_prefix(original_baseline, projected[STORY], 5)
    test_n = min(original_targets.shape[1], len(original_baseline), len(test_combined))
    baseline_model, baseline_mean, baseline_scale = loso.fit_transfer_model(
        primary_train_b, primary_train_y, 10.0
    )
    combined_model, combined_mean, combined_scale = loso.fit_transfer_model(
        primary_train_c, primary_train_y, 10.0
    )
    baseline_prediction = baseline_model.predict(
        (original_baseline[:test_n] - baseline_mean) / baseline_scale
    )
    combined_prediction = combined_model.predict(
        (test_combined[:test_n] - combined_mean) / combined_scale
    )
    equal_n = int(targets[REFERENCE_FOLD].shape[0])
    rng = np.random.default_rng(20260806)
    equal_n_rows = []
    for repetition in range(1000):
        indices = rng.choice(len(original_targets), size=equal_n, replace=False)
        target = original_targets[indices, :test_n].mean(axis=0)
        target = loso.detrend_and_standardize(target)
        baseline_r = correlation_by_column(target, baseline_prediction)
        combined_r = correlation_by_column(target, combined_prediction)
        delta = combined_r - baseline_r
        equal_n_rows.append(
            {
                "repetition": repetition,
                "n_participants": equal_n,
                "mean_baseline_r": float(np.mean(baseline_r)),
                "mean_combined_r": float(np.mean(combined_r)),
                "mean_delta_r": float(np.mean(delta)),
                "positive_rois": int(np.sum(delta > 0)),
            }
        )
    equal_n_frame = pd.DataFrame(equal_n_rows)
    equal_n_frame.to_csv(output / "equal_n_subsampling.csv", index=False)

    summary_rows = []
    for story in [STORY, REFERENCE_FOLD, "bronx"]:
        if story in [STORY, REFERENCE_FOLD]:
            row = component_frame.query("story == @story and n_components == 5").iloc[0]
            delta = float(row["mean_delta_r"])
            baseline_r = float(row["mean_baseline_r"])
            combined_r = float(row["mean_combined_r"])
        else:
            delta = float(stored[story])
            roi = pd.read_csv(
                ROOT / "results" / "jnl_13_story_extension" / "loso" / "cross_story_transfer_long.csv"
            ).query("story == @story")
            baseline_r = float(roi["rich_audio"].mean())
            combined_r = float(roi["combined"].mean())
        summary_rows.append(
            {
                "story": story,
                "sample": len(selected) if story == STORY else targets[story].shape[0],
                "mean_baseline_r": baseline_r,
                "mean_combined_r": combined_r,
                "mean_delta_r": delta,
            }
        )
    pd.DataFrame(summary_rows).to_csv(output / "story_comparison_summary.csv", index=False)

    audio_rows = []
    source_dirs = {
        STORY: output / "regressors" / "pieman_shared_pca_regressors.csv",
        REFERENCE_FOLD: loso.source_regressor_path(central_args(), REFERENCE_FOLD),
        "bronx": loso.source_regressor_path(central_args(), "bronx"),
    }
    for story, path in source_dirs.items():
        frame = pd.read_csv(path)
        row = {"story": story}
        for column in RICH_AUDIO_COLUMNS:
            row[f"mean_{column}"] = float(frame[column].mean())
        audio_rows.append(row)
    pd.DataFrame(audio_rows).to_csv(output / "audio_comparison.csv", index=False)

    regressor_frames = {
        story: pd.read_csv(loso.source_regressor_path(central_args(), story))
        for story in loso.STORIES
    }
    regressor_frames[STORY] = original_frame
    delivery_frame, transcript_similarity = transcript_delivery_comparison(
        args, output, regressor_frames
    )
    lexical_columns = list(loso.LEXICAL_BASELINE_COLS)
    acoustic_columns = [f"{column}_hrf" for column in ACOUSTIC_CONTROL_COLUMNS]
    timing_columns = [f"{column}_hrf" for column in ACOUSTIC_CONTROL_COLUMNS[:5]]
    rich_columns = [f"{column}_hrf" for column in RICH_AUDIO_COLUMNS]
    baseline_variants = {
        "lexical_time": lexical_columns,
        "speech_timing": lexical_columns + timing_columns,
        "current_rich": lexical_columns + acoustic_columns + rich_columns,
    }

    def normalize_story_acoustics(values: np.ndarray, columns: list[str]) -> np.ndarray:
        normalized = values.copy()
        indices = [
            columns.index(column)
            for column in acoustic_columns + rich_columns
            if column in columns
        ]
        mean_values = normalized[:, indices].mean(axis=0)
        scale_values = normalized[:, indices].std(axis=0)
        scale_values[(scale_values <= 0) | ~np.isfinite(scale_values)] = 1.0
        normalized[:, indices] = (
            normalized[:, indices] - mean_values
        ) / scale_values
        return normalized

    baseline_rows = []
    target_lookup = {REFERENCE_FOLD: targets[REFERENCE_FOLD], STORY: original_targets}
    for variant, columns in baseline_variants.items():
        train_b = []
        train_c = []
        train_y = []
        for story in training:
            baseline = regressor_frames[story][columns].to_numpy(dtype=float)
            combined = with_component_prefix(baseline, projected[story], 5)
            y = targets[story].mean(axis=0)
            n = min(len(baseline), len(combined), len(y))
            train_b.append(baseline[:n])
            train_c.append(combined[:n])
            train_y.append(y[:n])
        for story in [STORY, REFERENCE_FOLD]:
            baseline = regressor_frames[story][columns].to_numpy(dtype=float)
            combined = with_component_prefix(baseline, projected[story], 5)
            test_targets = target_lookup[story]
            n = min(len(baseline), len(combined), test_targets.shape[1])
            baseline_r, combined_r, delta = fit_scores(
                train_b,
                train_c,
                train_y,
                baseline[:n],
                combined[:n],
                test_targets.mean(axis=0)[:n],
            )
            baseline_rows.append(
                {
                    "variant": variant,
                    "story": story,
                    "mean_baseline_r": float(np.mean(baseline_r)),
                    "mean_combined_r": float(np.mean(combined_r)),
                    "mean_delta_r": float(np.mean(delta)),
                    "positive_rois": int(np.sum(delta > 0)),
                }
            )

    columns = baseline_variants["current_rich"]
    train_b = []
    train_c = []
    train_y = []
    for story in training:
        baseline = normalize_story_acoustics(
            regressor_frames[story][columns].to_numpy(dtype=float), columns
        )
        combined = with_component_prefix(baseline, projected[story], 5)
        y = targets[story].mean(axis=0)
        n = min(len(baseline), len(combined), len(y))
        train_b.append(baseline[:n])
        train_c.append(combined[:n])
        train_y.append(y[:n])
    for story in [STORY, REFERENCE_FOLD]:
        baseline = normalize_story_acoustics(
            regressor_frames[story][columns].to_numpy(dtype=float), columns
        )
        combined = with_component_prefix(baseline, projected[story], 5)
        test_targets = target_lookup[story]
        n = min(len(baseline), len(combined), test_targets.shape[1])
        baseline_r, combined_r, delta = fit_scores(
            train_b,
            train_c,
            train_y,
            baseline[:n],
            combined[:n],
            test_targets.mean(axis=0)[:n],
        )
        baseline_rows.append(
            {
                "variant": "rich_story_normalized",
                "story": story,
                "mean_baseline_r": float(np.mean(baseline_r)),
                "mean_combined_r": float(np.mean(combined_r)),
                "mean_delta_r": float(np.mean(delta)),
                "positive_rois": int(np.sum(delta > 0)),
            }
        )
    baseline_frame = pd.DataFrame(baseline_rows)
    baseline_frame.to_csv(output / "baseline_sensitivity.csv", index=False)

    audit = {
        "analysis": "exploratory original Pie Man comparison",
        "reference_training_set": training,
        "reference_fold": REFERENCE_FOLD,
        "run_rule": "first valid run after published scan exclusions",
        "motion_qc": {
            "max_mean_fd": MAX_MEAN_FD,
            "max_pct_fd_gt_0_5": MAX_PCT_FD_GT_0_5,
        },
        "selected_participants": len(selected),
        "reference_fold_stored_delta": float(stored[REFERENCE_FOLD]),
        "reference_fold_reproduced_delta": reproduced,
        "participant_holdout_mean_delta": float(
            participant_frame.groupby("subject")["delta_r"].mean().mean()
        ),
        "equal_n_subsampling": {
            "seed": 20260806,
            "repetitions": len(equal_n_frame),
            "participants_per_repetition": equal_n,
            "mean_delta": float(equal_n_frame["mean_delta_r"].mean()),
            "median_delta": float(equal_n_frame["mean_delta_r"].median()),
            "minimum_delta": float(equal_n_frame["mean_delta_r"].min()),
            "maximum_delta": float(equal_n_frame["mean_delta_r"].max()),
            "fraction_positive": float((equal_n_frame["mean_delta_r"] > 0).mean()),
        },
        "transcript_delivery": delivery_frame.to_dict(orient="records"),
        "transcript_similarity": transcript_similarity,
        "baseline_sensitivity": baseline_frame.to_dict(orient="records"),
        "files": {
            "audio_sha256": sha256(output / "gentle" / STORY / "a.wav"),
            "alignment_sha256": sha256(output / "gentle" / STORY / "align.json"),
            "script_sha256": sha256(Path(__file__)),
        },
    }
    (output / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--stage",
        choices=["inventory", "qc", "regressors", "rois", "evaluate", "all"],
        default="all",
    )
    parser.add_argument("--recompute-embeddings", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.stage in {"inventory", "all"}:
        manifest = inventory(args)
        print(f"Selected {len(manifest)} independent original Pie Man scans", flush=True)
        if args.stage == "inventory":
            return
    qc_frame, selected = motion_qc(args)
    print(
        f"Motion QC retained {len(selected)}/{len(qc_frame)} original Pie Man scans",
        flush=True,
    )
    if args.stage == "qc":
        return
    if args.stage in {"regressors", "all"}:
        path = create_regressors(args, selected)
        print(f"Saved regressors: {path}", flush=True)
        if args.stage == "regressors":
            return
    if args.stage in {"rois", "all"}:
        extract_rois(args, selected)
        if args.stage == "rois":
            return
    evaluate(args, selected)
    print(f"Saved comparison outputs under {output}", flush=True)


if __name__ == "__main__":
    main()
