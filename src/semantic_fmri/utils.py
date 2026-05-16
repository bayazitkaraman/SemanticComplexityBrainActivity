"""
Common utilities for the semantic-complexity / fMRI revision analyses.

This file is used by:
- semantic_baseline_regressors.py
- shared_pca_regressors.py
- gpt2_layer_pc_analysis.py
- roi_encoding_model_comparison.py

It assumes each story dictionary has the same structure as your current project:

{
    "name": "lucy",
    "subject": "sub-026",
    "bold_file": "data/lucy/sub-026_task-lucy_bold.nii.gz",
    "events_file": "data/lucy/task-lucy_events.tsv",
    "transcript_file": "data/lucy/lucy_transcript.txt"
}
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import nibabel as nib

from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from nilearn import datasets
from nilearn.image import resample_to_img, smooth_img
from nilearn.masking import compute_brain_mask
from nilearn.glm.first_level import glover_hrf
from nilearn.signal import clean as nilearn_clean


ROI_LABELS_TO_TEST = [
    "Superior Temporal Gyrus, anterior division",
    "Superior Temporal Gyrus, posterior division",
    "Middle Temporal Gyrus, anterior division",
    "Middle Temporal Gyrus, posterior division",
    "Inferior Frontal Gyrus, pars triangularis",
    "Inferior Frontal Gyrus, pars opercularis",
    "Angular Gyrus",
    "Supramarginal Gyrus, anterior division",
    "Supramarginal Gyrus, posterior division",
    "Frontal Pole",
    "Precuneous Cortex",
    "Temporal Fusiform Cortex, anterior division",
    "Temporal Fusiform Cortex, posterior division",
    "Parahippocampal Gyrus, anterior division",
    "Parahippocampal Gyrus, posterior division",
    "Cingulate Gyrus, anterior division",
    "Cingulate Gyrus, posterior division",
]


def normalize_path(path: str | os.PathLike) -> str:
    """Make Windows-style paths usable on macOS/Linux too."""
    return str(path).replace("\\", os.sep)


def ensure_dir(path: str | os.PathLike) -> None:
    os.makedirs(path, exist_ok=True)


def load_story_list(subject_list: str = "experiment_subject_list") -> List[Dict]:
    """
    Load `stories` from a Python module name or .py file.

    Examples
    --------
    load_story_list("experiment_subject_list")
    load_story_list("experiment_subject_list_all")
    load_story_list("experiment_subject_list_all.py")
    """
    if subject_list.endswith(".py") or os.path.exists(subject_list):
        module_path = Path(subject_list)
        if not module_path.exists():
            module_path = Path(subject_list + ".py")
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load subject list from {subject_list}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(subject_list)

    if not hasattr(module, "stories"):
        raise AttributeError(f"{subject_list} does not contain a variable named `stories`.")
    return list(module.stories)


def get_unique_stories(stories: Sequence[Dict]) -> List[Dict]:
    """Return one representative dictionary per story name."""
    by_name = {}
    for s in stories:
        name = s["name"]
        if name not in by_name:
            by_name[name] = dict(s)
    return list(by_name.values())


def read_story_event(events_file: str) -> Tuple[float, float]:
    """Return onset and duration for the row where trial_type == story."""
    events_file = normalize_path(events_file)
    events_df = pd.read_csv(events_file, sep="\t")
    story_event = events_df[events_df["trial_type"] == "story"]
    if story_event.empty:
        raise ValueError(f"No trial_type == 'story' row found in {events_file}")
    onset = float(story_event.iloc[0]["onset"])
    duration = float(story_event.iloc[0]["duration"])
    return onset, duration


def get_env_timing_strategy() -> str:
    """
    Select transcript timing strategy.

    PowerShell examples:
        $env:SEMANTIC_FMRI_TIMING="gentle"
        $env:SEMANTIC_FMRI_TIMING="uniform"

    Default is "gentle" when alignment files are available.
    """
    return os.environ.get("SEMANTIC_FMRI_TIMING", "gentle").strip().lower()


def get_env_gentle_dir() -> str:
    """Gentle alignment root directory."""
    return os.environ.get(
        "SEMANTIC_FMRI_GENTLE_DIR",
        "<path-to-narratives>/stimuli/gentle",
    )


def story_name_to_gentle_name(story_name: str) -> str:
    """Map BIDS task/story names to Gentle stimulus folder names."""
    mapping = {
        "milkyway": "milkywayoriginal",
    }
    return mapping.get(story_name, story_name)


def infer_gentle_align_json(story: Dict) -> str | None:
    """
    Infer Gentle align.json path.

    Priority:
    1. story["align_file"] if present
    2. SEMANTIC_FMRI_GENTLE_DIR/<story>/align.json
    3. SEMANTIC_FMRI_GENTLE_DIR/<mapped_story>/align.json
    """
    if "align_file" in story and story["align_file"]:
        path = normalize_path(story["align_file"])
        if os.path.exists(path):
            return path

    gentle_dir = Path(normalize_path(get_env_gentle_dir()))
    candidates = [
        gentle_dir / story["name"] / "align.json",
        gentle_dir / story_name_to_gentle_name(story["name"]) / "align.json",
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    return None


def load_gentle_words(align_json: str | os.PathLike) -> List[Dict]:
    """Load successful Gentle-aligned words with start/end times."""
    path = normalize_path(align_json)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = []
    for w in data.get("words", []):
        if w.get("case") != "success":
            continue
        if "start" not in w or "end" not in w:
            continue
        word_text = str(w.get("word", w.get("alignedWord", ""))).strip()
        if not word_text:
            continue
        try:
            start = float(w["start"])
            end = float(w["end"])
        except Exception:
            continue
        if end < start:
            continue
        words.append({
            "word": word_text,
            "alignedWord": str(w.get("alignedWord", word_text)),
            "start": start,
            "end": end,
        })

    if not words:
        raise ValueError(f"No successful words found in Gentle file: {align_json}")

    return words


def choose_gentle_origin(words: Sequence[Dict], duration: float, max_candidates: int = 30) -> float:
    """
    Choose timing origin for a story.

    Gentle alignments sometimes include a stray early word before the true story
    begins. Example: notthefallintact has an early "So" followed by a long gap.
    We choose an early word start that makes:
        last_word_end - origin
    closest to the BIDS story duration.

    This preserves silence after the true story onset while avoiding isolated
    pre-story tokens.
    """
    if not words:
        raise ValueError("No words provided.")

    last_end = float(words[-1]["end"])
    candidate_starts = [float(w["start"]) for w in words[: min(max_candidates, len(words))]]

    # Also include first word by default.
    if float(words[0]["start"]) not in candidate_starts:
        candidate_starts.insert(0, float(words[0]["start"]))

    best_origin = candidate_starts[0]
    best_score = float("inf")

    for origin in candidate_starts:
        span = last_end - origin
        # Penalize origins that produce very short spans.
        if span < 0.75 * duration:
            continue
        score = abs(span - duration)
        if score < best_score:
            best_score = score
            best_origin = origin

    return float(best_origin)


def words_to_tr_chunks_timed(
    words: Sequence[Dict],
    duration: float,
    tr: float = 1.5,
    origin: float | None = None,
) -> Tuple[List[str], int, float]:
    """
    Convert word-level Gentle timings to one text chunk per TR.

    Words are assigned by midpoint:
        midpoint = ((start + end) / 2) - origin

    Words outside [0, duration) are clipped.
    Empty TRs are represented by empty strings. The GPT-2 embedding function
    handles empty strings using the tokenizer's EOS token.
    """
    n_trs = int(duration // tr)
    if n_trs <= 0:
        raise ValueError("Non-positive n_trs.")

    if origin is None:
        origin = choose_gentle_origin(words, duration)

    bins: List[List[str]] = [[] for _ in range(n_trs)]

    for w in words:
        midpoint = ((float(w["start"]) + float(w["end"])) / 2.0) - origin
        if midpoint < 0 or midpoint >= duration:
            continue
        idx = int(midpoint // tr)
        if 0 <= idx < n_trs:
            bins[idx].append(str(w["word"]))

    chunks = [" ".join(tokens) for tokens in bins]
    return chunks, n_trs, float(origin)


def transcript_to_tr_chunks_uniform_only(story: Dict, tr: float = 1.5) -> Tuple[List[str], float, float, int]:
    """
    Original fallback method: split transcript into contiguous word chunks,
    one chunk per TR using uniform word allocation.
    """
    transcript_file = normalize_path(story["transcript_file"])
    with open(transcript_file, "r", encoding="utf-8") as f:
        words = f.read().strip().split()

    onset, duration = read_story_event(story["events_file"])
    n_trs = int(duration // tr)
    if n_trs <= 0:
        raise ValueError(f"Non-positive number of TRs for story {story['name']}.")

    words_per_tr = int(np.floor(len(words) / n_trs))
    if words_per_tr <= 0:
        raise ValueError(f"words_per_tr=0 for story {story['name']}.")

    chunks = [
        " ".join(words[i * words_per_tr : (i + 1) * words_per_tr])
        for i in range(n_trs)
    ]

    leftover = words[n_trs * words_per_tr :]
    if leftover:
        chunks[-1] += " " + " ".join(leftover)

    return chunks, onset, duration, n_trs


def transcript_to_tr_chunks_gentle(story: Dict, tr: float = 1.5) -> Tuple[List[str], float, float, int]:
    """
    Use Gentle word-level timing to create one text chunk per TR.

    Timing origin is chosen automatically using BIDS story duration. This handles
    stories where the first aligned word is a stray pre-story token.

    Returns the same tuple as the old uniform function:
        chunks, onset, duration, n_trs
    """
    onset, duration = read_story_event(story["events_file"])
    align_json = infer_gentle_align_json(story)
    if align_json is None:
        raise FileNotFoundError(
            f"No Gentle align.json found for story {story['name']}. "
            "Set SEMANTIC_FMRI_GENTLE_DIR or add align_file to the story dict."
        )

    words = load_gentle_words(align_json)
    origin = choose_gentle_origin(words, duration)
    chunks, n_trs, origin = words_to_tr_chunks_timed(words, duration, tr=tr, origin=origin)

    # Lightweight diagnostics for the first call per process/story.
    nonempty = sum(1 for c in chunks if c.strip())
    if os.environ.get("SEMANTIC_FMRI_VERBOSE_TIMING", "0") in ["1", "true", "yes"]:
        print(
            f"[Gentle timing] {story['name']}: align={align_json}, "
            f"origin={origin:.2f}, duration={duration:.2f}, "
            f"TRs={n_trs}, nonempty_TRs={nonempty}/{len(chunks)}"
        )

    return chunks, onset, duration, n_trs


def transcript_to_tr_chunks_uniform(story: Dict, tr: float = 1.5) -> Tuple[List[str], float, float, int]:
    """
    Main transcript-to-TR function used by the pipeline.

    Despite the historical name, this now supports two strategies:
    - SEMANTIC_FMRI_TIMING=gentle  (default): use Gentle align.json if available
    - SEMANTIC_FMRI_TIMING=uniform: use the original uniform transcript chunking

    The function name is kept for backward compatibility with existing scripts.
    """
    strategy = get_env_timing_strategy()

    if strategy in ["uniform", "old", "none"]:
        return transcript_to_tr_chunks_uniform_only(story, tr=tr)

    if strategy in ["gentle", "word", "wordlevel", "word-level"]:
        try:
            return transcript_to_tr_chunks_gentle(story, tr=tr)
        except Exception as e:
            if os.environ.get("SEMANTIC_FMRI_ALLOW_UNIFORM_FALLBACK", "0") in ["1", "true", "yes"]:
                print(f"WARNING: Gentle timing failed for {story['name']}: {e}. Falling back to uniform chunking.")
                return transcript_to_tr_chunks_uniform_only(story, tr=tr)
            raise

    raise ValueError(f"Unknown SEMANTIC_FMRI_TIMING strategy: {strategy}")


def compute_text_baselines(word_chunks: Sequence[str]) -> pd.DataFrame:
    """Simple lexical/time regressors at TR resolution."""
    rows = []
    for i, chunk in enumerate(word_chunks):
        tokens = chunk.split()
        if len(tokens) == 0:
            word_count = 0
            mean_char_len = 0.0
            total_chars = 0
        else:
            char_lens = [len(w) for w in tokens]
            word_count = len(tokens)
            mean_char_len = float(np.mean(char_lens))
            total_chars = int(np.sum(char_lens))

        rows.append({
            "time_TR": i,
            "word_count": word_count,
            "mean_char_len": mean_char_len,
            "total_chars": total_chars,
        })

    return pd.DataFrame(rows)


def safe_pearson(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    """Pearson correlation with NaN and zero-variance protection."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = ~np.isnan(x) & ~np.isnan(y)

    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return np.nan, np.nan
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan

    r, p = pearsonr(x, y)
    return float(r), float(p)


def zscore_1d(x: Sequence[float]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if sd == 0 or np.isnan(sd):
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sd


def hrf_convolve(x: Sequence[float], tr: float = 1.5) -> np.ndarray:
    """Z-score then convolve a 1D regressor with the canonical Glover HRF."""
    x = zscore_1d(x)
    hrf = glover_hrf(t_r=tr, oversampling=20, time_length=30, onset=0)
    y = np.convolve(x, hrf)[: len(x)]
    return zscore_1d(y)


def load_gpt2(device: str | None = None):
    """
    Load GPT-2 tokenizer/model with hidden states enabled.

    Use device="cuda" if available and desired.
    """
    from transformers import GPT2Tokenizer, GPT2Model

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)
    model.eval()
    model.to(device)
    return tokenizer, model, device


def compute_gpt2_embeddings_by_layer(
    word_chunks: Sequence[str],
    tokenizer,
    model,
    layers: Sequence[int],
    max_length: int = 128,
    device: str | None = None,
) -> Dict[int, np.ndarray]:
    """
    Compute mean-pooled GPT-2 embeddings for selected hidden-state layers.

    GPT-2 hidden_states length is 13:
    - layer 0: token embedding output
    - layer 1..12: transformer block outputs
    """
    if device is None:
        device = next(model.parameters()).device.type

    layers = list(layers)
    output = {layer: [] for layer in layers}

    with torch.no_grad():
        for chunk in word_chunks:
            text = chunk.strip() if isinstance(chunk, str) else str(chunk).strip()
            if not text:
                text = tokenizer.eos_token or "."
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states

            for layer in layers:
                actual_layer = layer
                if actual_layer < 0:
                    actual_layer = len(hidden_states) + actual_layer
                if actual_layer < 0 or actual_layer >= len(hidden_states):
                    raise ValueError(f"Invalid GPT-2 layer {layer}. Valid range: 0..{len(hidden_states)-1}")

                emb = hidden_states[actual_layer].squeeze(0).mean(dim=0)
                output[layer].append(emb.detach().cpu().numpy())

    return {layer: np.stack(v, axis=0) for layer, v in output.items()}


def fit_pca_scores(X: np.ndarray, n_components: int = 10) -> Tuple[np.ndarray, PCA, StandardScaler]:
    """Fit StandardScaler + PCA and return scores."""
    n_components = min(n_components, X.shape[0], X.shape[1])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)
    return scores, pca, scaler


def shift_with_nan(x: Sequence[float], lag: int) -> np.ndarray:
    """Shift a vector using np.roll but mark wrapped values as NaN."""
    x = np.asarray(x, dtype=float)
    shifted = np.roll(x, lag)

    if lag > 0:
        shifted[:lag] = np.nan
    elif lag < 0:
        shifted[lag:] = np.nan

    return shifted


def best_lag_correlation(
    regressor: Sequence[float],
    y: Sequence[float],
    lag_min: int = 0,
    lag_max: int = 8,
) -> Tuple[float, float, int]:
    """
    Search lags and return the best absolute Pearson correlation.

    After HRF convolution, non-negative lags are often safer:
    lag_min=0, lag_max=8.
    """
    regressor = np.asarray(regressor, dtype=float)
    y = np.asarray(y, dtype=float)

    n = min(len(regressor), len(y))
    regressor = regressor[:n]
    y = y[:n]

    best_r, best_p, best_lag = np.nan, np.nan, np.nan

    for lag in range(lag_min, lag_max + 1):
        shifted = shift_with_nan(regressor, lag)
        valid = ~np.isnan(shifted) & ~np.isnan(y)

        if valid.sum() > 5:
            r, p = safe_pearson(shifted[valid], y[valid])
            if not np.isnan(r) and (np.isnan(best_r) or abs(r) > abs(best_r)):
                best_r, best_p, best_lag = r, p, lag

    return float(best_r), float(best_p), int(best_lag) if not np.isnan(best_lag) else np.nan


def permutation_p_for_best_lag(
    regressor: Sequence[float],
    y: Sequence[float],
    observed_r: float,
    lag_min: int = 0,
    lag_max: int = 8,
    n_perm: int = 1000,
    seed: int = 42,
) -> float:
    """
    Empirical two-sided p-value using the same max-over-lag procedure.
    """
    if n_perm <= 0 or np.isnan(observed_r):
        return np.nan

    rng = np.random.default_rng(seed)
    regressor = np.asarray(regressor, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(regressor), len(y))
    regressor = regressor[:n]
    y = y[:n]

    null_rs = []
    for _ in range(n_perm):
        shuffled = rng.permutation(regressor)
        r_perm, _, _ = best_lag_correlation(shuffled, y, lag_min=lag_min, lag_max=lag_max)
        if not np.isnan(r_perm):
            null_rs.append(r_perm)

    if len(null_rs) == 0:
        return np.nan

    null_rs = np.asarray(null_rs, dtype=float)
    p_emp = (1.0 + np.sum(np.abs(null_rs) >= abs(observed_r))) / (len(null_rs) + 1.0)
    return float(p_emp)


def fetch_harvard_oxford_atlas():
    atlas = datasets.fetch_atlas_harvard_oxford("cort-prob-2mm")
    atlas_img = atlas.maps
    atlas_labels = [str(x) for x in atlas.labels]
    return atlas_img, atlas_labels




# ---------------------------------------------------------------------
# fMRIPrep confound utilities
# ---------------------------------------------------------------------


MOTION6_COLUMNS = [
    "trans_x", "trans_y", "trans_z",
    "rot_x", "rot_y", "rot_z",
]

MOTION_DERIVATIVE_COLUMNS = [
    "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
    "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1",
]

MOTION_POWER_COLUMNS = [
    "trans_x_power2", "trans_y_power2", "trans_z_power2",
    "rot_x_power2", "rot_y_power2", "rot_z_power2",
]

MOTION_DERIVATIVE_POWER_COLUMNS = [
    "trans_x_derivative1_power2", "trans_y_derivative1_power2", "trans_z_derivative1_power2",
    "rot_x_derivative1_power2", "rot_y_derivative1_power2", "rot_z_derivative1_power2",
]

MOTION24_COLUMNS = (
    MOTION6_COLUMNS
    + MOTION_DERIVATIVE_COLUMNS
    + MOTION_POWER_COLUMNS
    + MOTION_DERIVATIVE_POWER_COLUMNS
)

WM_CSF_COLUMNS = [
    "white_matter", "csf",
    "white_matter_derivative1", "csf_derivative1",
    "white_matter_power2", "csf_power2",
    "white_matter_derivative1_power2", "csf_derivative1_power2",
]

QUALITY_COLUMNS = [
    "framewise_displacement",
    "dvars", "std_dvars",
]

GLOBAL_SIGNAL_COLUMNS = [
    "global_signal", "global_signal_derivative1", "global_signal_power2",
    "global_signal_derivative1_power2",
]


def get_env_cleaning_strategy() -> str:
    """
    Select nuisance strategy from environment variable.

    PowerShell example:
        $env:SEMANTIC_FMRI_CONFOUND_STRATEGY="motion24"
        $env:SEMANTIC_FMRI_HIGH_PASS="none"

    Supported strategies:
        none
        motion6
        motion24
        motion24_wmcsf
        acompcor6
        full
    """
    return os.environ.get("SEMANTIC_FMRI_CONFOUND_STRATEGY", "motion24").strip().lower()


def get_env_high_pass(default: float | None = None) -> float | None:
    """
    Read high-pass cutoff from environment.

    PowerShell examples:
        $env:SEMANTIC_FMRI_HIGH_PASS="none"
        $env:SEMANTIC_FMRI_HIGH_PASS="0.008"
    """
    value = os.environ.get("SEMANTIC_FMRI_HIGH_PASS", None)
    if value is None or str(value).strip() == "":
        return default
    value = str(value).strip().lower()
    if value in ["none", "no", "false", "0"]:
        return None
    try:
        return float(value)
    except ValueError:
        return default


def select_confound_columns(df: pd.DataFrame, strategy: str, n_compcor: int = 6, include_global_signal: bool = False) -> list[str]:
    """Select confound columns according to a moderate-to-aggressive strategy ladder."""
    strategy = (strategy or "motion24").strip().lower()

    if strategy in ["none", "no", "off"]:
        return []

    cols: list[str] = []

    if strategy == "motion6":
        cols.extend(MOTION6_COLUMNS)

    elif strategy == "motion24":
        cols.extend(MOTION24_COLUMNS)

    elif strategy == "motion24_wmcsf":
        cols.extend(MOTION24_COLUMNS)
        cols.extend(WM_CSF_COLUMNS)

    elif strategy == "acompcor6":
        cols.extend(MOTION24_COLUMNS)
        acompcor_cols = sorted([c for c in df.columns if c.startswith("a_comp_cor_")])[:n_compcor]
        cosine_cols = sorted([c for c in df.columns if c.startswith("cosine")])
        cols.extend(acompcor_cols)
        cols.extend(cosine_cols)

    elif strategy == "full":
        cols.extend(MOTION24_COLUMNS)
        cols.extend(WM_CSF_COLUMNS)
        cols.extend(QUALITY_COLUMNS)
        acompcor_cols = sorted([c for c in df.columns if c.startswith("a_comp_cor_")])[:n_compcor]
        cosine_cols = sorted([c for c in df.columns if c.startswith("cosine")])
        cols.extend(acompcor_cols)
        cols.extend(cosine_cols)

    else:
        # Safe default
        cols.extend(MOTION24_COLUMNS)

    if include_global_signal:
        cols.extend(GLOBAL_SIGNAL_COLUMNS)

    # Keep only available columns, preserve order, remove duplicates.
    seen = set()
    selected = []
    for col in cols:
        if col in df.columns and col not in seen:
            selected.append(col)
            seen.add(col)

    return selected


def infer_confounds_file_from_bold(bold_file: str | os.PathLike) -> str | None:
    """
    Infer the fMRIPrep confounds TSV path from a preprocessed BOLD filename.

    Handles filenames like:
      sub-049_task-shapessocial_space-MNI..._desc-preproc_bold.nii.gz

    by searching in the same func folder for:
      sub-049_task-shapessocial*_desc-confounds_timeseries.tsv
      sub-049_task-shapessocial*_desc-confounds_regressors.tsv
    """
    bold_path = Path(normalize_path(bold_file))
    func_dir = bold_path.parent
    name = bold_path.name

    # Extract BIDS prefix up to task/run if present.
    # Examples:
    # sub-049_task-shapessocial_space-MNI... -> sub-049_task-shapessocial
    # sub-001_task-pieman_run-1_space-MNI... -> sub-001_task-pieman_run-1
    stem = name
    for suffix in [
        "_space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz",
        "_space-MNI152NLin6Asym_res-native_desc-preproc_bold.nii.gz",
        "_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
        "_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz",
        "_space-T1w_desc-preproc_bold.nii.gz",
        "_desc-preproc_bold.nii.gz",
        "_bold.nii.gz",
    ]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    patterns = [
        f"{stem}_desc-confounds_timeseries.tsv",
        f"{stem}_desc-confounds_regressors.tsv",
        f"{stem}*desc-confounds_timeseries.tsv",
        f"{stem}*desc-confounds_regressors.tsv",
    ]

    for pattern in patterns:
        matches = sorted(func_dir.glob(pattern))
        if matches:
            return str(matches[0])

    return None


def get_confounds_file_for_story(story: Dict) -> str | None:
    """Return the confounds file from the story dictionary or infer it from bold_file."""
    if "confounds_file" in story and story["confounds_file"]:
        path = normalize_path(story["confounds_file"])
        if os.path.exists(path):
            return path

    inferred = infer_confounds_file_from_bold(story["bold_file"])
    if inferred is not None and os.path.exists(inferred):
        return normalize_path(inferred)

    return None


def load_fmriprep_confounds(
    confounds_file: str | os.PathLike | None,
    n_volumes: int,
    n_compcor: int = 6,
    include_global_signal: bool = False,
    strategy: str | None = None,
) -> np.ndarray | None:
    """
    Load fMRIPrep confounds using a strategy ladder.

    Strategies
    ----------
    none:
        No nuisance regressors.
    motion6:
        Six rigid-body motion parameters.
    motion24:
        Motion parameters + derivatives + squares + derivative squares.
    motion24_wmcsf:
        motion24 plus white matter and CSF signals when available.
    acompcor6:
        motion24 plus first six aCompCor components and fMRIPrep cosine regressors.
        This follows fMRIPrep's recommendation that CompCor should be accompanied
        by the corresponding cosine regressors.
    full:
        motion24 + WM/CSF + FD/DVARS + first six aCompCor + cosine regressors.

    The default strategy is read from:
        SEMANTIC_FMRI_CONFOUND_STRATEGY

    Recommended first sensitivity run:
        motion24 with SEMANTIC_FMRI_HIGH_PASS=none
    """
    if confounds_file is None:
        return None

    if strategy is None:
        strategy = get_env_cleaning_strategy()

    if strategy in ["none", "no", "off"]:
        return None

    confounds_file = normalize_path(confounds_file)
    if not os.path.exists(confounds_file):
        return None

    try:
        df = pd.read_csv(confounds_file, sep="\t")
    except Exception:
        return None

    selected_cols = select_confound_columns(
        df,
        strategy=strategy,
        n_compcor=n_compcor,
        include_global_signal=include_global_signal,
    )

    if not selected_cols:
        return None

    confounds = df[selected_cols].copy()

    # Replace inf and fill NaNs. FD/DVARS often have NaN in first row.
    confounds = confounds.replace([np.inf, -np.inf], np.nan)
    confounds = confounds.bfill().ffill().fillna(0.0)

    X = confounds.to_numpy(dtype=float)

    # Match run length.
    if X.shape[0] < n_volumes:
        pad = np.zeros((n_volumes - X.shape[0], X.shape[1]), dtype=float)
        X = np.vstack([X, pad])
    elif X.shape[0] > n_volumes:
        X = X[:n_volumes, :]

    # Drop constant columns.
    keep = np.nanstd(X, axis=0) > 1e-8
    X = X[:, keep]

    if X.shape[1] == 0:
        return None

    return X


def clean_roi_matrix(
    roi_matrix: np.ndarray,
    tr: float,
    confounds: np.ndarray | None = None,
    detrend: bool = True,
    standardize: bool | str = "zscore_sample",
    high_pass: float | None = 0.008,
) -> np.ndarray:
    """
    Clean ROI time series with nilearn.signal.clean.

    Parameters
    ----------
    roi_matrix : ndarray, shape (n_volumes, n_rois)
    tr : fMRI TR
    confounds : ndarray or None
        fMRIPrep nuisance regressors.
    high_pass : float or None
        0.008 Hz corresponds roughly to a 128-second high-pass filter.
    """
    if roi_matrix.size == 0:
        return roi_matrix

    try:
        return nilearn_clean(
            roi_matrix,
            confounds=confounds,
            t_r=tr,
            detrend=detrend,
            standardize=standardize,
            high_pass=high_pass,
            low_pass=None,
            ensure_finite=True,
        )
    except TypeError:
        # Older nilearn versions may not support string standardize modes.
        return nilearn_clean(
            roi_matrix,
            confounds=confounds,
            t_r=tr,
            detrend=detrend,
            standardize=True,
            high_pass=high_pass,
            low_pass=None,
            ensure_finite=True,
        )


def extract_roi_timeseries(
    story: Dict,
    roi_labels: Sequence[str] = ROI_LABELS_TO_TEST,
    atlas_img=None,
    atlas_labels: Sequence[str] | None = None,
    fwhm: float = 6.0,
    atlas_threshold: float = 0.0,
    apply_confounds: bool = True,
    high_pass: float | None = None,
    n_compcor: int = 6,
    include_global_signal: bool = False,
    confound_strategy: str | None = None,
) -> Tuple[Dict[str, np.ndarray], float, float, float, int]:
    """
    Extract cleaned mean BOLD activity per ROI for the story segment.

    Important update:
    If `story["confounds_file"]` exists, fMRIPrep confounds are regressed from
    the full run ROI time series before the story segment is extracted.

    Returns
    -------
    roi_ts : dict
        roi label -> vector with one value per TR during the story.
    tr : float
        fMRI TR read from NIfTI header.
    onset : float
        Story onset in seconds.
    duration : float
        Story duration in seconds.
    n_trs : int
        Number of TRs in the story segment.
    """
    if atlas_img is None or atlas_labels is None:
        atlas_img, atlas_labels = fetch_harvard_oxford_atlas()

    bold_file = normalize_path(story["bold_file"])
    fmri_img = smooth_img(nib.load(bold_file), fwhm=fwhm)
    fmri_data = fmri_img.get_fdata()
    tr = float(fmri_img.header.get_zooms()[3])
    n_volumes = fmri_data.shape[3]

    onset, duration = read_story_event(story["events_file"])
    n_trs = int(duration // tr)
    bold_start = int(onset // tr)

    resampled_atlas = resample_to_img(
        atlas_img,
        fmri_img,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    atlas_data = resampled_atlas.get_fdata()

    # Build ROI masks first.
    selected_roi_labels = []
    roi_masks = []
    for roi_label in roi_labels:
        if roi_label not in atlas_labels:
            continue
        roi_index = atlas_labels.index(roi_label)
        roi_mask = atlas_data[..., roi_index] > atlas_threshold
        if np.any(roi_mask):
            selected_roi_labels.append(roi_label)
            roi_masks.append(roi_mask)

    if not selected_roi_labels:
        return {}, tr, onset, duration, n_trs

    # Extract full-run ROI matrix: shape (n_volumes, n_rois).
    roi_matrix = np.zeros((n_volumes, len(selected_roi_labels)), dtype=float)

    for j, roi_mask in enumerate(roi_masks):
        for t in range(n_volumes):
            vol = fmri_data[:, :, :, t]
            roi_matrix[t, j] = np.nanmean(vol[roi_mask])

    # Clean full run before extracting story segment.
    # Defaults are intentionally moderate: motion24 and no high-pass.
    # You can change behavior with environment variables:
    #   SEMANTIC_FMRI_CONFOUND_STRATEGY=motion6|motion24|motion24_wmcsf|acompcor6|full|none
    #   SEMANTIC_FMRI_HIGH_PASS=none|0.008
    confounds = None
    effective_strategy = confound_strategy or get_env_cleaning_strategy()
    effective_high_pass = get_env_high_pass(default=high_pass)

    if apply_confounds and effective_strategy not in ["none", "no", "off"]:
        confounds_file = get_confounds_file_for_story(story)
        confounds = load_fmriprep_confounds(
            confounds_file,
            n_volumes=n_volumes,
            n_compcor=n_compcor,
            include_global_signal=include_global_signal,
            strategy=effective_strategy,
        )

    roi_matrix = clean_roi_matrix(
        roi_matrix,
        tr=tr,
        confounds=confounds,
        detrend=True,
        standardize="zscore_sample",
        high_pass=effective_high_pass,
    )

    # Extract story segment.
    end = min(bold_start + n_trs, roi_matrix.shape[0])
    segment = roi_matrix[bold_start:end, :]

    roi_ts = {
        roi_label: segment[:, j].astype(float)
        for j, roi_label in enumerate(selected_roi_labels)
    }

    return roi_ts, tr, onset, duration, n_trs


