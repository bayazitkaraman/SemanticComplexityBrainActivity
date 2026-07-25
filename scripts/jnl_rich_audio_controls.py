from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import wavfile
from scipy.signal import detrend
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from jnl_pc_count_sensitivity import LEXICAL_BASELINE_COLS, safe_name, vectorized_cv_r  # noqa: E402
from roi_encoding_model_comparison import load_regressors  # noqa: E402
from semantic_fmri.utils import (  # noqa: E402
    ACOUSTIC_CONTROL_COLUMNS,
    choose_gentle_origin,
    get_unique_stories,
    hrf_convolve,
    load_gentle_words,
    load_story_list,
    read_story_event,
    story_name_to_gentle_name,
)


DEFAULT_ROI_CACHE = ROOT / "results" / "journal_neurolinguistics_robustness" / "pc_count_full" / "roi_time_series_cache"
DEFAULT_PRIMARY_DELTAS = ROOT / "results" / "model_comparison_gentle_motionQC_only" / "encoding_model_deltas.csv"


RICH_AUDIO_COLUMNS = [
    "rich_audio_rms",
    "rich_audio_abs_mean",
    "rich_audio_zero_crossing_rate",
    "rich_audio_spectral_centroid_hz",
    "rich_audio_spectral_bandwidth_hz",
    "rich_audio_spectral_rolloff85_hz",
    "rich_audio_low_band_power",
    "rich_audio_mid_band_power",
    "rich_audio_high_band_power",
    "rich_audio_spectral_flux",
    "rich_audio_f0_median_hz",
    "rich_audio_voiced_fraction",
]


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        denom = max(abs(info.min), abs(info.max))
        x = x.astype(np.float32) / float(denom)
    else:
        x = x.astype(np.float32)
        max_abs = float(np.nanmax(np.abs(x))) if x.size else 0.0
        if max_abs > 1.5:
            x = x / max_abs
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def gentle_audio_path(story_name: str, gentle_root: Path) -> Path:
    mapped = story_name_to_gentle_name(story_name)
    candidates = [
        gentle_root / mapped / "a.wav",
        gentle_root / story_name / "a.wav",
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 1000:
            return path
    raise FileNotFoundError(f"No real Gentle audio found for {story_name} under {gentle_root}")


def gentle_align_path(story_name: str, gentle_root: Path) -> Path:
    mapped = story_name_to_gentle_name(story_name)
    candidates = [
        gentle_root / mapped / "align.json",
        gentle_root / story_name / "align.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No Gentle align.json found for {story_name} under {gentle_root}")


def safe_band_power(freqs: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    idx = (freqs >= low) & (freqs < high)
    if not np.any(idx):
        return 0.0
    return float(np.sum(power[idx]) / max(np.sum(power), 1e-12))


def rough_pitch_features(seg: np.ndarray, sample_rate: int) -> tuple[float, float]:
    if seg.size < int(0.08 * sample_rate):
        return 0.0, 0.0

    frame_len = max(32, int(round(0.04 * sample_rate)))
    hop = max(16, int(round(0.02 * sample_rate)))
    min_lag = max(1, int(sample_rate / 400.0))
    max_lag = max(min_lag + 1, int(sample_rate / 50.0))
    f0s = []
    n_frames = 0

    for start in range(0, max(1, seg.size - frame_len + 1), hop):
        frame = seg[start : start + frame_len].astype(float)
        if frame.size < frame_len:
            continue
        frame = frame - np.mean(frame)
        energy = float(np.mean(frame**2))
        if energy < 1e-6:
            continue
        n_frames += 1
        ac = np.correlate(frame, frame, mode="full")[frame_len - 1 :]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        hi = min(max_lag, ac.size - 1)
        if hi <= min_lag:
            continue
        search = ac[min_lag:hi]
        lag = int(np.argmax(search) + min_lag)
        peak = float(ac[lag])
        if peak >= 0.25:
            f0s.append(float(sample_rate / lag))

    if not f0s:
        return 0.0, 0.0
    return float(np.median(f0s)), float(len(f0s) / max(n_frames, 1))


def spectral_features(seg: np.ndarray, sample_rate: int, prev_mag: np.ndarray | None) -> tuple[dict, np.ndarray]:
    if seg.size == 0:
        empty = np.zeros(1, dtype=float)
        return {
            "rich_audio_rms": 0.0,
            "rich_audio_abs_mean": 0.0,
            "rich_audio_zero_crossing_rate": 0.0,
            "rich_audio_spectral_centroid_hz": 0.0,
            "rich_audio_spectral_bandwidth_hz": 0.0,
            "rich_audio_spectral_rolloff85_hz": 0.0,
            "rich_audio_low_band_power": 0.0,
            "rich_audio_mid_band_power": 0.0,
            "rich_audio_high_band_power": 0.0,
            "rich_audio_spectral_flux": 0.0,
            "rich_audio_f0_median_hz": 0.0,
            "rich_audio_voiced_fraction": 0.0,
        }, empty

    y = np.asarray(seg, dtype=float)
    y = y - np.mean(y)
    rms = float(np.sqrt(np.mean(y**2)))
    abs_mean = float(np.mean(np.abs(y)))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(y)).astype(float)))) if y.size > 1 else 0.0

    window = np.hanning(y.size)
    spec = np.fft.rfft(y * window)
    mag = np.abs(spec)
    power = mag**2
    freqs = np.fft.rfftfreq(y.size, d=1.0 / sample_rate)
    total_power = float(np.sum(power))

    if total_power <= 1e-12:
        centroid = bandwidth = rolloff = 0.0
    else:
        centroid = float(np.sum(freqs * power) / total_power)
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total_power))
        cumulative = np.cumsum(power)
        roll_idx = int(np.searchsorted(cumulative, 0.85 * total_power))
        rolloff = float(freqs[min(roll_idx, len(freqs) - 1)])

    norm_mag = mag / max(float(np.linalg.norm(mag)), 1e-12)
    if prev_mag is None or prev_mag.shape != norm_mag.shape:
        flux = 0.0
    else:
        flux = float(np.sqrt(np.mean((norm_mag - prev_mag) ** 2)))

    f0, voiced_fraction = rough_pitch_features(y, sample_rate)
    return {
        "rich_audio_rms": rms,
        "rich_audio_abs_mean": abs_mean,
        "rich_audio_zero_crossing_rate": zcr,
        "rich_audio_spectral_centroid_hz": centroid,
        "rich_audio_spectral_bandwidth_hz": bandwidth,
        "rich_audio_spectral_rolloff85_hz": rolloff,
        "rich_audio_low_band_power": safe_band_power(freqs, power, 80.0, 300.0),
        "rich_audio_mid_band_power": safe_band_power(freqs, power, 300.0, 3000.0),
        "rich_audio_high_band_power": safe_band_power(freqs, power, 3000.0, 8000.0),
        "rich_audio_spectral_flux": flux,
        "rich_audio_f0_median_hz": f0,
        "rich_audio_voiced_fraction": voiced_fraction,
    }, norm_mag


def compute_rich_audio_features(story: dict, gentle_root: Path, tr: float) -> pd.DataFrame:
    story_name = story["name"]
    onset, duration = read_story_event(story["events_file"])
    n_trs = int(duration // tr)
    audio_path = gentle_audio_path(story_name, gentle_root)
    align_path = gentle_align_path(story_name, gentle_root)
    words = load_gentle_words(align_path)
    origin = choose_gentle_origin(words, duration)

    sample_rate, audio = wavfile.read(audio_path)
    x = normalize_audio(audio)

    rows = []
    prev_mag = None
    for i in range(n_trs):
        start_sec = max(0.0, origin + i * tr)
        end_sec = max(0.0, origin + (i + 1) * tr)
        start = int(round(start_sec * sample_rate))
        end = int(round(end_sec * sample_rate))
        seg = x[start : min(end, x.size)]
        feats, prev_mag = spectral_features(seg, sample_rate, prev_mag)
        feats.update(
            {
                "story": story_name,
                "time_TR": i,
                "audio_file": str(audio_path),
                "align_file": str(align_path),
                "gentle_origin_sec": float(origin),
                "sample_rate": int(sample_rate),
            }
        )
        rows.append(feats)
    return pd.DataFrame(rows)


def bootstrap_mean_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan, np.nan
    ci = stats.bootstrap(
        (values,),
        np.mean,
        confidence_level=0.95,
        n_resamples=10000,
        random_state=42,
        method="basic",
    ).confidence_interval
    return float(ci.low), float(ci.high)


def summarize(values: pd.Series) -> dict:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if x.size == 0:
        return {"n": 0}
    t_stat, p_t = stats.ttest_1samp(x, 0.0)
    try:
        p_w = stats.wilcoxon(x, alternative="greater", zero_method="wilcox").pvalue
    except ValueError:
        p_w = np.nan
    ci_low, ci_high = bootstrap_mean_ci(x)
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else np.nan,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "t": float(t_stat),
        "p_ttest_two_sided": float(p_t),
        "p_wilcoxon_greater": float(p_w) if not pd.isna(p_w) else np.nan,
        "fraction_positive": float(np.mean(x > 0)),
    }


def mixed_effect_intercept(scan: pd.DataFrame, value_col: str) -> dict:
    data = scan[["story", "subject", value_col]].dropna().copy()
    data = data.rename(columns={value_col: "y"})
    data["all_scans"] = "all"
    model = smf.mixedlm(
        "y ~ 1",
        data,
        groups=data["all_scans"],
        vc_formula={"subject": "0 + C(subject)", "story": "0 + C(story)"},
    )
    result = None
    for method in ["lbfgs", "powell", "cg", "bfgs", "nm"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            candidate = model.fit(reml=True, method=method, maxiter=2000, disp=False)
        result = candidate
        if bool(candidate.converged):
            break
    ci = result.conf_int().loc["Intercept"]
    z_value = float(result.tvalues["Intercept"])
    return {
        "mixed_estimate": float(result.params["Intercept"]),
        "mixed_ci95_low": float(ci[0]),
        "mixed_ci95_high": float(ci[1]),
        "mixed_p_two_sided": float(result.pvalues["Intercept"]),
        "mixed_p_one_sided_positive": float(stats.norm.sf(z_value)),
        "mixed_converged": bool(result.converged),
    }


def add_rich_audio_to_regressors(
    unique_stories: list[dict],
    regressor_dir: Path,
    gentle_root: Path,
    out_dir: Path,
    tr: float,
) -> dict[str, pd.DataFrame]:
    reg_out = out_dir / "rich_audio_regressors"
    reg_out.mkdir(parents=True, exist_ok=True)
    feature_rows = []
    regressors = {}

    for story in unique_stories:
        print(f"Computing rich audio features: {story['name']}", flush=True)
        reg = load_regressors(regressor_dir, story["name"]).copy()
        feats = compute_rich_audio_features(story, gentle_root, tr=tr)
        feature_rows.append(feats)

        for col in RICH_AUDIO_COLUMNS:
            if col not in feats.columns:
                feats[col] = 0.0
            feats[f"{col}_hrf"] = hrf_convolve(feats[col].to_numpy(), tr=tr)

        merged = reg.merge(
            feats[["time_TR"] + RICH_AUDIO_COLUMNS + [f"{c}_hrf" for c in RICH_AUDIO_COLUMNS]],
            on="time_TR",
            how="left",
        )
        for col in RICH_AUDIO_COLUMNS + [f"{c}_hrf" for c in RICH_AUDIO_COLUMNS]:
            merged[col] = merged[col].fillna(0.0)
        merged.to_csv(reg_out / f"{story['name']}_shared_pca_regressors.csv", index=False)
        regressors[story["name"]] = merged

    all_features = pd.concat(feature_rows, ignore_index=True)
    all_features.to_csv(out_dir / "jnl_rich_audio_features_by_story.csv", index=False)
    return regressors


def evaluate_models(
    stories: list[dict],
    regressors: dict[str, pd.DataFrame],
    roi_cache_dir: Path,
    out_dir: Path,
    alpha: float,
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pc_cols = [f"PC{i}_hrf" for i in range(1, 6)]
    existing_acoustic_cols = [
        f"{col}_hrf"
        for col in ACOUSTIC_CONTROL_COLUMNS
        if f"{col}_hrf" in next(iter(regressors.values())).columns
    ]
    rich_cols = [f"{col}_hrf" for col in RICH_AUDIO_COLUMNS]

    model_columns = {
        "baseline_lexical": LEXICAL_BASELINE_COLS,
        "combined_lexical_gpt2_pc1_to_pc5": LEXICAL_BASELINE_COLS + pc_cols,
        "baseline_existing_acoustic": LEXICAL_BASELINE_COLS + existing_acoustic_cols,
        "combined_existing_acoustic_gpt2_pc1_to_pc5": LEXICAL_BASELINE_COLS + existing_acoustic_cols + pc_cols,
        "baseline_rich_audio": LEXICAL_BASELINE_COLS + existing_acoustic_cols + rich_cols,
        "combined_rich_audio_gpt2_pc1_to_pc5": LEXICAL_BASELINE_COLS + existing_acoustic_cols + rich_cols + pc_cols,
    }

    for idx, story in enumerate(stories, start=1):
        print(f"[rich audio {idx}/{len(stories)}] {story['name']} {story['subject']}", flush=True)
        cache_path = roi_cache_dir / f"{safe_name(story['name'])}_{safe_name(story['subject'])}.npz"
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing ROI cache: {cache_path}")
        cached = np.load(cache_path, allow_pickle=False)
        roi_names = [str(v) for v in cached["roi_names"].tolist()]
        y_matrix = cached["y_matrix"]
        reg = regressors[story["name"]]

        cv_by_model = {}
        for model_name, cols in model_columns.items():
            missing = [c for c in cols if c not in reg.columns]
            if missing:
                raise ValueError(f"Missing {missing} in {story['name']} regressors")
            cv_by_model[model_name], n_used = vectorized_cv_r(
                reg[cols].to_numpy(),
                y_matrix,
                n_splits=n_splits,
                alpha=alpha,
                do_detrend=True,
            )

        for roi_idx, roi in enumerate(roi_names):
            row = {
                "story": story["name"],
                "subject": story["subject"],
                "roi": roi,
                "n_timepoints": int(n_used),
                "alpha": float(alpha),
            }
            for model_name, values in cv_by_model.items():
                row[model_name] = values[roi_idx]
            row["delta_lexical_gpt2"] = (
                row["combined_lexical_gpt2_pc1_to_pc5"] - row["baseline_lexical"]
            )
            row["delta_existing_acoustic_gpt2"] = (
                row["combined_existing_acoustic_gpt2_pc1_to_pc5"] - row["baseline_existing_acoustic"]
            )
            row["delta_rich_audio_gpt2"] = (
                row["combined_rich_audio_gpt2_pc1_to_pc5"] - row["baseline_rich_audio"]
            )
            rows.append(row)

    long = pd.DataFrame(rows)
    long.to_csv(out_dir / "jnl_rich_audio_encoding_long.csv", index=False)
    scan = (
        long.groupby(["story", "subject"], as_index=False)
        .agg(
            lexical_delta=("delta_lexical_gpt2", "mean"),
            existing_acoustic_delta=("delta_existing_acoustic_gpt2", "mean"),
            rich_audio_delta=("delta_rich_audio_gpt2", "mean"),
            lexical_baseline_cv_r=("baseline_lexical", "mean"),
            rich_audio_baseline_cv_r=("baseline_rich_audio", "mean"),
            rich_audio_combined_cv_r=("combined_rich_audio_gpt2_pc1_to_pc5", "mean"),
        )
        .sort_values(["story", "subject"])
    )
    scan.to_csv(out_dir / "jnl_rich_audio_scan_level.csv", index=False)
    return long, scan


def compare_to_original(scan: pd.DataFrame, primary_deltas: Path, out_dir: Path) -> pd.DataFrame:
    orig = pd.read_csv(primary_deltas)
    orig_scan = (
        orig.groupby(["story", "subject"], as_index=False)
        .agg(original_delta=("delta_cv_r_combined_pc1_to_pc5_minus_baseline", "mean"))
    )
    comp = scan.merge(orig_scan, on=["story", "subject"], how="left")
    comp["rich_minus_original_delta"] = comp["rich_audio_delta"] - comp["original_delta"]
    comp.to_csv(out_dir / "jnl_rich_audio_comparison_vs_original_scan_level.csv", index=False)
    return comp


def write_summary(scan: pd.DataFrame, comparison: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for col in ["lexical_delta", "existing_acoustic_delta", "rich_audio_delta"]:
        row = {"analysis": col}
        row.update(summarize(scan[col]))
        row.update(mixed_effect_intercept(scan.rename(columns={col: "value"}), "value"))
        rows.append(row)
    row = {"analysis": "rich_minus_original_delta"}
    row.update(summarize(comparison["rich_minus_original_delta"]))
    rows.append(row)
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(out_dir / "jnl_rich_audio_scan_stats.csv", index=False)

    stats_by_name = stats_df.set_index("analysis")
    final_rows = []
    for source_name, final_name in [
        ("lexical_delta", "primary_lexical_time_delta"),
        ("rich_audio_delta", "rich_audio_control_delta"),
        ("rich_minus_original_delta", "rich_audio_minus_primary_delta"),
    ]:
        item = stats_by_name.loc[source_name]
        final_rows.append(
            {
                "analysis": final_name,
                "n": int(item["n"]),
                "mean_delta_cv_r": item["mean"],
                "median_delta_cv_r": item["median"],
                "ci95_low": item["ci95_low"],
                "ci95_high": item["ci95_high"],
                "t": item["t"],
                "p_ttest_two_sided": item["p_ttest_two_sided"],
                "p_wilcoxon_greater": item["p_wilcoxon_greater"],
                "fraction_positive": item["fraction_positive"],
                "mixed_estimate": item.get("mixed_estimate", np.nan),
                "mixed_ci95_low": item.get("mixed_ci95_low", np.nan),
                "mixed_ci95_high": item.get("mixed_ci95_high", np.nan),
                "mixed_p_two_sided": item.get("mixed_p_two_sided", np.nan),
            }
        )
    final_table = ROOT / "results" / "final_tables" / "table21_rich_audio_control_sensitivity.csv"
    final_table.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(final_rows).to_csv(final_table, index=False)

    lines = ["# Journal of Neurolinguistics rich audio controls", ""]
    for _, item in stats_df.iterrows():
        lines.append(
            f"- `{item['analysis']}`: mean = {item['mean']:.4f}, "
            f"95% CI [{item['ci95_low']:.4f}, {item['ci95_high']:.4f}], "
            f"p = {item['p_ttest_two_sided']:.4g}, fraction positive = {item['fraction_positive']:.3f}."
        )
    lines.extend(
        [
            "",
            "Interpretation: the key line is `rich_audio_delta`. If it stays positive, GPT-2 still adds information after stronger waveform/spectral/prosodic controls. If `rich_minus_original_delta` is negative, the richer controls explain part of the original increment.",
        ]
    )
    (out_dir / "jnl_rich_audio_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stats_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-list", default="configs/stories_all_fmriprep_confounds_motionfiltered.py")
    parser.add_argument("--regressor-dir", default="results/regressors")
    parser.add_argument("--roi-cache-dir", default=str(DEFAULT_ROI_CACHE))
    parser.add_argument("--primary-deltas", default=str(DEFAULT_PRIMARY_DELTAS))
    parser.add_argument("--gentle-root", default="D:/OpenNeuroData/narratives/stimuli/gentle")
    parser.add_argument("--output-dir", default="results/jnl_revision_major_review/rich_audio_controls")
    parser.add_argument("--tr", type=float, default=1.5)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-subjects-per-story", type=int, default=0)
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stories = load_story_list(str(ROOT / args.subject_list))
    if args.max_subjects_per_story > 0:
        selected = []
        counts: dict[str, int] = {}
        for story in stories:
            counts.setdefault(story["name"], 0)
            if counts[story["name"]] < args.max_subjects_per_story:
                selected.append(story)
                counts[story["name"]] += 1
        stories = selected
    unique_stories = get_unique_stories(stories)

    regressors = add_rich_audio_to_regressors(
        unique_stories,
        ROOT / args.regressor_dir,
        Path(args.gentle_root),
        out_dir,
        tr=args.tr,
    )
    _, scan = evaluate_models(
        stories,
        regressors,
        Path(args.roi_cache_dir),
        out_dir,
        alpha=args.alpha,
        n_splits=args.n_splits,
    )
    comparison = compare_to_original(scan, Path(args.primary_deltas), out_dir)
    stats_df = write_summary(scan, comparison, out_dir)
    print(f"Saved rich-audio outputs in {out_dir}")
    print(stats_df[["analysis", "mean", "ci95_low", "ci95_high", "p_ttest_two_sided"]].to_string(index=False))


if __name__ == "__main__":
    main()
