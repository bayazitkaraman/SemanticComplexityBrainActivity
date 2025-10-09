# neuroimage_fmri_analysis.py

import os
import torch
import nibabel as nib
import numpy as np
import pandas as pd
import csv

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from transformers import GPT2Tokenizer, GPT2Model
from nilearn import datasets, plotting
from nilearn.image import resample_to_img, smooth_img, new_img_like, mean_img
from nilearn.masking import compute_brain_mask
from nilearn.glm.first_level import glover_hrf
from statsmodels.stats.multitest import multipletests

# ---------------------- SETUP ----------------------
RESULT_DIR = "results"
CSV_DIR = os.path.join(RESULT_DIR, "csv")
MAP_DIR = os.path.join(RESULT_DIR, "maps")
FIG_DIR = os.path.join(RESULT_DIR, "figures")
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(MAP_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Permutations & lag window (set LAG_MIN=0 to restrict to non-negative lags)
N_PERM = 10000          # set to 1000 for quick tests, 10000 for final runs
LAG_MIN, LAG_MAX = -3, 8  # inclusive; corresponds to -3..+8 TRs

# ---------------------- GLOBAL MODEL LOADING ----------------------
def load_models():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model

# ---------------------- SEMANTIC COMPLEXITY COMPUTATION ----------------------
def compute_complexity_hrf(story, tokenizer, model, TR):
    with open(story['transcript_file'], 'r', encoding='utf-8') as f:
        words = f.read().strip().split()

    events_df = pd.read_csv(story['events_file'], sep='\t')
    story_event = events_df[events_df['trial_type'] == 'story']
    if story_event.empty:
        return None, None, None, None, None

    onset = float(story_event.iloc[0]['onset'])
    duration = float(story_event.iloc[0]['duration'])
    story_TRs = int(duration // TR)

    # Split transcript into chunks per TR
    words_per_TR = int(np.floor(len(words) / story_TRs)) if story_TRs > 0 else 0
    if story_TRs <= 0 or words_per_TR == 0:
        return None, None, None, None, None

    word_chunks = [" ".join(words[i * words_per_TR:(i + 1) * words_per_TR]) for i in range(story_TRs)]
    leftover = words[story_TRs * words_per_TR:]
    if leftover:
        word_chunks[-1] += " " + " ".join(leftover)

    # Get GPT-2 embeddings
    embeddings = []
    for chunk in word_chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        # mean-pool tokens -> 768-d
        embeddings.append(outputs.last_hidden_state.squeeze(0).mean(dim=0))

    # PCA over normalized embeddings
    X_scaled = StandardScaler().fit_transform(np.stack([e.numpy() for e in embeddings]))
    complexity = PCA(n_components=1).fit_transform(X_scaled).flatten()

    # Apply Glover HRF
    hrf = glover_hrf(t_r=TR, oversampling=20, time_length=30, onset=0)
    complexity_hrf = np.convolve(complexity, hrf)[:len(complexity)]

    return complexity_hrf, onset, duration, story_TRs, events_df

# ---------------------- MAIN ANALYSIS ----------------------
def analyze_subject(story, tokenizer, model, atlas_img, atlas_labels, roi_labels_to_test):
    fmri_img = smooth_img(nib.load(story['bold_file']), fwhm=6)
    fmri_data = fmri_img.get_fdata()
    TR = fmri_img.header.get_zooms()[3]
    brain_mask = compute_brain_mask(fmri_img).get_fdata().astype(bool)

    complexity_hrf, onset, duration, story_TRs, events_df = compute_complexity_hrf(story, tokenizer, model, TR)
    if complexity_hrf is None:
        return [], None, []

    # Resample atlas to match fMRI space
    resampled_atlas = resample_to_img(atlas_img, fmri_img, interpolation='nearest', force_resample=True, copy_header=True)
    resampled_data = resampled_atlas.get_fdata()

    results = []
    lag_curve_records = []  # for saving lag vs r per ROI
    np.random.seed(42)  # reproducibility

    # Precompute bold start index
    bold_start = int(onset // TR)

    for roi_label in roi_labels_to_test:
        if roi_label not in atlas_labels:
            continue
        roi_index = atlas_labels.index(roi_label)

        # Probabilistic Harvard–Oxford maps -> use threshold > 0 as mask
        roi_mask = resampled_data[..., roi_index] > 0

        # Extract mean ROI activation per TR
        roi_activation = []
        for i in range(story_TRs):
            bold_index = bold_start + i
            if bold_index >= fmri_data.shape[3]:
                break
            roi_activation.append(np.nanmean(fmri_data[:, :, :, bold_index][roi_mask]))

        if len(roi_activation) == 0:
            # No data -> record NaNs and continue
            results.append({
                "story": story['name'],
                "subject": story['subject'],
                "roi": roi_label,
                "r": np.nan, "p": np.nan, "p_emp": np.nan, "lag": np.nan,
                "mean_r_shuffled": np.nan, "std_r_shuffled": np.nan, "z_score": np.nan,
                "TR": TR, "duration": duration, "onset": onset
            })
            continue

        # Correlate with complexity_hrf (with lag search)
        roi_activation = np.array(roi_activation[:len(complexity_hrf)], dtype=float)

        best_r, best_p, best_lag = np.nan, np.nan, np.nan
        # lag range (supports negative lags)
        for lag in range(LAG_MIN, LAG_MAX + 1):
            shifted = np.roll(complexity_hrf, lag)
            if lag > 0:
                shifted[:lag] = np.nan  # leading NaNs when BOLD lags stimulus
            elif lag < 0:
                shifted[lag:] = np.nan  # trailing NaNs when regressor leads
            valid = ~np.isnan(shifted) & ~np.isnan(roi_activation)
            if valid.sum() > 1:
                r, p = pearsonr(shifted[valid], roi_activation[valid])
                lag_curve_records.append({
                    "story": story['name'],
                    "subject": story['subject'],
                    "roi": roi_label,
                    "lag": lag,
                    "r": r
                })
                if np.isnan(best_r) or abs(r) > abs(best_r):
                    best_r, best_p, best_lag = r, p, lag

        # If no valid correlation could be computed, record NaNs and continue
        if np.isnan(best_r):
            results.append({
                "story": story['name'],
                "subject": story['subject'],
                "roi": roi_label,
                "r": np.nan, "p": np.nan, "p_emp": np.nan, "lag": np.nan,
                "mean_r_shuffled": np.nan, "std_r_shuffled": np.nan, "z_score": np.nan,
                "TR": TR, "duration": duration, "onset": onset
            })
            continue

        # Shuffled control correlations (empirical p, best-lag null)
        shuffled_rs = []
        for _ in range(N_PERM):
            shuffled = np.random.permutation(complexity_hrf)
            best_r_shuf = np.nan
            for lag in range(LAG_MIN, LAG_MAX + 1):
                shifted = np.roll(shuffled, lag)
                if lag > 0:
                    shifted[:lag] = np.nan
                elif lag < 0:
                    shifted[lag:] = np.nan
                valid = ~np.isnan(shifted) & ~np.isnan(roi_activation)
                if valid.sum() > 1:
                    r = pearsonr(shifted[valid], roi_activation[valid])[0]
                    if np.isnan(best_r_shuf) or abs(r) > abs(best_r_shuf):
                        best_r_shuf = r
            shuffled_rs.append(best_r_shuf)

        shuffled_rs = np.asarray(shuffled_rs, dtype=float)
        mean_shuf = float(np.nanmean(shuffled_rs))
        std_shuf  = float(np.nanstd(shuffled_rs))
        z_score   = (best_r - mean_shuf) / std_shuf if std_shuf > 0 else np.nan

        # two-sided empirical p-value using absolute correlations
        p_emp = (1.0 + np.sum(np.abs(shuffled_rs) >= abs(best_r))) / (N_PERM + 1.0)

        results.append({
            "story": story['name'],
            "subject": story['subject'],
            "roi": roi_label,
            "r": round(best_r, 4),
            "p": round(best_p, 6),          # parametric p from pearsonr (kept for reference)
            "p_emp": round(p_emp, 6),       # empirical permutation p
            "lag": best_lag,
            "mean_r_shuffled": round(mean_shuf, 4),
            "std_r_shuffled": round(std_shuf, 4),
            "z_score": round(z_score, 4),
            "TR": TR,
            "duration": duration,
            "onset": onset
        })

    # ---- FDR across ROIs for this (story, subject) ----
    if results:
        p_vals = np.array([row["p_emp"] for row in results], dtype=float)
        # if any p_emp is nan (e.g., no valid data), set to 1 so it won't be significant
        p_vals = np.where(np.isnan(p_vals), 1.0, p_vals)
        reject, q_vals, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')
        for i in range(len(results)):
            results[i]["q_roi"] = round(float(q_vals[i]), 6)
            results[i]["signif_roi_fdr"] = bool(reject[i])

    # --- Voxelwise Correlation Map ---
    voxel_shape = fmri_data.shape[:3]
    corr_map = np.zeros(voxel_shape)
    p_map = np.ones(voxel_shape)

    for x in range(voxel_shape[0]):
        for y in range(voxel_shape[1]):
            for z in range(voxel_shape[2]):
                if not brain_mask[x, y, z]:
                    continue
                bold_series = fmri_data[x, y, z, bold_start:bold_start + story_TRs]
                bold_series = bold_series[:len(complexity_hrf)]
                valid = ~np.isnan(bold_series)
                if valid.sum() > 5:
                    corr_map[x, y, z], p_map[x, y, z] = pearsonr(complexity_hrf[valid], bold_series[valid])

    # Simple BH-FDR for voxelwise p-map (already have your manual FDR—keeping your logic)
    p_flat = p_map.flatten()
    p_sorted = np.sort(p_flat[p_flat < 1])
    N = len(p_sorted)
    q = 0.05
    thresholds = (np.arange(1, N + 1) / N) * q
    passed = p_sorted <= thresholds
    fdr_threshold = p_sorted[np.where(passed)[0][-1]] if np.any(passed) else 0
    mask_sig = p_map < fdr_threshold

    masked_corr_map = corr_map * mask_sig
    masked_corr_map[~brain_mask] = 0
    corr_img = new_img_like(fmri_img, masked_corr_map)

    # Save NIfTI and PNG
    nii_filename = os.path.join(MAP_DIR, f"corr_map_{story['name']}_{story['subject']}.nii.gz")
    png_filename = os.path.join(FIG_DIR, f"corr_map_{story['name']}_{story['subject']}.png")
    corr_img.to_filename(nii_filename)
    display = plotting.plot_stat_map(
        corr_img, threshold=0.3, cmap='coolwarm',
        title=f"Voxel-wise Correlation: {story['name']} {story['subject']}",
        display_mode='ortho', draw_cross=False, cut_coords=[-18, 53, 11], colorbar=True
    )
    display.savefig(png_filename)
    print(f"Saved: {png_filename}")
    display.close()

    return results, corr_img, lag_curve_records

# ---------------------- MAIN ENTRY POINT ----------------------
def run_all():
    tokenizer, model = load_models()
    atlas = datasets.fetch_atlas_harvard_oxford('cort-prob-2mm')
    atlas_img = atlas.maps
    atlas_labels = [str(x) for x in atlas.labels]

    # Define ROIs of interest
    roi_labels_to_test = [
        'Superior Temporal Gyrus, anterior division',
        'Superior Temporal Gyrus, posterior division',
        'Middle Temporal Gyrus, anterior division',
        'Middle Temporal Gyrus, posterior division',
        'Inferior Frontal Gyrus, pars triangularis',
        'Inferior Frontal Gyrus, pars opercularis',
        'Angular Gyrus',
        'Supramarginal Gyrus, anterior division',
        'Supramarginal Gyrus, posterior division',
        'Frontal Pole',
        'Precuneous Cortex',
        'Temporal Fusiform Cortex, anterior division',
        'Temporal Fusiform Cortex, posterior division',
        'Parahippocampal Gyrus, anterior division',
        'Parahippocampal Gyrus, posterior division',
        'Cingulate Gyrus, anterior division',
        'Cingulate Gyrus, posterior division'
    ]

    missing = [r for r in roi_labels_to_test if r not in atlas_labels]
    if missing:
        print("WARNING: the following ROI labels were not found in the atlas:", missing)

    from experiment_subject_list import stories

    all_results = []
    group_maps = []
    all_lag_curve_records = []

    for story in stories:
        print(f"Processing: {story['name']} ({story['subject']})")
        results, corr_img, lag_curves = analyze_subject(
            story, tokenizer, model, atlas_img, atlas_labels, roi_labels_to_test
        )
        all_results.extend(results)
        all_lag_curve_records.extend(lag_curves)
        if corr_img:
            group_maps.append(corr_img)

    pd.DataFrame(all_results).to_csv(os.path.join(CSV_DIR, "roi_language_correlation_summary.csv"), index=False)
    print("Saved summary CSV.")

    # Save lag tuning curves to CSV
    with open(os.path.join(CSV_DIR, "lag_tuning_curves.csv"), "w", newline='') as csvfile:
        fieldnames = ["story", "subject", "roi", "lag", "r"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_lag_curve_records:
            writer.writerow(row)
    print("Saved lag tuning curve CSV.")

if __name__ == "__main__":
    run_all()
