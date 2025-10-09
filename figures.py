# figures.py
"""
Generates publication-ready Figures 2–8.

Key upgrades:
- Fig2: 95% CI error bars
- Fig3: NaN-safe group mean maps + combined 1x3 panel saved to results/figures
- Fig4: jittered dots, stricter stars (≥50% pairs FDR<.05), short ROI labels
- Fig5: auto-pick best 3 maps, shared colorbar, threshold=0.35
- Fig6: error bars (SD) + inferred permutation count in title
- Fig7: SEM shading + seconds axis, uses actual lag values (e.g., −3…+8)
- Fig8: short ROI labels, mean ± SD by story

Run: python figures.py
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting, image
from nilearn.masking import compute_brain_mask
import nibabel as nib

# ----------------------------- Paths & discovery -----------------------------
CSV_DIR = "results/csv"
MAP_DIR = "results/maps"
FIG_DIR = "results/figures"
GROUP_DIR = "results/group_averages"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(GROUP_DIR, exist_ok=True)

def pick_first_existing(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    # fall back to first path even if missing, so errors are informative
    return candidates[0]

CSV_ROI = pick_first_existing([
    os.path.join(CSV_DIR, "roi_language_correlation_summary(-3,8).csv"),
    os.path.join(CSV_DIR, "roi_language_correlation_summary.csv"),
    os.path.join(CSV_DIR, "roi_language_correlation_summary(-2,6).csv"),
])
CSV_LAG = pick_first_existing([
    os.path.join(CSV_DIR, "lag_tuning_curves(-3,8).csv"),
    os.path.join(CSV_DIR, "lag_tuning_curves.csv"),
    os.path.join(CSV_DIR, "lag_tuning_curves(-2,6).csv"),
])

# Stories to expect for group means / panel (others will still be handled)
STORY_NAMES = ["lucy", "merlin", "notthefallintact"]

# Common plotting params
CMAP = "coolwarm"
VTHRESH_SUBJ = 0.35   # slightly higher to reduce speckle
VTHRESH_GROUP = 0.25  # group visualization threshold
CUT_COORDS = [-18, 53, 11]

# Short ROI labels for readability (used in Fig4 and Fig8 only)
ROI_ABBR = {
 "Superior Temporal Gyrus, anterior division":"STG a",
 "Superior Temporal Gyrus, posterior division":"STG p",
 "Middle Temporal Gyrus, anterior division":"MTG a",
 "Middle Temporal Gyrus, posterior division":"MTG p",
 "Inferior Frontal Gyrus, pars triangularis":"IFG tri",
 "Inferior Frontal Gyrus, pars opercularis":"IFG operc",
 "Angular Gyrus":"AG",
 "Supramarginal Gyrus, anterior division":"SMG a",
 "Supramarginal Gyrus, posterior division":"SMG p",
 "Frontal Pole":"FP",
 "Precuneous Cortex":"PreCun",
 "Temporal Fusiform Cortex, anterior division":"TFC a",
 "Temporal Fusiform Cortex, posterior division":"TFC p",
 "Parahippocampal Gyrus, anterior division":"PHG a",
 "Parahippocampal Gyrus, posterior division":"PHG p",
 "Cingulate Gyrus, anterior division":"CG a",
 "Cingulate Gyrus, posterior division":"CG p",
}

# ----------------------------- Utilities -----------------------------
def infer_perm_label(df_roi: pd.DataFrame) -> str:
    if "p_emp" not in df_roi.columns:
        return "10,000 permutations"
    p = pd.to_numeric(df_roi["p_emp"], errors="coerce"); p = p[p > 0]
    if p.empty: return "10,000 permutations"
    mn = p.min()
    if mn <= 1.2e-4: return "10,000 permutations"
    if mn <= 1.2e-3: return "1,000 permutations"
    approx = int(round(1.0 / mn)) - 1
    return f"~{approx:,} permutations"

def tr_mode(df_roi: pd.DataFrame, default=1.5) -> float:
    if "TR" in df_roi.columns and not df_roi["TR"].dropna().empty:
        return float(df_roi["TR"].mode()[0])
    return float(default)

def robust_score(nii_path: str) -> float:
    try:
        arr = image.load_img(nii_path).get_fdata()
        return float(np.nanpercentile(np.abs(arr), 95))
    except Exception:
        return -np.inf

# ----------------------------- Load ROI CSV -----------------------------
if not os.path.exists(CSV_ROI):
    raise FileNotFoundError(f"Missing ROI CSV: {CSV_ROI}")

df = pd.read_csv(CSV_ROI)
required_cols = {"roi","story","subject","r"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"[Fig2–8] Missing required columns in {CSV_ROI}: {missing}")

# Normalize types
for col in ["roi","story","subject"]:
    df[col] = df[col].astype(str)

# ----------------------------- FIGURE 2: Top-5 (95% CI) -----------------------------
g = df.groupby("roi")["r"]
roi_stats = pd.DataFrame({"mean_r": g.mean(), "sem": g.sem()}).reset_index()
top = roi_stats.sort_values("mean_r", ascending=False).head(5)
ci95 = 1.96 * top["sem"]

plt.figure(figsize=(10, 5))
plt.bar(top["roi"], top["mean_r"], yerr=ci95, capsize=5)
plt.title("Top 5 ROIs by Mean Correlation (r) with Semantic Complexity")
plt.ylabel("Mean Pearson r (95% CI)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig2_top5_rois_mean_r.png"), dpi=300, bbox_inches="tight")
plt.close()

# ----------------------------- FIGURE 3: Group means + panel -----------------------------
for story in STORY_NAMES:
    nii_files = [f for f in os.listdir(MAP_DIR) if f.startswith(f"corr_map_{story}_") and f.endswith(".nii.gz")]
    nii_paths = [os.path.join(MAP_DIR, f) for f in nii_files]
    if not nii_paths:
        print(f"[Fig3] No maps found for {story}. Skipping.")
        continue

    imgs = [nib.load(p) for p in nii_paths]
    data_stack = np.stack([img.get_fdata() for img in imgs], axis=0)

    # Mask using first image to avoid averaging outside brain; silence warnings
    try:
        brain_mask = compute_brain_mask(imgs[0]).get_fdata().astype(bool)
        data_stack[:, ~brain_mask] = np.nan
    except Exception:
        pass

    stack = np.ma.masked_invalid(data_stack)
    mean_data = stack.mean(axis=0).filled(0.0)
    mean_img = nib.Nifti1Image(mean_data, affine=imgs[0].affine)

    nii_out = os.path.join(GROUP_DIR, f"group_mean_corr_map_{story}.nii.gz")
    png_out = os.path.join(GROUP_DIR, f"group_mean_corr_map_{story}.png")
    mean_img.to_filename(nii_out)
    disp = plotting.plot_stat_map(
        mean_img, title=f"Group-level Correlation: {story}",
        threshold=VTHRESH_GROUP, display_mode="ortho",
        draw_cross=False, cut_coords=CUT_COORDS, cmap=CMAP, colorbar=True
    )
    disp.savefig(png_out); disp.close()
    # also copy to figures/ as single images if you like
    shutil.copyfile(png_out, os.path.join(FIG_DIR, f"figure3_{story}.png"))
    print(f"[Fig3] Saved: {png_out}")

# Build a 1x3 panel into results/figures
stories_for_panel = [s for s in STORY_NAMES if os.path.exists(os.path.join(GROUP_DIR, f"group_mean_corr_map_{s}.nii.gz"))]
if stories_for_panel:
    vmin, vmax = -0.8, 0.8
    fig, axes = plt.subplots(1, len(stories_for_panel), figsize=(15, 5), constrained_layout=True)
    if len(stories_for_panel) == 1: axes = [axes]
    for ax, s in zip(axes, stories_for_panel):
        img = image.load_img(os.path.join(GROUP_DIR, f"group_mean_corr_map_{s}.nii.gz"))
        plotting.plot_stat_map(img, threshold=VTHRESH_GROUP, display_mode="ortho",
                               cut_coords=CUT_COORDS, draw_cross=False, cmap=CMAP,
                               colorbar=False, vmin=vmin, vmax=vmax, axes=ax, annotate=False)
        ax.set_title(s)
    # shared colorbar
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap=CMAP, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02)
    out_path = os.path.join(FIG_DIR, "figure3_group_mean_corr_maps.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[Fig3 panel] Saved: {out_path}")

# ----------------------------- FIGURE 4: Jittered dots + stricter stars -----------------------------
STAR_PROP = 0.50  # star if >=50% subject–story pairs in that ROI are FDR-significant

df_fig4 = df.copy()
roi_order = df_fig4["roi"].drop_duplicates().tolist()
stories = df_fig4["story"].drop_duplicates().tolist()
rng = np.random.default_rng(0)
offsets = np.linspace(-0.25, 0.25, num=len(stories))
xs_by_story = {s: [] for s in stories}
ys_by_story = {s: [] for s in stories}

for i, roi in enumerate(roi_order):
    for j, s in enumerate(stories):
        sub = df_fig4[(df_fig4["roi"] == roi) & (df_fig4["story"] == s)]
        x_vals = i + offsets[j] + 0.04 * rng.normal(size=len(sub))
        y_vals = sub["r"].to_numpy()
        xs_by_story[s].extend(x_vals.tolist()); ys_by_story[s].extend(y_vals.tolist())

plt.figure(figsize=(16, 9))
for s in stories:
    plt.scatter(xs_by_story[s], ys_by_story[s], s=40, alpha=0.6, label=s)
plt.axhline(0, color="black", linewidth=1, linestyle="--")

# Stars from q_roi if present
if "q_roi" in df_fig4.columns:
    def frac_sig(series):
        q = pd.to_numeric(series, errors="coerce")
        return float((q < 0.05).mean()) if q.notna().any() else 0.0
    prop_by_roi = df_fig4.groupby("roi")["q_roi"].apply(frac_sig).reindex(roi_order).fillna(0.0).to_list()
else:
    prop_by_roi = [0.0] * len(roi_order)

# Build tick labels with abbreviations + optional star
def abbr_label(long):
    return ROI_ABBR.get(long, long.replace(" Cortex","").split(",")[0])
xticklabels = [f"{abbr_label(roi)}{'*' if prop >= STAR_PROP else ''}" for roi, prop in zip(roi_order, prop_by_roi)]

plt.xticks(ticks=range(len(roi_order)), labels=xticklabels, rotation=60, ha="right", fontsize=12)
plt.xlabel("ROI"); plt.ylabel("Correlation (r)")
plt.title("Subject-level Correlations by ROI and Story")
plt.legend(title="Story", loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig4_subject_dots_matplotlib.png"), dpi=300, bbox_inches="tight")
plt.close()

# ----------------------------- FIGURE 5: Best voxelwise examples (shared colorbar) -----------------------------
all_maps = [os.path.join(MAP_DIR, f) for f in os.listdir(MAP_DIR) if f.startswith("corr_map_") and f.endswith(".nii.gz")]
scored = [(os.path.basename(p), p, robust_score(p)) for p in all_maps]
scored = [t for t in scored if np.isfinite(t[2])]
scored.sort(key=lambda t: t[2], reverse=True)
example_list = [(name.replace("corr_map_", "").replace(".nii.gz", ""), path) for name, path, _ in scored[:3]]

if example_list:
    vmin, vmax = -0.8, 0.8
    fig, axes = plt.subplots(1, len(example_list), figsize=(14, 5), constrained_layout=True)
    if len(example_list) == 1: axes = [axes]
    for ax, (title, file_path) in zip(axes, example_list):
        img = image.load_img(file_path)
        plotting.plot_stat_map(img, threshold=VTHRESH_SUBJ, cut_coords=CUT_COORDS,
                               display_mode="ortho", draw_cross=False, cmap=CMAP,
                               colorbar=False, vmin=vmin, vmax=vmax, axes=ax, annotate=False)
        ax.text(0.5, -0.1, title, ha='center', va='top', fontsize=10, transform=ax.transAxes)
    # shared colorbar
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap=CMAP, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    out_path = os.path.join(FIG_DIR, "figure5_best_voxelwise_examples.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[Fig5] Saved: {out_path}")
else:
    print("[Fig5] No example maps found to plot.")

# ----------------------------- FIGURE 6: Actual vs shuffled (with SD) -----------------------------
perm_label = infer_perm_label(df)
agg = (df.groupby("roi")
         .agg(actual_r=("r","mean"),
              actual_sd=("r","std"),
              shuf_r=("mean_r_shuffled","mean"),
              shuf_sd=("mean_r_shuffled","std"))
         .reset_index())
agg = agg.sort_values("actual_r", ascending=False).reset_index(drop=True)

x = np.arange(len(agg)); w = 0.35
plt.figure(figsize=(14, 6))
plt.bar(x - w/2, agg["actual_r"], width=w, yerr=agg["actual_sd"], capsize=3, label="Actual")
plt.bar(x + w/2, agg["shuf_r"],  width=w, yerr=agg["shuf_sd"],  capsize=3, label="Shuffled")
plt.xticks(x, [abbr_label(r) for r in agg["roi"]], rotation=45, ha='right')
plt.ylabel("Mean Pearson r ± SD")
plt.title(f"Actual vs. Shuffled ROI Correlation (Mean Across Subjects; {perm_label})")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig6_actual_vs_shuffled.png"), dpi=400, bbox_inches="tight")
plt.close()

# ----------------------------- FIGURE 7: Lag tuning (SEM shading + seconds axis) -----------------------------
if os.path.exists(CSV_LAG):
    df_lag = pd.read_csv(CSV_LAG)
    if {"roi","lag","r"} <= set(df_lag.columns):
        df_lag["roi"] = df_lag["roi"].astype(str)
        # pick top 5 ROIs by overall mean r across lags
        selected_rois = (df_lag.groupby("roi")["r"].mean()
                         .sort_values(ascending=False).head(5).index.tolist())

        plt.figure(figsize=(12, 8))
        for roi in selected_rois:
            d = (df_lag[df_lag["roi"] == roi].groupby("lag")["r"]
                    .agg(["mean","sem"]).sort_index())
            plt.plot(d.index.values, d["mean"].values, marker='o', label=roi)
            plt.fill_between(d.index.values, d["mean"]-d["sem"], d["mean"]+d["sem"], alpha=0.15)

        plt.xlabel("Lag (TRs)"); plt.ylabel("Mean Pearson r")
        plt.title("Lag Tuning Curves of Semantic Complexity Correlation (top ROIs)")

        all_lags = sorted(df_lag["lag"].dropna().unique().tolist())
        plt.xticks(all_lags, [str(int(l)) for l in all_lags])
        if 0 in all_lags:
            plt.axvline(0, color="black", linewidth=1, linestyle="--")

        TR = tr_mode(df, default=1.5)
        ax = plt.gca()
        def tr_to_sec(x): return x * TR
        def sec_to_tr(s): return s / TR
        sec_ax = ax.secondary_xaxis('top', functions=(tr_to_sec, sec_to_tr))
        sec_ax.set_xlabel("Lag (seconds)")
        sec_ax.set_xticks(all_lags)
        sec_ax.set_xticklabels([f"{l*TR:.1f}" for l in all_lags])

        plt.grid(True); plt.legend(loc="best"); plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "fig7_lag_tuning_curves.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("[Fig7] Saved lag tuning figure.")
    else:
        print(f"[Fig7] {CSV_LAG} missing required columns (need roi, lag, r).")
else:
    print(f"[Fig7] Missing lag CSV: {CSV_LAG}")

# ----------------------------- FIGURE 8: ROI × story (mean ± SD) -----------------------------
storywise = df.groupby(["roi","story"]).agg(mean_r=("r","mean"), sd_r=("r","std")).reset_index()
roi_means = storywise.groupby("roi")["mean_r"].mean().sort_values(ascending=False).head(10).index
top_df = storywise[storywise["roi"].isin(roi_means)]
pivot_mean = top_df.pivot(index="roi", columns="story", values="mean_r").loc[roi_means]
pivot_sd   = top_df.pivot(index="roi", columns="story", values="sd_r").loc[roi_means]

stories_order = list(pivot_mean.columns)
x = np.arange(len(pivot_mean))
width = 0.25 if len(stories_order) <= 4 else 0.8 / max(1, len(stories_order))

plt.figure(figsize=(14, 6))
for i, story in enumerate(stories_order):
    means = pivot_mean[story]; sds = pivot_sd[story]
    plt.bar(x + i*width, means, width=width, yerr=sds, capsize=4, label=story)

plt.xticks(x + (len(stories_order)-1)*width/2, [abbr_label(r) for r in pivot_mean.index], rotation=45, ha='right')
plt.ylabel("Mean Pearson r ± SD")
plt.title("ROI-wise Correlation by Story (Mean ± SD)")
plt.legend(title="Story"); plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig8_storywise_roi_bars.png"), dpi=300, bbox_inches="tight")
plt.close()

print("All figures saved to:", FIG_DIR)
