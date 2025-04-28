import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the correlation summary
df = pd.read_csv("results/csv/roi_language_correlation_summary.csv")

# Compute mean and std for each ROI
roi_stats = df.groupby("roi").agg(mean_r=("r", "mean"), std_r=("r", "std"),
                                  mean_z=("z_score", "mean"), std_z=("z_score", "std")).reset_index()

# Sort by mean_r for top 5 ROIs
top_r = roi_stats.sort_values("mean_r", ascending=False).head(5)
top_z = roi_stats.sort_values("mean_z", ascending=False).head(5)

# Create results directory
os.makedirs("results/figures", exist_ok=True)

# === Figure 2: Bar plot of top ROIs by mean r ===
plt.figure(figsize=(10, 5))
plt.bar(top_r["roi"], top_r["mean_r"], yerr=top_r["std_r"], capsize=5)
plt.title("Top 5 ROIs by Mean Correlation (r) with Semantic Complexity")
plt.ylabel("Mean Pearson r")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("results/figures/fig2_top5_rois_mean_r.png")
plt.close()

# === Figure 3: Group-level voxel-wise correlation map for the Lucy narrative ===
# You can generate the figure by calling generate_group_voxelwise_maps.py function


# === Figure 4: Heatmap of ROI r values across subjects ===
df["story_subject"] = df["story"] + "_" + df["subject"]
pivot_df = df.pivot(index="story_subject", columns="roi", values="r")
plt.figure(figsize=(18, 10))
sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar_kws={'label': 'Pearson r'}, annot_kws={"fontsize": 7})
plt.title("ROI Correlation Heatmap Across Subjects")
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig("results/figures/fig4_heatmap_roi_subject.png")
plt.close()

# === Figure 6: Actual vs Shuffled ===
roi_avg = df.groupby("roi").agg(
    actual_r=("r", "mean"),
    shuffled_r=("mean_r_shuffled", "mean")
).reset_index()

# Sort ROIs by actual_r
roi_avg = roi_avg.sort_values(by="actual_r", ascending=False).reset_index(drop=True)
x = np.arange(len(roi_avg))

plt.figure(figsize=(14, 6))
bar_width = 0.35
plt.bar(x - bar_width / 2, roi_avg["actual_r"], width=bar_width, label="Actual", color="steelblue")
plt.bar(x + bar_width / 2, roi_avg["shuffled_r"], width=bar_width, label="Shuffled", color="orange")

plt.xticks(x, roi_avg["roi"], rotation=45, ha='right')
plt.ylabel("Mean Pearson r")
plt.title("Actual vs. Shuffled ROI Correlation (Mean Across Subjects)")
plt.legend()
plt.tight_layout()

# Save high-quality version
plt.savefig("results/figures/fig6_actual_vs_shuffled.png", dpi=400, bbox_inches="tight")
plt.close()

# === Figure 8: ROI-by-Story Correlation ===
storywise = df.groupby(["roi", "story"]).agg(mean_r=("r", "mean"), std_r=("r", "std")).reset_index()

# Get top 10 ROIs by average mean_r across all stories
roi_means = storywise.groupby("roi")["mean_r"].mean().sort_values(ascending=False).head(10).index
top_df = storywise[storywise["roi"].isin(roi_means)]

# Pivot for plotting
pivot_mean = top_df.pivot(index="roi", columns="story", values="mean_r").loc[roi_means]
pivot_std = top_df.pivot(index="roi", columns="story", values="std_r").loc[roi_means]

# Plot grouped bar chart with error bars
stories = pivot_mean.columns
x = np.arange(len(pivot_mean))
width = 0.25

plt.figure(figsize=(14, 6))
for i, story in enumerate(stories):
    means = pivot_mean[story]
    stds = pivot_std[story]
    plt.bar(x + i * width, means, width=width, yerr=stds, capsize=4, label=story)

plt.xticks(x + width, pivot_mean.index, rotation=45, ha='right')
plt.ylabel("Mean Pearson r")
plt.title("ROI-wise Correlation by Story (Mean ± SD)")
plt.legend(title="Story")
plt.tight_layout()

plt.savefig("results/figures/fig8_storywise_roi_bars.png")
plt.close()

# === Figure 7: Lag Tuning Curves ===
df_lag = pd.read_csv("results/csv/lag_tuning_curves.csv")

selected_rois = df_lag.groupby("roi")["r"].mean().sort_values(ascending=False).head(5).index.tolist()

plt.figure(figsize=(12, 8))
for roi in selected_rois:
    roi_df = df_lag[df_lag["roi"] == roi]
    means = roi_df.groupby("lag")["r"].mean()
    plt.plot(means.index, means.values, marker='o', label=roi)

plt.xlabel("Lag (TRs)")
plt.ylabel("Mean Pearson r")
plt.title("Lag Tuning Curves of Semantic Complexity Correlation for Top ROIs")
plt.xticks(ticks=range(7), labels=[f"{1.5 * i:.1f}" for i in range(7)])
plt.legend(loc="center right", bbox_to_anchor=(0.9, 0.8))
plt.grid(True)
plt.tight_layout()
plt.savefig("results/figures/fig7_lag_tuning_curves.png", dpi=300, bbox_inches='tight')
plt.close()
