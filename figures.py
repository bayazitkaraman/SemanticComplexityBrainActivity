import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
plt.figure(figsize=(14, 8))
sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar_kws={'label': 'Pearson r'})
plt.title("ROI Correlation Heatmap Across Subjects")
plt.tight_layout()
plt.savefig("results/figures/fig4_heatmap_roi_subject.png")
plt.close()

# === Figure 6: Actual vs Shuffled ===
roi_avg = df.groupby("roi").agg(
    actual_r=("r", "mean"),
    shuffled_r=("mean_r_shuffled", "mean")
).reset_index()

x = range(len(roi_avg))
plt.figure(figsize=(10, 5))
plt.bar([i - 0.2 for i in x], roi_avg["actual_r"], width=0.4, label="Actual", color="blue")
plt.bar([i + 0.2 for i in x], roi_avg["shuffled_r"], width=0.4, label="Shuffled", color="orange")
plt.xticks(x, roi_avg["roi"], rotation=90)
plt.title("Figure 6: Actual vs. Shuffled ROI Correlation")
plt.ylabel("Pearson r")
plt.legend()
plt.tight_layout()
plt.savefig("results/figures/fig6_actual_vs_shuffled.png")
plt.close()

# === Figure 8: ROI-by-Story Correlation ===
storywise = df.groupby(["roi", "story"]).agg(mean_r=("r", "mean")).reset_index()
pivot = storywise.pivot(index="roi", columns="story", values="mean_r").fillna(0)
top10 = pivot.mean(axis=1).sort_values(ascending=False).head(10).index
pivot = pivot.loc[top10]

x = np.arange(len(pivot))
width = 0.25
plt.figure(figsize=(12, 6))
for i, story in enumerate(pivot.columns):
    plt.bar(x + i * width, pivot[story], width=width, label=story)
plt.xticks(x + width, pivot.index, rotation=45, ha='right')
plt.ylabel("Mean Pearson r")
plt.title("Figure 8: ROI-wise Correlation by Story")
plt.legend()
plt.tight_layout()
plt.savefig("results/figures/fig7_storywise_roi_bars.png")
plt.close()
