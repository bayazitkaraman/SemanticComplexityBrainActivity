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

# Bar plot of top ROIs by mean r
plt.figure(figsize=(10, 5))
plt.bar(top_r["roi"], top_r["mean_r"], yerr=top_r["std_r"], capsize=5)
plt.title("Top 5 ROIs by Mean Correlation (r) with Semantic Complexity")
plt.ylabel("Mean Pearson r")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("results/figures/top5_rois_by_mean_r.png")
plt.close()

# Bar plot of top ROIs by mean z-score
plt.figure(figsize=(10, 5))
plt.bar(top_z["roi"], top_z["mean_z"], yerr=top_z["std_z"], capsize=5, color='orange')
plt.title("Top 5 ROIs by Mean Z-Score (vs. Shuffled Control)")
plt.ylabel("Mean Z-Score")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("results/figures/top5_rois_by_mean_z.png")
plt.close()

# Heatmap of ROI r values across subjects
df["story_subject"] = df["story"] + "_" + df["subject"]
pivot_df = df.pivot(index="story_subject", columns="roi", values="r")
plt.figure(figsize=(14, 8))
sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar_kws={'label': 'Pearson r'})
plt.title("ROI Correlation Heatmap Across Subjects")
plt.tight_layout()
plt.savefig("results/figures/roi_correlation_heatmap.png")
plt.close()