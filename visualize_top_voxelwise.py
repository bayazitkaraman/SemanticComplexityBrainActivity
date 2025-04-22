from nilearn import plotting, image
import matplotlib.pyplot as plt
import os

# Setup
output_dir = "results/figures"
os.makedirs(output_dir, exist_ok=True)

top_examples = [
    ("lucy sub-052", "results/maps/corr_map_lucy_sub-052.nii.gz"),
    ("notthefallintact sub-318", "results/maps/corr_map_notthefallintact_sub-318.nii.gz"),
    ("lucy sub-056", "results/maps/corr_map_lucy_sub-056.nii.gz"),
]

cut_coords = [-18, 53, 11]
vmax = 0.8
vmin = -0.8
cmap = "coolwarm"

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, (title, file_path) in enumerate(top_examples):
    img = image.load_img(file_path)
    display = plotting.plot_stat_map(
        img,
        threshold=0.3,
        cut_coords=cut_coords,
        display_mode="ortho",
        draw_cross=False,
        cmap=cmap,
        colorbar=True,
        vmax=vmax,
        vmin=vmin,
        axes=axes[i],
        annotate=False  # prevent automatic title
    )
    # Add label below
    axes[i].set_title("", fontsize=1)
    axes[i].text(0.5, -0.15, rf"$\mathbf{{{title}}}$", ha='center', va='top', fontsize=10, transform=axes[i].transAxes)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "figure_best_voxelwise_examples.png"), dpi=300, bbox_inches="tight")