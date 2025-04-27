import os
import nibabel as nib
import numpy as np
from nilearn import plotting, image
from nilearn.masking import compute_brain_mask

# --- Configuration ---
story_names = ['lucy', 'merlin', 'notthefallintact']
input_folder = 'results/maps'
output_folder = 'results/group_averages'
os.makedirs(output_folder, exist_ok=True)

# --- Group-Level Averaging ---
for story in story_names:
    nii_files = [f for f in os.listdir(input_folder) if f.startswith(f'corr_map_{story}') and f.endswith('.nii.gz')]
    nii_paths = [os.path.join(input_folder, f) for f in nii_files]
    
    if not nii_paths:
        print(f"No maps found for {story}. Skipping.")
        continue

    imgs = [nib.load(p) for p in nii_paths]
    data_stack = np.stack([img.get_fdata() for img in imgs])
    mean_data = np.nanmean(data_stack, axis=0)
    mean_img = nib.Nifti1Image(mean_data, affine=imgs[0].affine)

    # Compute brain mask for thresholding
    brain_mask = compute_brain_mask(imgs[0]).get_fdata().astype(bool)
    mean_data[~brain_mask] = 0
    mean_img_masked = nib.Nifti1Image(mean_data, affine=imgs[0].affine)

    # Save the NIfTI map
    nii_out_path = os.path.join(output_folder, f"group_mean_corr_map_{story}.nii.gz")
    mean_img_masked.to_filename(nii_out_path)

    # Save PNG plot
    png_out_path = os.path.join(output_folder, f"group_mean_corr_map_{story}.png")
    display =plotting.plot_stat_map(
        mean_img_masked, title=f"Group-level Correlation: {story}",
        threshold=0.25, display_mode='ortho', draw_cross=False,
        cut_coords=[-18, 53, 11], cmap="coolwarm", colorbar=True
    )
    display.savefig(png_out_path)
    display.close()

    print(f"✅ Saved average map for {story} → {png_out_path}")
