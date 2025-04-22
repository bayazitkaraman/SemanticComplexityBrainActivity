# Semantic Complexity and Brain Activity

This project investigates the relationship between semantic complexity in natural language and brain activity using fMRI data from the <a href="https://openneuro.org/datasets/ds002245/versions/1.0.1" target="_blank">Narratives dataset</a>. The pipeline aligns language features with fMRI signals and computes correlations across different brain regions.

---

## Project Structure

```
.
├── neuroimage_fmri_analysis.py       # Main analysis pipeline
├── generate_group_voxelwise_maps.py  # Group-level voxelwise average maps
├── visualize_top_voxelwise.py        # Example visualization of top voxelwise maps
├── figures.py                        # Plots for summary statistics
├── experiment_subject_list.py        # List of stories and corresponding subject files
├── results/
│   ├── csv/                          # ROI correlation summaries
│   ├── maps/                         # Individual voxelwise maps (NIfTI)
│   ├── figures/                      # Visual figures (PNG)
│   └── group_averages/               # Group-level mean maps
```

---

## Requirements

Install the required packages using pip:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch transformers nilearn scikit-learn matplotlib seaborn pandas nibabel
```

---

## How to Run

### Step 1: Prepare Your Data
Organize your dataset similar to this structure:
```
/data/
  ├── lucy/
  │    ├── sub-XXX_task-lucy_bold.nii.gz
  │    ├── task-lucy_events.tsv
  │    └── lucy_transcript.txt
  └── merlin/ ...
```
Note: The data/ directory should contain the transcript.csv and events.tsv files. Please download the necessary .nii.gz files directly from the source.​ Ensure paths in `experiment_subject_list.py` point to the correct files.

### Step 2: Run the Analysis
```bash
python neuroimage_fmri_analysis.py
```
This will:
- Compute GPT-2-based semantic complexity
- Align it with fMRI data using HRF convolution
- Run ROI and voxelwise correlation analyses
- Save results in the `results/` folder

### Step 3: Visualize Results
```bash
python generate_group_voxelwise_maps.py
python visualize_top_voxelwise.py
python figures.py
```
These scripts produce:
- Group-level correlation maps
- Best examples from individuals
- Bar and heatmap plots of ROI correlation results

---

## Method Summary
- Semantic complexity is derived by extracting GPT-2 embeddings per TR.
- Embeddings are PCA-reduced to a single complexity score.
- Complexity is convolved with a Glover HRF and aligned with BOLD data.
- Pearson correlations are computed between complexity and fMRI signal at both ROI and voxel levels.
- Permutation testing is used to compute z-scores.

---

## Outputs

- `results/csv/roi_language_correlation_summary.csv`: Main table of ROI correlation results per subject.
- `results/maps/`: Individual voxelwise NIfTI maps.
- `results/figures/`: Correlation heatmaps and bar plots.
- `results/group_averages/`: Mean correlation maps for each story.

---

## Notes
- The model uses GPT-2 and Harvard-Oxford Cortical Atlas for analysis.
- This pipeline is scalable for additional stories and subjects.

---

## Citation & Acknowledgments
Bayazit Karaman, "Tracking Brain Activity with Semantic Complexity During Naturalistic Narrative Comprehension." Submitted to NeuroImage, 2025.

## License
MIT License 
