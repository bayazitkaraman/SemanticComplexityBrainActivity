# Semantic Complexity and Brain Activity

This project investigates the relationship between semantic complexity in natural language and brain activity using fMRI data from the <a href="https://openneuro.org/datasets/ds002345/versions/1.0.1" target="_blank">Narratives dataset</a>. The pipeline aligns language features with fMRI signals and computes correlations across different brain regions.

---

## Project Structure

```
.
├── neuroimage_fmri_analysis.py       # Main analysis pipeline
├── figures.py                        # Generates all figures except Figure 1
├── experiment_subject_list.py        # Defines story names and file paths
├── results/
│   ├── csv/                          # ROI correlation summaries and lag tuning curves
│   ├── maps/                         # Individual voxelwise correlation maps (NIfTI)
│   ├── figures/                      # Visualization figures (PNG)
│   └── group_averages/               # Group-level mean maps (per story)
```

---

## Requirements

Install the required packages using pip:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch transformers nilearn scikit-learn matplotlib seaborn pandas nibabel statsmodels
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
  └── merlin/
       ├── sub-XXX_task-merlin_bold.nii.gz
       ├── task-merlin_events.tsv
       └── merlin_transcript.txt
```
Ensure that each story folder contains:
- *_bold.nii.gz - the preprocessed fMRI data
- *_events.tsv - the event timing file
- *_transcript.txt - the story transcript

Paths to these files must be specified correctly inside `experiment_subject_list.py` using:

{
  'name': 'lucy',
  'subject': 'sub-XXX',
  'bold_file': '/path/to/sub-XXX_task-lucy_bold.nii.gz',
  'events_file': '/path/to/task-lucy_events.tsv',
  'transcript_file': '/path/to/lucy_transcript.txt'
}

### Step 2: Run the Analysis
```bash
python neuroimage_fmri_analysis.py
```
This will:
- Compute GPT-2–based semantic complexity per TR
- Convolve it with a Glover HRF
- Perform ROI-wise and voxel-wise correlation analyses
- Apply FDR correction and permutation testing
- Save results under the `results/` directory

### Step 3: Visualize Results
```bash
python figures.py
```
These scripts produce:
- Group-level mean maps
- Representative voxelwise correlation figures
- ROI-level bar and heatmap visualizations

All outputs are saved in the `results/figures/` and `results/group_averages/` folders.

---

## Method Summary
- Semantic complexity is derived from GPT-2 hidden embeddings per TR.
- Embeddings are normalized and reduced to a single dimension via PCA.
- The resulting complexity signal is convolved with a Glover HRF and aligned to BOLD responses.
- Pearson correlations are computed between the complexity regressor and BOLD activity at both ROI and voxel levels.
- A permutation test (10,000 iterations) provides empirical p-values and z-scores.
- FDR correction (Benjamini–Hochberg) is applied within subject for ROI and voxelwise results.

---

## Outputs

- `results/csv/roi_language_correlation_summary.csv`: ROI correlation statistics per story and subject
- `results/csv/lag_tuning_curves.csv`: Correlation values across tested lags
- `results/maps/`: Individual voxelwise correlation maps (NIfTI format)
- `results/figures/`: Voxelwise and ROI visualization figures (PNG)
- `results/group_averages/`: Mean correlation maps across subjects per story

---

## Notes
- The analysis uses GPT-2 (Hugging Face Transformers) and the Harvard–Oxford Cortical Atlas.
- HRF convolution follows the canonical Glover model.
- The pipeline is scalable to additional stories, subjects, or models.
- Reproducibility: random seed is fixed (42) for deterministic behavior.

---

## Citation & Acknowledgments
Bayazit Karaman, "Tracking Brain Activity with Semantic Complexity During Naturalistic Narrative Comprehension.".

## License
MIT License 
