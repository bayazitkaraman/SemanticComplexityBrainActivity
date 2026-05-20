# Semantic Complexity and Brain Activity

This repository contains a reproducible Python pipeline for analyzing how low-dimensional GPT-2 contextual semantic predictors relate to fMRI responses during naturalistic spoken narrative comprehension.

The final manuscript analysis uses the public **Narratives** fMRI dataset, Gentle word-level alignments, GPT-2 contextual embeddings, shared PCA-derived semantic regressors, HRF convolution, ROI-level encoding models, motion-quality-control checks, and acoustic-envelope sensitivity analyses.

Data files, virtual environments, WAV files, and large neuroimaging outputs are intentionally **not** stored in this repository.

---

## Project goals

This project supports the final Brain and Language submission analyses:

1. Use the full selected Narratives dataset instead of a small subset.
2. Align spoken words to fMRI TRs using Gentle forced-alignment files.
3. Extract GPT-2 contextual embeddings at TR resolution.
4. Fit shared PCA across stories so GPT-2 PC dimensions are comparable.
5. Compare lexical/time baseline models, GPT-2-only models, and combined lexical-plus-GPT-2 models.
6. Evaluate whether GPT-2 PC1--PC5 improves prediction beyond lexical and temporal baseline features.
7. Test whether effects are robust to motion-quality-control and nuisance-regression sensitivity checks.
8. Add acoustic-envelope and speech-density sensitivity controls.
9. Generate final tables and figures for the manuscript and supplementary material.

---

## Repository structure

```text
SemanticComplexityBrainActivity/
│
├── configs/                  # Story/subject configuration files
├── scripts/                  # Runnable analysis scripts
├── src/semantic_fmri/         # Reusable package utilities
├── results/final_tables/      # Final manuscript/supplement tables
├── results/final_figures/     # Final manuscript/supplement figures
├── docs/                     # Notes and documentation
├── archive/old_scripts/       # Older scripts preserved for reference
├── data/                     # Local data only; not pushed to GitHub
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

The editable install is important because the scripts import utilities from:

```text
src/semantic_fmri/
```

---

## Data requirements

The repository does **not** include raw fMRI data, fMRIPrep outputs, WAV files, or OpenNeuro/DataLad data.

The final analysis used the public **Narratives** dataset from OpenNeuro/DataLad:

```text
OpenNeuro accession: ds002345
```

The selected stories are:

```text
black
bronx
forgot
milkyway
notthefallintact
piemanpni
shapesphysical
shapessocial
```

The exact local data layout can vary depending on how the dataset was downloaded. The pipeline uses the paths defined in:

```text
configs/stories_all.py
```

Each story/subject entry should point to the local BOLD image, events file, transcript file, and, when available, fMRIPrep confounds file.

---

## Gentle timing files

The final manuscript analysis used Gentle word-level forced alignments.

Set the Gentle alignment directory before running the manuscript pipeline:

Windows PowerShell:

```powershell
$env:SEMANTIC_FMRI_TIMING="gentle"
$env:SEMANTIC_FMRI_GENTLE_DIR="D:\OpenNeuroData\narratives\stimuli\gentle"
```

macOS/Linux:

```bash
export SEMANTIC_FMRI_TIMING="gentle"
export SEMANTIC_FMRI_GENTLE_DIR="/path/to/narratives/stimuli/gentle"
```

Expected Gentle layout:

```text
gentle/
├── black/align.json
├── bronx/align.json
├── forgot/align.json
├── milkywayoriginal/align.json
├── notthefallintact/align.json
├── piemanpni/align.json
├── shapesphysical/align.json
└── shapessocial/align.json
```

The code maps the story name `milkyway` to the Gentle folder `milkywayoriginal`.

---

## Manuscript analysis settings

The primary manuscript analysis used:

```text
Gentle word-level timing
fMRIPrep-preprocessed MNI-space BOLD images
motion-quality-control exclusion
no additional nuisance regression
no high-pass filter
```

Use these environment settings for the primary analysis:

Windows PowerShell:

```powershell
$env:SEMANTIC_FMRI_TIMING="gentle"
$env:SEMANTIC_FMRI_GENTLE_DIR="D:\OpenNeuroData\narratives\stimuli\gentle"
$env:SEMANTIC_FMRI_CONFOUND_STRATEGY="none"
$env:SEMANTIC_FMRI_HIGH_PASS="none"
```

macOS/Linux:

```bash
export SEMANTIC_FMRI_TIMING="gentle"
export SEMANTIC_FMRI_GENTLE_DIR="/path/to/narratives/stimuli/gentle"
export SEMANTIC_FMRI_CONFOUND_STRATEGY="none"
export SEMANTIC_FMRI_HIGH_PASS="none"
```

`SEMANTIC_FMRI_CONFOUND_STRATEGY="none"` means that the analysis uses motion-QC-passed fMRIPrep data without additional nuisance regression. Motion6 and motion24 are used only as sensitivity analyses.

Supported nuisance-regression strategies include:

```text
none
motion6
motion24
motion24_wmcsf
acompcor6
full
```

---

## Acoustic-envelope sensitivity analysis

The acoustic-control analysis adds:

```text
speech_duration_sec
mean_word_duration_sec
pause_fraction
articulation_rate
word_rate
audio_rms
audio_abs_mean
```

The speech-timing variables are derived from Gentle word timings. The waveform-envelope features are extracted from WAV files when available.

Set the audio directory before running acoustic controls:

Windows PowerShell:

```powershell
$env:SEMANTIC_FMRI_AUDIO_DIR="data/openneuro/ds002345/stimuli"
```

macOS/Linux:

```bash
export SEMANTIC_FMRI_AUDIO_DIR="data/openneuro/ds002345/stimuli"
```

Expected WAV filenames include:

```text
black_audio.wav
bronx_audio.wav
forgot_audio.wav
milkywayoriginal_audio.wav
notthefallintact_audio.wav
piemanpni_audio.wav
shapesphysical_audio.wav
shapessocial_audio.wav
```

The acoustic-control analysis is a sensitivity analysis, not the primary planned comparison.

---

## Build or update the subject list

If needed, build the subject/story configuration file:

```bash
python scripts/build_subject_list.py --data-dir data --output configs/stories_all.py
```

The final manuscript analyses use:

```text
configs/stories_all.py
```

---

## Quick test

Run a quick test before launching the full analysis:

```bash
python scripts/run_revision_pipeline.py --quick --subject-list configs/stories_all.py
```

---

## Full primary pipeline

Run the primary analysis:

```bash
python scripts/run_revision_pipeline.py --subject-list configs/stories_all.py
python scripts/final_stats_and_tables.py
python scripts/audit_results.py
```

The audit should end with:

```text
Audit finished: all expected final files were found.
```

---

## Full pipeline with acoustic controls

Run the acoustic-envelope and speech-density sensitivity analysis:

```bash
python scripts/run_revision_pipeline.py --subject-list configs/stories_all.py --include-acoustic
python scripts/final_stats_and_tables.py
python scripts/audit_results.py
```

If only the shared regressors or acoustic-control code changed, the following shorter rerun is usually sufficient:

```bash
python scripts/shared_pca_regressors.py --subject-list configs/stories_all.py
python scripts/roi_encoding_model_comparison.py --subject-list configs/stories_all.py --include-acoustic
python scripts/final_stats_and_tables.py
python scripts/audit_results.py
```

---

## Main scripts

```text
scripts/semantic_baseline_regressors.py
    Computes diagnostics comparing GPT-2 PC1 with lexical/time baselines.

scripts/shared_pca_regressors.py
    Extracts GPT-2 embeddings, fits shared PCA, creates TR-level regressors,
    and optionally adds acoustic/speech-density controls.

scripts/gpt2_layer_pc_analysis.py
    Performs descriptive GPT-2 layer-wise alignment analysis.

scripts/roi_encoding_model_comparison.py
    Fits cross-validated ROI encoding models and compares model families.

scripts/final_stats_and_tables.py
    Generates final manuscript and supplementary tables.

scripts/audit_results.py
    Checks that expected final tables and figures are present and internally consistent.

scripts/run_revision_pipeline.py
    Runs the main revision-analysis pipeline.
```

---

## Important outputs

Intermediate outputs:

```text
results/diagnostics/pc1_vs_baselines_correlations.csv
results/regressors/shared_pca_explained_variance.csv
results/regressors/*_shared_pca_regressors.csv
results/regressors/shared_pc1_vs_baselines_correlations.csv
results/layer_analysis/
results/model_comparison/
```

Final manuscript/supplement outputs:

```text
results/final_tables/
results/final_figures/
```

Key final tables include:

```text
results/final_tables/table1_dataset_story_counts.csv
results/final_tables/table2_primary_model_comparison.csv
results/final_tables/table3_primary_paired_model_delta_stats.csv
results/final_tables/table4_roi_delta_stats_combined_pc1_to_pc5_vs_baseline.csv
results/final_tables/table5_layer_overall_summary.csv
results/final_tables/table7_sensitivity_model_comparison_by_strategy.csv
results/final_tables/table8_motion_correlation_with_gpt2_delta.csv
```

---

## Reproducibility notes

The main manuscript result compares:

```text
baseline_only
vs.
combined_pc1_to_pc5
```

where:

```text
baseline_only = lexical/time predictors
combined_pc1_to_pc5 = lexical/time predictors + GPT-2 PC1--PC5
```

The acoustic sensitivity analysis compares:

```text
baseline_acoustic_only
vs.
combined_acoustic_pc1_to_pc5
```

where the acoustic baseline includes lexical, temporal, speech-density, and waveform-envelope predictors.

The primary performance metric is:

```text
cross-validated prediction correlation (cv_r)
```

Cross-validated R² is computed as a secondary diagnostic metric, but it is not used for primary interpretation because it is unstable and often negative for short naturalistic fMRI time series.

---

## Citation and data availability

Raw fMRI data, WAV files, virtual environments, and large intermediate outputs are not included in this repository. The fMRI data are publicly available through the Narratives dataset on OpenNeuro/DataLad under accession:

```text
ds002345
```

This repository contains the analysis code, configuration files, final result tables, and generated manuscript figures needed to reproduce the final analysis outputs, assuming the user has local access to the required fMRI, transcript, alignment, confound, and audio files.