# Semantic Complexity and Brain Activity

This repository contains a reproducible Python pipeline for analyzing how low-dimensional GPT-2-derived linguistic predictors relate to fMRI responses during naturalistic spoken narrative comprehension.

The current Journal of Neurolinguistics revision uses the public **Narratives** fMRI dataset, Gentle word-level alignments, autoregressive cumulative-context GPT-2 embeddings, PCA-derived predictors, TR-sampled HRF convolution, story-relative resampling of BOLD targets, cross-narrative shared-response ROI encoding models, participant-separated validation, and rich lexical and acoustic controls.

Data files, virtual environments, WAV files, and large neuroimaging outputs are intentionally **not** stored in this repository.

## Current revision analysis

The primary result is a 13-fold leave-one-story-out analysis of the response shared across listeners. In each fold, the scaler, PCA, predictor scaling, and neural encoding coefficients are fitted to 12 narratives and evaluated on the complete omitted narrative. Every training story receives equal total weight while all available TRs are retained. A stricter story-and-participant holdout removes each tested listener from every training-story group response before neural-model fitting.

After downloading the dataset and fMRIPrep derivatives, first create the local motion-filtered subject list:

```bash
python scripts/build_subject_list_fmriprep.py --subject-list configs/stories_all.py --fmriprep-dir /path/to/fmriprep --output configs/stories_all_fmriprep_confounds.py
python scripts/motion_qc_from_confounds.py --subject-list configs/stories_all_fmriprep_confounds.py --filtered-output configs/stories_all_fmriprep_confounds_motionfiltered.py
```

After preparing the rich-audio regressors and ROI caches described below, run the central 13-story analysis:

```bash
python scripts/jnl_13_story_loso.py --gentle-root /path/to/gentle --stage all
python scripts/jnl_13_story_transfer_figure.py
```

`jnl_13_story_loso.py` computes story-balanced leave-one-story-out PCA, the primary train-12/test-1 neural transfer analysis, the story-and-participant holdout, the 1,000-iteration temporal-shift null, and secondary within-story results. The primary setting uses five final-layer GPT-2 components and ridge `alpha=10`; separate output directories hold the ridge sensitivity runs at 1, 100, and 1,000. `jnl_13_story_transfer_figure.py` creates the three-panel cross-story result figure. Generated machine-specific configurations and ROI caches remain local because they contain local paths or derived imaging data.

The 13-story analysis specification is
`results/jnl_13_story_extension/locked_analysis_specification.json`
(SHA-256 `3B988713ECBE5038ECC8DA8381D22EC14FFCA9B407B412A000574531A7BB1E28`).
The inclusion logic, equal-weight implementation, principal results, and
implementation checks are documented in
`results/jnl_13_story_extension/analysis_audit.md`.

For every shared-response analysis, cached participant ROI series are placed at their actual acquisition times relative to story onset and linearly interpolated to the common 1.5-s story grid. Only time points supported by every participant in a story are retained; the stimulus designs are restricted to the same effective grid.

---

## Project goals

This project supports the final Journal of Neurolinguistics submission analyses:

1. Include all 13 intact narratives meeting the objective run, duration, and participant criteria.
2. Align spoken words and participant BOLD volumes to a common story-relative 1.5-s grid.
3. Extract GPT-2 contextual embeddings at TR resolution.
4. In each fold, fit a story-balanced scaler and PCA to 12 narratives and project the omitted narrative without refitting.
5. Test whether cumulative-context GPT-2 PC1--PC5 adds prediction beyond a rich lexical/acoustic baseline.
6. Fit neural encoding coefficients on the same 12 narratives and evaluate the complete omitted narrative.
7. Test participant generalization by removing the tested listener from all training-story responses.
8. Evaluate temporal specificity, regularization, within-story prediction, and ROI sensitivity.
9. Retain the corrected individual-scan, independent-chunk, and BERT analyses as complementary checks.
10. Generate final tables and figures for the manuscript and supplementary material.

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
pip install -r requirements-lock.txt
pip install -e .
```

`requirements-lock.txt` records the exact direct-dependency versions used for
the revision. `requirements.txt` remains the unpinned convenience list.

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

The 13 intact eligible narratives are:

```text
21styear
black
bronx
forgot
merlin
milkyway
notthefallintact
piemanpni
prettymouth
shapesphysical
shapessocial
sherlock
tunnel
```

The objective candidate scan required at least 20 usable participants, at
least 5 minutes' median duration, and a single BOLD run per participant.
The two temporally scrambled `notthefall` controls were excluded because the
analysis concerns intact narratives. `pieman` and `schema` failed the run or
duration criteria, while `slumlordreach` and `lucy` had fewer than 20 usable
participants. All 13 remaining stories had the external Gentle alignments and
audio resources needed for modeling and all 13 entered the leave-one-story-out
analysis. See `scripts/scan_story_candidates.py` and
`results/jnl_13_story_extension/analysis_audit.md` for the complete inclusion
record.

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
├── shapessocial/align.json
├── prettymouth/align.json
├── sherlock/align.json
└── merlin/align.json
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
prettymouth_audio.wav
sherlock_audio.wav
merlin_audio.wav
```

This envelope-only control belongs to the earlier individual-scan pipeline. The current shared-response analysis uses the richer lexical/acoustic baseline described below.

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

## Legacy individual-scan pipeline

Run the earlier individual-scan pipeline retained for audit and supplementary comparisons:

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

## Legacy pipeline with envelope controls

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

Run the richer waveform/spectral/coarse-voicing audio-control analysis used in the Journal of Neurolinguistics revision:

```bash
python scripts/jnl_rich_audio_controls.py --subject-list configs/stories_all_fmriprep_confounds_motionfiltered.py --gentle-root /path/to/narratives/stimuli/gentle
```

Windows PowerShell example:

```powershell
python scripts\jnl_rich_audio_controls.py --subject-list configs\stories_all_fmriprep_confounds_motionfiltered.py --gentle-root "D:\OpenNeuroData\narratives\stimuli\gentle"
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

scripts/jnl_additional_robustness.py
    Creates Journal of Neurolinguistics robustness tables, including
    participant-level, story-level, leave-one-story-out, and nuisance-strategy checks.

scripts/jnl_pc_count_sensitivity.py
    Evaluates combined lexical-plus-GPT-2 models using PC1 through PC1--PC10.

scripts/jnl_circular_shift_null.py
    Runs circular-shift null tests for the temporal alignment of GPT-2 PC regressors.

scripts/static_embedding_regressors.py
    Creates static or contextual BERT embedding PCA regressors for exploratory
    embedding-model comparisons.

scripts/jnl_rich_audio_controls.py
    Runs the richer waveform/spectral/coarse-voicing audio-control sensitivity
    analysis and writes table21_rich_audio_control_sensitivity.csv.

scripts/jnl_heldout_story_replication.py
    Runs the locked held-out cumulative-context shared-response analysis,
    including participant split-half and leave-one-participant-out tests.

scripts/jnl_heldout_story_sensitivity.py
    Runs sensitivity analyses for the held-out shared-response specification.

scripts/jnl_shared_response_context_validation.py
    Runs the eight-story development shared-response analysis after mapping each
    participant's ROI series to the common story-relative grid.

scripts/jnl_fold_safe_subject_normalization.py
    Repeats the full-group, split-half, and leave-one-participant-out analyses
    with participant normalization estimated inside each temporal training fold.

scripts/jnl_shared_response_hemisphere_sensitivity.py
    Repeats the central shared-response analyses separately for left and right
    hemisphere ROI sets and evaluates paired right-minus-left differences.

scripts/jnl_shared_response_pc_count_sensitivity.py
    Preserves the locked PC1--PC5 regressors, appends development-fitted
    PC6--PC10, and repeats the full-group and cross-participant analyses while
    requiring exact reproduction of the locked PC5 story results.

scripts/jnl_submission_reporting_tables.py
    Combines development and held-out absolute baseline, combined-model, and
    incremental correlations and computes pooled leave-one-participant-out
    story-level inference for manuscript reporting.
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
results/jnl_shared_response_context_validation/
results/jnl_heldout_story_replication/
results/jnl_fold_safe_subject_normalization/
results/jnl_shared_response_hemisphere_sensitivity/
results/jnl_shared_response_pc_count_sensitivity/
results/jnl_submission_reporting/
```

Key final tables include:

```text
results/final_tables/table1_dataset_story_counts.csv
results/final_tables/table2_primary_model_comparison.csv
results/final_tables/table3_primary_paired_model_delta_stats.csv
results/final_tables/table3c_participant_level_primary_delta_stats.csv
results/final_tables/table4_roi_delta_stats_combined_pc1_to_pc5_vs_baseline.csv
results/final_tables/table5_layer_overall_summary.csv
results/final_tables/table7_sensitivity_model_comparison_by_strategy.csv
results/final_tables/table8_motion_correlation_with_gpt2_delta.csv
results/final_tables/table9_pc_count_sensitivity_scan_stats.csv
results/final_tables/table9b_pc_count_sensitivity_participant_stats.csv
results/final_tables/table10_motion_strategy_scan_level_delta_tests.csv
results/final_tables/table11_leave_one_story_out_primary_delta.csv
results/final_tables/table12_circular_shift_null_summary.csv
results/final_tables/table13_embedding_model_comparison_summary.csv
results/final_tables/table13b_embedding_model_paired_scan_comparisons.csv
results/final_tables/table21_rich_audio_control_sensitivity.csv
```

---

## Reproducibility notes

The current central shared-response analysis compares:

```text
baseline_rich_audio
vs.
combined_rich_audio_gpt2_pc1_to_pc5
```

where the rich baseline includes lexical/time predictors, acoustic-envelope/speech-density controls, and additional waveform, spectral, pitch, and coarse voicing-related predictors. The primary target is the full-story-standardized ROI response averaged across listeners within each story. This participant-level full-story standardization is distinct from the fold-specific detrending and scaling of the resulting shared encoding target. A sensitivity analysis additionally estimates participant means and scales only from the training portion of each temporal fold. The earlier individual subject--story comparison is retained as a corrected null audit result, not as the central evidence.

The primary performance metric is:

```text
cross-validated prediction correlation (cv_r)
```

Cross-validated R-squared is computed as a secondary diagnostic metric, but it is not used for primary interpretation because it is unstable and often negative for short naturalistic fMRI time series.

---

## Citation and data availability

Raw fMRI data, WAV files, virtual environments, and large intermediate outputs are not included in this repository. The fMRI data are publicly available through the Narratives dataset on OpenNeuro/DataLad under accession:

```text
ds002345
```

This repository contains the analysis code, portable configuration templates, final result tables, and generated manuscript figures needed to reproduce the final analysis outputs, assuming the user has local access to the required fMRI, transcript, alignment, confound, and audio files. Machine-specific configurations and derived imaging caches remain local.
