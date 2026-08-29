# Semantic Complexity and Brain Activity

This repository contains the analysis code and aggregate results for examining
whether low-dimensional GPT-2 features improve prediction of fMRI responses
during naturalistic spoken narrative comprehension.

The final design uses all 13 eligible intact narratives from the public
Narratives dataset (`ds002345`). After motion quality control, the dataset
contains 567 subject-story scans from 277 participants.

## Final analysis

The central analysis uses 13 leave-one-story-out folds. In each fold:

1. One complete narrative is reserved for testing.
2. The feature scaler and story-balanced PCA are fitted to the other 12 stories.
3. Neural encoding coefficients are fitted to those same 12 stories.
4. The fitted model is evaluated on the complete omitted story.

Every training story receives equal total weight while all available TRs are
retained. The primary comparison adds five cumulative-context GPT-2 components
to a rich lexical-acoustic baseline.

The mean story-level increase in prediction correlation is `0.017398`
(`p = 0.007782`, two-sided story-level t test; 10/13 stories positive). A
story-and-participant holdout sensitivity gives a smaller positive effect of
`0.004500` (`p = 0.034289`; 10/13 stories positive).

Independent local GPT-2 features do not reliably improve cross-narrative
prediction (`Delta r = 0.004316`, `p = 0.292879`). The paired
cumulative-minus-independent difference is `0.013082` (`p = 0.036636`). The
complementary individual subject-story analysis is null (`Delta cv_r =
-0.001433`, `p = 0.849227`).

## Repository layout

```text
configs/                                           Story and participant configurations
docs/                                             Data and analysis documentation
scripts/                                          Analysis programs
src/semantic_fmri/                                Shared Python utilities
results/FINAL_RESULTS_SUMMARY.md                  Current result index
results/jnl_13_story_extension/loso/              Central cross-narrative outputs
results/jnl_13_story_context_comparison/           Cumulative-versus-local comparison
results/jnl_13_story_individual_local/             Individual subject-story outputs
results/jnl_13_story_auxiliary_embeddings/         BERT and GPT-2 layer checks
results/jnl_13_story_hemisphere_sensitivity/       Hemisphere sensitivity
results/jnl_13_story_fold_safe_normalization/      Normalization sensitivity
results/jnl_story_heterogeneity/original_pieman/   Matched-narrative comparison
```

## Setup

Create a virtual environment and install the package:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
```

Windows PowerShell activation:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux activation:

```bash
source .venv/bin/activate
```

## Data configuration

The tracked portable configuration contains 572 available subject-story rows
before motion QC:

```text
configs/stories_all.py
```

Map those rows to local fMRIPrep outputs and apply motion QC:

```bash
python scripts/build_subject_list_fmriprep.py --subject-list configs/stories_all.py --fmriprep-dir /path/to/fmriprep --output configs/stories_all_fmriprep_confounds.py
python scripts/motion_qc_from_confounds.py --subject-list configs/stories_all_fmriprep_confounds.py --filtered-output configs/stories_all_fmriprep_confounds_motionfiltered.py
```

The final motion-filtered file has 567 scans. Machine-specific configurations
are excluded from Git. See `configs/README.md` and `docs/DATA_DOWNLOAD.md` for
the full configuration chain and story counts.

## Timing and preprocessing

The analysis uses Gentle word-level alignments and a common story-relative
1.5-s grid. Configure the alignment directory before running the pipeline:

```powershell
$env:SEMANTIC_FMRI_TIMING="gentle"
$env:SEMANTIC_FMRI_GENTLE_DIR="D:\OpenNeuroData\narratives\stimuli\gentle"
$env:SEMANTIC_FMRI_CONFOUND_STRATEGY="none"
$env:SEMANTIC_FMRI_HIGH_PASS="none"
```

The reported analysis uses motion-QC-passed fMRIPrep MNI-space BOLD data
without additional nuisance regression or high-pass filtering.

## Local resources

The current scripts use one unified set of local intermediate resources:

```text
results/jnl_13_story_resources/rich_audio_regressors/
results/jnl_13_story_resources/roi_cache/
results/jnl_13_story_resources/hemisphere_roi_cache/
```

These directories are excluded from Git because they contain local paths or
participant-level derived data. Each directory must contain all 13 stories.
The validated local copy contains 13 regressor files and 567 ROI files in each
ROI cache.

## Run the analyses

Central cross-narrative analysis and representation comparison:

```bash
python scripts/jnl_13_story_loso.py --gentle-root /path/to/gentle --stage all
python scripts/jnl_13_story_context_comparison.py --gentle-root /path/to/gentle
python scripts/jnl_13_story_component_sensitivity.py
python scripts/jnl_13_story_transfer_figure.py
```

Complementary all-13 analyses:

```bash
python scripts/jnl_13_story_individual_local.py --gentle-root /path/to/gentle --stage all
python scripts/jnl_13_story_individual_direct_extension.py --gentle-root /path/to/gentle
python scripts/jnl_13_story_auxiliary_embeddings.py --gentle-root /path/to/gentle
python scripts/jnl_13_story_hemisphere_sensitivity.py --gentle-root /path/to/gentle
python scripts/jnl_13_story_fold_safe_normalization.py --gentle-root /path/to/gentle
python scripts/jnl_original_pieman_comparison.py --dataset-dir /path/to/narratives
```

All central programs default to the unified subject configuration, regressor
directory, and ROI cache described above.

## Current outputs

The current numerical index is `results/FINAL_RESULTS_SUMMARY.md`. Principal
machine-readable files include:

```text
results/jnl_13_story_extension/loso/cross_story_transfer_summary.csv
results/jnl_13_story_extension/loso/cross_story_transfer_story_means.csv
results/jnl_13_story_extension/loso/cross_story_participant_transfer_summary.csv
results/jnl_13_story_extension/loso/cross_story_temporal_shift_summary.csv
results/jnl_13_story_context_comparison/cross_story_direct_comparison_inference.csv
results/jnl_13_story_extension/component_count_sensitivity/component_count_summary.csv
results/jnl_13_story_individual_local/summary.csv
results/jnl_13_story_auxiliary_embeddings/representation_summary.csv
results/jnl_13_story_hemisphere_sensitivity/left_right_paired_summary.csv
results/jnl_13_story_fold_safe_normalization/fold_safe_summary.csv
```

## Interpretation

The results support a small cross-narrative contribution from a compact GPT-2
representation beyond the measured lexical and acoustic controls. They do not
establish a complete model of semantic comprehension, a uniquely localized
neural mechanism, or reliable prediction at the individual scan level.

Raw fMRI data, WAV files, participant-level time series, model caches, and
machine-specific paths are not distributed in this repository. The public fMRI
data are available under OpenNeuro accession `ds002345`.
