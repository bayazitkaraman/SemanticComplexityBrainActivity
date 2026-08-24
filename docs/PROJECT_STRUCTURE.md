# Project Structure

This repository is a reproducible analysis pipeline for studying how
low-dimensional GPT-2-derived predictors relate to fMRI responses during
naturalistic spoken narrative comprehension.

## Main folders

```text
configs/                                      Portable and generated story configurations
scripts/                                      Runnable analysis scripts
src/semantic_fmri/                            Reusable Python utilities
results/jnl_13_story_extension/loso/          Central cross-narrative outputs
results/jnl_story_heterogeneity/              Exploratory matched-narrative outputs
results/final_tables/                         Complementary and legacy result tables
results/final_figures/                        Generated manuscript figures
docs/                                         Analysis and data documentation
archive/old_scripts/                          Older scripts preserved for reference
data/                                         Local data only; excluded from Git
```

## Central workflow

The central analysis uses:

1. Gentle word-level forced alignments and a common story-relative 1.5-s grid.
2. Autoregressive cumulative-context GPT-2 embeddings at TR resolution.
3. A story-balanced scaler and PCA fitted to 12 narratives in each fold.
4. HRF-convolved lexical, acoustic, and GPT-2 predictors.
5. Neural encoding coefficients fitted on 12 narratives and evaluated on the
   complete omitted narrative.
6. Story-and-participant holdout and temporal-shift validation.
7. Ridge, component-count, within-story, ROI, and matched-performance checks.

## Central scripts

```text
scripts/jnl_13_story_loso.py
```

Runs the 13-fold story-balanced analysis, including full shared-response,
story-and-participant holdout, temporal-shift, and within-story outputs.

```text
scripts/jnl_13_story_component_sensitivity.py
```

Repeats all 13 folds with 1--10 retained GPT-2 components and refits the scaler,
PCA, predictor scaling, and neural coefficients inside every fold.

```text
scripts/jnl_13_story_transfer_figure.py
```

Creates the cross-narrative result figure from the aggregate output tables.

```text
scripts/jnl_original_pieman_comparison.py
```

Compares the scanner-recorded `piemanpni` retelling with the original `pieman`
performance while holding the reference fold's feature transform and neural
model fixed. This is exploratory and does not enter the 13-story inference.

## Supporting scripts

```text
scripts/build_subject_list_from_openneuro.py
scripts/build_subject_list_fmriprep.py
scripts/motion_qc_from_confounds.py
scripts/jnl_rich_audio_controls.py
scripts/jnl_pc_count_sensitivity.py
```

These scripts prepare local configurations, apply motion quality control,
construct the rich lexical-acoustic baseline, and create reusable ROI caches.

The historical `run_revision_pipeline.py` workflow and its associated scripts
remain available for the complementary individual-scan, BERT, layer,
hemisphere, nuisance-regression, and earlier robustness analyses. They are not
the central inferential workflow.

## Public outputs

Aggregate cross-narrative outputs are stored under:

```text
results/jnl_13_story_extension/loso/
results/jnl_13_story_extension/component_count_sensitivity/
results/jnl_story_heterogeneity/original_pieman/
```

Raw fMRI data, WAV files, participant-level time series, machine-specific
configurations, ROI caches, and large model intermediates are not included in
the repository.
