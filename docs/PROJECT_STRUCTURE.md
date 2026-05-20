# Project Structure

This repository is organized as a reproducible analysis pipeline for studying how low-dimensional GPT-2 semantic predictors relate to fMRI responses during naturalistic spoken narrative comprehension.

## Main folders

```text
configs/                  Story and subject configuration files
scripts/                  Runnable analysis scripts
src/semantic_fmri/         Reusable Python utilities
results/final_tables/      Final manuscript and supplementary tables
results/final_figures/     Final manuscript and supplementary figures
docs/                     Project documentation
archive/old_scripts/       Older scripts preserved for reference
data/                     Local data only; not included in GitHub
```

## Main analysis workflow

The main analysis uses:

1. Gentle word-level forced alignments.
2. GPT-2 contextual embeddings at TR resolution.
3. Shared PCA across selected stories.
4. HRF-convolved lexical, temporal, semantic, and optional acoustic predictors.
5. Cross-validated ROI encoding models.
6. Final summary tables and figures.

## Main scripts

```text
scripts/semantic_baseline_regressors.py
```

Computes diagnostic relationships between GPT-2 PC1 and lexical/time baseline regressors.

```text
scripts/shared_pca_regressors.py
```

Extracts GPT-2 embeddings, fits shared PCA across stories, and saves TR-level regressors. When acoustic controls are enabled, this script also saves speech-density and audio-envelope predictors.

```text
scripts/gpt2_layer_pc_analysis.py
```

Runs the descriptive GPT-2 layer-wise alignment analysis.

```text
scripts/roi_encoding_model_comparison.py
```

Fits cross-validated ROI encoding models and compares baseline-only, GPT-2-only, combined lexical-plus-GPT-2, and optional acoustic-control models.

```text
scripts/final_stats_and_tables.py
```

Generates final manuscript and supplementary result tables.

```text
scripts/audit_results.py
```

Checks that expected final tables and figures exist and that key consistency checks pass.

```text
scripts/run_revision_pipeline.py
```

Runs the main analysis pipeline. The filename is historical; this is the current full analysis pipeline.

## Recommended run order

Primary analysis:

```bash
python scripts/run_revision_pipeline.py --subject-list configs/stories_all.py
python scripts/final_stats_and_tables.py
python scripts/audit_results.py
```

Acoustic-control analysis:

```bash
python scripts/run_revision_pipeline.py --subject-list configs/stories_all.py --include-acoustic
python scripts/final_stats_and_tables.py
python scripts/audit_results.py
```

If only acoustic-control code or regressors changed, this shorter rerun is usually sufficient:

```bash
python scripts/shared_pca_regressors.py --subject-list configs/stories_all.py
python scripts/roi_encoding_model_comparison.py --subject-list configs/stories_all.py --include-acoustic
python scripts/final_stats_and_tables.py
python scripts/audit_results.py
```

## Final outputs

Final tables are saved in:

```text
results/final_tables/
```

Final figures are saved in:

```text
results/final_figures/
```

Large intermediate outputs, raw data, WAV files, and neuroimaging files are not included in the repository.