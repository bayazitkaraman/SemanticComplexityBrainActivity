# Semantic Complexity / Semantic Variation and Brain Activity

This repository contains an organized Python pipeline for analyzing how GPT-2-based contextual semantic variation relates to fMRI activity during naturalistic narrative comprehension.

The project is organized so that data and virtual environments are **not** included in GitHub. You can add your local `data/` folder later and run the full pipeline from the beginning.

## Main goals

This version supports the revision analyses needed after reviewer feedback:

1. Use the full available dataset instead of a small subset.
2. Test whether GPT-2 PC1 is reducible to simple lexical/time baselines.
3. Fit a shared PCA space across stories so PC dimensions are comparable.
4. Compare GPT-2 layers instead of assuming the final layer is best.
5. Compare baseline-only, GPT2-only, and combined encoding models.
6. Generate updated figures for the revised manuscript.

## Folder structure

```text
SemanticComplexityBrainActivity/
│
├── configs/                  # story/subject lists
├── scripts/                  # runnable scripts
├── src/semantic_fmri/         # reusable utilities
├── data/                     # local data only, not pushed to GitHub
├── results/                  # generated outputs
├── docs/                     # project notes
└── archive/old_scripts/       # original scripts preserved for reference
```

## Setup

Create and activate your environment outside GitHub tracking:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

Install requirements:

```bash
pip install -r requirements.txt
pip install -e .
```

## Add your data

Put your local dataset under `data/`. This folder is intentionally ignored by Git.

Expected simple layout:

```text
data/
├── lucy/
│   ├── sub-XXX_task-lucy_bold.nii.gz
│   ├── task-lucy_events.tsv
│   └── lucy_transcript.txt
├── merlin/
│   ├── sub-XXX_task-merlin_bold.nii.gz
│   ├── task-merlin_events.tsv
│   └── merlin_transcript.txt
└── notthefallintact/
    ├── sub-XXX_task-notthefallintact_bold.nii.gz
    ├── task-notthefallintact_events.tsv
    └── notthefallintact_transcript.txt
```

## Build the full subject list

```bash
python scripts/build_subject_list.py --data-dir data --output configs/stories_all.py
```

A small old 30-subject example list is already kept in:

```text
configs/stories_small.py
```

## Quick test

Run a small test before launching the full analysis:

```bash
python scripts/run_revision_pipeline.py --quick --subject-list configs/stories_all.py
```

## Full revision pipeline

```bash
python scripts/run_revision_pipeline.py --subject-list configs/stories_all.py
```

This runs:

1. `semantic_baseline_regressors.py`
2. `shared_pca_regressors.py`
3. `gpt2_layer_pc_analysis.py`
4. `roi_encoding_model_comparison.py`
5. `figures_revision.py`

## Original main analysis

The original ROI/voxel correlation pipeline is kept as:

```bash
python scripts/run_main_analysis.py --subject-list configs/stories_all.py
```

The original figure-generation script is kept as:

```bash
python scripts/make_figures.py
```

## Important outputs

```text
results/diagnostics/pc1_vs_baselines_correlations.csv
results/regressors/shared_pca_explained_variance.csv
results/regressors/*_shared_pca_regressors.csv
results/layer_analysis/layer_overall_summary.csv
results/model_comparison/encoding_model_comparison.csv
results/model_comparison/encoding_model_deltas.csv
results/revision_figures/
```

## GitHub notes

Do not push:

- `.venv/`
- `data/`
- large `.nii.gz` files
- generated maps under `results/maps/`
- group-average NIfTI files

Recommended Git commands:

```bash
git init
git add .
git commit -m "Organize semantic fMRI project with revision analysis pipeline"
git branch -M main
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main
```
