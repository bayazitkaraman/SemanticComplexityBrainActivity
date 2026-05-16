# Project structure

```text
configs/                  story/subject lists
scripts/                  runnable scripts
src/semantic_fmri/         reusable functions
data/                     local data only, ignored by Git
results/                  generated outputs
docs/                     notes and documentation
archive/old_scripts/       old scripts preserved for reference
```

## Main workflow

```bash
python scripts/build_subject_list.py --data-dir data --output configs/stories_all.py
python scripts/run_revision_pipeline.py --quick --subject-list configs/stories_all.py
python scripts/run_revision_pipeline.py --subject-list configs/stories_all.py
```

## Reviewer-focused analyses

- `semantic_baseline_regressors.py`: PC1 vs lexical/time sanity checks.
- `shared_pca_regressors.py`: shared PCA across stories.
- `gpt2_layer_pc_analysis.py`: GPT-2 layer comparison.
- `roi_encoding_model_comparison.py`: baseline-only vs GPT2 vs combined models.
