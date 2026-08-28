# Results Folder

Generated outputs are saved here. Raw or preprocessed neuroimaging files,
participant-level time series, ROI caches, model intermediates, regressor
directories, machine-specific paths, and WAV files are excluded from Git.

## Central public outputs

```text
results/jnl_13_story_extension/analysis_audit.md
results/jnl_13_story_extension/locked_analysis_specification.json
results/jnl_13_story_extension/loso/
results/jnl_13_story_context_comparison/
results/jnl_13_story_extension/component_count_sensitivity/
results/jnl_13_story_individual_local/
results/jnl_13_story_auxiliary_embeddings/
results/jnl_13_story_hemisphere_sensitivity/
results/jnl_13_story_fold_safe_normalization/
results/jnl_story_heterogeneity/original_pieman/
```

The `loso` directory contains aggregate results from the 13-fold
train-12/test-1 analysis, the story-and-participant holdout, the temporal-shift
null, ridge sensitivity, within-story comparisons, and the cross-narrative
figure. The component-count directory contains the 1--10 component sensitivity
analysis. The `original_pieman` directory contains aggregate outputs from the
exploratory matched-performance comparison; it does not contribute to the
13-story inference.

The `jnl_13_story_individual_local` directory contains the locked specification,
aggregate story-level results, mixed-effects summary, PCA audit, and sensitivity
results for the complementary individual analysis across the same 13 stories.
Large independent-embedding arrays and participant-level imaging derivatives
remain local.

The `jnl_13_story_context_comparison` directory contains cumulative-versus-local
results for the same 13 cross-narrative folds and the secondary within-story
analyses. The auxiliary-embedding, hemisphere, and normalization directories
contain the corresponding all-13-story sensitivity tables. Large raw embedding
arrays and hemisphere ROI caches remain local.

## Complementary outputs

```text
results/final_tables/
results/final_figures/
results/jnl_heldout_story_replication/
results/jnl_shared_response_context_validation/
results/jnl_fold_safe_subject_normalization/
results/jnl_shared_response_hemisphere_sensitivity/
results/jnl_shared_response_pc_count_sensitivity/
results/jnl_submission_reporting/
```

These directories preserve earlier subset-based shared-response,
participant-normalization, hemisphere, component-count, BERT, nuisance,
acoustic, and reporting analyses for the audit trail. They do not generate the
current manuscript tables; the active all-13 outputs are listed above.

Machine-specific configurations and participant-level time-series caches remain
local. Portable result tables needed to verify the reported analyses are
tracked.
