# Results Folder

Generated outputs are saved here. Raw or preprocessed neuroimaging files,
participant-level time series, ROI caches, model intermediates, regressor
directories, machine-specific paths, and WAV files are excluded from Git.

## Central public outputs

```text
results/jnl_13_story_extension/analysis_audit.md
results/jnl_13_story_extension/locked_analysis_specification.json
results/jnl_13_story_extension/loso/
results/jnl_13_story_extension/component_count_sensitivity/
results/jnl_story_heterogeneity/original_pieman/
```

The `loso` directory contains aggregate results from the 13-fold
train-12/test-1 analysis, the story-and-participant holdout, the temporal-shift
null, ridge sensitivity, within-story comparisons, and the cross-narrative
figure. The component-count directory contains the 1--10 component sensitivity
analysis. The `original_pieman` directory contains aggregate outputs from the
exploratory matched-performance comparison; it does not contribute to the
13-story inference.

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

These directories preserve the earlier shared-response, individual-scan,
participant-normalization, hemisphere, component-count, BERT, nuisance,
acoustic, and reporting analyses used as complementary checks and for the audit
trail. They are not the primary cross-narrative result.

Machine-specific configurations and participant-level time-series caches remain
local. Portable result tables needed to verify the reported analyses are
tracked.
