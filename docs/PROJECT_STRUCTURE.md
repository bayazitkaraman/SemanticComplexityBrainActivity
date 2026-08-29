# Project Structure

The repository is organized around the final all-13-story analysis.

## Inputs

```text
configs/stories_all.py
configs/stories_all_fmriprep_confounds.py
configs/stories_all_fmriprep_confounds_motionfiltered.py
results/jnl_13_story_resources/rich_audio_regressors/
results/jnl_13_story_resources/roi_cache/
results/jnl_13_story_resources/hemisphere_roi_cache/
```

The portable raw-BIDS configuration is tracked. Machine-specific fMRIPrep
configurations and participant-level caches remain local.

## Central scripts

```text
scripts/jnl_13_story_loso.py
scripts/jnl_13_story_context_comparison.py
scripts/jnl_13_story_component_sensitivity.py
scripts/jnl_13_story_transfer_figure.py
```

These scripts run the train-12/test-1 cross-narrative analysis, compare
cumulative and local representations, evaluate component-count sensitivity,
and create the principal result figure.

## Complementary scripts

```text
scripts/jnl_13_story_individual_local.py
scripts/jnl_13_story_individual_direct_extension.py
scripts/jnl_13_story_auxiliary_embeddings.py
scripts/jnl_13_story_hemisphere_sensitivity.py
scripts/jnl_13_story_fold_safe_normalization.py
scripts/jnl_original_pieman_comparison.py
```

These scripts run the individual, alternative-embedding, hemisphere,
normalization, and matched-narrative analyses across the final story set.

## Public outputs

```text
results/FINAL_RESULTS_SUMMARY.md
results/jnl_13_story_extension/loso/
results/jnl_13_story_extension/component_count_sensitivity/
results/jnl_13_story_context_comparison/
results/jnl_13_story_individual_local/
results/jnl_13_story_auxiliary_embeddings/
results/jnl_13_story_hemisphere_sensitivity/
results/jnl_13_story_fold_safe_normalization/
results/jnl_story_heterogeneity/original_pieman/
```

Only aggregate and reproducibility outputs are tracked. Raw imaging data,
participant-level result rows, embeddings, and derived ROI caches are excluded.
