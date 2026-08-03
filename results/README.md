# Results folder

Generated outputs will be saved here. Large NIfTI files, model-comparison intermediates,
regressor folders, null-draw files, and local maps are ignored by Git.

Public reproducibility outputs are kept in:

```text
results/final_tables/
results/final_figures/
results/jnl_heldout_story_replication/
results/jnl_shared_response_hemisphere_sensitivity/
```

The final tables include the primary GPT-2 encoding results plus the added
Journal of Neurolinguistics robustness checks: PC-count sensitivity, nuisance
strategy deltas, leave-one-story-out, circular-shift null summary,
exploratory BERT embedding comparisons, and the richer audio-control
sensitivity table.

The held-out and hemisphere folders contain portable numerical outputs and analysis records. Generated subject configurations with absolute local paths and cached ROI time series are excluded from Git.
