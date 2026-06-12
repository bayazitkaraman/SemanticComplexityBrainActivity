# Results folder

Generated outputs will be saved here. Large NIfTI files, model-comparison intermediates,
regressor folders, null-draw files, and local maps are ignored by Git.

Public reproducibility outputs are kept in:

```text
results/final_tables/
results/final_figures/
```

The final tables include the primary GPT-2 encoding results plus the added
Journal of Neurolinguistics robustness checks: PC-count sensitivity, nuisance
strategy deltas, leave-one-story-out, circular-shift null summary, and
exploratory BERT embedding comparisons.
