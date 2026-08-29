# Final All-13 Results Summary

This file indexes the aggregate results used by the current manuscript and
supplement.

## Dataset

All 13 intact narratives meeting the run, duration, and participant criteria
entered the analysis. The portable configuration has 572 subject-story rows
before motion QC. The final analysis has 567 scans from 277 unique participants.

| Story | Scans |
|---|---:|
| shapessocial | 58 |
| shapesphysical | 58 |
| notthefallintact | 55 |
| milkyway | 53 |
| bronx | 47 |
| forgot | 46 |
| black | 45 |
| piemanpni | 46 |
| prettymouth | 40 |
| sherlock | 36 |
| merlin | 36 |
| 21styear | 24 |
| tunnel | 23 |
| **Total** | **567** |

## Central cross-narrative result

The rich lexical-acoustic baseline was compared with the same baseline plus
five cumulative-context GPT-2 components. In every fold, the feature transform,
predictor scaling, and neural encoding coefficients were fitted to 12 stories
and evaluated on the complete thirteenth.

| Analysis | Mean Delta r | Positive stories | Two-sided t-test p |
|---|---:|---:|---:|
| Story held out | 0.017398 | 10/13 | 0.007782 |
| Story and participant held out (exploratory sensitivity) | 0.004500 | 10/13 | 0.034289 |

The exploratory 1,000-iteration temporal-shift sensitivity gave an empirical
one-sided `p = 0.000999`. The primary five-component result remained positive across the
reported ridge sensitivities. Results with 3 through 10 components were also
positive; component count was not retuned after observing those sensitivities.

Primary sources:

```text
results/jnl_13_story_extension/loso/cross_story_transfer_summary.csv
results/jnl_13_story_extension/loso/cross_story_transfer_story_means.csv
results/jnl_13_story_extension/loso/cross_story_participant_transfer_summary.csv
results/jnl_13_story_extension/loso/cross_story_temporal_shift_summary.csv
results/jnl_13_story_extension/loso/ridge_sensitivity/ridge_sensitivity_summary.csv
results/jnl_13_story_extension/component_count_sensitivity/component_count_summary.csv
```

## Representation comparison

Independent local GPT-2 features did not improve cross-narrative prediction
reliably (`mean Delta r = 0.004316`, two-sided `p = 0.292879`). The paired
cumulative-minus-independent difference was `0.013082` (`p = 0.036636`). This
comparison evaluates two complete representation pipelines and does not by
itself isolate a single mechanism of discourse context.

Source:

```text
results/jnl_13_story_context_comparison/cross_story_direct_comparison_inference.csv
```

## Complementary individual result

The all-13 individual subject-story analysis was null (`mean Delta cv_r =
-0.001433`, 9/13 positive stories, two-sided `p = 0.849227`; 567 scans). Its
alternative baseline, temporal-gap, and pooled-PCA settings were also null.

Source:

```text
results/jnl_13_story_individual_local/summary.csv
results/jnl_13_story_individual_local/crossed_mixed_effects.csv
```

## Additional sensitivities

All-13 BERT, GPT-2 layer, hemisphere, and fold-specific normalization outputs
are stored in:

```text
results/jnl_13_story_auxiliary_embeddings/
results/jnl_13_story_hemisphere_sensitivity/
results/jnl_13_story_fold_safe_normalization/
```

These analyses are complementary or exploratory and do not replace the central
cross-narrative test. The matched `pieman` analysis is likewise exploratory and
does not contribute to the 13-story inference.

## Audit records

Story inclusion, equal weighting, implementation checks, and the immutable
specification checksum are recorded in:

```text
docs/STORY_INCLUSION.md
results/jnl_13_story_extension/analysis_audit.md
results/jnl_13_story_extension/locked_analysis_specification.json
```
