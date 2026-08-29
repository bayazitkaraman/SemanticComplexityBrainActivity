# Thirteen-Story Cross-Narrative Analysis Audit

## Dataset

The candidate scan identified 19 fMRI task labels. The two temporally scrambled
`notthefall` controls were excluded because they are not intact narratives.
`pieman` and `schema` failed the single-run or duration criteria, and
`slumlordreach` and `lucy` had fewer than 20 usable participants. All 13
remaining eligible intact stories entered the analysis.

The portable configuration contains 572 subject-story rows. Applying the
reported motion thresholds leaves 567 scans from 277 participants.

## Cross-narrative design

Every available TR is retained. In each fold, scaler moments and the PCA
covariance are averaged across the 12 training stories, so each story makes the
same total contribution regardless of duration. Ridge sample weights likewise
give every training story the same total loss weight.

The omitted story does not contribute to PCA fitting, predictor scaling, or
neural encoding coefficients. The complete omitted story is used only for
evaluation.

## Central result

The rich lexical-acoustic baseline was compared with the same baseline plus
five cumulative-context GPT-2 components.

- Mean story-level change in prediction correlation: 0.017398
- Median: 0.023603
- Positive stories: 10/13
- Story-level two-sided one-sample t test: p = 0.007782
- Story-level one-sided Wilcoxon test: p = 0.001709
- Story-level one-sided sign test: p = 0.046143

The result remains positive at ridge alpha values of 1, 10, 100, and 1000.

## Sensitivity analyses

The story-and-participant holdout removes each tested participant from every
training-story group response before fitting and evaluates that participant in
the omitted narrative.

- Mean story-level change in prediction correlation: 0.004500
- Median: 0.006713
- Positive stories: 10/13
- Story-level two-sided one-sample t test: p = 0.034289

A 1,000-iteration temporal-shift null gave an empirical one-sided
`p = 0.000999`. These analyses are treated as exploratory sensitivities.

## Implementation checks

- All 13 folds use the same story list and model settings.
- All 7,877 available TRs are represented in the cumulative feature cache.
- PCA is fitted to the other 12 stories separately in every fold.
- Predictor scaling and neural coefficients are fitted to training stories only.
- The participant sensitivity removes the test participant from every training target.
- No story is selected or excluded according to effect direction.
- No nonfinite prediction scores occur in the central outputs.

Current analysis script SHA-256:

`E2DE4E81EF2B6CBE9C0F94873DA39718A26F6A529A521AE3D7059E24BF1B71CF`
