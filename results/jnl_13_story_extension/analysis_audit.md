# Thirteen-story cross-narrative analysis audit

## Scope

The candidate scan identified 19 fMRI task labels. The two temporally scrambled
`notthefall` controls were excluded because they are not intact narratives.
`pieman` and `schema` failed the single-run or duration criteria, and
`slumlordreach` and `lucy` had fewer than 20 usable participants. All 13
remaining intact, eligible stories were analyzed. After the same motion-QC
rules used in the revision, the analysis contains 567 subject-story scans from
277 unique participants.

The locked specification was written before outcomes for `21styear` and
`tunnel` were computed. Its SHA-256 checksum is:

`3B988713ECBE5038ECC8DA8381D22EC14FFCA9B407B412A000574531A7BB1E28`

## Equal treatment of stories

No beginning, middle, ending, or other segment was selected. Every available
TR is retained. In each leave-one-story-out fold, scaler moments and the PCA
covariance are averaged across the 12 training stories, so every story has the
same total contribution regardless of duration. Ridge sample weights are also
normalized so that every training story has the same total loss weight. This
uses more information and introduces fewer arbitrary choices than truncating
all stories to the shortest duration.

The omitted story does not contribute to PCA fitting, predictor scaling, or
neural encoding coefficients. The complete omitted story is used only for
evaluation.

## Primary cross-story result

The rich lexical-acoustic baseline was compared with the same baseline plus
five cumulative-context GPT-2 components. Neural encoding coefficients were
trained on 12 stories and applied to the complete thirteenth story.

- Mean story-level change in prediction correlation: 0.017398
- Median: 0.023603
- Positive stories: 10/13
- Story-level two-sided one-sample t test: p = 0.007782
- Story-level one-sided Wilcoxon test: p = 0.001709
- Story-level one-sided sign test: p = 0.046143

The result remains positive at ridge alpha values of 1, 10, 100, and 1000;
the corresponding mean changes range from 0.016918 to 0.023179, and all four
two-sided t-test p values are below 0.009.

A 1,000-iteration temporal null circularly shifted only the omitted story's
five already HRF-convolved GPT-2 component series while leaving its rich
baseline aligned. The aligned mean was 0.017398, the null mean was -0.011404,
and the null 95th percentile was -0.005453. None of the 1,000 null means
reached the observed value (empirical p = 0.000999).

## Story-and-participant holdout sensitivity

This sensitivity analysis was added after the primary cross-story outcomes
were known and must be described as exploratory. For every participant in an
omitted story, that participant was removed from all 12 training-story group
responses before model fitting. The model was then evaluated on that person's
response to the complete omitted story. Participant estimates were averaged
within story before inference.

- Mean story-level change in prediction correlation: 0.004500
- Median: 0.006713
- Positive stories: 10/13
- Story-level two-sided one-sample t test: p = 0.034289
- Story-level one-sided Wilcoxon test: p = 0.019897
- Story-level one-sided sign test: p = 0.046143

This result supports transfer across both narratives and listeners. It is
smaller than the shared-response result, as expected for noisier individual
responses.

## Secondary within-story result

When only the PCA transform was fitted leave-one-story-out and neural
coefficients were still estimated within the omitted story, the full-group
mean change was 0.036335. The two-sided t test was p = 0.051652 and the
one-sided Wilcoxon test was p = 0.047119. This analysis is secondary because it
does not test cross-story transfer of the neural encoding coefficients.

## Implementation checks

- All 13 folds use the same predefined story list and model settings.
- All 7,877 available TRs are represented in the feature cache.
- PCA is fitted to the other 12 stories separately in every fold.
- Predictor scaling and neural coefficients are fitted to training stories only.
- The participant sensitivity removes the test participant from every training target.
- No nonfinite prediction scores occur in the primary or participant-level outputs.
- The primary result is restored at the locked alpha of 10 after sensitivity runs.

Analysis script SHA-256:

`E72499BD77A303A5F79EF100A7EE16CEB8E1A86B9CADE07029FE0AC524C5312F`
