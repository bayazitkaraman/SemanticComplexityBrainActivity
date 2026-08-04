# Held-out story replication audit

## Status

The held-out outcomes were kept separate from model development. The three
held-out story outcomes were not used to choose stories, ROIs, component count,
ridge alpha, or cross-validation gap.

The local locked specification was first written on August 2, 2026 at 10:37
EDT, before the held-out fMRI outcomes were scored. The first held-out result
files were generated at 11:15 EDT. The lock is recorded in
`locked_analysis_specification.json` (SHA-256:
`7B2C01AB580554A8CD0149E0017D3CABBAF2E424C6CCE424ACE85736505F6682`).
This was a local analysis lock, not a public preregistration.

The held-out outcomes had been generated before the final end-to-end timing
audit on August 3. The later corrections used actual story-relative volume
times and a common supported 1.5-s grid. They were outcome-independent, were
applied identically to development and held-out stories, and did not alter the
stories, predictors, ROIs, component count, regularization, folds, temporal
gap, or model contrast.

## Story selection and data completeness

The objective candidate scan required at least 20 usable participants, at
least 5 minutes' median duration, and a single BOLD run per participant.
Eligible stories were ranked by usable participant count and then median
duration. The scan was run with `--allow-missing-transcript` because the Gentle
alignments used for modeling were maintained in an external directory rather
than the dataset locations searched by the scanner. This explains why the
table's local transcript-candidate field is false although all 11 analyzed
stories had the required external Gentle and audio resources. The complete
sanitized candidate table is stored in
`results/jnl_heldout_story_replication/candidate_story_ranking.csv` and is
exported from the local diagnostic table by
`scripts/jnl_submission_reporting_tables.py`. Its first 11 eligible entries
were:

1. `shapessocial`
2. `shapesphysical`
3. `notthefallintact`
4. `milkyway`
5. `bronx`
6. `forgot`
7. `black`
8. `piemanpni`
9. `prettymouth`
10. `sherlock`
11. `merlin`

The original eight stories were the first eight entries. The held-out set
contains the next three eligible entries: `prettymouth`, `sherlock`, and
`merlin`.

- `prettymouth`: 40 participants, 450 story TRs, 17 ROIs
- `sherlock`: 36 participants, 700 retained story TRs, 17 ROIs
- `merlin`: 36 participants, 589 retained story TRs, 17 ROIs
- Total held-out scans: 112
- Missing or non-finite ROI outcomes: 0
- Motion-QC exclusions: 0 under the locked thresholds

The event onsets were used when extracting each participant's story segment
from the full BOLD run. Each volume was placed at its actual acquisition time
relative to story onset using the run's recorded TR. Participant ROI series
were then linearly interpolated to the common 1.5-s story grid. No extrapolation
was used: only time points supported by every participant were retained, and
the word, audio, and GPT-2 designs were restricted to the same effective grid.

## Model audit

- GPT-2 embeddings were computed autoregressively from cumulative preceding
  context within rolling windows of at most 1,024 tokens.
- The scaler and five-component PCA were fitted only to the original eight
  stories. The three held-out stories were projected without refitting.
- Recomputed discovery-story components reproduced the saved components with
  a minimum correlation greater than 0.9999996.
- The baseline contains lexical, speech-timing, waveform, spectral, pitch, and
  voicing controls.
- Predictors are standardized inside each training fold.
- Target detrending and scaling use training-fold estimates.
- Five forward-chaining time-series folds and a 10-TR gap were used.
- All 17 predefined ROIs were averaged; no outcome-driven ROI subset was used.

## Locked held-out result

The primary unit is one mean incremental cross-validated correlation per story,
averaged over the 17 ROIs.

| Analysis | Mean delta r | Story values | Directional evidence |
|---|---:|---|---|
| Full group | 0.0913 | 0.0685, 0.0878, 0.1175 | 3/3 positive; two-sided t-test p = .023 |
| Participant split-half | 0.0620 | 0.0352, 0.0649, 0.0858 | 3/3 positive |
| Leave-one-participant-out | 0.0271 | 0.0131, 0.0372, 0.0311 | 3/3 positive |

A 1,000-iteration circular-shift null gave `p = .003` for the full-group
held-out effect. The null mean was -0.0069 and its 95th percentile was 0.0453.

Across the original eight and held-out three stories, the full-group mean was
0.0593 (`p = .0009` by two-sided story-level t-test; 9/11 positive). The
participant split-half mean was 0.0437 (`p = .0013`; 10/11 positive).
The leave-one-participant-out mean was 0.0188 (`p = .0021`; 10/11 positive).
Absolute baseline, combined-model, and incremental correlations are stored in
`results/jnl_submission_reporting/absolute_prediction_story_means.csv`.

## Robustness checks

These checks were run after seeing the locked result and should be labeled as
sensitivity analyses.

- Adding an explicit silence control: full-group mean = 0.0928; split-half
  mean = 0.0638; all three stories positive.
- Zeroing GPT-2-component values during silent TRs before HRF convolution: full-group
  mean = 0.0939; split-half mean = 0.0651; all three stories positive.
- Ridge alpha 1, 10, 100, and 1,000: the full-group mean remained positive at
  every value (0.0944, 0.0913, 0.0561, and 0.0274).
- CV gaps of 5, 10, and 15 TRs: the full-group means were 0.0497, 0.0913, and
  0.0782. At gap 5, `merlin` was slightly negative (-0.0017), while the other
  two stories remained positive.
- For each tested setting, the mean across all three stories was positive for
  every one of the 50 participant split seeds.
- Estimating each participant's normalization parameters only from the
  training portion of each temporal fold gave held-out means of 0.0848,
  0.0578, and 0.0207 for full-group, split-half, and leave-one-participant-out
  analyses. All three story values remained positive in each analysis.

## Interpretation limits

The result supports a focused claim: low-dimensional cumulative-context GPT-2
features improve prediction of shared, stimulus-locked fMRI responses beyond
the detailed lexical and acoustic baseline, and this effect generalizes to
three previously unused narratives.

It does not establish that cumulative context is better than an independent-
chunk GPT-2 model. The direct contrast in the original eight stories was
test-dependent: the full-group two-sided t-test gave `p = .073`, the one-sided
Wilcoxon test gave `p = .039`, and the sign test gave `p = .145`. With no
prospectively recorded hierarchy among these development tests, the comparison
is treated as inconclusive. It was not part of the locked held-out test.
It also does not support ROI-specific localization: the held-out set has only
three stories, and no ROI subset was selected. The corrected original
individual-participant encoding result remains null and must be reported rather
than replaced by the shared-response analysis.

The held-out narratives are new stimuli, but their participants can overlap
with participants in the original Narratives dataset. The split-half and
leave-one-participant-out analyses establish cross-participant prediction
within each held-out story; they do not make the three-story dataset an entirely
new participant cohort.
