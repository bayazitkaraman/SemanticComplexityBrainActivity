# Held-out story replication audit

## Status

The held-out outcomes were kept separate from model development. The three
held-out story outcomes were not used to choose stories, ROIs, component count,
ridge alpha, or cross-validation gap.

The locked specification was written before the held-out fMRI outcomes were
scored. It is recorded in `locked_analysis_specification.json`. This was a
local analysis lock, not a public preregistration.

## Story selection and data completeness

The original eight stories were the first eight entries in the pre-existing
objective candidate ranking. The held-out set contains the next three eligible
entries in that same ranking: `prettymouth`, `sherlock`, and `merlin`.

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

- GPT-2 embeddings were computed causally from cumulative preceding context.
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

It does not establish that cumulative context is significantly better than an
independent-chunk GPT-2 model. The direct contrast in the original eight stories
was not significant, and that contrast was not part of the locked held-out test.
It also does not support ROI-specific localization: the held-out set has only
three stories, and no ROI subset was selected. The corrected original
individual-participant encoding result remains null and must be reported rather
than replaced by the shared-response analysis.

The held-out narratives are new stimuli, but their participants can overlap
with participants in the original Narratives dataset. The split-half and
leave-one-participant-out analyses establish cross-participant prediction
within each held-out story; they do not make the three-story dataset an entirely
new participant cohort.
