# Story Selection Criteria

The final analysis uses eight stories from the public Narratives fMRI dataset:

```text
black
bronx
forgot
milkyway
notthefallintact
piemanpni
shapesphysical
shapessocial
```

## Selection goals

The selected stories were chosen to support stable TR-level encoding analyses across multiple narratives and many subject--story scans.

The selection prioritized stories with:

1. Available fMRIPrep-preprocessed BOLD data.
2. Available transcript and event timing files.
3. Available Gentle word-level forced alignments.
4. Sufficient usable subject--story scans after motion quality control.
5. Story durations suitable for TR-level modeling.
6. Compatibility with shared PCA across stories.

## Why multiple stories were used

Using multiple stories improves the analysis in several ways:

- It reduces dependence on one narrative stimulus.
- It allows story-level variability to be examined.
- It supports shared PCA across different narrative contexts.
- It makes the encoding results more general than a single-story analysis.

## Motion quality control

The final sample was determined after motion quality control. Runs were excluded if they exceeded predefined framewise-displacement criteria. This produced the final set of 408 motion-QC-passed subject--story scans from 204 unique participants.

## Important principle

Stories were selected using data availability, timing quality, and sample-size criteria. They were not selected based on which stories produced the strongest GPT-2/fMRI effects.

This helps keep the analysis objective and avoids choosing stories based on the final encoding-model results.