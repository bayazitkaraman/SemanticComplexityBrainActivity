# Analysis Notes

This project evaluates whether low-dimensional GPT-2-derived linguistic
predictors improve fMRI encoding during naturalistic spoken narrative
comprehension.

## Central analysis

The central analysis:

1. Includes all 13 intact Narratives tasks meeting the predefined run,
   duration, and participant criteria.
2. Aligns Gentle word timings and participant BOLD series to a common
   story-relative 1.5-s grid.
3. Fits a story-balanced scaler and PCA to 12 narratives in each fold.
4. Projects the omitted narrative without refitting the feature transform.
5. Compares a rich lexical-acoustic baseline with the same baseline plus five
   cumulative-context GPT-2 components.
6. Fits ROI encoding coefficients on the 12 training narratives and evaluates
   the complete omitted narrative.
7. Tests transfer across both stories and listeners by removing each tested
   participant from every available training-story group response.
8. Evaluates temporal-shift, ridge, component-count, within-story, ROI, and
   matched-performance sensitivity analyses.

The individual-scan, BERT, layer, hemisphere, fold-specific normalization, and
independent-chunk analyses use the same 13 stories as complementary checks.
They are not the central inferential basis of the manuscript.

## Interpretation

The main interpretation should remain modest:

> A compact GPT-2 representation adds small but reliable predictive information
> beyond a rich lexical-acoustic baseline and transfers across naturalistic
> narratives at the level of shared regional responses.

The results should not be interpreted as a complete model of semantic
comprehension, a complete prosodic or acoustic model, strong semantic decoding,
or evidence for a uniquely localized neural mechanism.
