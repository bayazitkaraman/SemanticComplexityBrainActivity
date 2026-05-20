# Analysis Notes

This project evaluates whether low-dimensional GPT-2 contextual semantic predictors improve fMRI encoding during naturalistic spoken narrative comprehension.

The main analysis goals are:

1. Use the selected full Narratives dataset stories with all usable subject--story scans.
2. Align spoken words to fMRI TRs using Gentle forced-alignment files.
3. Fit shared PCA across stories so GPT-2 component dimensions are comparable.
4. Compare lexical/time baseline models, GPT-2-only models, and combined lexical-plus-GPT-2 models.
5. Evaluate GPT-2 layer-wise alignment as a descriptive analysis.
6. Test motion-quality-control and nuisance-regression sensitivity.
7. Test acoustic-envelope and speech-density sensitivity controls.

The main interpretation should remain modest:

> A compact GPT-2 contextual semantic representation adds small but reliable predictive information beyond lexical and temporal baseline features during naturalistic narrative comprehension.

The analysis should not be interpreted as a complete model of semantic comprehension, a complete prosodic/acoustic model, or a strong semantic decoding result.