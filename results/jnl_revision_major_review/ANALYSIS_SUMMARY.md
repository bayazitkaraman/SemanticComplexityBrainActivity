# Journal of Neurolinguistics Major Revision Analyses

Generated for manuscript JNL-D-26-00113 on 2026-06-29. These outputs document the additional robustness and sensitivity analyses used in the revised manuscript and supplementary material.

## Primary Inference

- Legacy scan-level inference remains positive: mean scan-level delta cv_r = 0.0151, 95% CI [0.0083, 0.0218], t(407) = 4.35, p = 1.74e-5.
- Participant-averaged evidence is weaker: mean = 0.0059, 95% CI [-0.0033, 0.0155], t(203) = 1.23, p = 0.220; one-sided Wilcoxon p = 0.0050.
- Crossed mixed-effects model with participant and story random intercepts is marginal for the original alpha=10 final-layer PC1-PC5 model: estimate = 0.0164, 95% CI approximately [-0.0012, 0.0340], p = 0.068.

Manuscript interpretation: the scan-level test is not presented as the sole inference. The revised claim is modest and reports scan-level, participant-level, and crossed mixed-effects evidence.

## Context Test

Compared the existing independent final-layer GPT-2 TR chunks with a cumulative-context layer-12 model. The cumulative model used prior story tokens up to GPT-2's context window and pooled the current TR tokens.

- Existing independent final layer: mean delta = 0.0151.
- Cumulative-context final layer: mean delta = 0.0131, 95% CI [0.0063, 0.0199], scan-level p = 2.15e-4.
- Paired difference, cumulative context minus existing independent final layer: mean = -0.0020, 95% CI [-0.0090, 0.0050], p = 0.572.

Manuscript interpretation: cumulative context does not improve prediction over independent TR chunks. The revised text scales back discourse-context claims and frames the result as local GPT-2-derived linguistic predictors, not direct evidence for accumulated narrative-context tracking.

## Layer 4 Full Encoding Test

The layer-wise result raised the question of whether layer 4 should replace the final layer. A full PC1-PC5 encoding model was run for independent GPT-2 layer 4.

- Independent layer 4 PC1-PC5: mean delta = 0.0084, 95% CI [0.0009, 0.0156], scan-level p = 0.029.
- Paired difference, layer 4 minus existing final layer: mean = -0.0068, 95% CI [-0.0126, -0.0010], p = 0.0217.

Manuscript interpretation: layer 4 does not outperform the final layer in the full encoding model. The earlier layer-wise PC1 correlation is described as descriptive only.

## Ridge Alpha Sensitivity

Alpha sensitivity for combined PC1-PC5 minus baseline:

- alpha=0.1: mean = -0.0045, p = 0.330.
- alpha=1: mean = 0.0069, p = 0.114.
- alpha=10: mean = 0.0151, p = 1.74e-5.
- alpha=100: mean = 0.0126, p = 2.29e-5.
- alpha=1000: mean = 0.0328, p = 9.86e-32.

Manuscript interpretation: the main result is not unique to alpha=10, but the effect size is alpha-sensitive. The revised text states alpha=10 as the main working setting and includes sensitivity results rather than claiming alpha is fully optimized.

## PC Count Sensitivity

Crossed mixed-effects models across PC counts show that PC1-PC5 is not uniquely strongest under participant/story random effects. PC1-PC3 is significant in the crossed model; PC1-PC5 is marginal.

Manuscript interpretation: PC1-PC5 is framed as the main compact representation used in the submitted analysis, with PC-count sensitivity reported transparently. The revised text avoids implying that PC1-PC5 was definitively selected by independent model optimization.

## Story-Level Heterogeneity

Story descriptors were added: duration/TR count, empty-TR fraction, word rate, mean words per TR, pause fraction, articulation rate, and lexical diversity.

- Story effects remain heterogeneous: strongest for bronx, black, and forgot; weakest/negative for piemanpni and milkyway.
- With only eight stories, descriptor correlations are descriptive. Duration/TR count correlated positively with story-level mean delta, but this is treated as exploratory.

Manuscript interpretation: a small supplementary table was added, and the revised text avoids strong general claims across all narratives.

## Motion/Absolute Prediction Distribution

Fraction of scans with positive GPT-2 delta:

- Motion-QC only: 0.596.
- Motion6, no high-pass: 0.559.
- Motion24, no high-pass: 0.561.

Fraction of scans where both baseline and combined absolute cv_r were negative:

- Motion-QC only: 0.294.
- Motion6, no high-pass: 0.417.
- Motion24, no high-pass: 0.498.

Manuscript interpretation: the GPT-2 increment remains positive for a majority of scans, but stronger denoising lowers absolute prediction and increases below-zero absolute cv_r. The revised discussion states this directly.

## Hemisphere Sensitivity

Left and right Harvard-Oxford masks were split at MNI x=0 using the same atlas threshold (>0) and the same extraction/model settings.

- Left hemisphere: mean delta = 0.0171, 95% CI [0.0101, 0.0239], scan-level p = 2.23e-6; mixed-effects p = 0.0278.
- Right hemisphere: mean delta = 0.0163, 95% CI [0.0095, 0.0230], scan-level p = 3.04e-6; mixed-effects p = 0.0663.
- Left minus right: mean = 0.0008, 95% CI [-0.0042, 0.0056], p = 0.762.

Manuscript interpretation: the effect is not strongly lateralized in this ROI-level analysis. This is useful as a sensitivity check, but it is not interpreted as precise anatomical localization.
