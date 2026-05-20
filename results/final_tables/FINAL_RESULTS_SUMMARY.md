# Final Results Summary

## Dataset

| story            |   n_subject_story_rows |   n_unique_subjects | bold_source   | has_confounds_file   |
|:-----------------|-----------------------:|--------------------:|:--------------|:---------------------|
| black            |                     45 |                  45 | fmriprep_mni  | True                 |
| bronx            |                     47 |                  47 | fmriprep_mni  | True                 |
| forgot           |                     46 |                  46 | fmriprep_mni  | True                 |
| milkyway         |                     53 |                  53 | fmriprep_mni  | True                 |
| notthefallintact |                     55 |                  55 | fmriprep_mni  | True                 |
| piemanpni        |                     46 |                  46 | fmriprep_mni  | True                 |
| shapesphysical   |                     58 |                  58 | fmriprep_mni  | True                 |
| shapessocial     |                     58 |                  58 | fmriprep_mni  | True                 |
| TOTAL            |                    408 |                 204 | fmriprep_mni  | True                 |


## Primary model comparison

| model               |   cv_r_mean |   cv_r_median |   cv_r_std |   cv_r_sem |   cv_r_count |   cv_r2_mean |   cv_r2_median |   cv_r2_std |   cv_r2_sem |   cv_r2_count |
|:--------------------|------------:|--------------:|-----------:|-----------:|-------------:|-------------:|---------------:|------------:|------------:|--------------:|
| combined_pc1_to_pc5 |  0.0253566  |    0.021579   |   0.136216 | 0.00163559 |         6936 |     -7.39511 |      -2.30534  |    15.1529  |   0.181946  |          6936 |
| combined_pc1        |  0.0177226  |    0.0145179  |   0.13436  | 0.0016133  |         6936 |     -4.67386 |      -1.27247  |    11.0638  |   0.132847  |          6936 |
| baseline_only       |  0.0102337  |    0.00702432 |   0.136635 | 0.00164061 |         6936 |     -3.35606 |      -1.00621  |     8.83513 |   0.106086  |          6936 |
| gpt2_pc1_to_pc5     |  0.00750007 |    0.00376465 |   0.13669  | 0.00164128 |         6936 |     -4.88244 |      -1.37342  |    12.1051  |   0.145349  |          6936 |
| gpt2_pc1            | -0.0261873  |   -0.0259645  |   0.13656  | 0.00163971 |         6936 |     -2.2348  |      -0.400164 |     8.18293 |   0.0982549 |          6936 |


## Planned paired model delta statistics

| comparison                          | model               | baseline      |    n |        mean |       median |       std |         sem |    ci95_low |    ci95_high |    t_stat |   p_ttest_two_sided |   wilcoxon_stat |   p_wilcoxon_greater |
|:------------------------------------|:--------------------|:--------------|-----:|------------:|-------------:|----------:|------------:|------------:|-------------:|----------:|--------------------:|----------------:|---------------------:|
| combined_pc1_to_pc5 - baseline_only | combined_pc1_to_pc5 | baseline_only | 6936 |  0.0151229  |  0.00597314  | 0.108014  | 0.00129696  |  0.0125572  |  0.0176701   |  11.6603  |         3.96293e-31 |     1.39245e+07 |          3.04422e-30 |
| combined_pc1 - baseline_only        | combined_pc1        | baseline_only | 6936 |  0.00748896 |  0.000421474 | 0.0614201 | 0.000737489 |  0.00609094 |  0.00900888  |  10.1547  |         4.65105e-24 |     1.32284e+07 |          3.15667e-13 |
| gpt2_pc1_to_pc5 - baseline_only     | gpt2_pc1_to_pc5     | baseline_only | 6936 | -0.00273362 | -0.00354854  | 0.13398   | 0.00160874  | -0.00586562 |  0.000516462 |  -1.69923 |         0.0893213   |     1.16088e+07 |          0.994105    |
| gpt2_pc1 - baseline_only            | gpt2_pc1            | baseline_only | 6936 | -0.036421   | -0.0183558   | 0.155538  | 0.00186759  | -0.0400455  | -0.0327577   | -19.5016  |         1.67159e-82 |     8.98627e+06 |          1           |


## Participant-level robustness check

| comparison                                               |   n |       mean |    median |       std |        sem |    ci95_low |   ci95_high |   t_stat |   p_ttest_two_sided |   wilcoxon_stat |   p_wilcoxon_greater |
|:---------------------------------------------------------|----:|-----------:|----------:|----------:|-----------:|------------:|------------:|---------:|--------------------:|----------------:|---------------------:|
| participant-averaged combined_pc1_to_pc5 - baseline_only | 204 | 0.00588806 | 0.0112935 | 0.0682923 | 0.00478141 | -0.00326905 |   0.0148744 |  1.23145 |             0.21958 |           12627 |           0.00504348 |


## Top ROI deltas: combined PC1-PC5 minus baseline

| roi                                          |   n |       mean |     median |    ci95_low |   ci95_high |   p_wilcoxon_greater |   q_wilcoxon_greater_fdr |
|:---------------------------------------------|----:|-----------:|-----------:|------------:|------------:|---------------------:|-------------------------:|
| Middle Temporal Gyrus, anterior division     | 408 | 0.0216124  | 0.00658299 | 0.0105139   |   0.0325834 |          0.000428847 |               0.00350849 |
| Superior Temporal Gyrus, posterior division  | 408 | 0.0200483  | 0.00707581 | 0.00703852  |   0.033011  |          0.00362391  |               0.00560058 |
| Frontal Pole                                 | 408 | 0.0190624  | 0.00607449 | 0.00833211  |   0.0299523 |          0.00025198  |               0.00350849 |
| Angular Gyrus                                | 408 | 0.0166604  | 0.00660209 | 0.00666738  |   0.0272873 |          0.00259366  |               0.00489914 |
| Inferior Frontal Gyrus, pars triangularis    | 408 | 0.0158008  | 0.00663176 | 0.00532107  |   0.0264614 |          0.00298875  |               0.00508088 |
| Temporal Fusiform Cortex, anterior division  | 408 | 0.0151595  | 0.00459604 | 0.00394413  |   0.0272056 |          0.0126112   |               0.0139166  |
| Superior Temporal Gyrus, anterior division   | 408 | 0.0151443  | 0.00322576 | 0.00450327  |   0.0257924 |          0.013098    |               0.0139166  |
| Supramarginal Gyrus, anterior division       | 408 | 0.0147416  | 0.0050079  | 0.00543287  |   0.0245223 |          0.000837364 |               0.00350849 |
| Temporal Fusiform Cortex, posterior division | 408 | 0.0147059  | 0.0071211  | 0.00510412  |   0.0248183 |          0.000937269 |               0.00350849 |
| Supramarginal Gyrus, posterior division      | 408 | 0.0142192  | 0.00469981 | 0.00453265  |   0.0243874 |          0.00463755  |               0.00656986 |
| Cingulate Gyrus, anterior division           | 408 | 0.0141933  | 0.00746284 | 0.00457588  |   0.0243189 |          0.00190327  |               0.00489914 |
| Parahippocampal Gyrus, anterior division     | 408 | 0.0140745  | 0.00448497 | 0.00321054  |   0.0250818 |          0.00681133  |               0.0082709  |
| Cingulate Gyrus, posterior division          | 408 | 0.013622   | 0.0065241  | 0.00415615  |   0.0237122 |          0.00103191  |               0.00350849 |
| Inferior Frontal Gyrus, pars opercularis     | 408 | 0.0133523  | 0.00646479 | 0.00301758  |   0.0242604 |          0.00238613  |               0.00489914 |
| Middle Temporal Gyrus, posterior division    | 408 | 0.0133299  | 0.00488606 | 0.00375917  |   0.022843  |          0.00571681  |               0.00747583 |
| Parahippocampal Gyrus, posterior division    | 408 | 0.012061   | 0.00753044 | 0.00219971  |   0.022381  |          0.00229412  |               0.00489914 |
| Precuneous Cortex                            | 408 | 0.00930213 | 0.00404293 | 0.000208692 |   0.0185936 |          0.0245469   |               0.0245469  |


## Layer summary

|   layer |   mean_abs_r |      mean_r |    n |
|--------:|-------------:|------------:|-----:|
|       4 |     0.126918 |  0.00448367 | 6936 |
|       1 |     0.125831 |  0.00765994 | 6936 |
|       3 |     0.125718 |  0.00454819 | 6936 |
|       5 |     0.12563  |  0.00440111 | 6936 |
|       2 |     0.125364 |  0.00595175 | 6936 |
|       0 |     0.124617 |  0.00701753 | 6936 |
|       6 |     0.124496 |  0.00566852 | 6936 |
|      12 |     0.123089 | -0.0108865  | 6936 |
|       7 |     0.122993 |  0.00542668 | 6936 |
|       8 |     0.122736 |  0.00578374 | 6936 |
|      11 |     0.122383 |  0.0114957  | 6936 |
|       9 |     0.121872 |  0.0080008  | 6936 |
|      10 |     0.12184  |  0.011037   | 6936 |


## Top ROIs for best layer

|   layer | roi                                          |   mean_abs_r |       mean_r |   n |
|--------:|:---------------------------------------------|-------------:|-------------:|----:|
|       4 | Temporal Fusiform Cortex, anterior division  |    0.153667  | -0.0131598   | 408 |
|       4 | Superior Temporal Gyrus, posterior division  |    0.151581  |  0.00114597  | 408 |
|       4 | Middle Temporal Gyrus, anterior division     |    0.145161  |  0.00904875  | 408 |
|       4 | Inferior Frontal Gyrus, pars opercularis     |    0.138518  | -0.000960933 | 408 |
|       4 | Angular Gyrus                                |    0.12947   |  0.0277429   | 408 |
|       4 | Frontal Pole                                 |    0.129197  | -0.0218258   | 408 |
|       4 | Middle Temporal Gyrus, posterior division    |    0.127588  |  0.0200539   | 408 |
|       4 | Supramarginal Gyrus, posterior division      |    0.126941  |  0.0397797   | 408 |
|       4 | Temporal Fusiform Cortex, posterior division |    0.125682  | -0.00537924  | 408 |
|       4 | Supramarginal Gyrus, anterior division       |    0.125247  |  0.0290979   | 408 |
|       4 | Superior Temporal Gyrus, anterior division   |    0.124865  | -0.0131797   | 408 |
|       4 | Parahippocampal Gyrus, anterior division     |    0.124384  | -0.0192642   | 408 |
|       4 | Inferior Frontal Gyrus, pars triangularis    |    0.123838  |  0.00269344  | 408 |
|       4 | Parahippocampal Gyrus, posterior division    |    0.111799  | -0.0122189   | 408 |
|       4 | Cingulate Gyrus, anterior division           |    0.111063  |  0.00677094  | 408 |
|       4 | Cingulate Gyrus, posterior division          |    0.111038  |  0.0184922   | 408 |
|       4 | Precuneous Cortex                            |    0.0975593 |  0.00738549  | 408 |