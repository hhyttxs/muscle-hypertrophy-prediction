# Stage 2 model results

Secondary analysis of 67 participants. All five primary predictors were declared before fitting new models.

| Model | Mean OOF MAE (cm) | Mean OOF RMSE (cm) | Mean POST R² | Mean change R² |
|---|---:|---:|---:|---:|
| Mean POST reference | 0.2806 | 0.3287 | -0.038 | -0.818 |
| Mean-change reference | 0.1913 | 0.2479 | 0.410 | -0.033 |
| Initial thickness only | 0.1861 | 0.2377 | 0.457 | 0.049 |
| Initial + composition + nutrition | 0.1949 | 0.2489 | 0.404 | -0.043 |
| Same features, tuned ridge | 0.1966 | 0.2493 | 0.403 | -0.046 |

Extended OLS minus baseline OLS mean MAE: +0.0088 cm. Tuned ridge minus baseline: +0.0105 cm.

Under this fixed specification, neither the extended linear model nor tuned ridge improved mean held-out MAE over the initial-thickness-only model. This is not evidence that nutrition has no physiological effect.

Metrics pool the five out-of-fold partitions per repeat, then summarize 20 repeats. This is not the same as averaging fold R². The repeats reuse the same people; split SD is not a population confidence interval.

Ridge scaling and tuning used only outer training data with inner 5-fold validation. The same splits were used for every model. Both age and age-plus-categorical-group sensitivity comparisons are exported.

HC3 covariance was implemented from the documented formula and verified by independent delete-one least-squares fits. t-reference intervals and block Wald F tests are approximate; individual p-values are not multiplicity adjusted.

The reported coefficients are conditional associations, not intervention effects. High predictability of POST size does not establish high predictability of growth. No external validation, personal prescription, or stable responder-class claim is made.

Raw data: Mobley et al. (2018), PLOS ONE 13(4):e0195203, S2. https://doi.org/10.1371/journal.pone.0195203. New analysis code is project-specific and AI-assisted.
