# Stage 2 implementation decisions

Date: 2026-10-13. Written before running the new models. The main question, five features, alpha grid, participant-level 20 x 5 outer cross-validation and seed are unchanged from `analysis_protocol.md`. The following implementation details resolve choices that were not explicit in that protocol.

1. Inner ridge selection uses shuffled 5-fold CV and negative mean squared error. Inner seed is `20261011 + outer_split_index + 1`. Alpha grid remains `[0.1, 1, 10, 100]`; the search is a `StandardScaler` plus `Ridge` pipeline. No preprocessing or tuning is fitted to outer test data.
2. Main metric summaries are means and SDs of 20 per-repeat pooled out-of-fold metrics. Each repeat's predictions come from five different held-out models. These pooled metrics are not identical to averaging fold metrics, so export both. Every model uses the same outer splits. Split SD and min/max describe sensitivity to partitioning of this one dataset, not a population confidence interval. Repeated appearances of the same 67 participants never count as independent new people.
3. Sensitivity comparisons adjust baseline and extended linear models for age, then age plus supplement code. Codes are categorical; their names remain unresolved. Scaler/one-hot encoding is fitted in each training fold. No named supplement effect is interpreted.
4. OLS inference uses the documented HC3 sandwich covariance. Coefficient intervals and p-values use a t reference with `n - p` residual degrees of freedom; the joint test of four added features uses an approximate HC3 Wald F reference. These finite-sample references are approximations, not exact heteroskedastic finite-sample guarantees. Because statsmodels is not available in the execution environment, implement HC3 using NumPy/SciPy and verify it against independent delete-one least-squares fits. Coefficients are expressed per 0.1 cm initial thickness, 10 kg lean mass, 10 kg fat mass, 25 g/day protein and 500 kcal/day energy. Centering is only for full-sample inference; it is not reused in CV.
5. The block test covers all four added features jointly. Individual p-values are descriptive, unadjusted for multiplicity. Avoid claims based on one p-value. No feature selection follows inference results.
6. Diagnose VIF, residual distribution, residual-versus-fitted behavior, leverage, Cook's distance and leave-one-participant-out coefficient sensitivity. Screening thresholds (`4/n` for Cook's distance, `2p/n` for leverage) flag observations for review, not deletion. Keep all participants in primary models.
7. Report both POST R-squared and derived-change R-squared. Predicted change equals predicted POST minus observed PRE; MAE and RMSE are algebraically identical for the two targets, but their R-squared denominators differ. Explicitly verify this to avoid misinterpreting good prediction of initial size as good prediction of growth.
8. No random forest, XGBoost, feature-selection search or new interactions are added after seeing results. No new data are collected or merged. Results are secondary analysis of an existing training/supplement intervention, not an effect of changing someone's protein intake.

Technical references:

- HC3 definition: https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.HC3_se.html
- scikit-learn 1.8 GridSearchCV: https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.GridSearchCV.html
- Cross-validation and training-only transformations: https://scikit-learn.org/1.8/modules/cross_validation.html
