# Analysis protocol v1

Date: 2026-10-11. This is a local analysis specification, not a registered preregistration. The original publication and its reported results were read before this document was written. The feature set below is fixed before fitting new association or prediction models in this project. Changes must be recorded with their reason and marked exploratory when driven by observed results.

## Research question

Do pre-training body composition, protein intake and energy intake add explanatory and out-of-sample predictive information beyond initial vastus lateralis thickness for thickness after 12 weeks of resistance training?

Scope: a secondary analysis of the 67 participants in Mobley et al. (2018), S2 raw data. This is a methods and reproducibility project, not a claim of a newly discovered physiological mechanism or a personal nutrition prescription.

## Outcomes and predictors

- Primary outcome: `post_vl_cm`, measured by ultrasound, in cm.
- Baseline adjustment: `pre_vl_cm`.
- Four additional primary features: `pre_lean_mass_kg`, `pre_fat_mass_kg`, `pre_protein_g_day`, `pre_energy_kcal_day`.
- Age and supplement-code adjustment are declared sensitivity analyses, not automatic feature selection.
- Secondary descriptive outcomes: absolute muscle thickness change, percentage change, DXA lean body mass change, squat 3RM change. Do not interpret lean mass as pure muscle mass.
- No primary feature is chosen by its univariate p-value or by correlation with the outcome.
- Do not add all biomarkers, all macronutrients, polynomial terms or interactions simply to improve apparent fit. Any later addition requires a documented scientific reason.

The raw source has no direct body-weight or height column. Use protein in g/day. Do not silently substitute DXA lean plus fat mass for body weight, and do not label protein per lean mass as protein per body weight.

## Sampling and cleaning

Each real `MPS` subject is one participant. Exclude source summary rows and legend rows. Keep original Excel row numbers. Do not increase the effective sample size by stacking PRE and POST as independent people.

The six primary fields are complete for all 67 participants. Keep 67 for the primary analysis. Three participants lack squat 3RM; only analyses that require squat should use the corresponding available subset. Do not drop people based on missing unused biomarkers. No extreme value is removed without source verification; any verified change is documented. The original XLSX remains unchanged.

## Association analysis, planned

Fit an ordinary linear model for POST thickness with PRE thickness and the four extra features. Report coefficients in meaningful units, their uncertainty, and model diagnostics. Use heteroskedasticity-robust HC3 standard errors if implemented and verified; do not substitute naive p-values silently. Assess collinearity, influential observations and residual behavior. Interpret coefficients as conditional associations in this dataset, not causal effects. The five correlated predictors and the sample size limit precision.

Sensitivity analyses add age and then supplement code as categorical indicators. The supplement codes can adjust group differences without knowing the names, but biological interpretation of named supplements awaits the verified codebook. Fit the same controls in both models of an adjusted comparison. Unknown labels must not be guessed from group counts or dietary means.

Avoid explaining a negative PRE-to-change correlation as evidence that smaller muscles inherently grow more: PRE is also part of the change calculation. Retain baseline-adjusted POST as the main modeling outcome.

## Prediction analysis, planned

Use the same participant-level splits for every model:

| Model | Inputs or rule | Purpose |
|---|---|---|
| Mean POST reference | Training-fold mean POST | Context only |
| Mean-change reference | Test PRE plus training-fold mean POST-minus-PRE | Simple growth-aware reference |
| Baseline model | PRE thickness only, linear regression | Main comparator |
| Extended linear model | PRE thickness plus the four primary baseline features | Incremental information |
| Extended ridge model | Same five features; regularization tuned within training data | Stabilized exploratory prediction |

Primary validation: repeated 5-fold cross-validation, 20 repeats, a fixed random seed of 20261011. Keep the participant ID in the split record. A repeat assigns every participant once to a test fold. Aggregate that repeat's out-of-fold predictions before computing MAE, RMSE and R-squared. Compare models using paired differences within the same repeats. Across-repeat variation is split sensitivity, not a conventional independent-sample confidence interval.

Ridge tuning: inner 5-fold validation within each outer training set, with a fixed candidate alpha grid `[0.1, 1, 10, 100]`. Select by inner validation error. Learn all scaling, encoding and any necessary future imputation inside the training folds. Primary data currently need no imputation. If categorical adjustment is used, fit encoding in training data and define how unseen groups are handled. Never select features or hyperparameters on all participants before cross-validation.

Report performance improvements relative to the PRE-only model and the mean-change reference, not only absolute POST R-squared. Much POST predictability may come from initial muscle size. Predicted growth is predicted POST minus observed PRE; evaluate this explicitly if discussed. Accept negative out-of-sample R-squared or no improvement as valid findings.

No independent external validation is currently available. Do not claim a validated general-use predictor, or turn cross-validation into an admission of clinical utility. A final held-out set of about 13 people would be unstable; use the declared cross-validation approach and report its limitations.

## Time availability and leakage

Only PRE information belongs in pre-training prediction. Exclude POST thickness, every POST biomarker, POST dietary intake, whole-intervention volume, derived growth, and original low/moderate/high response labels from the predictor matrix. Original response classes were defined from the same outcome; they are not independent baseline phenotypes.

POST diet and total volume may later be analyzed as clearly labeled process associations. Such analyses do not answer what can be predicted before training begins. Two 4-day food logs do not establish habitual intake throughout 12 weeks.

## Interpretation boundaries

All groups trained, so this file has no non-training control. Observed individual change contains measurement error and day-to-day variation. Single-cycle PRE/POST data cannot establish stable biological response heterogeneity or identify an individual's optimal protein dose. Restrict generalization to the represented young, previously untrained male sample under this training/supplement setting.

The mother study collected baseline, week-6 and week-12 nutrition, but this attachment has PRE and POST only. Do not manufacture week-6 data. Supplement-code names and POST supplement accounting are unresolved source questions. They do not block baseline descriptive EDA but limit interpretation of process nutrition and group effects.

## Deliverables and current stage

Stage 1 completed by `src/run_eda.py`: participant cleaning, raw-to-clean dictionary, source manifest, missingness, descriptive tables and figures. No association or prediction model has been fit in this stage. Model fitting, diagnostic checks and declared cross-validation are the next stage.

References:

1. Mobley et al. (2018), PLOS ONE, 13(4):e0195203. https://doi.org/10.1371/journal.pone.0195203
2. Mobley et al. (2017), Nutrients, 9(9):972. https://doi.org/10.3390/nu9090972
