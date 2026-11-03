# Model card

## Question and intended scope

Incremental prediction of 12-week vastus lateralis thickness beyond PRE thickness, using PRE DXA lean mass, fat mass, protein and energy. Intended use is reproducible research and learning from one public cohort, not a production prediction service. No serialized deployment model is distributed.

Data: Mobley et al. (2018), PLOS ONE 13(4):e0195203, S2 raw XLSX; 67 young previously untrained men. https://doi.org/10.1371/journal.pone.0195203 . All primary fields are complete. Source checksum and selection rules are in `data/source_manifest.json`.

## Inputs and outcome

Output: POST ultrasound thickness (cm). Mandatory baseline predictor: PRE ultrasound thickness (cm). Extended predictors: PRE lean/fat mass (kg), protein (g/day) and energy (kcal/day). Age and supplement categories appear in matched sensitivity analyses. POST measurements, whole-intervention volume and outcome-derived response labels are excluded from baseline prediction.

The model requires initial ultrasound thickness; it is not a lifestyle questionnaire model, a disease-risk model or a predictor of whole-body muscle growth.

## Evaluation and results

Same participant-level 20 x 5 outer CV splits for all models; inner 5-fold ridge selection; training-only scaling/encoding. Per-repeat metrics pool held-out partitions, then summarize repeats. Mean MAE: baseline 0.1861 cm, extended OLS 0.1949 cm, extended ridge 0.1966 cm. Corresponding mean observed-change R²: 0.049, -0.043, -0.046. Added features did not improve mean error. Repeat counts are not independent evidence of superiority. No external cohort was evaluated.

Full-sample association models use HC3 covariance with approximate t/F references. The four added features jointly have approximate p=0.589. Individual p-values are unadjusted and coefficients are not dietary intervention effects.

## Known gaps

- Small, selected sample. Subgroup performance and generalization to women, trained people or older adults are not established.
- Short self-reported baseline diet is not continuous intake; actual body weight, height and sleep are absent from the modeled source fields.
- Supplement-code name mapping and POST supplement accounting remain unresolved and are not guessed.
- DXA lean mass is not pure skeletal muscle. Observed ultrasound changes include measurement/day-to-day variation.
- This analysis has no non-training control or independent repeated training cycle. Stable biological responder classes and individual causal training responses are not identified.

The findings do not establish that nutrition lacks physiological effects and do not support personal coaching or protein prescriptions.

## Verification and provenance

Checks cover all outer/inner split containment, unique held-out predictions, independent metrics, SciPy baseline coefficients, independent delete-one HC3 covariance, and one direct reconstruction of nested ridge. See `results/models/verification_checks.json`.

The original investigators collected the data. New scripts were generated with AI assistance for this project. User-side running reproduced reported primary metrics and checks; the user's files were not remotely inspected. Personal role statements should reflect the work actually understood and performed.
