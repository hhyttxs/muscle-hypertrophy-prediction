# Reproducibility record

## User-side evidence

The user supplied a successful Windows execution log on 2026-10-21: Python 3.12.10, sklearn 1.8.0, NumPy 2.3.5 and SciPy 1.17.0. The reported main MAEs match the project run (0.1860867014971304, 0.1948827361330371 and 0.1966044729749788 cm), and all reported verification flags were true. The block p-value differs only in final floating-point digits. This supports reported execution and metric reproducibility; the user's files were not remotely inspected and it is not external validation.

## Byte-hash difference

Project-side LF CSV SHA-256:

`72033c47ad7c7cd820833c7363d3369c25de5d629947c19d380ea4246f829213`

User's reported CSV SHA-256:

`b7d3e82962e04e462db9b0006d3edc94f92f1e039eb467f719779597e6350601`

Converting only project-side LF line endings to CRLF reproduces the latter hash exactly. This explains the reported byte-hash difference without changing CSV values. A byte hash alone is not a semantic data comparison. Version 3 explicitly sets `lineterminator="\n"` in all CSV exports and LF in `.gitattributes`.

## Verification coverage

Automated checks include participant selection and primary completeness; source/derived thickness changes; all 100 outer disjoint splits; all 500 inner contained splits; one held-out prediction per participant/model/repeat; independently recomputed saved metrics; a SciPy baseline coefficient; delete-one HC3 covariance; and a direct closed-form reconstruction of the first nested ridge grid and predictions.

The direct ridge check independently rebuilds one search, not every fitted ridge model. Split checks cover all saved partitions. Clean-extraction rerunning covers the complete pipeline under pinned dependencies.

## Version 3 scope

No participant, primary predictor, model family, validation seed, repeat count or alpha grid changed. Version 3 adds reporting/documentation and portable CSV output. `analysis_specification_date` identifies the fixed 2026-10-13 design; report preparation is dated 2026-10-21. PDF metadata/fonts may differ across systems. Compare numerical values with suitable floating-point tolerance and raw-source bytes by exact checksum.
