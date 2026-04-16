# AutoDerm Autoresearch Loop Program

## Held-out-primary keep rule

A mutation is KEPT if and only if all of the following hold relative to the current best kept iteration (the "reference"):

1. locked_eval_scored_map50_95 does not regress by more than 0.005 (small noise tolerance).
2. research_val_scored_map50_95 improves by at least 0.003, OR locked_eval_scored_map50_95 improves by at least 0.005.
3. locked_eval_nodule_cyst_recall does not drop below 0.50 (clinical safety guardrail).
4. No class precision on locked_eval drops by more than 0.10 absolute from the reference.

The locked_eval signal is the primary keep signal. research_val is used as a secondary corroborating signal and as the early-stopping target during training itself. A mutation that improves research_val but regresses locked_eval is DISCARDED — this is the failure mode the previous loop's keep rule produced and we are explicitly correcting it.

When research_val and locked_eval disagree, locked_eval wins.

## Methodological context (read this before proposing any mutation)

The previous autoresearch run on this task ran 56 mutations across three loops using a research_val-only keep rule. Post-hoc analysis revealed that three discarded iterations outperformed the kept "winner" on locked_eval — by margins larger than anything the loop produced over baseline. The keep rule overfit a 30-image research_val set.

This means:

- Mutations that look like research_val wins of less than 0.01 are likely noise on a 30-image val set. Do not propose mutations whose mechanism only acts on val-set-specific features.
- Mutations that improve research_val by overfitting (e.g., aggressive class-rebalancing toward classes that happen to be common in val, or threshold tuning that targets val confusion patterns) will be caught by locked_eval and discarded. Do not propose them.
- Generalization-improving mutations (regularization, augmentation diversity, robust loss formulations, calibration based on training-set statistics rather than val-set statistics) are more likely to survive the held-out-primary rule than the previous one. Bias proposals toward this class.
- The training loop's early-stopping `patience` is the only place research_val should drive a decision. It does not drive the keep rule.

## Eval integrity rules

- Any mutation that changes input preprocessing (cropping, resizing, color space, normalization) MUST apply the same preprocessing at eval time. The eval pipeline materializes research_val and locked_eval images using the same preprocessing function as train. Cross-preprocessing comparisons are invalid and any results.tsv row produced from one is invalid. This is enforced in code by scripts/check_preprocessing_consistency.py.
- Any mutation that changes the train/val/locked_eval split is forbidden. The split is fixed by patient ID and committed to disk. Mutations that touch the split MUST be discarded automatically and the proposing iteration MUST be marked invalid.
- Any mutation that introduces locked_eval images into training (e.g., pseudo-labeling on locked_eval, or accidentally including locked_eval in train via globbing) is forbidden and invalidates the loop. If detected, halt the loop.

## Saturation tracking

This loop starts with no inherited saturation assumptions from previous loops. A neighborhood is only considered saturated if THIS loop's results.tsv shows three consecutive discards on closely related mutations within that neighborhood. Mutations that the previous loop discarded MAY be re-proposed if the proposer believes the new keep rule changes their outcome.

## Required logging per iteration

Each iteration MUST produce an experiment_row.json containing all fields in the ExperimentRow TypedDict (src/types.py). A row missing any required field is treated as a failed iteration and is not eligible for keep.

Required fields for keep eligibility:
- research_val_scored_map50_95
- locked_eval_scored_map50_95
- locked_eval_nodule_cyst_recall
- locked_eval_per_class_precision (must contain all 5 scored classes as keys)
