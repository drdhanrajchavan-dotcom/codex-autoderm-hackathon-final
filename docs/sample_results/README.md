# Sample Results

This directory contains curated, public-safe examples of the artifacts produced by the AutoDerm autoresearch loop. They are small text examples for judges and reviewers, not training data, PHI, patient images, private datasets, model weights, or full run artifacts.

## Files

- `sample_results.tsv`: representative loop rows with KEEP, DISCARD, and FAILED outcomes.
- `sample_experiment_row.json`: one complete kept `experiment_row.json` example.
- `sample_train_diff.txt`: the most clinically interpretable Codex-authored `train.py` mutation diff.
- `sample_prompt.txt`: a representative Codex loop prompt showing the constraints Codex received before proposing a mutation.
- `baseline_summary.json`: crop-gate baseline numbers and the chosen starting checkpoint.

The sample values mirror the current loop artifact schema: five scored acne classes, `post_acne_mark` excluded from the scored metric, `research_val` as the secondary signal, and `locked_eval` as the primary keep signal.
