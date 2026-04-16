# AutoDerm Memory

## 2026-04-16 Baseline State

- First baseline run completed on EC2 A10G using the uncropped split only.
- Cropped/cheek-view baseline was deferred; do not treat crop-gate as decided.
- Baseline model recipe: `yolo26l-obb.pt`, `IMAGE_SIZE = 512`, `BATCH_SIZE = 8`, 300 second experiment budget.
- Baseline direct eval:
  - `locked_eval_scored_map50_95 = 0.023492275253707305`
  - `research_val_scored_map50_95 = 0.00449117036357333`
  - `locked_eval_nodule_cyst_recall = 0.34615384615384615`
  - `timeout = true`, `crashed = false`
- Pulled local artifacts are under `.pulled_artifacts/runs/baseline_uncropped/` and must remain untracked.

## Implementation Notes

- `src/train.py` uses Ultralytics `time` in hours, so budget seconds must be converted with `budget_seconds / 3600`.
- `locked_eval` remains the primary keep signal and must not be referenced by Ultralytics training-time validation.
