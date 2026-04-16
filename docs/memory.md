# AutoDerm Memory

## 2026-04-16 Baseline State

- First baseline run completed on EC2 A10G using the uncropped split.
- FaceMesh cropping was revised from cheek-only to an expanded face close-up crop. The crop should include the full nose tip, visible opposite cheek when present, forehead just above the hairline, below the chin, and the mid-ear/outer-cheek region.
- Cropped dataset was generated on EC2 from 203 image/label pairs with 142 train, 30 research_val, and 31 locked_eval examples.
- Cropped dataset crop methods: `face_mesh_closeup = 202`, `center_fallback = 1`.
- Cropped dataset preprocessing hash: `6a4caf0ca14f91ec7e0981d38076402170f9e1710d5c19ed5f616487b9bc1187`.
- Baseline model recipe: `yolo26l-obb.pt`, `IMAGE_SIZE = 512`, `BATCH_SIZE = 8`, 300 second experiment budget.
- Direct cropped baseline run completed on EC2 with YOLO26l OBB.
- Direct cropped eval:
  - `locked_eval_scored_map50_95 = 0.037870647353228024`
  - `research_val_scored_map50_95 = 0.05216926768026558`
  - `locked_eval_nodule_cyst_recall = 0.4230769230769231`
  - YOLO training validation final row: `metrics/mAP50(B) = 0.24307`, `metrics/mAP50-95(B) = 0.10766`
  - `timeout = true`, `crashed = false`
- Existing uncropped direct eval:
  - `locked_eval_scored_map50_95 = 0.023492275253707305`
  - `research_val_scored_map50_95 = 0.00449117036357333`
  - `locked_eval_nodule_cyst_recall = 0.34615384615384615`
  - `timeout = true`, `crashed = false`
- The direct cropped run improved over the existing uncropped run on locked_eval mAP and nodule/cyst recall, but a strict crop-gate would rerun uncropped and cropped under the same current preprocessing code before declaring the gate final.
- Pulled local artifacts are under `.pulled_artifacts/` and must remain untracked.

## Implementation Notes

- `src/train.py` uses Ultralytics `time` in hours, so budget seconds must be converted with `budget_seconds / 3600`.
- `locked_eval` remains the primary keep signal and must not be referenced by Ultralytics training-time validation.
