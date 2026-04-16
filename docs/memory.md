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

## 2026-04-16 Track C Demo Build

- Track C web demo is implemented in `web/` with Next.js pages for Doctor, Patient, Research, and iteration detail views.
- `src/api_server.py` provides the FastAPI backend for active checkpoint status, single-image inference, before/after inference, iteration listings, and iteration artifacts.
- `scripts/dev.sh` starts both services: FastAPI on port 8000 and Next.js on port 3000.
- Doctor tab supports single-photo and pre/post comparison workflows, approved sample-photo buttons, YOLO inference calls, lesion overlays, counts, Hayashi badge, and deterministic clinical observation text.
- Patient tab supports the same single-photo/sample flow, overlay rendering, plain-language findings, lesion-type explanations, placeholder care guidance, dermatologist guidance, and the research demo disclaimer.
- Research tab polls active checkpoint and iteration data every 30 seconds, shows checkpoint status, latest kept iteration status, a Recharts validation scatter plot, sortable iteration table, and per-run artifact detail pages.
- Demo photos under `web/public/demo_photos/` are the operator-approved public ACNE04 samples only: `levle0_151.jpg`, `levle0_156.jpg`, `levle0_491.jpg`, `levle1_33.jpg`, `levle1_129.jpg`, and `levle1_191.jpg`.
- Do not add or replace demo acne photos without explicit operator approval.
- Tailwind v4 requires `web/postcss.config.mjs` with `@tailwindcss/postcss`; without it, the app renders as raw unstyled HTML.
- Local smoke verification passed with `npm run build`, `scripts/dev.sh` route/API checks, and `/api/infer` on the approved demo samples.

## 2026-04-16 AWS Public Demo Plan

- Local read-only AWS checks confirmed the machine is AWS-ready: AWS CLI works, default region is `ap-south-1`, caller identity succeeds, the configured IAM user has `AdministratorAccess`, App Runner/ECR/EC2 are reachable, a default VPC exists, and Docker is installed.
- The recommended public hackathon path is Docker image -> private ECR -> AWS App Runner public HTTPS URL.
- Keep tracked deployment docs public-safe: do not hard-code AWS account IDs, credentials, private bucket names, PHI, patient images, or model artifacts.
- App Runner is CPU/container hosting. If the demo requires GPU inference latency, use EC2 G5/G6 with Docker and a reverse proxy instead.
- Before launch, add production container files, same-origin API routing, a health endpoint, upload limits, and private checkpoint/model artifact handling.
- Detailed sanitized deployment steps live in `docs/aws_app_runner_deployment.md`.
