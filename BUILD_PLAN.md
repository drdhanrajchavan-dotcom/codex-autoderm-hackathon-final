Built during OpenAI Codex Community Hackathon, Bengaluru, April 16, 2026.
Solo submission by Dhanraj Chavan, MD -- ClearSkin Private Limited.

## Architecture

- **Code:** developed locally on M5 Max
- **Training + loop:** runs on EC2 g5.xlarge (A10G GPU)
- **Web app (Next.js + FastAPI):** runs locally on M5 Max
- **Data flow:** `scripts/pullback.sh` rsyncs EC2 artifacts (including weights) to local `.pulled_artifacts/`
- **Inference:** local API serves from `.pulled_artifacts/` weights
- **Promotion:** manual edit of `config/active_checkpoint.json`

## Build order

### Track A -- Foundation -> Loop (sequential)

| Step | What | Depends on |
|------|------|------------|
| A0 | Codex session context paste | nothing |
| A1 | Repo skeleton | A0 |
| A2 | `src/types.py`, `src/eval.py`, `src/prepare.py`, check scripts | A1 |
| A3 | `src/train.py` (Codex writes from clinical spec), `src/worker_train.py`, `src/run_experiment.py` | A2 |
| A4 | `docs/program.md` (paste verbatim, do not generate) | A1 |
| A5 | `src/keep_rule.py` + `tests/test_keep_rule.py` (8 tests must pass) | A2 |
| A6 | `src/run_loop_codex_exec.py` | A3, A4, A5 |
| A7 | `scripts/run_baselines.py` + `scripts/render_baseline_gallery.py` (runs on EC2) | A3 |
| A8 | README, demo script, writeup (last) | everything |

### Track B -- EC2 setup (parallel from minute one)

| Step | What | Depends on |
|------|------|------------|
| B1 | Provision EC2 | nothing |
| B2 | Push code + deps + dataset to EC2 | A2 (re-push after A3, A5, A6, A7) |
| B3 | `codex exec` auth test on EC2 | B2 |
| B4 | `scripts/pullback.sh` + `scripts/watch_pullback.sh` | A1 |

### Track C -- Web app (starts when loop is kicked off)

| Step | What | Depends on |
|------|------|------------|
| C1 | Next.js skeleton, layout, types | A2 (for type schemas) |
| C2 | `src/api_server.py` (FastAPI) + `scripts/dev.sh` | A2, A3 |
| C3 | Doctor tab (single photo + pre/post) | C1, C2 |
| C4 | Patient tab (explanation + guidance) | C1, C2 |
| C5 | Research tab (scatter plot + iteration detail) | C1, C2 |
| C6 | Demo polish | C3, C4, C5 |

## Execution timeline

### Morning -- foundation + baselines

1. Start Track A (A0 -> A1 -> A2 -> A3) and Track B (B1 -> B2 -> B3) in parallel.
2. After A3: implement A5 (keep rule + tests), A4 (paste program.md), A6 (loop runner), A7 (baselines script).
3. Push code to EC2 (B2 re-push).
4. Run baselines on EC2: `python -m scripts.run_baselines --source-data source_data --output-root .`
5. Wait for crop gate decision (~15-20 min).

### Midday -- kick off loop + start app

6. Pull baseline artifacts locally: `scripts/pullback.sh`
7. Edit `config/active_checkpoint.json` -> baseline checkpoint.
8. Start loop on EC2: `python -m src.run_loop_codex_exec --data-yaml ... --reference-experiment-row runs/baseline_<chosen>/experiment_row.json &`
9. Start `scripts/watch_pullback.sh` in third terminal.
10. Start `scripts/dev.sh` for local web app.
11. Begin Track C: C1 -> C2 -> C3 -> C4 -> C5.

### Afternoon -- build app + monitor loop

12. Build Doctor, Patient, Research tabs against live pullback data.
13. Check loop progress periodically via Research tab scatter plot.

### Late afternoon -- polish + demo

14. C6 demo polish.
15. A8 submission docs.
16. Promote best kept iteration: edit `config/active_checkpoint.json`.
17. Demo practice + backup video recording.

## Key contracts

### Dataset layout
```
data/<variant>/
  train/images/, train/labels/
  research_val/images/, research_val/labels/
  locked_eval/images/, locked_eval/labels/
  preprocessing_hash.txt
  dataset.yaml  (locked_eval NOT referenced here)
```

### results.tsv columns
```
run_id, timestamp, decision, discard_reason,
research_val_scored_map50_95, locked_eval_scored_map50_95,
locked_eval_nodule_cyst_recall,
locked_eval_comedone_open_precision, locked_eval_comedone_closed_precision,
locked_eval_papule_precision, locked_eval_pustule_precision,
locked_eval_nodule_cyst_precision,
train_duration_seconds, timeout, crashed, preprocessing_hash, notes
```

### Held-out-primary keep rule
1. locked_eval regression <= 0.005
2. research_val improves >= 0.003 OR locked_eval improves >= 0.005
3. locked_eval nodule_cyst recall >= 0.50
4. No locked_eval class precision drops > 0.10

locked_eval wins all disagreements with research_val.

### Crop gate
- Permissive: cropped wins if `cropped_locked_eval >= uncropped_locked_eval`
- No live judgment call -- threshold is in code

### Scored classes
comedone_open, comedone_closed, papule, pustule, nodule_cyst
Auxiliary only: post_acne_mark

## Fallback table

| Failure | Fallback |
|---|---|
| Loop produces no kept iterations by 4pm | Demo = trajectory chart + held-out-primary methodology + baselines only |
| App broken by 4pm | Demo = Research tab only, scatter plot + iteration walkthrough |
| EC2 codex exec fails to auth | Loop runs locally on M5 Max (slower) |
| Pullback fails | Manual scp of kept iteration weights + experiment_row |
| All else fails | Pre-recorded demo video |

## Non-negotiables

- Keep rule is enforced in CODE (`src/keep_rule.py`), not just in prompts
- Eval integrity enforced in CODE (`scripts/check_preprocessing_consistency.py`)
- `pytest tests/test_keep_rule.py` must show 8 passing tests before the loop runs
- `docs/program.md` is pasted verbatim, never Codex-generated
- No code or artifacts imported from previous repos
- Public-safe: no PHI, no patient images, no weights in git
- **Codex-native: every line of train.py is Codex-authored from a clinical specification. The operator provides domain expertise; Codex provides the implementation.**

## 2-minute demo beats

| Beat | Time | Tab | What to show | What to say |
|------|------|-----|-------------|-------------|
| 1 | 0:00-0:25 | Doctor (single) | Pre-staged photo -> overlay + counts + GAGS badge | "Clinical photo, Fitzpatrick V. Five lesion types, severity grade. 3-4 min manual -> 2 sec." |
| 2 | 0:25-0:55 | Doctor (pre/post) | Anonymized before/after of same patient -> both overlays + count delta + badge change | "Same engine, two visits. Inflammatory count dropped from X to Y. Severity moderate -> mild. Treatment tracking, automatically." |
| 3 | 0:55-1:30 | Research | Scatter plot -> click dot -> code diff on screen | "I didn't write the training code. Codex wrote it from my clinical spec, then improved its own code." |
| 4 | 1:30-1:50 | Patient | Same photo, patient language, scale claim | "Same engine, patient-facing. Across 50K consultations a year, finally measure which treatments work -- publishable research, automatically." |
| 5 | 1:50-2:00 | -- | Close | "Solo build. I wrote the clinical spec. Codex wrote the code. The loop is still running." |

**Rehearse this one-liner:** "I didn't write the training code. I wrote the clinical specification. Codex wrote the code, ran the experiments, and improved the model."

**Critical Wednesday-night verification:** Test the pre/post photo pair against the active checkpoint. The "after" must visibly show fewer active lesions AND the model must correctly detect the reduction. If the model adds false positives on the after photo or misses the improvement, swap pairs. This visual is the proof of the treatment-tracking story -- it has to land.

**Rehearse this one-liner:** "I didn't write the training code. I wrote the clinical specification. Codex wrote the code, ran the experiments, and improved the model."

## Judge-visible repo artifacts

These are TRACKED (not gitignored) so judges can understand the project from the repo:

| Path | What | Why tracked |
|------|------|-------------|
| `web/public/demo_photos/` | 3-4 ACNE04 dataset images + LICENSE.md | Public research dataset, powers "Try a sample" buttons |
| `docs/sample_results/` | Curated iteration examples (TSV rows, experiment_row, diff, prompt, baseline_summary) | Shows what the loop produces without needing full dataset |
| `config/active_checkpoint.example.json` | Schema template | Shows judges the expected config format |
| `docs/program.md` | The loop contract | The held-out-primary keep rule specification |
| `BUILD_PLAN.md` | This file | Full build methodology |
