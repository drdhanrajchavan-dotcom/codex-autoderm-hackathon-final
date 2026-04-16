# AutoDerm

AutoDerm is a clinician-assist system for acne severity grading on Indian and darker skin. Upload one photo and it returns per-lesion detections, lesion counts, and a standardized GAGS severity grade. During the hackathon, Codex autonomously wrote the training code from a clinical specification, then improved the detector against a locked clinical validation set by editing its own code, running experiments, and keeping only the changes that improved the eval. Every line of train.py was written by Codex. The dermatologist provided domain expertise — what the lesion classes are, what 'better' means clinically, and what the safety guardrails are. Codex provided the implementation.

Built during OpenAI Codex Community Hackathon, Bengaluru, April 16, 2026. Solo submission by Dhanraj Chavan, MD, ClearSkin Private Limited.

All code, docs, web app surfaces, training/inference scripts, deployment scaffolding, and autoresearch artifacts in this repository were produced during the April 16, 2026 hackathon build; the only pre-existing project work was the clinical image annotation effort.

## Limited Autoresearch Result

Even under the limited hackathon autoresearch loop, the official kept checkpoint improved locked_eval scored mAP50-95 from 0.0379 to 0.0812: a +0.0433 absolute lift, or 2.14x over the cropped baseline. research_val scored mAP50-95 improved from 0.0522 to 0.1210: a +0.0688 absolute lift, or 2.32x. The kept model also raised locked_eval nodule/cyst recall from 0.4231 to 0.5000, meeting the clinical safety floor.

## Product Surfaces

- **Doctor:** Upload a single clinical photo or a before/after pair, then review lesion boxes, counts, deltas, and the GAGS severity badge.
- **Patient:** Show the same model output in patient-facing language with care guidance and escalation warnings for nodules or cysts.
- **Research:** Inspect the autoresearch trajectory, compare research_val and locked_eval metrics, and open the Codex-authored diff for each iteration.

## Codex-Native Story

Codex wrote train.py from a clinical spec. The autoresearch loop then improved Codex's own code. The operator (a practicing dermatologist) defined the problem; Codex solved it.

## Held-Out-Primary Keep Rule

A mutation is KEPT if and only if all of the following hold relative to the current best kept iteration (the "reference"):

1. locked_eval_scored_map50_95 does not regress by more than 0.005 (small noise tolerance).
2. research_val_scored_map50_95 improves by at least 0.003, OR locked_eval_scored_map50_95 improves by at least 0.005.
3. locked_eval_nodule_cyst_recall does not drop below 0.50 (clinical safety guardrail).
4. No class precision on locked_eval drops by more than 0.10 absolute from the reference.

The locked_eval signal is the primary keep signal. research_val is used as a secondary corroborating signal and as the early-stopping target during training itself. A mutation that improves research_val but regresses locked_eval is DISCARDED — this is the failure mode the previous loop's keep rule produced and we are explicitly correcting it.

When research_val and locked_eval disagree, locked_eval wins.

## Autoresearch Methodology

Codex proposes bounded mutations to its own `src/train.py`. `src/run_experiment.py` trains the candidate, evaluates it against `research_val` and `locked_eval`, writes an immutable `experiment_row.json`, and appends a flat row to `results.tsv`. `src/keep_rule.py` then decides KEEP, DISCARD, or FAILED under the held-out-primary rule.

The loop is inspired by Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) reference, used here only as conceptual background. AutoDerm does not import code, data, or artifacts from that repository.

## Architecture

- **Code and demo app:** run locally on the M5 Max laptop.
- **Training and autoresearch loop:** run on an EC2 g5.xlarge instance with an A10G GPU.
- **Pullback flow:** `scripts/pullback.sh` rsyncs EC2 loop outputs into local `.pulled_artifacts/`.
- **Inference:** the local FastAPI server reads the promoted checkpoint and serves the Next.js Doctor, Patient, and Research tabs.
- **Promotion:** manually edit `config/active_checkpoint.json` after reviewing a kept iteration; see `config/active_checkpoint.example.json` for the local-only schema.

## Clinical Differentiator

The dataset is predominantly Fitzpatrick type IV-VI (Indian skin). Codex's augmentation choices account for melanin-rich skin contrast. Performance on darker skin types is tracked as a first-class metric.

AutoDerm scores exactly five acne classes: `comedone_open`, `comedone_closed`, `papule`, `pustule`, and `nodule_cyst`. `post_acne_mark` is auxiliary only and is not part of the scored acne severity metric.

## Workflow ROI

Manual lesion counting takes 3-4 minutes per image. AutoDerm returns counts and severity in ~2 seconds.

## Public-Safe Policy

Tracked files must remain public-safe: code, docs, `.env.example`, sanitized examples, and approved non-relinkable renders only. Do not commit PHI, patient images, private datasets, model weights, training artifacts, or `config/active_checkpoint.json`.

## Demo Photos

Demo photos come from the ACNE04 public research dataset and are tracked with license attribution. The tracked demo images are for judge-visible app flow only and are kept separate from private training, `research_val`, and `locked_eval` data.

## Running Locally For Judges

1. Clone repo.
2. Install Python dependencies with `pip install -r requirements.txt` (Python 3.11+).
3. Install web dependencies with `cd web && npm install` (Node 18+).
4. Place dataset in `source_data/`.
5. Run baselines with `python -m scripts.run_baselines --source-data source_data --output-root .`.
6. Copy `config/active_checkpoint.example.json` to `config/active_checkpoint.json`, then edit the paths.
7. Start the app with `scripts/dev.sh`.
8. Trained weights are available on the demo laptop only.

## Production Demo Container

The repo includes App Runner-ready container scaffolding for the public demo. `Dockerfile` builds the Next.js app and FastAPI API together, `web/next.config.mjs` proxies same-origin `/api/*` and `/healthz` requests to FastAPI, and `scripts/start_prod.sh` starts both processes inside the container. The production image uses only the manually promoted official KEEP checkpoint and does not require judges to commit private training artifacts.

## Fallback Table

| Failure | Fallback |
|---|---|
| Loop produces no kept iterations by 4pm | Demo = trajectory + methodology + baselines only |
| App broken by 4pm | Demo = Research tab only, scatter plot + iteration walkthrough |
| EC2 codex exec fails | Loop runs locally on M5 Max |
| Pullback fails | Manual scp of kept iteration weights + experiment_row |
| All else fails | Pre-recorded demo video |
