# AutoDerm Operator Notes

This repository is a fresh build for the AutoDerm hackathon project. Do not copy code, data, or artifacts from any earlier repository.

## Mutable Files

These files are expected to change during the build:

- `README.md`
- `BUILD_PLAN.md`
- `docs/`
- `src/`
- `scripts/`
- `tests/`
- `web/`
- `config/active_checkpoint.example.json`
- `.env.example`
- `requirements.txt`
- `package.json`
- `local_env.py`

## Immutable Constraints

These constraints are not negotiable:

- Follow the documented build order in `BUILD_PLAN.md`; do not work ahead.
- Enforce the held-out-primary keep rule in code, not only in docs.
- Enforce eval integrity in code checks: preprocessing consistency, split immutability, and no leakage.
- Treat `locked_eval` as the primary keep signal.
- Treat `research_val` as the secondary signal and the early-stopping target only.
- Score exactly five acne classes: `comedone_open`, `comedone_closed`, `papule`, `pustule`, `nodule_cyst`.
- Treat `post_acne_mark` as auxiliary only.
- Keep tracked files public-safe: code, docs, `.env.example`, sanitized examples, and approved non-relinkable renders only.
- Do not commit PHI, patient images, private data, or training artifacts.
- Keep `config/active_checkpoint.json` local-only and gitignored.
- Keep `train.py` fully Codex-authored from the operator's clinical specification.

## Operator Guidance

- The operator defines the clinical problem, lesion classes, clinical priorities, and safety guardrails.
- Codex writes the implementation.
- The autoresearch loop improves Codex-authored code instead of importing outside training code.
