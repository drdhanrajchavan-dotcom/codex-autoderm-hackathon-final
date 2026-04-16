#!/usr/bin/env python3
"""Run AutoDerm uncropped/cropped baselines on EC2.

EC2 invocation flow:
1. Prepare the EC2 environment and place source data outside git.
2. From the repo root, run:
   python -m scripts.run_baselines --source-data /path/to/source_data --output-root .
3. Review runs/baseline_summary.json and the printed crop-gate decision.
4. Pull runs/, data/<chosen>/, and weights artifacts back to local with the pullback script.
5. Start the autoresearch loop manually with --reference-experiment-row pointing at the pulled chosen baseline row.

This script does not start the autoresearch loop.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.prepare import prepare_dataset
from src.run_experiment import run_experiment


BASELINE_BUDGET_SECONDS = 300


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_experiment_row(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Expected experiment row was not written: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _run_baseline(
    variant: str,
    data_dir: Path,
    runs_dir: Path,
    budget_seconds: int = BASELINE_BUDGET_SECONDS,
) -> dict:
    output_dir = runs_dir / f"baseline_{variant}"
    row = run_experiment(
        train_py=str(Path("src/train.py").resolve()),
        output_dir=str(output_dir),
        budget_seconds=budget_seconds,
        data_yaml=str(data_dir / "dataset.yaml"),
        research_val_dir=str(data_dir / "research_val"),
        locked_eval_dir=str(data_dir / "locked_eval"),
        run_id=f"baseline_{variant}",
    )
    row_path = output_dir / "experiment_row.json"
    return row if row else _read_experiment_row(row_path)


def _locked_eval_score(row: dict, variant: str) -> float:
    try:
        return float(row["locked_eval_scored_map50_95"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Baseline {variant} row is missing locked_eval_scored_map50_95") from exc


def run_baselines(source_data: str, output_root: str) -> dict:
    output_root_path = Path(output_root).expanduser().resolve()
    source_data_path = Path(source_data).expanduser().resolve()
    data_root = output_root_path / "data"
    runs_root = output_root_path / "runs"
    data_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    uncropped_data = data_root / "uncropped"
    cropped_data = data_root / "cropped"

    print("Materializing uncropped dataset: data/uncropped/")
    prepare_dataset(
        source_dir=str(source_data_path),
        output_dir=str(uncropped_data),
        split_seed=42,
        apply_face_crop=False,
    )
    print("Running uncropped baseline: runs/baseline_uncropped/")
    uncropped_row = _run_baseline("uncropped", uncropped_data, runs_root)
    uncropped_locked_eval = _locked_eval_score(uncropped_row, "uncropped")

    print("Materializing cropped dataset: data/cropped/")
    prepare_dataset(
        source_dir=str(source_data_path),
        output_dir=str(cropped_data),
        split_seed=42,
        apply_face_crop=True,
    )
    print("Running cropped baseline: runs/baseline_cropped/")
    cropped_row = _run_baseline("cropped", cropped_data, runs_root)
    cropped_locked_eval = _locked_eval_score(cropped_row, "cropped")

    chosen = "cropped" if cropped_locked_eval >= uncropped_locked_eval else "uncropped"
    summary = {
        "uncropped_locked_eval": uncropped_locked_eval,
        "cropped_locked_eval": cropped_locked_eval,
        "chosen": chosen,
        "chosen_experiment_row_path": f"runs/baseline_{chosen}/experiment_row.json",
        "chosen_weights_path": f"runs/baseline_{chosen}/weights/best.pt",
        "chosen_data_yaml_path": f"data/{chosen}/dataset.yaml",
        "timestamp": _timestamp(),
    }
    (runs_root / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== CROP GATE DECISION ===")
    print(f"uncropped: {uncropped_locked_eval}, cropped: {cropped_locked_eval}")
    print(f"CHOSEN: {chosen}")
    print(f"LOOP REFERENCE: runs/baseline_{chosen}/experiment_row.json")
    print(f"LOOP DATA YAML: data/{chosen}/dataset.yaml")
    print(
        "Pull these to local before starting the loop. Then start the loop with "
        "--reference-experiment-row pointing at the pulled experiment_row.json."
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run uncropped and cropped AutoDerm baselines on EC2.")
    parser.add_argument("--source-data", required=True, help="Private source data directory on EC2.")
    parser.add_argument("--output-root", required=True, help="Repo/output root that contains data/ and runs/.")
    args = parser.parse_args()

    run_baselines(source_data=args.source_data, output_root=args.output_root)


if __name__ == "__main__":
    main()
