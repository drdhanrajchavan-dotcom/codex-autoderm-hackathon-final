"""Run one bounded AutoDerm training experiment and write immutable artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .eval import evaluate
from .types import SCORED_CLASSES
from .worker_train import TrainResult, run_worker


RESULT_COLUMNS = [
    "run_id",
    "timestamp",
    "decision",
    "discard_reason",
    "research_val_scored_map50_95",
    "locked_eval_scored_map50_95",
    "locked_eval_nodule_cyst_recall",
    "locked_eval_comedone_open_precision",
    "locked_eval_comedone_closed_precision",
    "locked_eval_papule_precision",
    "locked_eval_pustule_precision",
    "locked_eval_nodule_cyst_precision",
    "train_duration_seconds",
    "timeout",
    "crashed",
    "preprocessing_hash",
    "notes",
]
REPO_ROOT = Path(__file__).resolve().parent.parent


def _empty_per_class_metrics() -> dict[str, float]:
    return {class_name: 0.0 for class_name in SCORED_CLASSES}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_hash(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing preprocessing hash: {path}")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError(f"Empty preprocessing hash: {path}")
    return value


def _preprocessing_hash(data_yaml: Path, research_val_dir: Path) -> str:
    candidates = [
        data_yaml.parent / "preprocessing_hash.txt",
        research_val_dir / "preprocessing_hash.txt",
        research_val_dir.parent / "preprocessing_hash.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return _read_hash(candidate)
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(f"Unable to locate preprocessing hash. Checked: {checked}")


def _write_weights_hash(output_dir: Path, preprocessing_hash: str) -> None:
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / "preprocessing_hash.txt").write_text(preprocessing_hash + "\n", encoding="utf-8")


def _error_class(error_message: Optional[str]) -> str:
    if not error_message:
        return "UnknownError"
    return error_message.split(":", 1)[0].strip() or "UnknownError"


def _failure_row(
    run_id: str,
    timestamp: str,
    discard_reason: str,
    train_result: TrainResult | None,
    preprocessing_hash: str,
    notes: str = "",
) -> dict:
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "decision": "FAILED",
        "discard_reason": discard_reason,
        "research_val_scored_map50_95": 0.0,
        "locked_eval_scored_map50_95": 0.0,
        "locked_eval_nodule_cyst_recall": 0.0,
        "locked_eval_per_class_precision": _empty_per_class_metrics(),
        "locked_eval_per_class_recall": _empty_per_class_metrics(),
        "research_val_per_class_precision": _empty_per_class_metrics(),
        "research_val_per_class_recall": _empty_per_class_metrics(),
        "train_duration_seconds": train_result["duration_seconds"] if train_result else 0.0,
        "timeout": train_result["timeout"] if train_result else False,
        "crashed": train_result["crashed"] if train_result else True,
        "preprocessing_hash": preprocessing_hash,
        "notes": notes,
    }


def _success_row(
    run_id: str,
    timestamp: str,
    train_result: TrainResult,
    preprocessing_hash: str,
    research_val_metrics: dict,
    locked_eval_metrics: dict,
) -> dict:
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "decision": "PENDING_KEEP_RULE",
        "discard_reason": None,
        "research_val_scored_map50_95": float(research_val_metrics["scored_map50_95"]),
        "locked_eval_scored_map50_95": float(locked_eval_metrics["scored_map50_95"]),
        "locked_eval_nodule_cyst_recall": float(locked_eval_metrics["per_class_recall"]["nodule_cyst"]),
        "locked_eval_per_class_precision": locked_eval_metrics["per_class_precision"],
        "locked_eval_per_class_recall": locked_eval_metrics["per_class_recall"],
        "research_val_per_class_precision": research_val_metrics["per_class_precision"],
        "research_val_per_class_recall": research_val_metrics["per_class_recall"],
        "train_duration_seconds": train_result["duration_seconds"],
        "timeout": train_result["timeout"],
        "crashed": train_result["crashed"],
        "preprocessing_hash": preprocessing_hash,
        "notes": "used_last_pt_due_to_timeout" if train_result["used_last_pt"] else "",
    }


def _write_experiment_row(output_dir: Path, row: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_row.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _flat_row(row: dict) -> dict[str, object]:
    locked_precision = row["locked_eval_per_class_precision"]
    return {
        "run_id": row["run_id"],
        "timestamp": row["timestamp"],
        "decision": row["decision"],
        "discard_reason": row["discard_reason"] or "",
        "research_val_scored_map50_95": row["research_val_scored_map50_95"],
        "locked_eval_scored_map50_95": row["locked_eval_scored_map50_95"],
        "locked_eval_nodule_cyst_recall": row["locked_eval_nodule_cyst_recall"],
        "locked_eval_comedone_open_precision": locked_precision["comedone_open"],
        "locked_eval_comedone_closed_precision": locked_precision["comedone_closed"],
        "locked_eval_papule_precision": locked_precision["papule"],
        "locked_eval_pustule_precision": locked_precision["pustule"],
        "locked_eval_nodule_cyst_precision": locked_precision["nodule_cyst"],
        "train_duration_seconds": row["train_duration_seconds"],
        "timeout": row["timeout"],
        "crashed": row["crashed"],
        "preprocessing_hash": row["preprocessing_hash"],
        "notes": row["notes"],
    }


def _append_results_tsv(row: dict, results_path: Path = Path("runs/results.tsv")) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not results_path.exists() or results_path.stat().st_size == 0
    with results_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS, delimiter="\t", lineterminator="\n")
        if needs_header:
            writer.writeheader()
        writer.writerow(_flat_row(row))


def _finish(output_dir: Path, row: dict) -> None:
    _write_experiment_row(output_dir, row)
    _append_results_tsv(row)


def _check_preprocessing(weights_path: str, split_dir: Path) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "check_preprocessing_consistency.py"),
        "--weights",
        weights_path,
        "--split",
        str(split_dir),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip().replace("\n", " ")
        raise RuntimeError(details or f"check exited with code {completed.returncode}")


def _snapshot_train_py(train_py: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(train_py, output_dir / "train.py.snapshot")


def run_experiment(
    train_py: str,
    output_dir: str,
    budget_seconds: int,
    data_yaml: str,
    research_val_dir: str,
    locked_eval_dir: str,
    run_id: str,
) -> dict:
    output_path = Path(output_dir).expanduser().resolve()
    train_py_path = Path(train_py).expanduser().resolve()
    data_yaml_path = Path(data_yaml).expanduser().resolve()
    research_val_path = Path(research_val_dir).expanduser().resolve()
    locked_eval_path = Path(locked_eval_dir).expanduser().resolve()
    timestamp = _timestamp()

    _snapshot_train_py(train_py_path, output_path)
    preprocessing_hash = _preprocessing_hash(data_yaml_path, research_val_path)
    _write_weights_hash(output_path, preprocessing_hash)

    train_result = run_worker(
        train_py=str(train_py_path),
        data_yaml=str(data_yaml_path),
        output_dir=str(output_path),
        budget_seconds=budget_seconds,
    )
    _write_weights_hash(output_path, preprocessing_hash)

    if train_result["crashed"] or train_result["weights_path"] is None:
        if train_result["weights_path"] is None and train_result["timeout"] and not train_result["crashed"]:
            discard_reason = "timeout_no_weights"
        else:
            discard_reason = f"train_crash:{_error_class(train_result['error_message'])}"
        row = _failure_row(
            run_id=run_id,
            timestamp=timestamp,
            discard_reason=discard_reason,
            train_result=train_result,
            preprocessing_hash=preprocessing_hash,
            notes="used_last_pt_due_to_timeout" if train_result["used_last_pt"] else "",
        )
        _finish(output_path, row)
        return row

    try:
        _check_preprocessing(train_result["weights_path"], research_val_path)
        _check_preprocessing(train_result["weights_path"], locked_eval_path)
    except Exception as exc:
        row = _failure_row(
            run_id=run_id,
            timestamp=timestamp,
            discard_reason=f"preprocessing_mismatch:{exc}",
            train_result=train_result,
            preprocessing_hash=preprocessing_hash,
        )
        _finish(output_path, row)
        return row

    try:
        research_val_metrics = evaluate(train_result["weights_path"], str(research_val_path))
    except Exception as exc:
        row = _failure_row(
            run_id=run_id,
            timestamp=timestamp,
            discard_reason=f"eval_crash_research_val:{exc.__class__.__name__}",
            train_result=train_result,
            preprocessing_hash=preprocessing_hash,
        )
        _finish(output_path, row)
        return row

    try:
        locked_eval_metrics = evaluate(train_result["weights_path"], str(locked_eval_path))
    except Exception as exc:
        row = _failure_row(
            run_id=run_id,
            timestamp=timestamp,
            discard_reason=f"eval_crash_locked_eval:{exc.__class__.__name__}",
            train_result=train_result,
            preprocessing_hash=preprocessing_hash,
        )
        _finish(output_path, row)
        return row

    row = _success_row(
        run_id=run_id,
        timestamp=timestamp,
        train_result=train_result,
        preprocessing_hash=preprocessing_hash,
        research_val_metrics=research_val_metrics,
        locked_eval_metrics=locked_eval_metrics,
    )
    _finish(output_path, row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one bounded AutoDerm experiment.")
    parser.add_argument("--train-py", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget-seconds", type=int, required=True)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--research-val-dir", required=True)
    parser.add_argument("--locked-eval-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    run_experiment(
        train_py=args.train_py,
        output_dir=args.output_dir,
        budget_seconds=args.budget_seconds,
        data_yaml=args.data_yaml,
        research_val_dir=args.research_val_dir,
        locked_eval_dir=args.locked_eval_dir,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
