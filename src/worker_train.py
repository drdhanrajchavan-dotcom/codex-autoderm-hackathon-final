"""Bounded subprocess wrapper for AutoDerm training candidates."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, TypedDict


class TrainResult(TypedDict):
    weights_path: Optional[str]
    duration_seconds: float
    timeout: bool
    crashed: bool
    used_last_pt: bool
    error_message: Optional[str]


CHILD_CODE = r"""
import importlib.util
import json
import pathlib
import sys
import traceback

train_py = pathlib.Path(sys.argv[1]).resolve()
data_yaml = sys.argv[2]
output_dir = sys.argv[3]
budget_seconds = int(sys.argv[4])
result_path = pathlib.Path(sys.argv[5])

try:
    sys.path.insert(0, str(train_py.parent))
    spec = importlib.util.spec_from_file_location("autoderm_candidate_train", train_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import train module from {train_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    weights_path, used_last_pt = module.train(data_yaml, output_dir, budget_seconds)
    result_path.write_text(
        json.dumps(
            {
                "weights_path": weights_path,
                "used_last_pt": bool(used_last_pt),
                "error_message": None,
            }
        ),
        encoding="utf-8",
    )
except Exception as exc:
    traceback.print_exc()
    result_path.write_text(
        json.dumps(
            {
                "weights_path": None,
                "used_last_pt": False,
                "error_message": f"{exc.__class__.__name__}: {exc}",
            }
        ),
        encoding="utf-8",
    )
    raise
"""


def _fallback_weights(output_dir: Path) -> tuple[Optional[str], bool]:
    best_path = output_dir / "weights" / "best.pt"
    if best_path.exists():
        return str(best_path), False
    last_path = output_dir / "weights" / "last.pt"
    if last_path.exists():
        return str(last_path), True
    return None, False


def _kill_process(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        process.terminate()

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            process.kill()
        process.wait()


def _read_child_payload(result_path: Path) -> dict[str, object]:
    if not result_path.exists():
        return {}
    try:
        return json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def run_worker(train_py: str, data_yaml: str, output_dir: str, budget_seconds: int) -> TrainResult:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / "worker_stdout.log"
    child_result_path = output_path / "worker_train_child_result.json"
    child_result_path.unlink(missing_ok=True)

    command = [
        sys.executable,
        "-c",
        CHILD_CODE,
        str(Path(train_py).expanduser().resolve()),
        str(Path(data_yaml).expanduser().resolve()),
        str(output_path),
        str(int(budget_seconds)),
        str(child_result_path),
    ]

    start_time = time.monotonic()
    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=max(1, int(budget_seconds)))
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process(process)
            return_code = process.returncode if process.returncode is not None else -signal.SIGKILL

    duration_seconds = time.monotonic() - start_time
    weights_path, fallback_used_last = _fallback_weights(output_path)

    if timed_out:
        return {
            "weights_path": weights_path,
            "duration_seconds": duration_seconds,
            "timeout": True,
            "crashed": False,
            "used_last_pt": fallback_used_last,
            "error_message": "timeout",
        }

    child_payload = _read_child_payload(child_result_path)
    child_weights = child_payload.get("weights_path")
    child_used_last = bool(child_payload.get("used_last_pt", False))
    error_message = child_payload.get("error_message")

    if isinstance(child_weights, str) and Path(child_weights).exists():
        weights_path = child_weights
        fallback_used_last = child_used_last

    crashed = return_code != 0
    if crashed and not isinstance(error_message, str):
        error_message = f"train_process_exit:{return_code}"

    return {
        "weights_path": weights_path,
        "duration_seconds": duration_seconds,
        "timeout": False,
        "crashed": crashed,
        "used_last_pt": fallback_used_last,
        "error_message": error_message if isinstance(error_message, str) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a train.py candidate with a hard wall-clock budget.")
    parser.add_argument("--train-py", required=True)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget-seconds", type=int, required=True)
    parser.add_argument("--out", help="Optional JSON path for the TrainResult.")
    args = parser.parse_args()

    result = run_worker(
        train_py=args.train_py,
        data_yaml=args.data_yaml,
        output_dir=args.output_dir,
        budget_seconds=args.budget_seconds,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).expanduser().write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
