"""Outer Codex-driven autoresearch loop for AutoDerm."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from .keep_rule import candidate_beats_reference
from .run_experiment import RESULT_COLUMNS
from .types import SCORED_CLASSES


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = REPO_ROOT / "src" / "train.py"
PROGRAM_PATH = REPO_ROOT / "docs" / "program.md"
CODEx_RETRY_DELAY_SECONDS = 30


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_per_class_metrics() -> dict[str, float]:
    return {class_name: 0.0 for class_name in SCORED_CLASSES}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _results_path(output_root: Path) -> Path:
    return output_root / "results.tsv"


def _ensure_results_tsv(results_path: Path) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if results_path.exists() and results_path.stat().st_size > 0:
        return
    with results_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()


def _read_results_rows(results_path: Path) -> list[dict[str, str]]:
    if not results_path.exists():
        return []
    data_lines = [line for line in results_path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    if not data_lines:
        return []
    with io.StringIO("\n".join(data_lines) + "\n") as buffer:
        return list(csv.DictReader(buffer, delimiter="\t"))


def _append_results_row(results_path: Path, row: dict) -> None:
    _ensure_results_tsv(results_path)
    flat_row = {
        "run_id": row["run_id"],
        "timestamp": row["timestamp"],
        "decision": row["decision"],
        "discard_reason": row["discard_reason"] or "",
        "research_val_scored_map50_95": row["research_val_scored_map50_95"],
        "locked_eval_scored_map50_95": row["locked_eval_scored_map50_95"],
        "locked_eval_nodule_cyst_recall": row["locked_eval_nodule_cyst_recall"],
        "locked_eval_comedone_open_precision": row["locked_eval_per_class_precision"]["comedone_open"],
        "locked_eval_comedone_closed_precision": row["locked_eval_per_class_precision"]["comedone_closed"],
        "locked_eval_papule_precision": row["locked_eval_per_class_precision"]["papule"],
        "locked_eval_pustule_precision": row["locked_eval_per_class_precision"]["pustule"],
        "locked_eval_nodule_cyst_precision": row["locked_eval_per_class_precision"]["nodule_cyst"],
        "train_duration_seconds": row["train_duration_seconds"],
        "timeout": row["timeout"],
        "crashed": row["crashed"],
        "preprocessing_hash": row["preprocessing_hash"],
        "notes": row["notes"],
    }
    with results_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writerow(flat_row)


def _update_results_row(results_path: Path, run_id: str, decision: str, discard_reason: str | None) -> None:
    lines = results_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise RuntimeError(f"Cannot update empty results file: {results_path}")

    header_line = next((line for line in lines if line and not line.startswith("#")), "")
    header = next(csv.reader([header_line], delimiter="\t"))
    updated_lines: list[str] = []
    updated = False

    for line in lines:
        if not line or line.startswith("#") or line == header_line:
            updated_lines.append(line)
            continue
        row_values = next(csv.reader([line], delimiter="\t"))
        row = dict(zip(header, row_values, strict=False))
        if row.get("run_id") == run_id:
            row["decision"] = decision
            row["discard_reason"] = discard_reason or ""
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=header, delimiter="\t", lineterminator="")
            writer.writerow(row)
            updated_lines.append(buffer.getvalue())
            updated = True
        else:
            updated_lines.append(line)

    if not updated:
        raise RuntimeError(f"Unable to find run_id={run_id} in {results_path}")

    results_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")


def _append_comment(results_path: Path, message: str) -> None:
    _ensure_results_tsv(results_path)
    with results_path.open("a", encoding="utf-8") as file:
        file.write(f"# {message}\n")


def _candidate_iteration_dirs(output_root: Path) -> list[Path]:
    return sorted(path for path in output_root.glob("iter_*") if path.is_dir())


def _next_iteration_number(output_root: Path) -> int:
    existing_numbers = []
    for path in _candidate_iteration_dirs(output_root):
        suffix = path.name.split("_")[-1]
        if suffix.isdigit():
            existing_numbers.append(int(suffix))
    return (max(existing_numbers) + 1) if existing_numbers else 1


def _read_preprocessing_hash(data_yaml: Path, research_val_dir: Path) -> str:
    candidates = [
        data_yaml.parent / "preprocessing_hash.txt",
        research_val_dir / "preprocessing_hash.txt",
        research_val_dir.parent / "preprocessing_hash.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            value = candidate.read_text(encoding="utf-8").strip()
            if value:
                return value
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(f"Unable to locate preprocessing hash. Checked: {checked}")


def _failure_row(run_id: str, preprocessing_hash: str, discard_reason: str) -> dict:
    return {
        "run_id": run_id,
        "timestamp": _timestamp(),
        "decision": "FAILED",
        "discard_reason": discard_reason,
        "research_val_scored_map50_95": 0.0,
        "locked_eval_scored_map50_95": 0.0,
        "locked_eval_nodule_cyst_recall": 0.0,
        "locked_eval_per_class_precision": _empty_per_class_metrics(),
        "locked_eval_per_class_recall": _empty_per_class_metrics(),
        "research_val_per_class_precision": _empty_per_class_metrics(),
        "research_val_per_class_recall": _empty_per_class_metrics(),
        "train_duration_seconds": 0.0,
        "timeout": False,
        "crashed": True,
        "preprocessing_hash": preprocessing_hash,
        "notes": "",
    }


def _load_reference_train_text(reference_row_path: Path) -> str:
    snapshot_path = reference_row_path.parent / "train.py.snapshot"
    if snapshot_path.exists():
        return snapshot_path.read_text(encoding="utf-8")
    return TRAIN_PATH.read_text(encoding="utf-8")


def _find_iteration_row(output_root: Path, run_id: str) -> tuple[dict | None, str | None]:
    for experiment_row_path in output_root.glob("iter_*/experiment_row.json"):
        row = _read_json(experiment_row_path)
        if row.get("run_id") == run_id:
            snapshot_path = experiment_row_path.parent / "train.py.snapshot"
            train_text = snapshot_path.read_text(encoding="utf-8") if snapshot_path.exists() else None
            return row, train_text
    return None, None


def _initial_reference(reference_row_path: Path, output_root: Path) -> tuple[dict, str]:
    reference = _read_json(reference_row_path)
    train_text = _load_reference_train_text(reference_row_path)
    results_rows = _read_results_rows(_results_path(output_root))
    best_row = reference
    best_train_text = train_text
    for row in results_rows:
        if row.get("decision") != "KEEP":
            continue
        try:
            locked_eval = float(row["locked_eval_scored_map50_95"])
        except (KeyError, TypeError, ValueError):
            continue
        if locked_eval > float(best_row["locked_eval_scored_map50_95"]):
            candidate_row, candidate_train_text = _find_iteration_row(output_root, row["run_id"])
            if candidate_row is not None:
                best_row = candidate_row
                if candidate_train_text is not None:
                    best_train_text = candidate_train_text
    return best_row, best_train_text


def _summarize_rows(rows: list[dict[str, str]], limit: int) -> str:
    if not rows:
        return "No prior rows."
    selected = rows[-limit:]
    summaries = []
    for row in selected:
        summaries.append(
            " | ".join(
                [
                    row.get("run_id", ""),
                    f"decision={row.get('decision', '')}",
                    f"rv={row.get('research_val_scored_map50_95', '')}",
                    f"le={row.get('locked_eval_scored_map50_95', '')}",
                    f"nr={row.get('locked_eval_nodule_cyst_recall', '')}",
                    f"reason={row.get('discard_reason', '') or '-'}",
                ]
            )
        )
    return "\n".join(summaries)


def _build_prompt(
    program_text: str,
    current_train_text: str,
    results_rows: list[dict[str, str]],
    reference_row: dict,
) -> str:
    return f"""You are mutating AutoDerm's src/train.py for one bounded autoresearch iteration.

Return ONLY a unified diff against src/train.py. No prose, no markdown fences, no explanations.

Constraints:
- Mutate only src/train.py.
- Keep every tunable choice as top-level module constants.
- Favor generalization under the held-out-primary rule.
- Do not touch preprocessing, splits, or locked_eval integrity.

Current reference metrics:
- research_val_scored_map50_95: {reference_row.get("research_val_scored_map50_95")}
- locked_eval_scored_map50_95: {reference_row.get("locked_eval_scored_map50_95")}
- locked_eval_nodule_cyst_recall: {reference_row.get("locked_eval_nodule_cyst_recall")}

Program contract:
{program_text}

Last 10 results rows:
{_summarize_rows(results_rows, 10)}

Last 5 iteration summaries:
{_summarize_rows(results_rows, 5)}

Current src/train.py:
```python
{current_train_text}
```
"""


def _run_codex_exec(prompt: str) -> str:
    attempts = 0
    last_error = ""
    while attempts < 2:
        attempts += 1
        try:
            completed = subprocess.run(
                ["codex", "exec"],
                input=prompt,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            last_error = f"{exc.__class__.__name__}: {exc}"
            completed = None
        else:
            if completed.returncode == 0:
                return completed.stdout
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            last_error = stderr or stdout or f"codex exec exited with code {completed.returncode}"

        if attempts < 2:
            time.sleep(CODEx_RETRY_DELAY_SECONDS)

    raise RuntimeError(f"codex exec failed after retry: {last_error}")


def _extract_unified_diff(response_text: str) -> str | None:
    fenced_match = re.findall(r"```(?:diff)?\n(.*?)```", response_text, flags=re.DOTALL)
    candidates = fenced_match + [response_text]
    for candidate in candidates:
        lines = candidate.splitlines()
        for index, line in enumerate(lines):
            if line.startswith("--- ") or line.startswith("diff --git "):
                diff_text = "\n".join(lines[index:]).strip()
                if diff_text:
                    return diff_text + ("\n" if not diff_text.endswith("\n") else "")
    return None


def _normalize_diff_path(path: str) -> str:
    normalized = path.strip().split("\t", 1)[0]
    if normalized.startswith("a/") or normalized.startswith("b/"):
        normalized = normalized[2:]
    return normalized


def _apply_unified_diff(base_text: str, diff_text: str) -> str:
    diff_lines = diff_text.splitlines()
    file_paths = []
    for line in diff_lines:
        if line.startswith("--- ") or line.startswith("+++ "):
            file_paths.append(_normalize_diff_path(line[4:]))
    if not file_paths or any(path != "src/train.py" for path in file_paths):
        raise RuntimeError("diff must target only src/train.py")

    base_lines = base_text.splitlines()
    output_lines: list[str] = []
    base_index = 0
    line_index = 0
    hunk_seen = False

    while line_index < len(diff_lines):
        line = diff_lines[line_index]
        if line.startswith(("diff --git ", "index ", "--- ", "+++ ")):
            line_index += 1
            continue
        if not line.startswith("@@"):
            line_index += 1
            continue

        hunk_seen = True
        match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if not match:
            raise RuntimeError(f"invalid hunk header: {line}")
        old_start = int(match.group(1))
        output_lines.extend(base_lines[base_index : old_start - 1])
        base_index = old_start - 1
        line_index += 1

        while line_index < len(diff_lines):
            hunk_line = diff_lines[line_index]
            if hunk_line.startswith("@@"):
                break
            if hunk_line.startswith("\\"):
                line_index += 1
                continue
            if not hunk_line:
                prefix = " "
                text = ""
            else:
                prefix = hunk_line[0]
                text = hunk_line[1:]
            if prefix == " ":
                if base_index >= len(base_lines) or base_lines[base_index] != text:
                    raise RuntimeError("diff context mismatch while applying mutation")
                output_lines.append(text)
                base_index += 1
            elif prefix == "-":
                if base_index >= len(base_lines) or base_lines[base_index] != text:
                    raise RuntimeError("diff removal mismatch while applying mutation")
                base_index += 1
            elif prefix == "+":
                output_lines.append(text)
            else:
                raise RuntimeError(f"unsupported diff line: {hunk_line}")
            line_index += 1

    if not hunk_seen:
        raise RuntimeError("no unified diff hunks found")

    output_lines.extend(base_lines[base_index:])
    return "\n".join(output_lines) + "\n"


def _run_experiment(
    output_root: Path,
    iteration_dir: Path,
    budget_seconds: int,
    data_yaml: Path,
    research_val_dir: Path,
    locked_eval_dir: Path,
    run_id: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix="autoderm-loop-") as temp_dir:
        temp_path = Path(temp_dir)
        (temp_path / "runs").symlink_to(output_root, target_is_directory=True)
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        command = [
            sys.executable,
            "-m",
            "src.run_experiment",
            "--train-py",
            str(iteration_dir / "train.py"),
            "--output-dir",
            str(iteration_dir),
            "--budget-seconds",
            str(budget_seconds),
            "--data-yaml",
            str(data_yaml),
            "--research-val-dir",
            str(research_val_dir),
            "--locked-eval-dir",
            str(locked_eval_dir),
            "--run-id",
            run_id,
        ]
        completed = subprocess.run(command, cwd=temp_path, env=env, capture_output=True, text=True, check=False)
        if completed.returncode != 0 and not (iteration_dir / "experiment_row.json").exists():
            error_message = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(error_message or f"src.run_experiment exited with code {completed.returncode}")


def _write_iteration_failure(iteration_dir: Path, row: dict, results_path: Path) -> None:
    _write_json(iteration_dir / "experiment_row.json", row)
    _append_results_row(results_path, row)


def _load_iteration_row(iteration_dir: Path) -> dict:
    experiment_row_path = iteration_dir / "experiment_row.json"
    if not experiment_row_path.exists():
        raise RuntimeError(f"Missing experiment_row.json for {iteration_dir}")
    return _read_json(experiment_row_path)


def run_loop(
    data_yaml: str,
    research_val_dir: str,
    locked_eval_dir: str,
    output_root: str,
    budget_seconds: int,
    max_iterations: int,
    reference_experiment_row: str,
) -> None:
    data_yaml_path = Path(data_yaml).expanduser().resolve()
    research_val_path = Path(research_val_dir).expanduser().resolve()
    locked_eval_path = Path(locked_eval_dir).expanduser().resolve()
    output_root_path = Path(output_root).expanduser().resolve()
    reference_row_path = Path(reference_experiment_row).expanduser().resolve()
    if not reference_row_path.exists():
        raise FileNotFoundError(f"Reference experiment row is required and was not found: {reference_row_path}")

    output_root_path.mkdir(parents=True, exist_ok=True)
    results_path = _results_path(output_root_path)
    _ensure_results_tsv(results_path)

    program_text = PROGRAM_PATH.read_text(encoding="utf-8")
    current_reference, current_train_text = _initial_reference(reference_row_path, output_root_path)
    preprocessing_hash = _read_preprocessing_hash(data_yaml_path, research_val_path)

    interrupt_requested = {"value": False}

    def _handle_sigint(signum: int, frame: object) -> None:
        interrupt_requested["value"] = True

    previous_handler = signal.signal(signal.SIGINT, _handle_sigint)
    completed_iterations = 0
    try:
        next_iteration = _next_iteration_number(output_root_path)
        while completed_iterations < max_iterations and not interrupt_requested["value"]:
            iteration_name = f"iter_{next_iteration:03d}"
            run_id = iteration_name
            iteration_dir = output_root_path / iteration_name
            iteration_dir.mkdir(parents=True, exist_ok=True)

            results_rows = _read_results_rows(results_path)
            prompt = _build_prompt(program_text, current_train_text, results_rows, current_reference)
            (iteration_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

            try:
                raw_response = _run_codex_exec(prompt)
            except RuntimeError:
                raise

            (iteration_dir / "codex_response.txt").write_text(raw_response, encoding="utf-8")
            diff_text = _extract_unified_diff(raw_response)
            if diff_text is None:
                row = _failure_row(run_id, preprocessing_hash, "codex_invalid_output:missing_unified_diff")
                (iteration_dir / "train.py.diff").write_text("", encoding="utf-8")
                _write_iteration_failure(iteration_dir, row, results_path)
                completed_iterations += 1
                next_iteration += 1
                continue

            (iteration_dir / "train.py.diff").write_text(diff_text, encoding="utf-8")
            try:
                mutated_train_text = _apply_unified_diff(current_train_text, diff_text)
            except RuntimeError:
                row = _failure_row(run_id, preprocessing_hash, "codex_invalid_output:diff_apply_failed")
                _write_iteration_failure(iteration_dir, row, results_path)
                completed_iterations += 1
                next_iteration += 1
                continue

            (iteration_dir / "train.py").write_text(mutated_train_text, encoding="utf-8")
            try:
                _run_experiment(
                    output_root=output_root_path,
                    iteration_dir=iteration_dir,
                    budget_seconds=budget_seconds,
                    data_yaml=data_yaml_path,
                    research_val_dir=research_val_path,
                    locked_eval_dir=locked_eval_path,
                    run_id=run_id,
                )
            except RuntimeError as exc:
                row = _failure_row(run_id, preprocessing_hash, f"run_experiment_error:{exc}")
                _write_iteration_failure(iteration_dir, row, results_path)
                completed_iterations += 1
                next_iteration += 1
                continue

            row = _load_iteration_row(iteration_dir)
            if row.get("decision") == "FAILED":
                completed_iterations += 1
                next_iteration += 1
                continue

            keep, discard_reason = candidate_beats_reference(candidate=row, reference=current_reference)
            row["decision"] = "KEEP" if keep else "DISCARD"
            row["discard_reason"] = None if keep else discard_reason
            _write_json(iteration_dir / "experiment_row.json", row)
            _update_results_row(results_path, run_id, row["decision"], row["discard_reason"])

            if keep:
                current_reference = row
                current_train_text = mutated_train_text

            completed_iterations += 1
            next_iteration += 1
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        if interrupt_requested["value"]:
            _append_comment(
                results_path,
                f"loop interrupted at {_timestamp()} after {completed_iterations} completed iteration(s)",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the outer Codex autoresearch loop for AutoDerm.")
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--research-val-dir", required=True)
    parser.add_argument("--locked-eval-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--budget-seconds", required=True, type=int)
    parser.add_argument("--max-iterations", required=True, type=int)
    parser.add_argument("--reference-experiment-row", required=True)
    args = parser.parse_args()

    run_loop(
        data_yaml=args.data_yaml,
        research_val_dir=args.research_val_dir,
        locked_eval_dir=args.locked_eval_dir,
        output_root=args.output_root,
        budget_seconds=args.budget_seconds,
        max_iterations=args.max_iterations,
        reference_experiment_row=args.reference_experiment_row,
    )


if __name__ == "__main__":
    main()
