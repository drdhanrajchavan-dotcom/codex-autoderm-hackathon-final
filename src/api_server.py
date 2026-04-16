"""FastAPI service for the local AutoDerm demo app."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .eval import hayashi_badge
from .types import ALL_CLASSES, SCORED_CLASSES


REPO_ROOT = Path(__file__).resolve().parent.parent
ACTIVE_CHECKPOINT_PATH = REPO_ROOT / "config" / "active_checkpoint.json"
PULLED_ARTIFACTS_DIR = REPO_ROOT / ".pulled_artifacts"
RUNS_DIR = PULLED_ARTIFACTS_DIR / "runs"
CONFIDENCE_THRESHOLD = 0.40
DEFAULT_MAX_UPLOAD_BYTES = 8 * 1024 * 1024
NO_ACTIVE_CHECKPOINT = {
    "error": "no active checkpoint",
    "message": "Baselines may still be running. Check Research tab.",
}
VALID_PREPROCESSING = {"uncropped", "cropped"}
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


MAX_UPLOAD_BYTES = _env_int("AUTODERM_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES)


app = FastAPI(title="AutoDerm Demo API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass(frozen=True)
class ActiveCheckpoint:
    weights_path: Path
    weights_path_raw: str
    preprocessing: str
    iteration_id: str
    config_mtime: float


@dataclass
class ModelCache:
    config_mtime: float | None = None
    weights_path: Path | None = None
    model: object | None = None


@dataclass(frozen=True)
class IterationCheckpointResolution:
    run_id: str
    run_dir: Path
    weights_path: Path | None
    weights_path_raw: str | None
    weight_file: str | None
    preprocessing: str | None
    row: dict[str, Any]
    unusable_reason: str | None

    @property
    def usable(self) -> bool:
        return self.unusable_reason is None and self.weights_path is not None and self.preprocessing is not None


_MODEL_CACHE = ModelCache()
_MODEL_LOCK = threading.Lock()


def _resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _repo_relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _read_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8", errors="replace").strip()
    return value or None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _active_checkpoint() -> ActiveCheckpoint | None:
    raw = _read_json(ACTIVE_CHECKPOINT_PATH)
    if raw is None:
        return None

    weights_path_raw = str(raw.get("weights_path", ""))
    preprocessing = str(raw.get("preprocessing", ""))
    iteration_id = str(raw.get("iteration_id", ""))
    if not weights_path_raw or preprocessing not in VALID_PREPROCESSING or not iteration_id:
        return None

    return ActiveCheckpoint(
        weights_path=_resolve_repo_path(weights_path_raw),
        weights_path_raw=weights_path_raw,
        preprocessing=preprocessing,
        iteration_id=iteration_id,
        config_mtime=ACTIVE_CHECKPOINT_PATH.stat().st_mtime,
    )


def _active_checkpoint_for_inference() -> ActiveCheckpoint | None:
    checkpoint = _active_checkpoint()
    if checkpoint is None or not checkpoint.weights_path.exists():
        return None
    return checkpoint


def _no_active_checkpoint_response() -> JSONResponse:
    return JSONResponse(status_code=503, content=NO_ACTIVE_CHECKPOINT)


def _get_model(checkpoint: ActiveCheckpoint) -> object:
    with _MODEL_LOCK:
        if (
            _MODEL_CACHE.model is not None
            and _MODEL_CACHE.config_mtime == checkpoint.config_mtime
            and _MODEL_CACHE.weights_path == checkpoint.weights_path
        ):
            return _MODEL_CACHE.model

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is required for inference") from exc

        model = YOLO(str(checkpoint.weights_path))
        _MODEL_CACHE.config_mtime = checkpoint.config_mtime
        _MODEL_CACHE.weights_path = checkpoint.weights_path
        _MODEL_CACHE.model = model
        return model


def _to_python_list(value: object) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)  # type: ignore[arg-type]


def _class_name(names: dict[int, str] | dict[str, str], class_index: object) -> str:
    index = int(class_index)
    configured = ALL_CLASSES[index] if 0 <= index < len(ALL_CLASSES) else str(index)
    model_name = str(names.get(index, configured))
    return model_name if model_name in ALL_CLASSES else configured


def _angle_from_points(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    return math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0])


def _xyxy_to_detection(
    class_name: str,
    confidence: float,
    xyxy: list[float],
    offset: tuple[int, int],
) -> dict[str, Any]:
    x1, y1, x2, y2 = [float(value) for value in xyxy[:4]]
    offset_x, offset_y = offset
    return {
        "class_name": class_name,
        "confidence": confidence,
        "bbox": [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y, 0.0],
    }


def _xywhr_to_detection(
    class_name: str,
    confidence: float,
    values: list[float],
    offset: tuple[int, int],
) -> dict[str, Any]:
    x_center, y_center, width, height, angle = [float(value) for value in values[:5]]
    offset_x, offset_y = offset
    return {
        "class_name": class_name,
        "confidence": confidence,
        "bbox": [
            x_center - width / 2 + offset_x,
            y_center - height / 2 + offset_y,
            x_center + width / 2 + offset_x,
            y_center + height / 2 + offset_y,
            angle,
        ],
    }


def _polygon_to_detection(
    class_name: str,
    confidence: float,
    polygon: list[list[float]] | list[tuple[float, float]],
    offset: tuple[int, int],
) -> dict[str, Any]:
    offset_x, offset_y = offset
    points = [(float(x) + offset_x, float(y) + offset_y) for x, y in polygon]
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return {
        "class_name": class_name,
        "confidence": confidence,
        "bbox": [min(xs), min(ys), max(xs), max(ys), _angle_from_points(points)],
    }


def _extract_detections(result: object, offset: tuple[int, int]) -> list[dict[str, Any]]:
    names = getattr(result, "names", {}) or {}
    obb = getattr(result, "obb", None)
    boxes = getattr(result, "boxes", None)
    detections: list[dict[str, Any]] = []

    if obb is not None and len(obb) > 0:
        class_indices = _to_python_list(obb.cls)
        confidences = _to_python_list(obb.conf)
        if hasattr(obb, "xyxyxyxy"):
            polygons = _to_python_list(obb.xyxyxyxy)
            for class_index, confidence, polygon in zip(class_indices, confidences, polygons):
                if float(confidence) < CONFIDENCE_THRESHOLD:
                    continue
                detections.append(
                    _polygon_to_detection(_class_name(names, class_index), float(confidence), polygon, offset)
                )
            return detections

        if hasattr(obb, "xywhr"):
            xywhr = _to_python_list(obb.xywhr)
            for class_index, confidence, values in zip(class_indices, confidences, xywhr):
                if float(confidence) < CONFIDENCE_THRESHOLD:
                    continue
                detections.append(_xywhr_to_detection(_class_name(names, class_index), float(confidence), values, offset))
            return detections

    if boxes is not None and len(boxes) > 0:
        class_indices = _to_python_list(boxes.cls)
        confidences = _to_python_list(boxes.conf)
        xyxy = _to_python_list(boxes.xyxy)
        for class_index, confidence, bbox in zip(class_indices, confidences, xyxy):
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue
            detections.append(_xyxy_to_detection(_class_name(names, class_index), float(confidence), bbox, offset))

    return detections


def _empty_counts() -> dict[str, int]:
    return {class_name: 0 for class_name in ALL_CLASSES}


def _preprocess_image(image: object, preprocessing: str) -> tuple[object, tuple[int, int]]:
    if preprocessing == "uncropped":
        return image, (0, 0)

    from .prepare import _cheek_region_crop_box

    crop_decision = _cheek_region_crop_box(image)
    left, top, right, bottom = crop_decision.box
    return image.crop((left, top, right, bottom)), (left, top)  # type: ignore[attr-defined]


async def _read_upload_image(upload: UploadFile) -> object:
    content = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"image upload exceeds {limit_mb:g} MB limit")
    if not content:
        raise HTTPException(status_code=400, detail="empty upload")

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for image upload handling") from exc

    try:
        with Image.open(io.BytesIO(content)) as image:
            return image.convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid image upload") from exc


def _inference_response(image: object, checkpoint: ActiveCheckpoint) -> dict[str, Any]:
    model = _get_model(checkpoint)
    preprocessed_image, offset = _preprocess_image(image, checkpoint.preprocessing)
    results = model.predict(source=preprocessed_image, conf=CONFIDENCE_THRESHOLD, verbose=False)  # type: ignore[attr-defined]
    result = results[0] if results else None
    detections = _extract_detections(result, offset) if result is not None else []

    counts = _empty_counts()
    for detection in detections:
        class_name = detection["class_name"]
        if class_name in counts:
            counts[class_name] += 1

    return {
        "detections": detections,
        "counts": counts,
        "hayashi_badge": hayashi_badge(counts),
    }


def _active_checkpoint_payload() -> dict[str, Any]:
    raw = _read_json(ACTIVE_CHECKPOINT_PATH)
    if raw is None:
        return {
            "weights_path": None,
            "preprocessing": None,
            "iteration_id": None,
            "weights_exists": False,
        }
    weights_path = str(raw.get("weights_path", ""))
    return {
        "weights_path": weights_path or None,
        "preprocessing": raw.get("preprocessing"),
        "iteration_id": raw.get("iteration_id"),
        "weights_exists": bool(weights_path and _resolve_repo_path(weights_path).exists()),
    }


def _candidate_results_paths() -> list[Path]:
    return [PULLED_ARTIFACTS_DIR / "results.tsv", RUNS_DIR / "results.tsv"]


def _read_tsv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not lines:
        return []
    return list(csv.DictReader(io.StringIO("\n".join(lines)), delimiter="\t"))


def _selected_results_path() -> Path | None:
    existing = [path for path in _candidate_results_paths() if path.exists()]
    for path in existing:
        rows = _read_tsv_rows(path)
        if rows and "run_id" in rows[0]:
            return path
        if not rows:
            header = path.read_text(encoding="utf-8").splitlines()
            if header and "run_id" in header[0].split("\t"):
                return path
    return existing[0] if existing else None


def _float_value(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value not in ("", None) else default
    except (TypeError, ValueError):
        return default


def _bool_value(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"1", "true", "yes"}


def _flat_precision(row: dict[str, str], prefix: str) -> dict[str, float]:
    return {
        class_name: _float_value(row, f"{prefix}_{class_name}_precision")
        for class_name in SCORED_CLASSES
    }


def _row_from_flat_tsv(row: dict[str, str]) -> dict[str, Any]:
    run_id = row.get("run_id") or row.get("iteration_id") or ""
    return {
        "run_id": run_id,
        "timestamp": row.get("timestamp", ""),
        "decision": row.get("decision") or "PENDING_KEEP_RULE",
        "discard_reason": row.get("discard_reason") or None,
        "research_val_scored_map50_95": _float_value(
            row,
            "research_val_scored_map50_95",
            _float_value(row, "research_val"),
        ),
        "locked_eval_scored_map50_95": _float_value(
            row,
            "locked_eval_scored_map50_95",
            _float_value(row, "locked_eval"),
        ),
        "locked_eval_nodule_cyst_recall": _float_value(row, "locked_eval_nodule_cyst_recall"),
        "locked_eval_per_class_precision": _flat_precision(row, "locked_eval"),
        "train_duration_seconds": _float_value(row, "train_duration_seconds"),
        "timeout": _bool_value(row, "timeout"),
        "crashed": _bool_value(row, "crashed"),
        "preprocessing_hash": row.get("preprocessing_hash", ""),
        "notes": row.get("notes", ""),
        "preprocessing": row.get("preprocessing"),
        "weights_path": row.get("weights_path"),
    }


def _iterations() -> list[dict[str, Any]]:
    path = _selected_results_path()
    if path is None:
        return []
    return [_row_from_flat_tsv(row) for row in _read_tsv_rows(path) if row.get("run_id") or row.get("iteration_id")]


def _iteration_row_from_results(run_id: str) -> dict[str, Any] | None:
    for row in _iterations():
        if row.get("run_id") == run_id:
            return row
    return None


def _merge_row(primary: dict[str, Any] | None, fallback: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(fallback or {})
    for key, value in (primary or {}).items():
        if value not in ("", None):
            merged[key] = value
        elif key not in merged:
            merged[key] = value
    return merged


def _row_for_run(run_id: str, run_dir: Path | None = None) -> dict[str, Any]:
    run_dir = run_dir or _safe_run_dir(run_id)
    row_path = run_dir / "experiment_row.json"
    json_row = _read_json(row_path) if row_path.exists() else None
    result_row = _iteration_row_from_results(run_id)
    return _merge_row(json_row, result_row)


def _safe_run_dir(run_id: str) -> Path:
    clean = run_id.strip()
    if clean != run_id or not RUN_ID_PATTERN.fullmatch(clean):
        raise HTTPException(status_code=400, detail=f"invalid iteration_id: {run_id}")

    runs_root = RUNS_DIR.resolve()
    run_dir = (RUNS_DIR / clean).resolve()
    if run_dir.parent != runs_root:
        raise HTTPException(status_code=400, detail=f"invalid iteration_id: {run_id}")
    return run_dir


def _preprocessing_hash_candidates(preprocessing: str) -> list[Path]:
    return [
        REPO_ROOT / "data" / preprocessing / "preprocessing_hash.txt",
        RUNS_DIR / f"baseline_{preprocessing}" / "weights" / "preprocessing_hash.txt",
        RUNS_DIR / f"baseline_{preprocessing}" / "experiment_row.json",
    ]


def _hash_from_candidate(path: Path) -> str | None:
    if path.suffix == ".json":
        payload = _read_json(path)
        value = payload.get("preprocessing_hash") if payload else None
        return str(value).strip() if value else None
    return _read_hash(path)


def _known_preprocessing_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for preprocessing in VALID_PREPROCESSING:
        for path in _preprocessing_hash_candidates(preprocessing):
            preprocessing_hash = _hash_from_candidate(path)
            if preprocessing_hash:
                hashes[preprocessing_hash] = preprocessing
    return hashes


def _weights_path_for_run(run_dir: Path) -> tuple[Path | None, str | None]:
    weights_dir = run_dir / "weights"
    for file_name in ("best.pt", "last.pt"):
        weights_path = weights_dir / file_name
        if weights_path.exists():
            return weights_path.resolve(), file_name
    return None, None


def _preprocessing_for_resolution(
    run_id: str,
    row: dict[str, Any],
    weights_path: Path | None,
) -> tuple[str | None, str | None]:
    candidates: list[tuple[str, str]] = []
    hash_to_preprocessing = _known_preprocessing_hashes()

    row_preprocessing = row.get("preprocessing")
    if row_preprocessing:
        if str(row_preprocessing) not in VALID_PREPROCESSING:
            return None, f"unknown preprocessing value: {row_preprocessing}"
        candidates.append(("row.preprocessing", str(row_preprocessing)))

    if run_id.startswith("baseline_"):
        baseline_preprocessing = run_id.removeprefix("baseline_")
        if baseline_preprocessing in VALID_PREPROCESSING:
            candidates.append(("run_id", baseline_preprocessing))

    if weights_path is not None:
        weights_hash = _read_hash(weights_path.parent / "preprocessing_hash.txt")
        if not weights_hash:
            return None, "missing weights preprocessing hash"
        weights_preprocessing = hash_to_preprocessing.get(weights_hash)
        if weights_preprocessing is None:
            return None, f"unknown preprocessing hash: {weights_hash}"
        candidates.append(("weights.preprocessing_hash", weights_preprocessing))

    row_hash = row.get("preprocessing_hash")
    if row_hash:
        row_hash_preprocessing = hash_to_preprocessing.get(str(row_hash))
        if row_hash_preprocessing is not None:
            candidates.append(("row.preprocessing_hash", row_hash_preprocessing))

    values = {preprocessing for _, preprocessing in candidates}
    if len(values) > 1:
        sources = ", ".join(f"{source}={preprocessing}" for source, preprocessing in candidates)
        return None, f"preprocessing mismatch: {sources}"
    if values:
        return next(iter(values)), None
    return None, "unknown preprocessing"


def _resolve_iteration_checkpoint(run_id: str) -> IterationCheckpointResolution:
    run_dir = _safe_run_dir(run_id)
    row = _row_for_run(run_id, run_dir)
    weights_path, weight_file = _weights_path_for_run(run_dir)
    weights_path_raw = _repo_relative_path(weights_path) if weights_path is not None else None

    if not run_dir.exists():
        return IterationCheckpointResolution(
            run_id=run_id,
            run_dir=run_dir,
            weights_path=weights_path,
            weights_path_raw=weights_path_raw,
            weight_file=weight_file,
            preprocessing=None,
            row=row,
            unusable_reason="run directory missing",
        )

    if weights_path is None:
        return IterationCheckpointResolution(
            run_id=run_id,
            run_dir=run_dir,
            weights_path=None,
            weights_path_raw=None,
            weight_file=None,
            preprocessing=None,
            row=row,
            unusable_reason="missing weights",
        )

    preprocessing, preprocessing_error = _preprocessing_for_resolution(run_id, row, weights_path)
    return IterationCheckpointResolution(
        run_id=run_id,
        run_dir=run_dir,
        weights_path=weights_path,
        weights_path_raw=weights_path_raw,
        weight_file=weight_file,
        preprocessing=preprocessing,
        row=row,
        unusable_reason=preprocessing_error,
    )


def _float_from_any(value: object, default: float = 0.0) -> float:
    try:
        return float(value) if value not in ("", None) else default
    except (TypeError, ValueError):
        return default


def _checkpoint_option_payload(resolution: IterationCheckpointResolution) -> dict[str, Any]:
    row = resolution.row
    decision = str(row.get("decision") or "PENDING_KEEP_RULE")
    return {
        "run_id": resolution.run_id,
        "decision": decision,
        "discard_reason": row.get("discard_reason") or None,
        "timestamp": row.get("timestamp", ""),
        "research_val_scored_map50_95": _float_from_any(row.get("research_val_scored_map50_95")),
        "locked_eval_scored_map50_95": _float_from_any(row.get("locked_eval_scored_map50_95")),
        "locked_eval_nodule_cyst_recall": _float_from_any(row.get("locked_eval_nodule_cyst_recall")),
        "preprocessing": resolution.preprocessing,
        "weights_path": resolution.weights_path_raw,
        "weight_file": resolution.weight_file,
        "weights_exists": resolution.weights_path is not None,
        "usable": resolution.usable,
        "unusable_reason": resolution.unusable_reason,
        "official_keep": decision == "KEEP",
        "exploratory_only": decision != "KEEP",
    }


def _checkpoint_for_iteration(run_id: str) -> ActiveCheckpoint:
    resolution = _resolve_iteration_checkpoint(run_id)
    if not resolution.usable or resolution.weights_path is None or resolution.preprocessing is None:
        reason = resolution.unusable_reason or "not usable"
        raise HTTPException(status_code=400, detail=f"iteration {run_id} is not usable: {reason}")

    return ActiveCheckpoint(
        weights_path=resolution.weights_path,
        weights_path_raw=resolution.weights_path_raw or str(resolution.weights_path),
        preprocessing=resolution.preprocessing,
        iteration_id=run_id,
        config_mtime=resolution.weights_path.stat().st_mtime,
    )


def _all_checkpoint_options() -> list[dict[str, Any]]:
    run_ids = {str(row.get("run_id")) for row in _iterations() if row.get("run_id")}
    if RUNS_DIR.exists():
        run_ids.update(
            path.name
            for path in RUNS_DIR.iterdir()
            if path.is_dir() and (path.name.startswith("iter_") or path.name.startswith("baseline_"))
        )

    options = [_checkpoint_option_payload(_resolve_iteration_checkpoint(run_id)) for run_id in run_ids]
    return sorted(
        options,
        key=lambda option: (
            str(option.get("timestamp") or ""),
            str(option.get("run_id") or ""),
        ),
        reverse=True,
    )


def _counts_diff(after: dict[str, Any], before: dict[str, Any]) -> dict[str, int]:
    return {
        class_name: int(after["counts"].get(class_name, 0)) - int(before["counts"].get(class_name, 0))
        for class_name in ALL_CLASSES
    }


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _tail_text(path: Path, line_count: int = 50) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    checkpoint = _active_checkpoint()
    return {
        "status": "ok",
        "active_checkpoint": checkpoint is not None,
        "weights_exists": bool(checkpoint and checkpoint.weights_path.exists()),
    }


@app.get("/api/active_checkpoint")
def active_checkpoint() -> dict[str, Any]:
    return _active_checkpoint_payload()


@app.get("/api/iterations")
def iterations() -> list[dict[str, Any]]:
    return _iterations()


@app.get("/api/inference_checkpoints")
def inference_checkpoints() -> list[dict[str, Any]]:
    return _all_checkpoint_options()


@app.get("/api/iteration/{run_id}")
def iteration_detail(run_id: str) -> dict[str, Any]:
    run_dir = _safe_run_dir(run_id)
    row_path = run_dir / "experiment_row.json"
    experiment_row = _read_json(row_path) if row_path.exists() else _iteration_row_from_results(run_id)
    if experiment_row is None and not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"iteration not found: {run_id}")

    return {
        "experiment_row": experiment_row,
        "train_diff_text": _read_text(run_dir / "train.py.diff"),
        "prompt_text": _read_text(run_dir / "prompt.txt"),
        "codex_response_text": _read_text(run_dir / "codex_response.txt"),
        "worker_stdout_tail": _tail_text(run_dir / "worker_stdout.log", line_count=50),
    }


@app.post("/api/infer")
async def infer(
    image: UploadFile = File(...),
    iteration_id: str | None = Form(None),
) -> Any:
    if iteration_id:
        checkpoint = _checkpoint_for_iteration(iteration_id)
    else:
        checkpoint = _active_checkpoint_for_inference()
        if checkpoint is None:
            return _no_active_checkpoint_response()
    upload_image = await _read_upload_image(image)
    return _inference_response(upload_image, checkpoint)


@app.post("/api/compare_iterations")
async def compare_iterations(
    image: UploadFile = File(...),
    iteration_a: str = Form(...),
    iteration_b: str = Form(...),
) -> Any:
    checkpoint_a = _checkpoint_for_iteration(iteration_a)
    checkpoint_b = _checkpoint_for_iteration(iteration_b)
    upload_image = await _read_upload_image(image)

    response_a = _inference_response(upload_image, checkpoint_a)
    response_b = _inference_response(upload_image, checkpoint_b)
    resolution_a = _resolve_iteration_checkpoint(iteration_a)
    resolution_b = _resolve_iteration_checkpoint(iteration_b)

    return {
        "iteration_a": _checkpoint_option_payload(resolution_a),
        "iteration_b": _checkpoint_option_payload(resolution_b),
        "a": response_a,
        "b": response_b,
        "deltas": {
            "counts_diff": _counts_diff(response_b, response_a),
            "badge_change": {
                "from": response_a["hayashi_badge"],
                "to": response_b["hayashi_badge"],
            },
        },
    }


@app.post("/api/infer_pair")
async def infer_pair(
    before: UploadFile = File(...),
    after: UploadFile = File(...),
) -> Any:
    checkpoint = _active_checkpoint_for_inference()
    if checkpoint is None:
        return _no_active_checkpoint_response()

    before_response = _inference_response(await _read_upload_image(before), checkpoint)
    after_response = _inference_response(await _read_upload_image(after), checkpoint)
    return {
        "before": before_response,
        "after": after_response,
        "deltas": {
            "counts_diff": _counts_diff(after_response, before_response),
            "badge_change": {
                "from": before_response["hayashi_badge"],
                "to": after_response["hayashi_badge"],
            },
        },
    }


def main() -> None:
    import uvicorn

    uvicorn.run("src.api_server:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
