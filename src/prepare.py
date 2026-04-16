"""Deterministic dataset preparation for AutoDerm.

The prepared Ultralytics dataset intentionally points validation at
``research_val`` only. ``locked_eval`` remains immutable and is evaluated
separately by ``src.eval``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import random
import shutil
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .types import ALL_CLASSES


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "research_val", "locked_eval")
SOURCE_SPLIT_NAMES = {"train", "val", "valid", "validation", "test", "research_val", "locked_eval"}
LEFT_CHEEK_LANDMARKS = (58, 93, 132, 136, 172, 234)
RIGHT_CHEEK_LANDMARKS = (288, 323, 361, 365, 397, 454)
FACE_ANCHOR_LANDMARKS = (
    10,
    21,
    54,
    58,
    67,
    93,
    103,
    109,
    127,
    132,
    136,
    148,
    149,
    150,
    152,
    162,
    172,
    176,
    234,
    251,
    284,
    288,
    297,
    323,
    332,
    338,
    356,
    361,
    365,
    377,
    378,
    379,
    389,
    397,
    400,
    454,
)
FACE_MESH_NOSE_TIP = 1
FACE_MESH_DETECT_MAX_SIDE = 1024
FACE_LANDMARKER_TASK_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
FACE_LANDMARKER_TASK_ENV = "AUTODERM_FACE_LANDMARKER_TASK"
FACE_CLOSEUP_FALLBACK_SHORT_SIDE_FRACTION = 0.82
FACE_CLOSEUP_MIN_WIDTH_SHORT_SIDE_FRACTION = 0.58
FACE_CLOSEUP_MIN_HEIGHT_SHORT_SIDE_FRACTION = 0.66
FACE_CLOSEUP_MAX_WIDTH_SHORT_SIDE_FRACTION = 0.84
FACE_CLOSEUP_MAX_HEIGHT_SHORT_SIDE_FRACTION = 0.92
FACE_CLOSEUP_SIDE_MARGIN_FRACTION = 0.12
FACE_CLOSEUP_TOP_MARGIN_FRACTION = 0.18
FACE_CLOSEUP_BOTTOM_MARGIN_FRACTION = 0.08


_FACE_MESH_DETECTOR: object | None = None
_FACE_MESH_UNAVAILABLE = False


@dataclass(frozen=True)
class SourceRecord:
    image_path: Path
    label_path: Path | None
    patient_id: str
    output_stem: str
    class_counts: dict[str, int]


@dataclass(frozen=True)
class CropDecision:
    box: tuple[int, int, int, int]
    method: str


@dataclass(frozen=True)
class FaceMeshDetector:
    backend: str
    detector: object


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES and not any(part.startswith(".") for part in path.parts)


def _safe_output_stem(relative_path: Path) -> str:
    parts = [part for part in relative_path.with_suffix("").parts if part not in {".", ""}]
    return "__".join(parts)


def _infer_patient_id(relative_path: Path) -> str:
    parts = [part for part in relative_path.parts[:-1] if part not in {"images", "labels"} | SOURCE_SPLIT_NAMES]
    if parts:
        return parts[0]

    stem = relative_path.stem
    for separator in ("__", "_", "-"):
        if separator in stem:
            return stem.split(separator, 1)[0]
    return stem


def _candidate_label_paths(source_dir: Path, image_path: Path) -> list[Path]:
    relative = image_path.relative_to(source_dir)
    candidates: list[Path] = []

    if "images" in relative.parts:
        parts = list(relative.parts)
        parts[parts.index("images")] = "labels"
        candidates.append(source_dir / Path(*parts).with_suffix(".txt"))

    candidates.extend(
        [
            image_path.with_suffix(".txt"),
            source_dir / "labels" / relative.with_suffix(".txt").name,
            source_dir / "labels" / relative.with_suffix(".txt"),
        ]
    )
    return candidates


def _find_label(source_dir: Path, image_path: Path) -> Path | None:
    for candidate in _candidate_label_paths(source_dir, image_path):
        if candidate.exists():
            return candidate
    return None


def _label_class_counts(label_path: Path | None) -> dict[str, int]:
    counts = {class_name: 0 for class_name in ALL_CLASSES}
    if label_path is None or not label_path.exists():
        return counts

    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = raw_line.split()
        if not parts:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number} has a non-numeric class id.") from exc
        _validate_class_id(class_id)
        counts[ALL_CLASSES[class_id]] += 1
    return counts


def _discover_records(source_dir: Path) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for image_path in sorted(path for path in source_dir.rglob("*") if _is_image(path)):
        relative = image_path.relative_to(source_dir)
        label_path = _find_label(source_dir, image_path)
        records.append(
            SourceRecord(
                image_path=image_path,
                label_path=label_path,
                patient_id=_infer_patient_id(relative),
                output_stem=_safe_output_stem(relative),
                class_counts=_label_class_counts(label_path),
            )
        )
    return records


def _stable_random_value(value: str, split_seed: int) -> float:
    digest = hashlib.sha256(f"{split_seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / 16**16


def _use_image_level_groups(records: list[SourceRecord]) -> bool:
    if not records:
        return False
    unique_patient_ids = {record.patient_id for record in records}
    minimum_reasonable_patient_ids = max(10, round(len(records) * 0.10))
    return len(unique_patient_ids) < minimum_reasonable_patient_ids


def _record_group_id(record: SourceRecord, use_image_level_groups: bool) -> str:
    return record.output_stem if use_image_level_groups else record.patient_id


def _target_split_sizes(total_images: int) -> dict[str, int]:
    train_count = round(total_images * 0.70)
    research_val_count = round(total_images * 0.15)
    locked_eval_count = total_images - train_count - research_val_count
    if total_images >= 3:
        train_count = max(1, train_count)
        research_val_count = max(1, research_val_count)
        locked_eval_count = max(1, locked_eval_count)
        while train_count + research_val_count + locked_eval_count > total_images:
            train_count -= 1
    return {
        "train": train_count,
        "research_val": research_val_count,
        "locked_eval": locked_eval_count,
    }


def _group_records(records: list[SourceRecord]) -> tuple[dict[str, list[SourceRecord]], bool]:
    image_level_groups = _use_image_level_groups(records)
    grouped: dict[str, list[SourceRecord]] = {}
    for record in records:
        grouped.setdefault(_record_group_id(record, image_level_groups), []).append(record)
    return grouped, image_level_groups


def _group_class_counts(group_records: list[SourceRecord]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in group_records:
        counts.update(record.class_counts)
    return counts


def _choose_split_for_group(
    group_size: int,
    group_counts: Counter[str],
    current_images: dict[str, int],
    current_counts: dict[str, Counter[str]],
    target_images: dict[str, int],
    target_counts: dict[str, dict[str, float]],
) -> str:
    best_split = "train"
    best_score = float("inf")
    any_under_capacity = any(current_images[split] < target_images[split] for split in SPLITS)

    for split in SPLITS:
        proposed_images = current_images[split] + group_size
        image_target = max(1, target_images[split])
        image_error = ((proposed_images - image_target) / image_target) ** 2
        if any_under_capacity and current_images[split] >= target_images[split]:
            image_error += 100.0

        class_error = 0.0
        for class_name in ALL_CLASSES:
            class_target = target_counts[split][class_name]
            if class_target <= 0:
                continue
            proposed_count = current_counts[split][class_name] + group_counts[class_name]
            class_error += ((proposed_count - class_target) / class_target) ** 2

        score = image_error + class_error
        if score < best_score:
            best_split = split
            best_score = score
    return best_split


def _image_level_assignment_cost(records: list[SourceRecord], assignment: dict[int, str], target_images: dict[str, int]) -> float:
    total_counts: Counter[str] = Counter()
    current_images: Counter[str] = Counter()
    current_counts = {split: Counter() for split in SPLITS}
    for index, record in enumerate(records):
        total_counts.update(record.class_counts)
        split = assignment[index]
        current_images[split] += 1
        current_counts[split].update(record.class_counts)

    cost = 0.0
    total_images = max(1, len(records))
    for split in SPLITS:
        split_fraction = target_images[split] / total_images
        image_target = max(1, target_images[split])
        cost += 100.0 * ((current_images[split] - image_target) / image_target) ** 2
        for class_name in ALL_CLASSES:
            class_target = total_counts[class_name] * split_fraction
            if class_target <= 0:
                continue
            # Research and locked eval class balance matters most for reliable model selection.
            split_weight = 5.0 if split in {"research_val", "locked_eval"} else 1.0
            cost += split_weight * ((current_counts[split][class_name] - class_target) / class_target) ** 2
    return cost


def _random_exact_image_assignment(records: list[SourceRecord], target_images: dict[str, int], split_seed: int) -> dict[int, str]:
    indexes = list(range(len(records)))
    random.Random(split_seed).shuffle(indexes)
    assignment: dict[int, str] = {}
    train_end = target_images["train"]
    research_val_end = train_end + target_images["research_val"]
    for index in indexes[:train_end]:
        assignment[index] = "train"
    for index in indexes[train_end:research_val_end]:
        assignment[index] = "research_val"
    for index in indexes[research_val_end:]:
        assignment[index] = "locked_eval"
    return assignment


def _optimized_image_level_split(records: list[SourceRecord], split_seed: int) -> dict[str, str]:
    target_images = _target_split_sizes(len(records))
    best_assignment: dict[int, str] | None = None
    best_cost = float("inf")

    for offset in range(256):
        assignment = _random_exact_image_assignment(records, target_images, split_seed + offset)
        cost = _image_level_assignment_cost(records, assignment, target_images)
        if cost < best_cost:
            best_assignment = assignment
            best_cost = cost

    if best_assignment is None:
        raise RuntimeError("Unable to create an image-level split assignment.")

    improved = True
    while improved:
        improved = False
        for left_index in range(len(records)):
            for right_index in range(left_index + 1, len(records)):
                if best_assignment[left_index] == best_assignment[right_index]:
                    continue
                best_assignment[left_index], best_assignment[right_index] = (
                    best_assignment[right_index],
                    best_assignment[left_index],
                )
                cost = _image_level_assignment_cost(records, best_assignment, target_images)
                if cost + 1e-12 < best_cost:
                    best_cost = cost
                    improved = True
                else:
                    best_assignment[left_index], best_assignment[right_index] = (
                        best_assignment[right_index],
                        best_assignment[left_index],
                    )

    return {record.output_stem: best_assignment[index] for index, record in enumerate(records)}


def _balanced_split_records(records: list[SourceRecord], split_seed: int) -> tuple[dict[str, str], str]:
    grouped, image_level_groups = _group_records(records)
    if image_level_groups:
        return _optimized_image_level_split(records, split_seed), "image_level_balanced"

    target_images = _target_split_sizes(len(records))
    total_counts: Counter[str] = Counter()
    for record in records:
        total_counts.update(record.class_counts)

    fractions = {
        "train": target_images["train"] / max(1, len(records)),
        "research_val": target_images["research_val"] / max(1, len(records)),
        "locked_eval": target_images["locked_eval"] / max(1, len(records)),
    }
    target_counts = {
        split: {class_name: total_counts[class_name] * fractions[split] for class_name in ALL_CLASSES}
        for split in SPLITS
    }

    group_items = sorted(
        grouped.items(),
        key=lambda item: (
            -sum(
                count / max(1, total_counts[class_name])
                for class_name, count in _group_class_counts(item[1]).items()
                if count
            ),
            -sum(record.class_counts[class_name] for record in item[1] for class_name in ALL_CLASSES),
            _stable_random_value(item[0], split_seed),
            item[0],
        ),
    )

    current_images = {split: 0 for split in SPLITS}
    current_counts = {split: Counter() for split in SPLITS}
    split_by_group: dict[str, str] = {}
    for group_id, group_records in group_items:
        group_counts = _group_class_counts(group_records)
        split = _choose_split_for_group(
            group_size=len(group_records),
            group_counts=group_counts,
            current_images=current_images,
            current_counts=current_counts,
            target_images=target_images,
            target_counts=target_counts,
        )
        split_by_group[group_id] = split
        current_images[split] += len(group_records)
        current_counts[split].update(group_counts)

    return split_by_group, "patient_level_balanced"


def _split_patients(patient_ids: list[str], split_seed: int) -> dict[str, str]:
    shuffled = sorted(set(patient_ids))
    random.Random(split_seed).shuffle(shuffled)

    train_cutoff = round(len(shuffled) * 0.70)
    research_val_cutoff = round(len(shuffled) * 0.85)

    split_by_patient: dict[str, str] = {}
    for patient_id in shuffled[:train_cutoff]:
        split_by_patient[patient_id] = "train"
    for patient_id in shuffled[train_cutoff:research_val_cutoff]:
        split_by_patient[patient_id] = "research_val"
    for patient_id in shuffled[research_val_cutoff:]:
        split_by_patient[patient_id] = "locked_eval"
    return split_by_patient


def _write_split_reports(
    output_dir: Path,
    records_by_split: dict[str, list[SourceRecord]],
    split_strategy: str,
    split_seed: int,
) -> None:
    summary = {
        "split_strategy": split_strategy,
        "split_seed": split_seed,
        "images": {split: len(records_by_split[split]) for split in SPLITS},
        "regions": {
            split: {
                class_name: sum(record.class_counts[class_name] for record in records_by_split[split])
                for class_name in ALL_CLASSES
            }
            for split in SPLITS
        },
    }
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    header = ["split", "image", *ALL_CLASSES, "total_regions"]
    lines = ["\t".join(header)]
    for split in SPLITS:
        for record in sorted(records_by_split[split], key=lambda item: item.output_stem):
            total_regions = sum(record.class_counts.values())
            row = [
                split,
                record.output_stem,
                *[str(record.class_counts[class_name]) for class_name in ALL_CLASSES],
                str(total_regions),
            ]
            lines.append("\t".join(row))
    (output_dir / "per_photo_region_counts.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _clamped_square_crop_box(width: int, height: int, center_x: float, center_y: float, side: float) -> tuple[int, int, int, int]:
    crop_side = max(1, min(width, height, round(side)))
    left = round(center_x - crop_side / 2)
    top = round(center_y - crop_side / 2)
    left = min(max(0, left), max(0, width - crop_side))
    top = min(max(0, top), max(0, height - crop_side))
    return left, top, left + crop_side, top + crop_side


def _face_region_crop_box(width: int, height: int) -> tuple[int, int, int, int]:
    side = min(width, height)
    crop_side = max(1, round(side * FACE_CLOSEUP_FALLBACK_SHORT_SIDE_FRACTION))
    return _clamped_square_crop_box(width, height, width / 2, height / 2, crop_side)


def _clamped_rect_crop_box(width: int, height: int, left: float, top: float, right: float, bottom: float) -> tuple[int, int, int, int]:
    crop_width = max(1, min(width, round(right - left)))
    crop_height = max(1, min(height, round(bottom - top)))
    crop_left = round((left + right) / 2 - crop_width / 2)
    crop_top = round((top + bottom) / 2 - crop_height / 2)
    crop_left = min(max(0, crop_left), max(0, width - crop_width))
    crop_top = min(max(0, crop_top), max(0, height - crop_height))
    return crop_left, crop_top, crop_left + crop_width, crop_top + crop_height


def _expand_crop_bounds(
    width: int,
    height: int,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> tuple[int, int, int, int]:
    short_side = min(width, height)
    target_width = right - left
    target_height = bottom - top
    min_width = short_side * FACE_CLOSEUP_MIN_WIDTH_SHORT_SIDE_FRACTION
    min_height = short_side * FACE_CLOSEUP_MIN_HEIGHT_SHORT_SIDE_FRACTION
    max_width = short_side * FACE_CLOSEUP_MAX_WIDTH_SHORT_SIDE_FRACTION
    max_height = short_side * FACE_CLOSEUP_MAX_HEIGHT_SHORT_SIDE_FRACTION
    target_width = min(max(target_width, min_width), max_width, width)
    target_height = min(max(target_height, min_height), max_height, height)

    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    return _clamped_rect_crop_box(
        width,
        height,
        center_x - target_width / 2,
        center_y - target_height / 2,
        center_x + target_width / 2,
        center_y + target_height / 2,
    )


def _face_landmarker_task_path(download: bool = False) -> Path | None:
    configured = os.environ.get(FACE_LANDMARKER_TASK_ENV)
    if configured:
        path = Path(configured).expanduser().resolve()
        return path if path.exists() else None

    cache_path = Path.home() / ".cache" / "autoderm" / "face_landmarker.task"
    if cache_path.exists():
        return cache_path
    if not download:
        return None

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".task.tmp")
        urllib.request.urlretrieve(FACE_LANDMARKER_TASK_URL, tmp_path)
        tmp_path.replace(cache_path)
    except Exception:
        return None
    return cache_path if cache_path.exists() else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _legacy_face_mesh_class() -> object | None:
    try:
        import mediapipe as mp
    except ImportError:
        return None

    solutions = getattr(mp, "solutions", None)
    face_mesh = getattr(solutions, "face_mesh", None) if solutions is not None else None
    return getattr(face_mesh, "FaceMesh", None) if face_mesh is not None else None


def _face_mesh_dependency_signature(allow_download: bool = False) -> str:
    try:
        version = importlib.metadata.version("mediapipe")
    except importlib.metadata.PackageNotFoundError:
        return "mediapipe=unavailable"

    if _legacy_face_mesh_class() is not None:
        return f"mediapipe={version};backend=solutions.face_mesh"

    task_path = _face_landmarker_task_path(download=allow_download)
    if task_path is None:
        return f"mediapipe={version};backend=center_fallback;face_landmarker_task=missing"
    return f"mediapipe={version};backend=tasks.FaceLandmarker;face_landmarker_task_sha256={_file_sha256(task_path)}"


def _face_mesh_detector() -> FaceMeshDetector | None:
    global _FACE_MESH_DETECTOR, _FACE_MESH_UNAVAILABLE
    if _FACE_MESH_UNAVAILABLE:
        return None
    if _FACE_MESH_DETECTOR is not None:
        return _FACE_MESH_DETECTOR

    legacy_face_mesh = _legacy_face_mesh_class()
    if legacy_face_mesh is not None:
        _FACE_MESH_DETECTOR = FaceMeshDetector(
            backend="solutions.face_mesh",
            detector=legacy_face_mesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.45,
            ),
        )
        return _FACE_MESH_DETECTOR

    task_path = _face_landmarker_task_path(download=True)
    if task_path is None:
        _FACE_MESH_UNAVAILABLE = True
        return None

    try:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core import base_options as base_options_module
    except ImportError:
        _FACE_MESH_UNAVAILABLE = True
        return None

    options = vision.FaceLandmarkerOptions(
        base_options=base_options_module.BaseOptions(model_asset_path=str(task_path)),
        num_faces=1,
        min_face_detection_confidence=0.45,
        min_face_presence_confidence=0.45,
    )
    _FACE_MESH_DETECTOR = FaceMeshDetector(
        backend="tasks.FaceLandmarker",
        detector=vision.FaceLandmarker.create_from_options(options),
    )
    return _FACE_MESH_DETECTOR


def _face_mesh_input_array(image: object) -> object:
    import numpy as np

    width, height = image.size  # type: ignore[attr-defined]
    max_side = max(width, height)
    if max_side > FACE_MESH_DETECT_MAX_SIDE:
        scale = FACE_MESH_DETECT_MAX_SIDE / max_side
        image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))))  # type: ignore[attr-defined]
    return np.ascontiguousarray(np.asarray(image))


def _detect_face_mesh_landmarks(image: object) -> list[tuple[float, float]] | None:
    detector = _face_mesh_detector()
    if detector is None:
        return None

    width, height = image.size  # type: ignore[attr-defined]
    image_array = _face_mesh_input_array(image)

    if detector.backend == "solutions.face_mesh":
        result = detector.detector.process(image_array)
        if not getattr(result, "multi_face_landmarks", None):
            return None
        landmarks = result.multi_face_landmarks[0].landmark
        return [(point.x * width, point.y * height) for point in landmarks]

    if detector.backend == "tasks.FaceLandmarker":
        try:
            import mediapipe as mp
        except ImportError:
            return None
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
        result = detector.detector.detect(mp_image)
        if not getattr(result, "face_landmarks", None):
            return None
        landmarks = result.face_landmarks[0]
        return [(point.x * width, point.y * height) for point in landmarks]

    return None


def _mean_landmark(landmarks: list[tuple[float, float]], indexes: tuple[int, ...]) -> tuple[float, float]:
    points = [landmarks[index] for index in indexes if index < len(landmarks)]
    if not points:
        raise ValueError("No requested face-mesh landmarks are available.")
    return sum(x for x, _ in points) / len(points), sum(y for _, y in points) / len(points)


def _cheek_region_crop_box_from_landmarks(
    width: int,
    height: int,
    landmarks: list[tuple[float, float]],
) -> tuple[int, int, int, int]:
    anchors = [landmarks[index] for index in FACE_ANCHOR_LANDMARKS if index < len(landmarks)]
    if not anchors:
        return _face_region_crop_box(width, height)

    min_x = min(x for x, _ in anchors)
    max_x = max(x for x, _ in anchors)
    min_y = min(y for _, y in anchors)
    max_y = max(y for _, y in anchors)
    face_width = max(1.0, max_x - min_x)
    face_height = max(1.0, max_y - min_y)

    nose = [landmarks[FACE_MESH_NOSE_TIP]] if FACE_MESH_NOSE_TIP < len(landmarks) else []
    protected_points = anchors + nose
    protected_min_x = min(x for x, _ in protected_points)
    protected_max_x = max(x for x, _ in protected_points)
    protected_min_y = min(y for _, y in protected_points)
    protected_max_y = max(y for _, y in protected_points)

    left = protected_min_x - face_width * FACE_CLOSEUP_SIDE_MARGIN_FRACTION
    right = protected_max_x + face_width * FACE_CLOSEUP_SIDE_MARGIN_FRACTION
    top = protected_min_y - face_height * FACE_CLOSEUP_TOP_MARGIN_FRACTION
    bottom = protected_max_y + face_height * FACE_CLOSEUP_BOTTOM_MARGIN_FRACTION
    return _expand_crop_bounds(width, height, left, top, right, bottom)


def _cheek_region_crop_box(image: object) -> CropDecision:
    width, height = image.size  # type: ignore[attr-defined]
    landmarks = _detect_face_mesh_landmarks(image)
    if landmarks is None:
        return CropDecision(_face_region_crop_box(width, height), "center_fallback")
    return CropDecision(_cheek_region_crop_box_from_landmarks(width, height, landmarks), "face_mesh_closeup")


def _crop_summary_payload(preprocessing_hash: str, crop_methods: Counter[str]) -> dict:
    return {
        "preprocessing_hash": preprocessing_hash,
        "crop_methods": dict(sorted(crop_methods.items())),
        "face_mesh_dependency": _face_mesh_dependency_signature(),
    }


def _validate_class_id(class_id: int) -> None:
    if class_id < 0 or class_id >= len(ALL_CLASSES):
        raise ValueError(f"Label class id {class_id} is outside the configured AutoDerm classes.")


def _transform_label_line(line: str, crop_box: tuple[int, int, int, int], original_size: tuple[int, int]) -> str | None:
    parts = line.split()
    if not parts:
        return None

    class_id = int(float(parts[0]))
    _validate_class_id(class_id)

    width, height = original_size
    left, top, right, bottom = crop_box
    crop_width = right - left
    crop_height = bottom - top
    values = [float(value) for value in parts[1:]]

    if len(values) == 8:
        points = [(values[index] * width, values[index + 1] * height) for index in range(0, 8, 2)]
        min_x = min(x for x, _ in points)
        max_x = max(x for x, _ in points)
        min_y = min(y for _, y in points)
        max_y = max(y for _, y in points)
        if max_x <= left or min_x >= right or max_y <= top or min_y >= bottom:
            return None

        transformed: list[float] = []
        for x, y in points:
            transformed.extend(
                [
                    min(1.0, max(0.0, (x - left) / crop_width)),
                    min(1.0, max(0.0, (y - top) / crop_height)),
                ]
            )
        coords = " ".join(f"{value:.6f}" for value in transformed)
        return f"{class_id} {coords}"

    if len(values) >= 5:
        x_center, y_center, box_width, box_height, angle = values[:5]
        x_abs = x_center * width
        y_abs = y_center * height
        box_width_abs = box_width * width
        box_height_abs = box_height * height

        min_x = x_abs - box_width_abs / 2
        max_x = x_abs + box_width_abs / 2
        min_y = y_abs - box_height_abs / 2
        max_y = y_abs + box_height_abs / 2
        if max_x <= left or min_x >= right or max_y <= top or min_y >= bottom:
            return None

        clipped_min_x = min(right, max(left, min_x))
        clipped_max_x = min(right, max(left, max_x))
        clipped_min_y = min(bottom, max(top, min_y))
        clipped_max_y = min(bottom, max(top, max_y))
        new_width = max(0.0, clipped_max_x - clipped_min_x) / crop_width
        new_height = max(0.0, clipped_max_y - clipped_min_y) / crop_height
        new_x = ((clipped_min_x + clipped_max_x) / 2 - left) / crop_width
        new_y = ((clipped_min_y + clipped_max_y) / 2 - top) / crop_height
        return f"{class_id} {new_x:.6f} {new_y:.6f} {new_width:.6f} {new_height:.6f} {angle:.6f}"

    raise ValueError(f"Unsupported YOLO OBB label format: {line}")


def _crop_image_and_label(source_image: Path, source_label: Path | None, destination_image: Path, destination_label: Path) -> str:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Face cropping requires Pillow, which is installed with Ultralytics.") from exc

    with Image.open(source_image) as image:
        image = image.convert("RGB")
        crop_decision = _cheek_region_crop_box(image)
        crop_box = crop_decision.box
        cropped = image.crop(crop_box)
        cropped.save(destination_image)
        original_size = image.size

    transformed_lines: list[str] = []
    if source_label and source_label.exists():
        for raw_line in source_label.read_text(encoding="utf-8").splitlines():
            transformed = _transform_label_line(raw_line.strip(), crop_box, original_size)
            if transformed is not None:
                transformed_lines.append(transformed)

    destination_label.write_text("\n".join(transformed_lines) + ("\n" if transformed_lines else ""), encoding="utf-8")
    return crop_decision.method


def _copy_image_and_label(source_image: Path, source_label: Path | None, destination_image: Path, destination_label: Path) -> None:
    shutil.copy2(source_image, destination_image)
    if source_label and source_label.exists():
        shutil.copy2(source_label, destination_label)
    else:
        destination_label.write_text("", encoding="utf-8")


def _preprocessing_hash(apply_face_crop: bool, split_seed: int) -> str:
    source = "\n".join(
        [
            inspect.getsource(_clamped_square_crop_box),
            inspect.getsource(_face_region_crop_box),
            inspect.getsource(_clamped_rect_crop_box),
            inspect.getsource(_expand_crop_bounds),
            inspect.getsource(_face_landmarker_task_path),
            inspect.getsource(_file_sha256),
            inspect.getsource(_legacy_face_mesh_class),
            inspect.getsource(_face_mesh_dependency_signature),
            inspect.getsource(_face_mesh_detector),
            inspect.getsource(_face_mesh_input_array),
            inspect.getsource(_detect_face_mesh_landmarks),
            inspect.getsource(_mean_landmark),
            inspect.getsource(_cheek_region_crop_box_from_landmarks),
            inspect.getsource(_cheek_region_crop_box),
            inspect.getsource(_transform_label_line),
            inspect.getsource(_crop_image_and_label),
            inspect.getsource(_copy_image_and_label),
            f"apply_face_crop={apply_face_crop}",
            f"face_mesh_dependency={_face_mesh_dependency_signature(allow_download=apply_face_crop) if apply_face_crop else 'not_used'}",
            f"split_seed={split_seed}",
        ]
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _write_dataset_yaml(output_dir: Path) -> None:
    names = "\n".join(f"      {index}: {class_name}" for index, class_name in enumerate(ALL_CLASSES))
    dataset_yaml = f"""path: {output_dir.resolve()}
train: train/images
val: research_val/images
names:
{names}
"""
    (output_dir / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")


def prepare_dataset(source_dir: str, output_dir: str, split_seed: int = 42, apply_face_crop: bool = False) -> str:
    source_path = Path(source_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_path}")

    records = _discover_records(source_path)
    split_by_group, split_strategy = _balanced_split_records(records, split_seed)
    grouped_records, image_level_groups = _group_records(records)
    preprocessing_hash = _preprocessing_hash(apply_face_crop, split_seed)

    for split in SPLITS:
        shutil.rmtree(output_path / split, ignore_errors=True)
        (output_path / split / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split / "labels").mkdir(parents=True, exist_ok=True)
        (output_path / split / "preprocessing_hash.txt").write_text(preprocessing_hash + "\n", encoding="utf-8")

    records_by_split = {split: [] for split in SPLITS}
    crop_methods: Counter[str] = Counter()
    for record in records:
        group_id = _record_group_id(record, image_level_groups)
        split = split_by_group[group_id]
        records_by_split[split].append(record)
        destination_image = output_path / split / "images" / f"{record.output_stem}{record.image_path.suffix.lower()}"
        destination_label = output_path / split / "labels" / f"{record.output_stem}.txt"
        if apply_face_crop:
            crop_methods[_crop_image_and_label(record.image_path, record.label_path, destination_image, destination_label)] += 1
        else:
            _copy_image_and_label(record.image_path, record.label_path, destination_image, destination_label)

    (output_path / "preprocessing_hash.txt").write_text(preprocessing_hash + "\n", encoding="utf-8")
    if apply_face_crop:
        (output_path / "crop_summary.json").write_text(
            json.dumps(_crop_summary_payload(preprocessing_hash, crop_methods), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _write_dataset_yaml(output_path)
    _write_split_reports(output_path, records_by_split, split_strategy, split_seed)
    return preprocessing_hash


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an immutable AutoDerm dataset split.")
    parser.add_argument("--source", required=True, help="Source directory containing images and YOLO OBB labels.")
    parser.add_argument("--output", required=True, help="Prepared output directory, e.g. data/uncropped.")
    parser.add_argument("--apply-face-crop", action="store_true", help="Apply the deterministic cheek-region crop.")
    parser.add_argument("--seed", type=int, default=42, help="Patient-level split seed.")
    args = parser.parse_args()

    preprocessing_hash = prepare_dataset(
        source_dir=args.source,
        output_dir=args.output,
        split_seed=args.seed,
        apply_face_crop=args.apply_face_crop,
    )
    print(preprocessing_hash)


if __name__ == "__main__":
    main()
