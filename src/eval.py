"""Direct split evaluation for AutoDerm.

This module deliberately evaluates ``images`` and ``labels`` directories
directly instead of reading ``dataset.yaml``. That keeps ``locked_eval`` out of
Ultralytics training-time validation paths.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .types import ALL_CLASSES, SCORED_CLASSES, PerClassMetrics


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
IOU_THRESHOLDS = [threshold / 100 for threshold in range(50, 100, 5)]
METRIC_CONFIDENCE_THRESHOLD = 0.25
METRIC_IOU_THRESHOLD = 0.50


@dataclass(frozen=True)
class Box:
    image_id: str
    class_name: str
    bbox: tuple[float, float, float, float]
    confidence: float = 1.0


def hayashi_badge(class_counts: dict[str, int]) -> Literal["clear", "almost_clear", "mild", "moderate", "severe"]:
    inflammatory = (
        class_counts.get("papule", 0)
        + class_counts.get("pustule", 0)
        + class_counts.get("nodule_cyst", 0)
    )
    comedonal = class_counts.get("comedone_open", 0) + class_counts.get("comedone_closed", 0)
    if inflammatory == 0 and comedonal == 0:
        return "clear"
    elif inflammatory == 0 and comedonal <= 5:
        return "almost_clear"
    elif inflammatory <= 5:
        return "mild"
    elif inflammatory <= 20:
        return "moderate"
    else:
        return "severe"


def _image_size(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Evaluation requires Pillow, which is installed with Ultralytics.") from exc

    with Image.open(image_path) as image:
        return image.size


def _images_dir(split_dir: Path) -> Path:
    candidate = split_dir / "images"
    return candidate if candidate.exists() else split_dir


def _labels_dir(split_dir: Path) -> Path:
    candidate = split_dir / "labels"
    return candidate if candidate.exists() else split_dir


def _list_images(split_dir: Path) -> list[Path]:
    images_dir = _images_dir(split_dir)
    return sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def _label_path_for_image(image_path: Path, split_dir: Path) -> Path:
    images_dir = _images_dir(split_dir)
    labels_dir = _labels_dir(split_dir)
    relative = image_path.relative_to(images_dir)
    return labels_dir / relative.with_suffix(".txt")


def _label_to_bbox(values: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    if len(values) == 8:
        xs = [values[index] * width for index in range(0, 8, 2)]
        ys = [values[index] * height for index in range(1, 8, 2)]
        return min(xs), min(ys), max(xs), max(ys)

    if len(values) >= 5:
        x_center, y_center, box_width, box_height = values[:4]
        x_abs = x_center * width
        y_abs = y_center * height
        width_abs = box_width * width
        height_abs = box_height * height
        return (
            x_abs - width_abs / 2,
            y_abs - height_abs / 2,
            x_abs + width_abs / 2,
            y_abs + height_abs / 2,
        )

    raise ValueError("YOLO OBB labels must use polygon points or x/y/w/h/angle values.")


def _parse_ground_truth(split_dir: Path) -> list[Box]:
    boxes: list[Box] = []
    for image_path in _list_images(split_dir):
        width, height = _image_size(image_path)
        label_path = _label_path_for_image(image_path, split_dir)
        if not label_path.exists():
            continue

        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.split()
            if not parts:
                continue
            class_index = int(float(parts[0]))
            if class_index < 0 or class_index >= len(ALL_CLASSES):
                raise ValueError(f"Class index {class_index} is outside the configured AutoDerm classes.")
            class_name = ALL_CLASSES[class_index]
            boxes.append(
                Box(
                    image_id=str(image_path.relative_to(_images_dir(split_dir))),
                    class_name=class_name,
                    bbox=_label_to_bbox([float(value) for value in parts[1:]], width, height),
                )
            )
    return boxes


def _empty_per_class_metrics() -> PerClassMetrics:
    return {class_name: 0.0 for class_name in SCORED_CLASSES}  # type: ignore[return-value]


def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _to_python_list(value: object) -> list:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)  # type: ignore[arg-type]


def _extract_result_boxes(result: object, image_id: str) -> list[Box]:
    extracted: list[Box] = []
    names = getattr(result, "names", {}) or {}
    obb = getattr(result, "obb", None)
    boxes = getattr(result, "boxes", None)

    if obb is not None and len(obb) > 0:
        class_indices = _to_python_list(obb.cls)
        confidences = _to_python_list(obb.conf)
        if hasattr(obb, "xyxyxyxy"):
            polygons = _to_python_list(obb.xyxyxyxy)
            for class_index, confidence, polygon in zip(class_indices, confidences, polygons):
                flat = [float(value) for point in polygon for value in point]
                xs = flat[0::2]
                ys = flat[1::2]
                class_name = str(names.get(int(class_index), ALL_CLASSES[int(class_index)]))
                extracted.append(Box(image_id, class_name, (min(xs), min(ys), max(xs), max(ys)), float(confidence)))
            return extracted

        if hasattr(obb, "xywhr"):
            xywhr = _to_python_list(obb.xywhr)
            for class_index, confidence, values in zip(class_indices, confidences, xywhr):
                x_center, y_center, width, height = [float(value) for value in values[:4]]
                class_name = str(names.get(int(class_index), ALL_CLASSES[int(class_index)]))
                extracted.append(
                    Box(
                        image_id,
                        class_name,
                        (
                            x_center - width / 2,
                            y_center - height / 2,
                            x_center + width / 2,
                            y_center + height / 2,
                        ),
                        float(confidence),
                    )
                )
            return extracted

    if boxes is not None and len(boxes) > 0:
        class_indices = _to_python_list(boxes.cls)
        confidences = _to_python_list(boxes.conf)
        xyxy = _to_python_list(boxes.xyxy)
        for class_index, confidence, bbox in zip(class_indices, confidences, xyxy):
            class_name = str(names.get(int(class_index), ALL_CLASSES[int(class_index)]))
            extracted.append(Box(image_id, class_name, tuple(float(value) for value in bbox), float(confidence)))

    return extracted


def _predict(weights_path: Path, split_dir: Path, images: list[Path]) -> list[Box]:
    if not images:
        return []
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file does not exist: {weights_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Evaluation requires ultralytics.") from exc

    model = YOLO(str(weights_path))
    results = model.predict(source=[str(path) for path in images], conf=0.001, verbose=False)
    predictions: list[Box] = []
    images_dir = _images_dir(split_dir)
    for image_path, result in zip(images, results):
        predictions.extend(_extract_result_boxes(result, str(image_path.relative_to(images_dir))))
    return predictions


def _ap_for_class(ground_truth: list[Box], predictions: list[Box], class_name: str, iou_threshold: float) -> float:
    class_ground_truth = [box for box in ground_truth if box.class_name == class_name]
    class_predictions = sorted(
        [box for box in predictions if box.class_name == class_name],
        key=lambda box: box.confidence,
        reverse=True,
    )
    if not class_ground_truth:
        return 0.0

    matched: set[int] = set()
    true_positives: list[int] = []
    false_positives: list[int] = []
    for prediction in class_predictions:
        best_index = -1
        best_iou = 0.0
        for index, target in enumerate(class_ground_truth):
            if index in matched or target.image_id != prediction.image_id:
                continue
            overlap = _iou(prediction.bbox, target.bbox)
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_index >= 0 and best_iou >= iou_threshold:
            matched.add(best_index)
            true_positives.append(1)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_positives.append(1)

    cumulative_tp = 0
    cumulative_fp = 0
    precision: list[float] = []
    recall: list[float] = []
    for tp, fp in zip(true_positives, false_positives):
        cumulative_tp += tp
        cumulative_fp += fp
        precision.append(cumulative_tp / max(1, cumulative_tp + cumulative_fp))
        recall.append(cumulative_tp / len(class_ground_truth))

    ap = 0.0
    for recall_threshold in [index / 100 for index in range(101)]:
        matching_precision = [p for p, r in zip(precision, recall) if r >= recall_threshold]
        ap += max(matching_precision, default=0.0) / 101
    return ap


def _scored_map50_95(ground_truth: list[Box], predictions: list[Box]) -> float:
    average_precisions = [
        _ap_for_class(ground_truth, predictions, class_name, threshold)
        for class_name in SCORED_CLASSES
        for threshold in IOU_THRESHOLDS
    ]
    return sum(average_precisions) / len(average_precisions)


def _precision_recall_at_threshold(ground_truth: list[Box], predictions: list[Box]) -> tuple[PerClassMetrics, PerClassMetrics]:
    precision = _empty_per_class_metrics()
    recall = _empty_per_class_metrics()
    filtered_predictions = [box for box in predictions if box.confidence >= METRIC_CONFIDENCE_THRESHOLD]

    for class_name in SCORED_CLASSES:
        class_ground_truth = [box for box in ground_truth if box.class_name == class_name]
        class_predictions = sorted(
            [box for box in filtered_predictions if box.class_name == class_name],
            key=lambda box: box.confidence,
            reverse=True,
        )
        matched: set[int] = set()
        true_positive = 0
        false_positive = 0
        for prediction in class_predictions:
            best_index = -1
            best_iou = 0.0
            for index, target in enumerate(class_ground_truth):
                if index in matched or target.image_id != prediction.image_id:
                    continue
                overlap = _iou(prediction.bbox, target.bbox)
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = index
            if best_index >= 0 and best_iou >= METRIC_IOU_THRESHOLD:
                matched.add(best_index)
                true_positive += 1
            else:
                false_positive += 1

        precision[class_name] = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall[class_name] = true_positive / len(class_ground_truth) if class_ground_truth else 0.0
    return precision, recall


def _count_ground_truth_classes(labels_dir: Path) -> dict[str, int]:
    counts = {class_name: 0 for class_name in ALL_CLASSES}
    if not labels_dir.exists():
        return counts

    for label_path in sorted(labels_dir.rglob("*.txt")):
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.split()
            if not parts:
                continue
            class_index = int(float(parts[0]))
            if 0 <= class_index < len(ALL_CLASSES):
                counts[ALL_CLASSES[class_index]] += 1
    return counts


def _training_counts_for_split(split_dir: Path) -> dict[str, int]:
    dataset_root = split_dir.parent if split_dir.name in {"train", "research_val", "locked_eval"} else split_dir
    return _count_ground_truth_classes(dataset_root / "train" / "labels")


def _weights_hash_path(weights_path: Path) -> Path:
    return (weights_path if weights_path.is_dir() else weights_path.parent) / "preprocessing_hash.txt"


def _split_hash_candidates(split_dir: Path) -> list[Path]:
    return [split_dir / "preprocessing_hash.txt", split_dir.parent / "preprocessing_hash.txt"]


def _read_first_existing_hash(paths: list[Path], label: str) -> str:
    for path in paths:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    candidates = ", ".join(str(path) for path in paths)
    raise RuntimeError(f"Missing {label} preprocessing hash. Checked: {candidates}")


def _verify_preprocessing_hash(weights_path: Path, split_dir: Path) -> str:
    weights_hash = _read_first_existing_hash([_weights_hash_path(weights_path)], "weights")
    split_hash = _read_first_existing_hash(_split_hash_candidates(split_dir), "split")
    if weights_hash != split_hash:
        raise RuntimeError(
            "Preprocessing hash mismatch: "
            f"weights={weights_hash} split={split_hash} split_dir={split_dir}"
        )
    return split_hash


def evaluate(weights_path: str, split_dir: str) -> dict:
    weights = Path(weights_path).expanduser().resolve()
    split = Path(split_dir).expanduser().resolve()
    if not split.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split}")

    preprocessing_hash = _verify_preprocessing_hash(weights, split)
    images = _list_images(split)
    ground_truth = [box for box in _parse_ground_truth(split) if box.class_name in SCORED_CLASSES]
    predictions = [box for box in _predict(weights, split, images) if box.class_name in SCORED_CLASSES]
    precision, recall = _precision_recall_at_threshold(ground_truth, predictions)
    training_counts = _training_counts_for_split(split)

    return {
        "weights_path": str(weights),
        "split_dir": str(split),
        "num_images": len(images),
        "scored_classes": SCORED_CLASSES,
        "scored_map50_95": _scored_map50_95(ground_truth, predictions),
        "per_class_precision": precision,
        "per_class_recall": recall,
        "class_counts": training_counts,
        "hayashi_badge": hayashi_badge(training_counts),
        "preprocessing_hash": preprocessing_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an AutoDerm split directly from images and labels.")
    parser.add_argument("--weights", required=True, help="Path to model weights.")
    parser.add_argument("--split", required=True, help="Path to split directory containing images/ and labels/.")
    parser.add_argument("--out", required=True, help="JSON output path.")
    args = parser.parse_args()

    metrics = evaluate(weights_path=args.weights, split_dir=args.split)
    output_path = Path(args.out).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
