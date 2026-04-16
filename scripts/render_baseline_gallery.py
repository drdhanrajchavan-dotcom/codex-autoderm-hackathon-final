#!/usr/bin/env python3
"""Render AutoDerm baseline prediction overlays for EC2 visual sanity checks.

EC2 invocation flow:
1. Run scripts.run_baselines and choose the crop-gated baseline.
2. Render overlays, for example:
   python -m scripts.render_baseline_gallery \
     --weights runs/baseline_cropped/weights/best.pt \
     --split data/cropped/locked_eval \
     --output runs/baseline_cropped/gallery
3. Review PNGs manually before pulling curated, non-PHI renders for judge-facing docs.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from src.types import ALL_CLASSES


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CLASS_COLORS = {
    "comedone_open": (37, 99, 235),
    "comedone_closed": (14, 165, 233),
    "papule": (249, 115, 22),
    "pustule": (234, 179, 8),
    "nodule_cyst": (220, 38, 38),
    "post_acne_mark": (107, 114, 128),
}


def _split_images_dir(split_dir: Path) -> Path:
    candidate = split_dir / "images"
    return candidate if candidate.exists() else split_dir


def _list_images(split_dir: Path) -> list[Path]:
    images_dir = _split_images_dir(split_dir)
    return sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def _to_python_list(value: object) -> list:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)  # type: ignore[arg-type]


def _xywhr_to_polygon(values: list[float]) -> list[tuple[float, float]]:
    x_center, y_center, width, height, angle = values[:5]
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    corners = [
        (-width / 2, -height / 2),
        (width / 2, -height / 2),
        (width / 2, height / 2),
        (-width / 2, height / 2),
    ]
    return [
        (x_center + x * cos_a - y * sin_a, y_center + x * sin_a + y * cos_a)
        for x, y in corners
    ]


def _extract_predictions(result: object) -> list[tuple[str, float, list[tuple[float, float]]]]:
    names = getattr(result, "names", {}) or {}
    obb = getattr(result, "obb", None)
    boxes = getattr(result, "boxes", None)
    predictions: list[tuple[str, float, list[tuple[float, float]]]] = []

    if obb is not None and len(obb) > 0:
        class_indices = _to_python_list(obb.cls)
        confidences = _to_python_list(obb.conf)
        if hasattr(obb, "xyxyxyxy"):
            polygons = _to_python_list(obb.xyxyxyxy)
            for class_index, confidence, polygon in zip(class_indices, confidences, polygons):
                class_name = str(names.get(int(class_index), ALL_CLASSES[int(class_index)]))
                points = [(float(x), float(y)) for x, y in polygon]
                predictions.append((class_name, float(confidence), points))
            return predictions
        if hasattr(obb, "xywhr"):
            xywhr = _to_python_list(obb.xywhr)
            for class_index, confidence, values in zip(class_indices, confidences, xywhr):
                class_name = str(names.get(int(class_index), ALL_CLASSES[int(class_index)]))
                predictions.append((class_name, float(confidence), _xywhr_to_polygon([float(value) for value in values])))
            return predictions

    if boxes is not None and len(boxes) > 0:
        class_indices = _to_python_list(boxes.cls)
        confidences = _to_python_list(boxes.conf)
        xyxy = _to_python_list(boxes.xyxy)
        for class_index, confidence, bbox in zip(class_indices, confidences, xyxy):
            x1, y1, x2, y2 = [float(value) for value in bbox]
            class_name = str(names.get(int(class_index), ALL_CLASSES[int(class_index)]))
            predictions.append((class_name, float(confidence), [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]))
    return predictions


def _draw_overlay(image_path: Path, predictions: list[tuple[str, float, list[tuple[float, float]]]], output_path: Path) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("render_baseline_gallery requires Pillow, installed with Ultralytics.") from exc

    with Image.open(image_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None

    for class_name, confidence, points in predictions:
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        closed_points = points + [points[0]]
        draw.line(closed_points, fill=color, width=3)
        label = f"{class_name} {confidence:.2f}"
        label_x = min(x for x, _ in points)
        label_y = max(0, min(y for _, y in points) - 12)
        draw.text((label_x, label_y), label, fill=color, font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def render_gallery(weights: str, split: str, output: str) -> None:
    weights_path = Path(weights).expanduser().resolve()
    split_path = Path(split).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    images = _list_images(split_path)
    if not images:
        raise RuntimeError(f"No images found in split: {split_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file does not exist: {weights_path}")

    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    results = model.predict(source=[str(image) for image in images], conf=0.25, verbose=False)
    images_root = _split_images_dir(split_path)
    for image_path, result in zip(images, results):
        relative = image_path.relative_to(images_root).with_suffix(".png")
        _draw_overlay(image_path, _extract_predictions(result), output_path / relative)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render baseline OBB prediction overlays.")
    parser.add_argument("--weights", required=True, help="Path to best.pt or last.pt.")
    parser.add_argument("--split", required=True, help="Split directory containing images/ and labels/.")
    parser.add_argument("--output", required=True, help="Output directory for PNG overlays.")
    args = parser.parse_args()

    render_gallery(weights=args.weights, split=args.split, output=args.output)


if __name__ == "__main__":
    main()
