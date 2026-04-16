"""Codex-authored initial training recipe for AutoDerm acne OBB detection."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Optional


BASE_MODEL_WEIGHTS = "yolo26l-obb.pt"
IMAGE_SIZE = 512
MAX_EPOCHS = 80
BATCH_SIZE = 8
EARLY_STOP_PATIENCE = 12
WORKERS = 4
DEVICE = 0
CACHE_IMAGES = False
SEED = 42
DETERMINISTIC = True

OPTIMIZER = "AdamW"
INITIAL_LR = 0.0012
FINAL_LR_FRACTION = 0.08
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0006
WARMUP_EPOCHS = 2.0
WARMUP_MOMENTUM = 0.8
WARMUP_BIAS_LR = 0.05
NOMINAL_BATCH_SIZE = 64
USE_COSINE_LR = True

BOX_LOSS_GAIN = 8.0
CLASS_LOSS_GAIN = 0.65
DFL_LOSS_GAIN = 1.5
FOCAL_LOSS_GAMMA = 1.5
CALIBRATION_CONFIDENCE_THRESHOLD = 0.25
NODULE_CYST_RECALL_FLOOR = 0.50

CLASS_OVERSAMPLE_WEIGHTS = {
    "comedone_open": 1.0,
    "comedone_closed": 1.0,
    "papule": 1.3,
    "pustule": 1.5,
    "nodule_cyst": 3.0,
    "post_acne_mark": 0.5,
}
MAX_OVERSAMPLE_COPIES_PER_IMAGE = 3

HSV_HUE = 0.008
HSV_SATURATION = 0.22
HSV_VALUE = 0.16
ROTATION_DEGREES = 8.0
TRANSLATE_FRACTION = 0.08
SCALE_FRACTION = 0.35
SHEAR_DEGREES = 2.0
PERSPECTIVE_FRACTION = 0.0005
HORIZONTAL_FLIP_PROBABILITY = 0.5
VERTICAL_FLIP_PROBABILITY = 0.0
MOSAIC_PROBABILITY = 0.35
MIXUP_PROBABILITY = 0.03
COPY_PASTE_PROBABILITY = 0.02
ERASING_PROBABILITY = 0.12
CLOSE_MOSAIC_EPOCHS = 8

SAVE_PERIOD = -1
ENABLE_VALIDATION_DURING_TRAINING = True
ENABLE_PLOTS = False


def _read_dataset_yaml_lines(data_yaml: Path) -> list[str]:
    return data_yaml.read_text(encoding="utf-8").splitlines()


def _simple_yaml_value(lines: list[str], key: str) -> str | None:
    prefix = f"{key}:"
    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped.startswith(prefix):
            return stripped.split(":", 1)[1].strip().strip("'\"")
    return None


def _dataset_root(data_yaml: Path, lines: list[str]) -> Path:
    root_value = _simple_yaml_value(lines, "path")
    if not root_value:
        return data_yaml.parent.resolve()
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root.resolve()


def _resolve_dataset_entry(data_yaml: Path, lines: list[str], key: str) -> Path:
    value = _simple_yaml_value(lines, key)
    if not value:
        raise RuntimeError(f"Dataset YAML is missing required key: {key}")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (_dataset_root(data_yaml, lines) / path).resolve()


def _label_path_for_image(train_images_dir: Path, image_path: Path) -> Path:
    relative_image = image_path.relative_to(train_images_dir)
    return train_images_dir.parent / "labels" / relative_image.with_suffix(".txt")


def _class_names_from_yaml(lines: list[str]) -> dict[int, str]:
    names: dict[int, str] = {}
    inside_names = False
    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped == "names:":
            inside_names = True
            continue
        if inside_names and line and not line.startswith((" ", "\t")):
            break
        if inside_names and ":" in stripped:
            key, value = stripped.split(":", 1)
            if key.strip().isdigit():
                names[int(key.strip())] = value.strip().strip("'\"")
    return names


def _image_weight(label_path: Path, class_names: dict[int, str]) -> float:
    if not label_path.exists():
        return 1.0

    weight = 1.0
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.split()
        if not parts:
            continue
        class_name = class_names.get(int(float(parts[0])))
        if class_name is None:
            continue
        weight = max(weight, CLASS_OVERSAMPLE_WEIGHTS.get(class_name, 1.0))
    return weight


def _materialize_weighted_train_manifest(data_yaml: Path, output_dir: Path) -> Path:
    lines = _read_dataset_yaml_lines(data_yaml)
    train_images_dir = _resolve_dataset_entry(data_yaml, lines, "train")
    class_names = _class_names_from_yaml(lines)
    image_paths = sorted(
        path for path in train_images_dir.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    manifest_lines: list[str] = []
    for image_path in image_paths:
        label_path = _label_path_for_image(train_images_dir, image_path)
        copies = max(1, min(MAX_OVERSAMPLE_COPIES_PER_IMAGE, round(_image_weight(label_path, class_names))))
        manifest_lines.extend([str(image_path.resolve())] * copies)

    manifest_path = output_dir / "weighted_train_images.txt"
    manifest_path.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8")
    return manifest_path


def _materialize_training_yaml(data_yaml: Path, output_dir: Path) -> Path:
    lines = _read_dataset_yaml_lines(data_yaml)
    train_manifest = _materialize_weighted_train_manifest(data_yaml, output_dir)

    rewritten: list[str] = []
    replaced_train = False
    for raw_line in lines:
        if raw_line.strip().startswith("train:"):
            rewritten.append(f"train: {train_manifest}")
            replaced_train = True
        else:
            rewritten.append(raw_line)
    if not replaced_train:
        rewritten.insert(0, f"train: {train_manifest}")

    training_yaml = output_dir / "autoderm_train_dataset.yaml"
    training_yaml.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    return training_yaml


def _weights_paths(output_dir: Path) -> tuple[Path, Path]:
    weights_dir = output_dir / "weights"
    return weights_dir / "best.pt", weights_dir / "last.pt"


def _clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for child_name in ("args.yaml", "results.csv", "weights", "labels.jpg", "labels_correlogram.jpg"):
        child = output_dir / child_name
        if child.is_dir():
            shutil.rmtree(child)
        elif child.exists():
            child.unlink()


def train(data_yaml: str, output_dir: str, budget_seconds: int) -> tuple[Optional[str], bool]:
    start_time = time.monotonic()
    data_yaml_path = Path(data_yaml).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    _clean_output_dir(output_path)
    training_yaml = _materialize_training_yaml(data_yaml_path, output_path)

    from ultralytics import YOLO

    model = YOLO(BASE_MODEL_WEIGHTS)
    model.train(
        data=str(training_yaml),
        project=str(output_path.parent),
        name=output_path.name,
        exist_ok=True,
        epochs=MAX_EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=EARLY_STOP_PATIENCE,
        time=max(1, budget_seconds) / 3600,
        workers=WORKERS,
        device=DEVICE,
        cache=CACHE_IMAGES,
        seed=SEED,
        deterministic=DETERMINISTIC,
        optimizer=OPTIMIZER,
        lr0=INITIAL_LR,
        lrf=FINAL_LR_FRACTION,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_momentum=WARMUP_MOMENTUM,
        warmup_bias_lr=WARMUP_BIAS_LR,
        nbs=NOMINAL_BATCH_SIZE,
        cos_lr=USE_COSINE_LR,
        box=BOX_LOSS_GAIN,
        cls=CLASS_LOSS_GAIN,
        dfl=DFL_LOSS_GAIN,
        hsv_h=HSV_HUE,
        hsv_s=HSV_SATURATION,
        hsv_v=HSV_VALUE,
        degrees=ROTATION_DEGREES,
        translate=TRANSLATE_FRACTION,
        scale=SCALE_FRACTION,
        shear=SHEAR_DEGREES,
        perspective=PERSPECTIVE_FRACTION,
        fliplr=HORIZONTAL_FLIP_PROBABILITY,
        flipud=VERTICAL_FLIP_PROBABILITY,
        mosaic=MOSAIC_PROBABILITY,
        mixup=MIXUP_PROBABILITY,
        copy_paste=COPY_PASTE_PROBABILITY,
        erasing=ERASING_PROBABILITY,
        close_mosaic=CLOSE_MOSAIC_EPOCHS,
        save=True,
        save_period=SAVE_PERIOD,
        val=ENABLE_VALIDATION_DURING_TRAINING,
        plots=ENABLE_PLOTS,
        verbose=True,
    )

    best_path, last_path = _weights_paths(output_path)
    if best_path.exists():
        return str(best_path), False
    if last_path.exists():
        return str(last_path), time.monotonic() - start_time >= max(1, budget_seconds) * 0.95
    return None, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the initial Codex-authored AutoDerm OBB detector.")
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget-seconds", type=int, default=300)
    args = parser.parse_args()

    weights_path, used_last_pt = train(
        data_yaml=args.data_yaml,
        output_dir=args.output_dir,
        budget_seconds=args.budget_seconds,
    )
    print({"weights_path": weights_path, "used_last_pt_due_to_timeout": used_last_pt})


if __name__ == "__main__":
    main()
