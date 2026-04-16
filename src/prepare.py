"""Deterministic dataset preparation for AutoDerm.

The prepared Ultralytics dataset intentionally points validation at
``research_val`` only. ``locked_eval`` remains immutable and is evaluated
separately by ``src.eval``.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .types import ALL_CLASSES


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "research_val", "locked_eval")
SOURCE_SPLIT_NAMES = {"train", "val", "valid", "validation", "test", "research_val", "locked_eval"}


@dataclass(frozen=True)
class SourceRecord:
    image_path: Path
    label_path: Path | None
    patient_id: str
    output_stem: str
    class_counts: dict[str, int]


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


def _face_region_crop_box(width: int, height: int) -> tuple[int, int, int, int]:
    side = min(width, height)
    crop_width = max(1, round(side * 0.90))
    crop_height = max(1, round(side * 0.90))
    left = max(0, round((width - crop_width) / 2))
    top = max(0, round((height - crop_height) / 2))
    return left, top, min(width, left + crop_width), min(height, top + crop_height)


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


def _crop_image_and_label(source_image: Path, source_label: Path | None, destination_image: Path, destination_label: Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Face cropping requires Pillow, which is installed with Ultralytics.") from exc

    with Image.open(source_image) as image:
        image = image.convert("RGB")
        crop_box = _face_region_crop_box(*image.size)
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


def _copy_image_and_label(source_image: Path, source_label: Path | None, destination_image: Path, destination_label: Path) -> None:
    shutil.copy2(source_image, destination_image)
    if source_label and source_label.exists():
        shutil.copy2(source_label, destination_label)
    else:
        destination_label.write_text("", encoding="utf-8")


def _preprocessing_hash(apply_face_crop: bool, split_seed: int) -> str:
    source = "\n".join(
        [
            inspect.getsource(_face_region_crop_box),
            inspect.getsource(_transform_label_line),
            inspect.getsource(_crop_image_and_label),
            inspect.getsource(_copy_image_and_label),
            f"apply_face_crop={apply_face_crop}",
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
    for record in records:
        group_id = _record_group_id(record, image_level_groups)
        split = split_by_group[group_id]
        records_by_split[split].append(record)
        destination_image = output_path / split / "images" / f"{record.output_stem}{record.image_path.suffix.lower()}"
        destination_label = output_path / split / "labels" / f"{record.output_stem}.txt"
        if apply_face_crop:
            _crop_image_and_label(record.image_path, record.label_path, destination_image, destination_label)
        else:
            _copy_image_and_label(record.image_path, record.label_path, destination_image, destination_label)

    (output_path / "preprocessing_hash.txt").write_text(preprocessing_hash + "\n", encoding="utf-8")
    _write_dataset_yaml(output_path)
    _write_split_reports(output_path, records_by_split, split_strategy, split_seed)
    return preprocessing_hash


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an immutable AutoDerm dataset split.")
    parser.add_argument("--source", required=True, help="Source directory containing images and YOLO OBB labels.")
    parser.add_argument("--output", required=True, help="Prepared output directory, e.g. data/uncropped.")
    parser.add_argument("--apply-face-crop", action="store_true", help="Apply the deterministic face-region crop.")
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
