"""Validate the train/val directories referenced by an Ultralytics dataset YAML."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _parse_simple_dataset_yaml(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line == "names:" or line[0].isdigit():
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip().strip("'\"")
    return parsed


def _resolve_dataset_path(dataset_yaml: Path, root_value: str | None, split_value: str) -> Path:
    split_path = Path(split_value)
    if split_path.is_absolute():
        return split_path

    root = Path(root_value).expanduser() if root_value else dataset_yaml.parent
    if not root.is_absolute():
        root = (dataset_yaml.parent / root).resolve()
    return (root / split_path).resolve()


def _contains_images(path: Path) -> bool:
    return any(candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES for candidate in path.rglob("*"))


def check_dataset_yaml(dataset_yaml_path: str) -> None:
    dataset_yaml = Path(dataset_yaml_path).expanduser().resolve()
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML does not exist: {dataset_yaml}")

    parsed = _parse_simple_dataset_yaml(dataset_yaml)
    missing_keys = [key for key in ("train", "val") if key not in parsed]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise RuntimeError(f"Dataset YAML is missing required key(s): {missing}")

    for key in ("train", "val"):
        directory = _resolve_dataset_path(dataset_yaml, parsed.get("path"), parsed[key])
        if not directory.exists() or not directory.is_dir():
            raise RuntimeError(f"{key} directory does not exist: {directory}")
        if not _contains_images(directory):
            raise RuntimeError(f"{key} directory contains no images: {directory}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Ultralytics train/val image directory references.")
    parser.add_argument("dataset_yaml", nargs="?", help="Path to dataset.yaml.")
    parser.add_argument("--dataset-yaml", dest="dataset_yaml_option", help="Path to dataset.yaml.")
    args = parser.parse_args()

    dataset_yaml = args.dataset_yaml_option or args.dataset_yaml
    if not dataset_yaml:
        print("ERROR: provide a dataset YAML path.", file=sys.stderr)
        raise SystemExit(2)

    try:
        check_dataset_yaml(dataset_yaml)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print("Ultralytics dataset hook check passed.")


if __name__ == "__main__":
    main()
