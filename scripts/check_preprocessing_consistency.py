"""Verify that weights and evaluation split use the same preprocessing pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _weights_hash_path(weights_path: Path) -> Path:
    return (weights_path if weights_path.is_dir() else weights_path.parent) / "preprocessing_hash.txt"


def _read_hash(path: Path, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} preprocessing hash: {path}")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError(f"Empty {label} preprocessing hash: {path}")
    return value


def _split_hash_path(split_path: Path) -> Path:
    direct = split_path / "preprocessing_hash.txt"
    if direct.exists():
        return direct
    return split_path.parent / "preprocessing_hash.txt"


def check_preprocessing_consistency(weights_path: str, split_path: str) -> None:
    weights = Path(weights_path).expanduser().resolve()
    split = Path(split_path).expanduser().resolve()
    weights_hash_path = _weights_hash_path(weights)
    split_hash_path = _split_hash_path(split)

    weights_hash = _read_hash(weights_hash_path, "weights")
    split_hash = _read_hash(split_hash_path, "split")
    if weights_hash != split_hash:
        raise RuntimeError(
            "Preprocessing hash mismatch: "
            f"weights={weights_hash} ({weights_hash_path}) "
            f"split={split_hash} ({split_hash_path})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check preprocessing hash consistency.")
    parser.add_argument("--weights", required=True, help="Path to weights file or weights directory.")
    parser.add_argument("--split", required=True, help="Path to split directory.")
    args = parser.parse_args()

    try:
        check_preprocessing_consistency(weights_path=args.weights, split_path=args.split)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print("Preprocessing consistency check passed.")


if __name__ == "__main__":
    main()
