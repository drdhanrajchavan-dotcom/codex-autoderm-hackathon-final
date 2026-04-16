"""Public-safety release checks for tracked files."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".cr2", ".heic"}
ALLOWED_TRACKED_IMAGE_PREFIXES = ("web/public/demo_photos/", "docs/sample_results/")
FORBIDDEN_TRACKED_PREFIXES = ("data/", "runs/", "weights/", ".pulled_artifacts/", "source_data/")
PHI_PATTERNS = {
    "medical record number": re.compile(r"\b(?:MRN|medical record(?: number)?)\s*[:=]\s*[A-Za-z0-9-]+", re.IGNORECASE),
    "date of birth": re.compile(r"\b(?:DOB|date of birth)\s*[:=]\s*\d{1,4}[-/]\d{1,2}[-/]\d{1,4}", re.IGNORECASE),
    "social security number": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone number": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
}


def _tracked_files() -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in completed.stdout.splitlines() if line]


def _is_binary(path: Path) -> bool:
    try:
        return b"\0" in path.read_bytes()[:4096]
    except OSError:
        return True


def _check_path_rules(path_text: str) -> list[str]:
    violations: list[str] = []
    path = Path(path_text)
    if path_text.startswith(FORBIDDEN_TRACKED_PREFIXES):
        violations.append(f"private data/artifact path is tracked: {path_text}")
    if path.suffix.lower() == ".pt":
        violations.append(f"model weights must not be tracked: {path_text}")
    if path.suffix.lower() in IMAGE_SUFFIXES and not path_text.startswith(ALLOWED_TRACKED_IMAGE_PREFIXES):
        violations.append(f"tracked image outside approved public-safe folders: {path_text}")
    return violations


def _check_phi_patterns(path_text: str) -> list[str]:
    path = Path(path_text)
    if not path.exists() or _is_binary(path):
        return []

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    violations: list[str] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if "re.compile(" in line or "PHI_PATTERNS" in line:
            continue
        for label, pattern in PHI_PATTERNS.items():
            if pattern.search(line):
                violations.append(f"possible PHI ({label}) in tracked file: {path_text}:{line_number}")
    return violations


def main() -> None:
    violations: list[str] = []
    for path_text in _tracked_files():
        violations.extend(_check_path_rules(path_text))
        violations.extend(_check_phi_patterns(path_text))

    if violations:
        for violation in violations:
            print(f"ERROR: {violation}", file=sys.stderr)
        raise SystemExit(1)

    print("Release check passed: tracked files are public-safe.")


if __name__ == "__main__":
    main()
