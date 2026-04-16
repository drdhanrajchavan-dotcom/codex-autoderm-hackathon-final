"""Minimal local environment loader for AutoDerm."""

from __future__ import annotations

from pathlib import Path
import os


REQUIRED_KEYS = ("EC2_HOST", "EC2_PROJECT_DIR", "EC2_SSH_KEY")


def load_env_file(env_path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines from a local .env file if it exists."""
    path = Path(env_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def validate_required_keys(required_keys: tuple[str, ...] = REQUIRED_KEYS) -> None:
    """Raise an error when any required environment key is missing."""
    missing = [key for key in required_keys if not os.getenv(key)]
    if missing:
        missing_csv = ", ".join(missing)
        raise RuntimeError(f"Missing required environment keys: {missing_csv}")


def load_local_env(env_path: str | Path = ".env") -> None:
    """Load .env and validate the keys required for local EC2 orchestration."""
    load_env_file(env_path)
    validate_required_keys()


if __name__ == "__main__":
    load_local_env()
