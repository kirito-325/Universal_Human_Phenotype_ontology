"""Repository-relative paths shared by every maintained pipeline stage."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "config"
DATA_ROOT = REPO_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
INTERIM_ROOT = DATA_ROOT / "interim"
PROCESSED_ROOT = DATA_ROOT / "processed"
PAPER_ROOT = REPO_ROOT / "paper"


def repo_path(value: str | Path) -> Path:
    """Resolve a configured repository-relative path without requiring cwd."""

    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path
