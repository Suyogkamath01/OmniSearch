"""Repository checks that can run without downloading data or installing ML packages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, ProjectConfig, load_config

REQUIRED_DIRECTORIES = (
    "src/omnisearch",
    "configs",
    "data/raw",
    "data/interim",
    "data/processed",
    "artifacts/experiments",
    "artifacts/metrics",
    "artifacts/models",
    "artifacts/logs",
    "docs",
    "tests",
)


@dataclass(frozen=True)
class BootstrapReport:
    """Machine-readable result of the Phase 0 repository check."""

    config: ProjectConfig
    missing_directories: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_directories


def check_repository(root: Path | str | None = None) -> BootstrapReport:
    """Check the repository contract without mutating the filesystem."""

    repository_root = (
        Path(root).resolve() if root else Path(__file__).resolve().parents[2]
    )
    config_path = repository_root / "configs" / DEFAULT_CONFIG_PATH.name
    config = load_config(config_path)
    missing = tuple(
        relative_path
        for relative_path in REQUIRED_DIRECTORIES
        if not (repository_root / relative_path).is_dir()
    )
    return BootstrapReport(config=config, missing_directories=missing)
