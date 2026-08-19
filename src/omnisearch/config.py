"""Small, dependency-free configuration loader used by the Phase 0 smoke test."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "default.toml"


@dataclass(frozen=True)
class ProjectConfig:
    """Validated subset of configuration needed before the ML stack exists."""

    name: str
    schema_version: int
    seed: int
    dataset_id: str
    split_group: str
    config_path: Path


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> ProjectConfig:
    """Load and validate a TOML project configuration."""

    config_path = Path(path).resolve()
    with config_path.open("rb") as file:
        raw = tomllib.load(file)

    project = raw.get("project", {})
    reproducibility = raw.get("reproducibility", {})
    dataset = raw.get("dataset", {})

    required = {
        "project.name": project.get("name"),
        "project.schema_version": project.get("schema_version"),
        "reproducibility.seed": reproducibility.get("seed"),
        "reproducibility.split_group": reproducibility.get("split_group"),
        "dataset.id": dataset.get("id"),
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Missing required configuration values: {', '.join(missing)}")
    if not isinstance(reproducibility["seed"], int) or reproducibility["seed"] < 0:
        raise ValueError("reproducibility.seed must be a non-negative integer")
    if reproducibility["split_group"] != "image_id":
        raise ValueError("split_group must remain image_id to prevent caption leakage")

    return ProjectConfig(
        name=str(project["name"]),
        schema_version=int(project["schema_version"]),
        seed=int(reproducibility["seed"]),
        dataset_id=str(dataset["id"]),
        split_group=str(reproducibility["split_group"]),
        config_path=config_path,
    )
