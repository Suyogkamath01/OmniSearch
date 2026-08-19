"""Native deployment configuration, preflight, and launch helpers.

The primary supported deployment is a native ``uv`` process on the host.  The
module keeps host/port/offline/OpenMP handling in one place and delegates all
model, cache, and FAISS work to the validated Phase 21 ``RetrievalService``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from . import __version__
from .api.config import ServiceConfig
from .api.errors import ServiceError
from .api.retrieval import RetrievalService
from .clip_baseline import select_device

PHASE24_SCHEMA_VERSION = 1
MIN_FREE_BYTES = 256 * 1024 * 1024
DEFAULT_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8000
DEFAULT_UI_PORT = 8501


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_size(path: Path) -> int:
    total = 0
    if not path.is_dir():
        return total
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def _relative_path(path: Path, root: Path) -> str | None:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def _record_path(path: Path, root: Path, *, required: bool = True) -> dict[str, Any]:
    is_directory = path.is_dir()
    exists = path.is_dir() if is_directory else path.is_file()
    record: dict[str, Any] = {
        "relative_path": _relative_path(path, root),
        "kind": "directory" if is_directory else "file",
        "required": required,
        "exists": exists,
        "readable": os.access(path, os.R_OK | (os.X_OK if is_directory else 0)) if exists else False,
        "bytes": _directory_size(path) if is_directory else path.stat().st_size if exists else 0,
    }
    if exists and path.is_file():
        record["sha256"] = _sha256(path)
    return record


def _required_paths(service: ServiceConfig) -> dict[str, tuple[Path, bool]]:
    return {
        "phase20_report": (service.phase20_report_path, True),
        "phase20_recommendations": (service.phase20_recommendations_path, True),
        "phase20_provenance": (service.phase20_provenance_path, True),
        "checkpoint": (service.checkpoint_path, True),
        "checkpoint_metadata": (service.checkpoint_path.with_name("checkpoint_metadata.json"), True),
        "manifest": (service.manifest_path, True),
        "image_root": (service.image_root, True),
        "embedding_cache": (service.cache_dir, True),
        "cache_metadata": (service.cache_dir / "metadata.json", True),
        "image_vectors": (service.cache_dir / "images.npy", True),
        "caption_vectors": (service.cache_dir / "captions.npy", True),
        "image_ids": (service.cache_dir / "image_ids.json", True),
        "caption_ids": (service.cache_dir / "caption_ids.json", True),
        "image_index": (service.image_index_path, True),
        "image_index_metadata": (service.image_index_metadata_path, True),
        "caption_index": (service.caption_index_path, True),
        "caption_index_metadata": (service.caption_index_metadata_path, True),
    }


@dataclass(frozen=True)
class DeploymentConfig:
    """Host-level deployment settings layered over ``ServiceConfig``."""

    root: Path
    service: ServiceConfig
    host: str = DEFAULT_HOST
    api_port: int = DEFAULT_API_PORT
    ui_port: int = DEFAULT_UI_PORT
    offline: bool = False
    openmp_workaround: bool = True

    @classmethod
    def from_env(cls, root: Path | None = None) -> DeploymentConfig:
        base = (root or Path(os.environ.get("OMNISEARCH_ROOT", Path.cwd()))).resolve()
        service = ServiceConfig.from_env(base)
        offline = _env_bool("OMNISEARCH_OFFLINE", False) or _env_bool("HF_HUB_OFFLINE", False) or _env_bool("TRANSFORMERS_OFFLINE", False)
        config = cls(
            root=base,
            service=service,
            host=os.environ.get("OMNISEARCH_HOST", DEFAULT_HOST),
            api_port=_env_int("OMNISEARCH_API_PORT", DEFAULT_API_PORT),
            ui_port=_env_int("OMNISEARCH_UI_PORT", DEFAULT_UI_PORT),
            offline=offline,
            openmp_workaround=_env_bool("OMNISEARCH_OPENMP_WORKAROUND", True),
        )
        config.validate()
        return config

    def validate(self) -> None:
        self.service.validate_settings()
        if not self.host.strip():
            raise ValueError("OMNISEARCH_HOST must not be empty")
        for name, port in (("OMNISEARCH_API_PORT", self.api_port), ("OMNISEARCH_UI_PORT", self.ui_port)):
            if not 1 <= port <= 65535:
                raise ValueError(f"{name} must be between 1 and 65535")

    def with_overrides(self, *, host: str | None = None, port: int | None = None) -> DeploymentConfig:
        if port is None and host is None:
            return self
        return replace(self, host=host or self.host, api_port=port if port is not None else self.api_port, ui_port=port if port is not None else self.ui_port)


def required_artifact_records(config: DeploymentConfig) -> dict[str, dict[str, Any]]:
    return {
        name: _record_path(path, config.root, required=required)
        for name, (path, required) in _required_paths(config.service).items()
    }


def build_deployment_manifest(config: DeploymentConfig, preflight: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a portable manifest without embedding machine-specific paths."""

    records = required_artifact_records(config)
    phase20_report = config.root / "artifacts/phase20/phase20_report.json"
    lockfile = config.root / "uv.lock"
    try:
        uv_version = subprocess.run(["uv", "--version"], capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        uv_version = "unavailable"
    startup = (preflight or {}).get("startup_validation", {})
    return {
        "schema_version": PHASE24_SCHEMA_VERSION,
        "project": "OmniSearch",
        "phase": 24,
        "code_version": __version__,
        "deployment_mode": "native_uv",
        "primary_deployment_path": True,
        "runtime": {
            "python_version": platform_python_version(),
            "python_requirement": ">=3.12,<3.14",
            "uv_version": uv_version,
            "lockfile_relative_path": _relative_path(lockfile, config.root),
            "lockfile_sha256": _sha256(lockfile) if lockfile.is_file() else None,
        },
        "configuration": {
            "root_from_environment": "OMNISEARCH_ROOT",
            "host": config.host,
            "api_port": config.api_port,
            "ui_port": config.ui_port,
            "offline_mode": config.offline,
            "device_preference": config.service.device,
            "selected_device": (preflight or {}).get("selected_device"),
            "openmp_workaround_on_macos": config.openmp_workaround,
        },
        "model": {
            "model_id": config.service.model_id,
            "checkpoint": records["checkpoint"],
            "checkpoint_metadata": records["checkpoint_metadata"],
            "local_hugging_face_cache_required_when_offline": config.offline,
            "cache_location_configuration": ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"],
        },
        "retrieval": {
            "system": "Phase 7 full-FT CLIP + Phase 10 cached embeddings + FAISS Flat exact search",
            "backend": "FAISS Flat exact inner-product search",
            "embedding_dimension": startup.get("embedding_dimension"),
            "manifest": records["manifest"],
            "image_root": records["image_root"],
            "embedding_cache": records["embedding_cache"],
            "image_index": records["image_index"],
            "image_index_metadata": records["image_index_metadata"],
            "caption_index": records["caption_index"],
            "caption_index_metadata": records["caption_index_metadata"],
            "phase20_report": _record_path(phase20_report, config.root),
        },
        "required_local_artifacts": records,
        "api": {"version": config.service.api_version, "entrypoint": "omnisearch-api"},
        "ui": {"framework": "Streamlit", "entrypoint": "omnisearch-ui", "in_process_retrieval_service": True},
        "docker": {
            "status": "optional_cpu_unvalidated",
            "mps_supported": False,
            "artifacts_mounted_not_baked": True,
            "daemon_validated": False,
        },
        "portable_components": ["source code", "tests", "fixtures", "uv.lock", ".python-version", "configuration example"],
        "local_artifact_dependent_components": ["checkpoint", "embedding cache", "FAISS indexes", "COCO image root", "offline model cache"],
        "hardware_dependent_components": ["MPS/CPU device selection", "cold-start time", "warm retrieval latency", "native library behavior"],
    }


def _model_cache_available(config: DeploymentConfig) -> tuple[bool, str | None]:
    """Check the local model cache without initiating a download."""

    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained(config.service.model_id, local_files_only=True)
        return True, None
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        return False, type(error).__name__


def _dependency_imports() -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "PIL", "faiss", "uvicorn", "streamlit")}


def run_preflight(config: DeploymentConfig) -> dict[str, Any]:
    """Validate deployment inputs without loading model weights into memory."""

    config.validate()
    records = required_artifact_records(config)
    missing = [name for name, record in records.items() if record["required"] and not record["exists"]]
    unreadable = [name for name, record in records.items() if record["exists"] and not record["readable"]]
    required_bytes = sum(int(record["bytes"]) for record in records.values())
    try:
        free_bytes = shutil.disk_usage(config.root).free
    except OSError:
        free_bytes = 0
    disk_ok = free_bytes >= max(MIN_FREE_BYTES, required_bytes // 10)

    dependencies = _dependency_imports()
    selected_device: str | None = None
    device_error: str | None = None
    try:
        import torch

        selected_device = select_device(config.service.device, torch)
    except (ImportError, RuntimeError, ValueError) as error:
        device_error = str(error)

    model_cache_available, model_cache_error = _model_cache_available(config)
    startup_compatibility = False
    startup_report: dict[str, Any] = {}
    startup_error: str | None = None
    if not missing and not unreadable:
        try:
            service = RetrievalService(config.service)
            startup_report = service.validate_startup()
            startup_compatibility = True
        except (OSError, RuntimeError, ServiceError, TypeError, ValueError) as error:
            startup_error = str(error)

    checks = {
        "runtime_python_supported": (3, 12) <= sys.version_info[:2] < (3, 14),
        "required_artifacts_present": not missing,
        "required_artifacts_readable": not unreadable,
        "disk_space_available": disk_ok,
        "runtime_dependencies_available": all(dependencies.values()),
        "device_selected": selected_device is not None,
        "startup_compatibility_validated": startup_compatibility,
        "offline_model_cache_available": model_cache_available if config.offline else True,
    }
    errors: list[str] = []
    if missing:
        errors.append("Missing required local artifacts: " + ", ".join(missing))
    if unreadable:
        errors.append("Unreadable required local artifacts: " + ", ".join(unreadable))
    if not disk_ok:
        errors.append(f"Insufficient free disk space: {free_bytes} bytes free; at least {max(MIN_FREE_BYTES, required_bytes // 10)} required")
    if device_error:
        errors.append(f"Device selection failed: {device_error}")
    if startup_error:
        errors.append(f"Artifact compatibility validation failed: {startup_error}")
    if config.offline and not model_cache_available:
        errors.append("Offline mode is enabled but the local Hugging Face model cache is unavailable; unset offline mode for first-time setup")
    warnings: list[str] = []
    if not config.offline and not model_cache_available:
        warnings.append("Model config is not cached locally; first launch may require network access")
    if sys.platform == "darwin" and config.openmp_workaround:
        warnings.append("macOS launchers set KMP_DUPLICATE_LIB_OK=TRUE as a FAISS/PyTorch OpenMP compatibility workaround")
    return {
        "schema_version": PHASE24_SCHEMA_VERSION,
        "phase": 24,
        "passed": all(checks.values()),
        "deployment_mode": "native_uv",
        "root_is_configured": True,
        "python_version": platform_python_version(),
        "selected_device": selected_device,
        "device_preference": config.service.device,
        "offline_mode": config.offline,
        "host": config.host,
        "api_port": config.api_port,
        "ui_port": config.ui_port,
        "checks": checks,
        "dependencies": dependencies,
        "required_artifacts": records,
        "required_artifact_bytes": required_bytes,
        "free_disk_bytes": free_bytes,
        "model_cache_available": model_cache_available,
        "model_cache_error_type": model_cache_error,
        "startup_validation": startup_report,
        "errors": errors,
        "warnings": warnings,
        "action": "Install the deployment extra, mount/provide the listed local artifacts, and rerun omnisearch-preflight" if errors else "Ready for native API/UI launch",
    }


def platform_python_version() -> str:
    return ".".join(str(part) for part in sys.version_info[:3])


def runtime_environment(config: DeploymentConfig) -> dict[str, str]:
    """Build one launch environment for API and UI processes."""

    environment = dict(os.environ)
    environment["OMNISEARCH_ROOT"] = str(config.root)
    if config.offline:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    if config.openmp_workaround and sys.platform == "darwin":
        environment.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        environment.setdefault("OMP_NUM_THREADS", "1")
        environment.setdefault("MKL_NUM_THREADS", "1")
        environment.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    return environment


def port_available(host: str, port: int) -> bool:
    """Return false without changing or terminating any existing process."""

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
        return True
    except OSError:
        return False


def api_command(config: DeploymentConfig) -> list[str]:
    return [sys.executable, "-m", "uvicorn", "omnisearch.api.app:create_app", "--factory", "--host", config.host, "--port", str(config.api_port)]


def ui_command(config: DeploymentConfig) -> list[str]:
    return [sys.executable, "-m", "streamlit", "run", "src/omnisearch/ui/streamlit_app.py", "--server.headless", "true", "--server.address", config.host, "--server.port", str(config.ui_port)]


def _launch(config: DeploymentConfig, command: Sequence[str], port: int, label: str) -> int:
    config.validate()
    if not port_available(config.host, port):
        print(f"Cannot start {label}: {config.host}:{port} is already in use. Set the corresponding port environment variable and retry.", file=sys.stderr)
        return 2
    try:
        completed = subprocess.run(list(command), cwd=config.root, env=runtime_environment(config), check=False)
    except OSError as error:
        print(f"Cannot start {label}: {error}. Install the deployment extra and retry.", file=sys.stderr)
        return 2
    return int(completed.returncode)


def _common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--root", type=Path, default=None, help="Repository/deployment root; defaults to OMNISEARCH_ROOT or cwd")
    parser.add_argument("--host", default=None, help="Override OMNISEARCH_HOST")
    parser.add_argument("--port", type=int, default=None, help="Override the API or UI port")
    return parser


def preflight_main() -> int:
    parser = _common_parser("Validate OmniSearch deployment artifacts and runtime")
    args = parser.parse_args()
    try:
        config = DeploymentConfig.from_env(args.root)
        if args.host:
            config = replace(config, host=args.host)
        result = run_preflight(config)
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"passed": False, "errors": [str(error)]}, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


def api_main() -> int:
    parser = _common_parser("Launch the OmniSearch FastAPI retrieval service")
    args = parser.parse_args()
    try:
        config = DeploymentConfig.from_env(args.root)
        config = config.with_overrides(host=args.host, port=args.port)
        return _launch(config, api_command(config), config.api_port, "FastAPI")
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Cannot configure FastAPI: {error}", file=sys.stderr)
        return 2


def ui_main() -> int:
    parser = _common_parser("Launch the OmniSearch Streamlit retrieval demo")
    args = parser.parse_args()
    try:
        config = DeploymentConfig.from_env(args.root)
        config = config.with_overrides(host=args.host, port=args.port)
        return _launch(config, ui_command(config), config.ui_port, "Streamlit UI")
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Cannot configure Streamlit: {error}", file=sys.stderr)
        return 2
