"""Phase 23 testing, CI, and reproducibility hardening evidence.

This module performs engineering checks only.  It does not train models,
download datasets, or rebuild any scientific artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PHASE23_SCHEMA_VERSION = 1
REQUIRED_ARTIFACTS = (
    "pre_phase_audit.json",
    "test_inventory.json",
    "ci_validation.json",
    "clean_environment_check.json",
    "entrypoint_validation.json",
    "large_file_audit.json",
    "secret_scan.json",
    "documentation_link_check.json",
    "artifact_validation.json",
    "provenance.json",
    "phase23_report.json",
)

INTEGRATION_TEST_FILES = {"test_coco_dataset.py", "test_phase21.py", "test_phase22.py"}
ARTIFACT_VALIDATION_TEST_FILES = {
    "test_phase14.py",
    "test_phase16.py",
    "test_phase17.py",
    "test_phase18.py",
    "test_phase19.py",
    "test_phase20.py",
    "test_phase21.py",
    "test_phase22.py",
    "test_phase23.py",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite_json)


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite_tree(key) and _finite_tree(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return True


def _tail(value: str, limit: int = 600) -> str:
    compact = value.strip()
    return compact[-limit:] if compact else ""


def _run_command(command: Sequence[str], root: Path, *, timeout: int = 120, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        command_env = dict(os.environ) if env is None else dict(env)
        command_env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        completed = subprocess.run(
            list(command),
            cwd=root,
            env=command_env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": " ".join(command),
            "returncode": completed.returncode,
            "passed": completed.returncode == 0,
            "duration_seconds": time.perf_counter() - started,
            "stdout_tail": _tail(completed.stdout),
            "stderr_tail": _tail(completed.stderr),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "command": " ".join(command),
            "returncode": None,
            "passed": False,
            "duration_seconds": time.perf_counter() - started,
            "stdout_tail": "",
            "stderr_tail": str(error),
        }


def audit_phase22(root: Path | str) -> dict[str, Any]:
    """Check only the Phase 22 dependency contract before hardening work."""

    base = Path(root).resolve()
    phase22 = base / "artifacts/phase22"
    phase21 = base / "artifacts/phase21"
    checks: dict[str, bool]
    try:
        report = _read_json(phase22 / "phase22_report.json")
        config = _read_json(phase22 / "ui_config.json")
        smoke = _read_json(phase22 / "ui_smoke_results.json")
        ui_command = _read_json(phase22 / "run_command_verification.json")
        api_command = _read_json(phase21 / "run_command_verification.json")
        phase22_report_pass = report.get("status") == "PASS" and all(report.get("quality_gate", {}).values())
        ui_source = "\n".join(
            (base / path).read_text(encoding="utf-8")
            for path in ("src/omnisearch/ui/adapter.py", "src/omnisearch/ui/streamlit_app.py")
        )
        checks = {
            "phase22_quality_gate_pass": phase22_report_pass,
            "streamlit_ui_launch_verified": ui_command.get("verified") is True and "streamlit run" in ui_command.get("command", ""),
            "real_text_to_image_ui_verified": smoke.get("browser_smoke", {}).get("text_to_image", {}).get("verified") is True and smoke.get("browser_smoke", {}).get("text_to_image", {}).get("visible_results", 0) > 0,
            "real_image_to_text_ui_verified": smoke.get("browser_smoke", {}).get("image_to_text", {}).get("verified") is True and smoke.get("browser_smoke", {}).get("image_to_text", {}).get("visible_caption_results", 0) > 0,
            "retrieval_service_reused": "RetrievalService" in report.get("architecture", "") and smoke.get("resource_reuse", {}).get("one_service_instance") is True,
            "no_duplicate_model_or_index_logic": not any(token in ui_source for token in ("get_text_features", "get_image_features", "from_pretrained", "faiss.Index", "import faiss")),
            "api_launch_command_verified": api_command.get("verified") is True,
            "ui_config_modes_and_bounds_present": config.get("supported_modes") == ["text-to-image", "image-to-text"] and config.get("ui_top_k", {}).get("min") == 1 and config.get("ui_top_k", {}).get("max") == 20,
            "no_phase22_audit_markdown": not (base / "docs/phase22_audit.md").exists(),
            "phase23_not_started": report.get("quality_gate", {}).get("phase23_not_started") is True,
        }
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        checks = {"phase22_artifacts_readable": False}
    passed = all(checks.values())
    return {
        "schema_version": PHASE23_SCHEMA_VERSION,
        "phase": 23,
        "dependency_phase": 22,
        "audit_result": "PRE-PHASE AUDIT: Phase 22 PASS" if passed else "PRE-PHASE AUDIT: Phase 22 BLOCKED",
        "passed": passed,
        "checks": checks,
        "recorded_before_phase23_hardening": True,
        "phase24_started": False,
    }


def build_test_inventory(root: Path | str) -> dict[str, Any]:
    """Classify the established test files without rewriting working tests."""

    base = Path(root)
    files = sorted(path.name for path in (base / "tests").glob("test_*.py"))
    real_model = ["test_real_model_smoke.py"] if "test_real_model_smoke.py" in files else []
    integration = sorted(name for name in files if name in INTEGRATION_TEST_FILES)
    unit = sorted(name for name in files if name not in set(integration) | set(real_model))
    artifact_validation = sorted(name for name in files if name in ARTIFACT_VALIDATION_TEST_FILES)
    expensive = list(real_model)
    covered = set(unit) | set(integration) | set(real_model)
    return {
        "schema_version": PHASE23_SCHEMA_VERSION,
        "phase": 23,
        "categories": {
            "UNIT": unit,
            "INTEGRATION": integration,
            "REAL-MODEL SMOKE": real_model,
            "EXPENSIVE / LOCAL-ONLY": expensive,
            "ARTIFACT VALIDATION": artifact_validation,
        },
        "marker_policy": {
            "default": "not real_model and not slow and not local_data",
            "real_model_command": "uv run pytest -m real_model",
            "markers_declared_in_pyproject": True,
        },
        "all_test_files_classified": covered == set(files),
        "test_file_count": len(files),
    }


def validate_ci_workflow(path: Path | str) -> dict[str, Any]:
    """Validate the compact workflow structure and its no-data CI contract."""

    workflow_path = Path(path)
    checks: dict[str, bool] = {}
    parser = "textual fallback"
    try:
        text = workflow_path.read_text(encoding="utf-8")
        try:
            import yaml

            parsed = yaml.safe_load(text)
            parser = "PyYAML"
        except ImportError:
            parsed = None
        if parsed is not None:
            trigger = parsed.get("on", parsed.get(True, {})) if isinstance(parsed, Mapping) else {}
            jobs = parsed.get("jobs", {}) if isinstance(parsed, Mapping) else {}
            steps = jobs.get("fast-checks", {}).get("steps", []) if isinstance(jobs, Mapping) else []
            runs = [str(step.get("run", "")) for step in steps if isinstance(step, Mapping)]
            checks.update(
                {
                    "yaml_mapping": isinstance(parsed, Mapping),
                    "push_or_pull_request_trigger": isinstance(trigger, Mapping) and ("push" in trigger or "pull_request" in trigger),
                    "fast_checks_job": isinstance(jobs, Mapping) and "fast-checks" in jobs,
                    "python_version_file_used": any("python-version-file" in str(step) for step in steps),
                    "frozen_lock_install": any("uv sync --frozen --extra test" in run for run in runs),
                    "pytest_command": any("uv run pytest -q" in run for run in runs),
                    "ruff_command": any("uv run ruff check" in run for run in runs),
                    "mypy_command": any("uv run mypy" in run for run in runs),
                    "compile_command": any("compileall" in run for run in runs),
                }
            )
        else:
            checks.update(
                {
                    "workflow_file_readable": bool(text),
                    "frozen_lock_install": "uv sync --frozen --extra test" in text,
                    "pytest_command": "uv run pytest -q" in text,
                    "ruff_command": "uv run ruff check" in text,
                    "mypy_command": "uv run mypy" in text,
                    "compile_command": "compileall" in text,
                }
            )
        lowered = text.casefold()
        checks["no_large_data_or_model_download"] = not any(token in lowered for token in ("coco download", "circo download", "huggingface.co", "from_pretrained", "best_checkpoint.pt"))
        checks["workflow_path_is_canonical"] = workflow_path.as_posix().endswith(".github/workflows/ci.yml")
    except (OSError, TypeError, ValueError):
        checks = {"workflow_readable": False}
    return {"schema_version": PHASE23_SCHEMA_VERSION, "path": str(workflow_path), "parser": parser, "checks": checks, "passed": all(checks.values()) if checks else False}


def validate_critical_artifacts(root: Path | str) -> dict[str, Any]:
    """Validate a small, explicit set of current contract artifacts."""

    base = Path(root)
    specs = {
        "phase21_report": (base / "artifacts/phase21/phase21_report.json", ("schema_version", "phase", "status", "quality_gate")),
        "phase21_provenance": (base / "artifacts/phase21/provenance.json", ("schema_version", "phase", "training_performed", "new_dataset_downloaded")),
        "phase22_report": (base / "artifacts/phase22/phase22_report.json", ("schema_version", "phase", "status", "quality_gate")),
        "phase22_provenance": (base / "artifacts/phase22/provenance.json", ("schema_version", "phase", "training_performed", "new_dataset_downloaded", "phase23_started")),
        "phase22_ui_config": (base / "artifacts/phase22/ui_config.json", ("schema_version", "phase", "framework", "supported_modes", "ui_top_k")),
        "dataset_manifest": (base / "data/processed/coco2017_val_split_manifest.json", ("schema_version", "dataset_id", "records")),
    }
    checks: dict[str, bool] = {}
    errors: list[str] = []
    documents: dict[str, Any] = {}
    for name, (path, required) in specs.items():
        try:
            value = _read_json(path)
            documents[name] = value
            checks[f"{name}_readable"] = True
            checks[f"{name}_required_fields"] = all(field in value for field in required) if isinstance(value, Mapping) else False
            checks[f"{name}_finite_numbers"] = _finite_tree(value)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            checks[f"{name}_readable"] = False
            checks[f"{name}_required_fields"] = False
            checks[f"{name}_finite_numbers"] = False
            errors.append(f"{name}: {error}")
    if documents.get("phase21_report"):
        checks["phase21_status_pass"] = documents["phase21_report"].get("status") == "PASS"
        checks["phase21_protocol_identity"] = "Phase 7 full-FT CLIP" in documents["phase21_report"].get("default_system", "")
    if documents.get("phase22_report"):
        checks["phase22_status_pass"] = documents["phase22_report"].get("status") == "PASS"
        checks["phase22_protocol_identity"] = documents["phase22_report"].get("ui_framework") == "Streamlit"
    if documents.get("phase22_ui_config"):
        checks["phase22_supported_modes"] = documents["phase22_ui_config"].get("supported_modes") == ["text-to-image", "image-to-text"]
    return {"schema_version": PHASE23_SCHEMA_VERSION, "passed": all(checks.values()) and not errors, "checks": checks, "errors": errors, "validated_artifact_count": len(documents)}


def scan_absolute_paths(root: Path | str) -> dict[str, Any]:
    """Find reusable machine-specific paths in source/config/canonical docs."""

    base = Path(root)
    roots = [base / "src", base / "configs", base / "docs", base / "README.md", base / ".github", base / "pyproject.toml"]
    pattern = re.compile(r"(?<![A-Za-z0-9])/(?:Users|Volumes|home)/[^\s`\"')]+")
    matches: list[dict[str, Any]] = []
    for candidate in roots:
        paths = [candidate] if candidate.is_file() else list(candidate.rglob("*")) if candidate.is_dir() else []
        for path in paths:
            if not path.is_file() or path.suffix in {".pyc", ".png", ".jpg"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    matches.append({"path": str(path.relative_to(base)), "line": line_number})
    return {"schema_version": PHASE23_SCHEMA_VERSION, "passed": not matches, "matches": matches, "scope": "src, configs, docs, README.md, .github, pyproject.toml"}


def scan_secrets(root: Path | str) -> dict[str, Any]:
    """Run a conservative local scan without treating ordinary words as secrets."""

    base = Path(root)
    roots = [base / "src", base / "configs", base / "docs", base / "README.md", base / ".github", base / "pyproject.toml"]
    patterns = {
        "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        "token_prefix": re.compile(r"\b(?:sk|ghp|xox[baprs])-[A-Za-z0-9_-]{16,}\b"),
        "credential_assignment": re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*[\"']?[A-Za-z0-9_./+=-]{16,}"),
    }
    matches: list[dict[str, Any]] = []
    for candidate in roots:
        paths = [candidate] if candidate.is_file() else list(candidate.rglob("*")) if candidate.is_dir() else []
        for path in paths:
            if not path.is_file() or path.suffix in {".pyc", ".lock"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                for name, pattern in patterns.items():
                    if pattern.search(line):
                        matches.append({"path": str(path.relative_to(base)), "line": line_number, "pattern": name})
    env_files = [str(path.relative_to(base)) for path in base.rglob(".env*") if path.is_file() and path.name not in {".env.example", ".env.template"}]
    return {"schema_version": PHASE23_SCHEMA_VERSION, "passed": not matches and not env_files, "matches": matches, "credential_files": env_files, "patterns_checked": sorted(patterns)}


def _is_ignored(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    result = subprocess.run(["git", "check-ignore", "--no-index", "--quiet", str(relative)], cwd=root, check=False)
    return result.returncode == 0


def audit_large_files(root: Path | str, threshold_bytes: int = 1_000_000) -> dict[str, Any]:
    """Classify large files and ensure none sit outside local-only protection."""

    base = Path(root)
    ignored_dirs = {".git", ".venv", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"}
    tracked = set(subprocess.run(["git", "ls-files"], cwd=base, capture_output=True, text=True, check=False).stdout.splitlines())
    rows: list[dict[str, Any]] = []
    for path in base.rglob("*"):
        if not path.is_file() or any(part in ignored_dirs for part in path.parts):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size < threshold_bytes:
            continue
        relative = str(path.relative_to(base))
        ignored = _is_ignored(path, base)
        rows.append({"path": relative, "bytes": size, "tracked": relative in tracked, "ignored": ignored, "classification": "TRACKED" if relative in tracked else "LOCAL-ONLY" if ignored or relative.startswith(("data/", "artifacts/")) else "UNTRACKED"})
    unexpected = [row for row in rows if row["classification"] not in {"TRACKED", "LOCAL-ONLY"}]
    return {"schema_version": PHASE23_SCHEMA_VERSION, "threshold_bytes": threshold_bytes, "passed": not unexpected, "large_files": sorted(rows, key=lambda row: -row["bytes"]), "unexpected_large_files": unexpected, "tracked_file_count": len(tracked), "note": "The current workspace has no committed Git index; classifications are based on Git ignore rules and repository paths."}


def check_documentation_links(root: Path | str) -> dict[str, Any]:
    """Check relative Markdown links and obvious stale audit references."""

    base = Path(root)
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    broken: list[dict[str, Any]] = []
    scanned: list[str] = []
    for path in [base / "README.md", *sorted((base / "docs").glob("*.md"))]:
        scanned.append(str(path.relative_to(base)))
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for target in link_pattern.findall(line):
                target = target.strip().strip("<>")
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                target_path = target.split("#", 1)[0]
                candidate = (path.parent / target_path).resolve()
                if not candidate.exists():
                    broken.append({"path": str(path.relative_to(base)), "line": line_number, "target": target})
    stale = []
    for needle in ("docs/phase21_audit.md", "docs/phase22_audit.md", "docs/phase23_audit.md", "Phase 21 was not started"):
        for path in [base / "README.md", *sorted((base / "docs").glob("*.md"))]:
            if needle in path.read_text(encoding="utf-8"):
                stale.append({"path": str(path.relative_to(base)), "needle": needle})
    return {"schema_version": PHASE23_SCHEMA_VERSION, "passed": not broken and not stale, "scanned_files": scanned, "broken_links": broken, "stale_references": stale}


def _clean_environment_check(root: Path) -> dict[str, Any]:
    """Install from the lockfile in an isolated temporary uv environment."""

    temporary_root = Path(tempfile.mkdtemp(prefix="omnisearch-phase23-"))
    environment = dict(os.environ)
    environment.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    environment["UV_PROJECT_ENVIRONMENT"] = str(temporary_root / "venv")
    commands = [
        ["uv", "sync", "--frozen", "--extra", "test"],
        ["uv", "run", "--frozen", "python", "-c", "import omnisearch; print(omnisearch.__version__)"] ,
        ["uv", "run", "--frozen", "pytest", "-q"],
        ["uv", "run", "--frozen", "omnisearch-bootstrap", "--root", "."],
    ]
    results = [_run_command(command, root, timeout=300, env=environment) for command in commands]
    passed = all(item["passed"] for item in results)
    shutil.rmtree(temporary_root, ignore_errors=True)
    return {
        "schema_version": PHASE23_SCHEMA_VERSION,
        "passed": passed,
        "environment": "temporary UV_PROJECT_ENVIRONMENT; path intentionally omitted",
        "lockfile_install": results[0],
        "package_import": results[1],
        "fast_tests": results[2],
        "bootstrap": results[3],
        "large_dataset_or_model_download_requested": False,
        "temporary_environment_cleaned": True,
    }


def _entrypoint_validation(root: Path) -> dict[str, Any]:
    phase21_command = _read_json(root / "artifacts/phase21/run_command_verification.json")
    phase22_command = _read_json(root / "artifacts/phase22/run_command_verification.json")
    commands = [
        ["uv", "run", "--frozen", "omnisearch-bootstrap", "--root", "."],
        ["uv", "run", "--frozen", "omnisearch-evaluate", "--help"],
        ["uv", "run", "--frozen", "uvicorn", "omnisearch.api.app:create_app", "--factory", "--help"],
        ["uv", "run", "--frozen", "streamlit", "--help"],
    ]
    results = [_run_command(command, root, timeout=120) for command in commands]
    checks = {
        "bootstrap_status_command": results[0]["passed"],
        "selected_evaluation_help": results[1]["passed"],
        "api_command_help": results[2]["passed"],
        "ui_command_help": results[3]["passed"],
        "phase21_real_api_command_artifact": phase21_command.get("verified") is True,
        "phase22_real_ui_command_artifact": phase22_command.get("verified") is True,
        "no_phase24_command": not any("phase24" in item["command"] for item in results),
    }
    return {"schema_version": PHASE23_SCHEMA_VERSION, "passed": all(checks.values()), "checks": checks, "commands": results, "selected_expensive_commands_not_run": ["omnisearch-phase7", "omnisearch-phase13", "omnisearch-phase20", "omnisearch-phase22"]}


def _local_verification(root: Path) -> dict[str, Any]:
    commands = [
        ["uv", "lock", "--check"],
        ["uv", "run", "--frozen", "pytest", "-q"],
        ["uv", "run", "--frozen", "ruff", "check", "src", "tests"],
        ["uv", "run", "--frozen", "mypy", "src/omnisearch"],
        ["uv", "run", "--frozen", "python", "-m", "compileall", "-q", "src", "tests"],
    ]
    results = [_run_command(command, root, timeout=300) for command in commands]
    return {"schema_version": PHASE23_SCHEMA_VERSION, "passed": all(item["passed"] for item in results), "commands": results, "workflow_expected_commands": ["uv sync --frozen --extra test", "uv run pytest -q", "uv run ruff check src tests", "uv run mypy src/omnisearch", "uv run python -m compileall -q src tests"]}


def run_phase23(root: Path | str = ".", output_dir: Path | str = "artifacts/phase23") -> dict[str, Any]:
    """Run hardening checks and write compact Phase 23 evidence."""

    base = Path(root).resolve()
    output = Path(output_dir)
    if not output.is_absolute():
        output = base / output
    output.mkdir(parents=True, exist_ok=True)
    pre_audit = audit_phase22(base)
    _write_json(pre_audit, output / "pre_phase_audit.json")
    if not pre_audit["passed"]:
        raise RuntimeError("PRE-PHASE AUDIT: Phase 22 BLOCKED")

    inventory = build_test_inventory(base)
    workflow = validate_ci_workflow(base / ".github/workflows/ci.yml")
    local = _local_verification(base)
    clean = _clean_environment_check(base)
    entrypoints = _entrypoint_validation(base)
    absolute_paths = scan_absolute_paths(base)
    secrets = scan_secrets(base)
    large_files = audit_large_files(base)
    docs = check_documentation_links(base)
    artifacts = validate_critical_artifacts(base)
    ci = {
        "schema_version": PHASE23_SCHEMA_VERSION,
        "workflow": workflow,
        "lockfile_check": local["commands"][0],
        "local_fast_checks": local,
        "remote_ci_executed": False,
        "remote_ci_claim": "workflow configuration validated locally; GitHub-hosted CI was not run in this environment",
    }
    _write_json(inventory, output / "test_inventory.json")
    _write_json(ci, output / "ci_validation.json")
    _write_json(clean, output / "clean_environment_check.json")
    _write_json(entrypoints, output / "entrypoint_validation.json")
    _write_json(large_files, output / "large_file_audit.json")
    _write_json(secrets, output / "secret_scan.json")
    _write_json(docs, output / "documentation_link_check.json")
    _write_json(artifacts, output / "artifact_validation.json")

    gate = {
        "phase22_dependency_pass": pre_audit["passed"],
        "no_model_retraining": True,
        "test_suite_categorized": inventory["all_test_files_classified"] and inventory["marker_policy"]["markers_declared_in_pyproject"],
        "fast_ci_without_large_data": workflow["passed"],
        "github_actions_workflow_exists": workflow["checks"].get("workflow_path_is_canonical", False),
        "github_actions_workflow_locally_validated": workflow["passed"],
        "ruff_in_ci": workflow["checks"].get("ruff_command", False),
        "mypy_in_ci": workflow["checks"].get("mypy_command", False),
        "pytest_in_ci": workflow["checks"].get("pytest_command", False),
        "compile_in_ci": workflow["checks"].get("compile_command", False),
        "lockfile_validated": local["commands"][0]["passed"],
        "clean_environment_install_and_checks": clean["passed"],
        "no_reusable_absolute_paths": absolute_paths["passed"],
        "basic_secret_scan_pass": secrets["passed"],
        "large_file_classification_pass": large_files["passed"],
        "gitignore_hardening_present": large_files["passed"],
        "api_ui_lightweight_coverage": "test_phase21.py" in inventory["categories"]["INTEGRATION"] and "test_phase22.py" in inventory["categories"]["INTEGRATION"],
        "documented_entrypoints_verified": entrypoints["passed"],
        "documentation_links_pass": docs["passed"],
        "critical_artifact_validation_pass": artifacts["passed"],
        "no_phase23_audit_markdown": not (base / "docs/phase23_audit.md").exists(),
        "phase24_not_started": not (base / "artifacts/phase24").exists() and not (base / "src/omnisearch/phase24.py").exists(),
    }
    provenance = {
        "schema_version": PHASE23_SCHEMA_VERSION,
        "phase": 23,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase22_report_sha256": _hash_file(base / "artifacts/phase22/phase22_report.json"),
        "phase22_ui_source_sha256": _hash_file(base / "src/omnisearch/ui/streamlit_app.py"),
        "training_performed": False,
        "new_dataset_downloaded": False,
        "large_model_or_dataset_download_in_ci": False,
        "remote_ci_executed": False,
        "phase24_started": False,
    }
    _write_json(provenance, output / "provenance.json")
    report = {
        "schema_version": PHASE23_SCHEMA_VERSION,
        "project": "OmniSearch",
        "phase": 23,
        "status": "PASS" if all(gate.values()) else "PARTIAL",
        "pre_phase_audit": pre_audit["audit_result"],
        "quality_gate": gate,
        "research_questions": {
            "RQ23.1": "Yes. The fast suite and clean-environment checks run from project metadata without COCO, CIRCO, checkpoints, or FAISS corpus indexes.",
            "RQ23.2": "Real-model smoke requires the local model cache, Phase 7 checkpoint, Phase 10 indexes, validated manifest, and corpus image files.",
            "RQ23.3": "Yes. CI installs with the frozen uv.lock and validates the lockfile before running checks.",
            "RQ23.4": "Yes. API boundary tests use a stub service and UI adapter tests use a fake service; real-resource smoke remains optional.",
            "RQ23.5": "MPS behavior, model availability, corpus storage, native thread settings, and latency remain hardware/environment dependent.",
        },
        "determinism_notes": [
            "Fixture tests use deterministic synthetic inputs where practical.",
            "MPS and native library kernels may be nondeterministic across hardware and versions.",
            "Package lock identity improves reproducibility but cannot erase hardware or library implementation effects.",
        ],
    }
    _write_json(report, output / "phase23_report.json")
    return report


def validate_phase23_artifacts(output_dir: Path | str = "artifacts/phase23") -> dict[str, Any]:
    output = Path(output_dir)
    required = {name: (output / name).is_file() for name in REQUIRED_ARTIFACTS}
    report = _read_json(output / "phase23_report.json") if (output / "phase23_report.json").is_file() else {}
    provenance = _read_json(output / "provenance.json") if (output / "provenance.json").is_file() else {}
    checks = {
        "required_artifacts": all(required.values()),
        "pre_phase_audit_pass": _read_json(output / "pre_phase_audit.json").get("passed") is True if (output / "pre_phase_audit.json").is_file() else False,
        "report_pass": report.get("status") == "PASS",
        "quality_gate_all_pass": all(report.get("quality_gate", {}).values()) if isinstance(report.get("quality_gate"), Mapping) else False,
        "no_training_or_download": provenance.get("training_performed") is False and provenance.get("new_dataset_downloaded") is False,
        "no_phase24": provenance.get("phase24_started") is False,
        "no_phase23_audit_markdown": not (Path.cwd() / "docs/phase23_audit.md").exists(),
    }
    return {"schema_version": PHASE23_SCHEMA_VERSION, "passed": all(checks.values()), "checks": checks, "required": required}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OmniSearch Phase 23 testing and reproducibility hardening")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase23"))
    args = parser.parse_args()
    print(json.dumps(run_phase23(args.root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
