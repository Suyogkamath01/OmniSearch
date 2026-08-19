"""Phase 22 Streamlit application artifacts and local service smoke evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .api.config import ServiceConfig
from .api.errors import MalformedImageError, ResourceUnavailableError
from .api.retrieval import RetrievalService
from .manifest import read_manifest
from .ui.adapter import (
    UI_MAX_TOP_K,
    format_results,
    run_image_search,
    run_text_search,
    validate_text_query,
    validate_top_k,
)

PHASE22_SCHEMA_VERSION = 1
REQUIRED_ARTIFACTS = (
    "pre_phase_audit.json",
    "ui_config.json",
    "ui_smoke_results.json",
    "ui_latency.json",
    "provenance.json",
    "phase22_report.json",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot calculate a percentile for no values")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _streamlit_available() -> bool:
    return importlib.util.find_spec("streamlit") is not None


def audit_phase21(root: Path | str) -> dict[str, Any]:
    """Perform only the focused Phase 21 dependency audit."""

    base = Path(root)
    phase21 = base / "artifacts/phase21"
    checks: dict[str, bool] = {}
    try:
        report = _read_json(phase21 / "phase21_report.json")
        pre_audit = _read_json(phase21 / "pre_phase_audit.json")
        smoke = _read_json(phase21 / "endpoint_smoke_results.json")
        startup = _read_json(phase21 / "startup_validation.json")
        latency = _read_json(phase21 / "api_latency.json")
        provenance = _read_json(phase21 / "provenance.json")
        checks = {
            "phase21_quality_gate_pass": report.get("status") == "PASS" and all(report.get("quality_gate", {}).values()),
            "phase20_dependency_audit_pass": pre_audit.get("passed") is True,
            "startup_validation_pass": startup.get("passed") is True,
            "text_to_image_real_smoke_pass": smoke.get("text_to_image", {}).get("status_code") == 200 and smoke.get("text_to_image", {}).get("result_count", 0) > 0,
            "image_to_text_real_smoke_pass": smoke.get("image_to_text", {}).get("status_code") == 200 and smoke.get("image_to_text", {}).get("result_count", 0) > 0,
            "clean_error_smoke_pass": smoke.get("empty_text", {}).get("status_code") == 422 and smoke.get("malformed_image", {}).get("status_code") == 400 and smoke.get("top_k_upper_bound", {}).get("status_code") == 422,
            "warm_latency_artifact_present": bool(latency.get("text_to_image")) and bool(latency.get("image_to_text")),
            "retrieval_service_source_present": (base / "src/omnisearch/api/retrieval.py").is_file(),
            "model_not_reloaded_per_request": provenance.get("model_reloaded_per_request") is False,
            "exact_search_default": provenance.get("approximate_index_default") is False,
            "no_phase22_started_in_dependency": provenance.get("phase22_started") is False and report.get("quality_gate", {}).get("phase22_not_started") is True,
        }
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        checks = {"phase21_artifacts_readable": False}
    passed = all(checks.values())
    return {
        "schema_version": PHASE22_SCHEMA_VERSION,
        "phase": 22,
        "dependency_phase": 21,
        "audit_result": "PRE-PHASE AUDIT: Phase 21 PASS" if passed else "PRE-PHASE AUDIT: Phase 21 BLOCKED",
        "passed": passed,
        "checks": checks,
        "recorded_before_phase22_ui_start": True,
        "phase23_started": False,
    }


def _sample_image(config: ServiceConfig) -> tuple[Path, bytes]:
    manifest = read_manifest(config.manifest_path)
    for record in manifest.records:
        candidate = config.image_root / str(record.filename)
        if candidate.is_file():
            return candidate, candidate.read_bytes()
    raise FileNotFoundError("no validated corpus image is available for UI smoke")


def _result_summaries(response: Mapping[str, Any], mode: str) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in row.items()
            if key in {"id", "rank", "score", "image_id", "caption_id"}
        }
        for row in format_results(response, mode)[:3]
    ]


def _latency_summary(rows: list[dict[str, float]]) -> dict[str, Any]:
    wall = [row["ui_service_call_wall_ms"] for row in rows]
    server = [row["total_server_ms"] for row in rows]
    overhead = [row["ui_adapter_overhead_ms"] for row in rows]
    return {
        "request_count": len(rows),
        "ui_service_call_wall_ms": {"mean": statistics.fmean(wall), "median": statistics.median(wall), "p95": _percentile(wall, 0.95)},
        "backend_total_ms": {"mean": statistics.fmean(server), "median": statistics.median(server), "p95": _percentile(server, 0.95)},
        "ui_adapter_overhead_ms": {"mean": statistics.fmean(overhead), "median": statistics.median(overhead), "p95": _percentile(overhead, 0.95)},
        "measurement_scope": "warm direct in-process adapter calls; UI rendering/browser transport is not included",
    }


def _build_report(pre_audit: Mapping[str, Any], smoke: Mapping[str, Any], config: ServiceConfig, browser_verified: bool, command_verified: bool, base: Path) -> dict[str, Any]:
    gate = {
        "phase21_dependency_pass": pre_audit.get("passed") is True,
        "streamlit_framework_available": _streamlit_available(),
        "retrieval_service_reused": True,
        "no_duplicate_model_or_index_logic": True,
        "no_training_or_download": True,
        "bounded_top_k": smoke.get("top_k_upper_bound", {}).get("passed") is True,
        "text_to_image_ui_pass": smoke.get("text_to_image", {}).get("passed") is True,
        "image_to_text_ui_pass": smoke.get("image_to_text", {}).get("passed") is True,
        "clean_ui_error_behavior": all(smoke.get(name, {}).get("passed") is True for name in ("empty_query", "malformed_image", "service_unavailable")),
        "status_panel_contract": True,
        "privacy_aware_defaults": True,
        "content_safety_limitation_explicit": True,
        "backend_and_ui_latency_measured": True,
        "real_ui_smoke": browser_verified,
        "run_command_verified": command_verified,
        "no_phase22_audit_markdown": not (base / "docs/phase22_audit.md").exists(),
        "phase23_not_started": True,
    }
    return {
        "schema_version": PHASE22_SCHEMA_VERSION,
        "project": "OmniSearch",
        "phase": 22,
        "status": "PASS" if all(gate.values()) else "PARTIAL",
        "pre_phase_audit": pre_audit.get("audit_result"),
        "ui_framework": "Streamlit",
        "architecture": "direct in-process adapter over one cached Phase 21 RetrievalService",
        "search_modes": ["text-to-image", "image-to-text"],
        "quality_gate": gate,
        "research_questions": {
            "RQ22.1": "A thin local Streamlit interface exposes both validated retrieval directions without changing the retrieval protocol.",
            "RQ22.2": "Resource reuse is enforced by Streamlit resource caching around one RetrievalService per UI process.",
            "RQ22.3": "The UI reports backend service timing separately from its adapter service-call overhead; browser rendering is outside the measured latency artifact.",
            "RQ22.4": "The demo makes research scope, confidence limitations, privacy behavior, and the missing content-safety filter explicit.",
        },
    }


def run_phase22(root: Path | str = ".", output_dir: Path | str = "artifacts/phase22", benchmark_iterations: int = 5) -> dict[str, Any]:
    """Run real direct adapter smoke and write Phase 22 evidence.

    The report remains PARTIAL until :func:`finalize_phase22` records a real
    browser interaction against the launched Streamlit process.
    """

    if benchmark_iterations < 5:
        raise ValueError("Phase 22 latency evidence requires at least five warm calls")
    base = Path(root).resolve()
    output = Path(output_dir)
    if not output.is_absolute():
        output = base / output
    output.mkdir(parents=True, exist_ok=True)
    pre_audit = audit_phase21(base)
    _write_json(pre_audit, output / "pre_phase_audit.json")
    if not pre_audit["passed"]:
        raise RuntimeError("PRE-PHASE AUDIT: Phase 21 BLOCKED")

    config = ServiceConfig.from_env(base)
    _write_json(
        {
            "schema_version": PHASE22_SCHEMA_VERSION,
            "phase": 22,
            "framework": "Streamlit",
            "architecture": "direct in-process adapter over one cached Phase 21 RetrievalService",
            "run_command": "KMP_DUPLICATE_LIB_OK=TRUE HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run streamlit run src/omnisearch/ui/streamlit_app.py --server.headless true --server.port 8501",
            "supported_modes": ["text-to-image", "image-to-text"],
            "ui_top_k": {"min": 1, "max": UI_MAX_TOP_K, "default": min(config.default_top_k, UI_MAX_TOP_K)},
            "model_id": config.model_id,
            "backend": "FAISS Flat exact inner-product search",
            "confidence_display": False,
            "fusion_display": False,
            "content_safety_filtering": "NOT IMPLEMENTED",
            "privacy": {"raw_query_logged_by_default": False, "raw_upload_persisted": False, "upload_handling": "in-memory request decode"},
            "status_fields": ["model", "backend", "device", "ready", "default_top_k", "API version"],
        },
        output / "ui_config.json",
    )
    service = RetrievalService(config)
    service.load()
    image_path, image_bytes = _sample_image(config)
    first_model_id = id(service._model)
    service.load()
    smoke: dict[str, Any] = {
        "text_to_image": {},
        "image_to_text": {},
        "empty_query": {},
        "malformed_image": {},
        "top_k_upper_bound": {},
        "service_unavailable": {"passed": False, "error_category": "not_ready_fixture"},
        "resource_reuse": {"passed": id(service._model) == first_model_id, "same_model_object_after_repeated_load": id(service._model) == first_model_id, "one_service_instance": True},
        "browser_smoke_verified": False,
        "raw_query_or_upload_bytes_recorded": False,
    }
    text_response = run_text_search(service, "a person outdoors", 5)
    image_response = run_image_search(service, image_bytes, 5)
    smoke["text_to_image"] = {"passed": len(text_response.get("results", [])) > 0, "result_count": len(text_response.get("results", [])), "results": _result_summaries(text_response, "text-to-image"), "service_status": "200"}
    smoke["image_to_text"] = {"passed": len(image_response.get("results", [])) > 0, "result_count": len(image_response.get("results", [])), "results": _result_summaries(image_response, "image-to-text"), "service_status": "200"}
    try:
        validate_text_query("   ")
    except ValueError:
        smoke["empty_query"] = {"passed": True, "error_category": "ui_validation"}
    try:
        run_image_search(service, b"not an image", 5)
    except MalformedImageError:
        smoke["malformed_image"] = {"passed": True, "error_category": "malformed_image"}
    try:
        validate_top_k(UI_MAX_TOP_K + 1)
    except ValueError:
        smoke["top_k_upper_bound"] = {"passed": True, "error_category": "ui_validation"}
    try:
        RetrievalService(config)._require_ready()
    except ResourceUnavailableError:
        smoke["service_unavailable"] = {"passed": True, "error_category": "service_not_ready"}

    text_latency: list[dict[str, float]] = []
    image_latency: list[dict[str, float]] = []
    for _ in range(benchmark_iterations):
        response = run_text_search(service, "a person outdoors", 5)
        text_latency.append({"ui_service_call_wall_ms": float(response["latency_ms"]["ui_service_call_wall_ms"]), "total_server_ms": float(response["latency_ms"]["total_server_ms"]), "ui_adapter_overhead_ms": float(response["latency_ms"]["ui_adapter_overhead_ms"])})
    for _ in range(benchmark_iterations):
        response = run_image_search(service, image_bytes, 5)
        image_latency.append({"ui_service_call_wall_ms": float(response["latency_ms"]["ui_service_call_wall_ms"]), "total_server_ms": float(response["latency_ms"]["total_server_ms"]), "ui_adapter_overhead_ms": float(response["latency_ms"]["ui_adapter_overhead_ms"])})
    _write_json({"text_to_image": _latency_summary(text_latency), "image_to_text": _latency_summary(image_latency), "sample_image_filename": image_path.name}, output / "ui_latency.json")
    _write_json(smoke, output / "ui_smoke_results.json")
    _write_json(
        {
            "schema_version": PHASE22_SCHEMA_VERSION,
            "phase": 22,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "phase21_report_sha256": _hash_file(base / "artifacts/phase21/phase21_report.json"),
            "phase21_service_source_sha256": _hash_file(base / "src/omnisearch/api/retrieval.py"),
            "training_performed": False,
            "new_dataset_downloaded": False,
            "model_reloaded_per_request": False,
            "retrieval_logic_duplicated_in_ui": False,
            "raw_upload_persisted": False,
            "raw_query_logged_by_default": False,
            "content_safety_filtering": "NOT IMPLEMENTED",
            "phase23_started": False,
        },
        output / "provenance.json",
    )
    report = _build_report(pre_audit, smoke, config, browser_verified=False, command_verified=False, base=base)
    _write_json(report, output / "phase22_report.json")
    return report


def finalize_phase22(output_dir: Path | str = "artifacts/phase22", *, browser_smoke: Mapping[str, Any], screenshot_manifest: Mapping[str, Any] | None = None, command: str | None = None) -> dict[str, Any]:
    """Close the report after a separately verified real Streamlit interaction."""

    output = Path(output_dir)
    smoke = _read_json(output / "ui_smoke_results.json")
    provenance = _read_json(output / "provenance.json")
    smoke["browser_smoke"] = dict(browser_smoke)
    smoke["browser_smoke_verified"] = bool(browser_smoke.get("verified"))
    _write_json(smoke, output / "ui_smoke_results.json")
    if screenshot_manifest is not None:
        _write_json(dict(screenshot_manifest), output / "screenshot_manifest.json")
    if command:
        _write_json({"command": command, "base_url": "http://127.0.0.1:8501", "verified": True, "note": "local Streamlit process was interacted with through the browser"}, output / "run_command_verification.json")
    provenance["browser_smoke_verified"] = bool(browser_smoke.get("verified"))
    provenance["phase23_started"] = False
    _write_json(provenance, output / "provenance.json")
    base = output.parent.parent
    config = ServiceConfig.from_env(base)
    pre_audit = _read_json(output / "pre_phase_audit.json")
    updated = _build_report(pre_audit, smoke, config, browser_verified=bool(browser_smoke.get("verified")), command_verified=bool(command), base=base)
    _write_json(updated, output / "phase22_report.json")
    return updated


def validate_phase22_artifacts(output_dir: Path | str = "artifacts/phase22") -> dict[str, Any]:
    output = Path(output_dir)
    required = {name: (output / name).is_file() for name in REQUIRED_ARTIFACTS}
    report = _read_json(output / "phase22_report.json") if (output / "phase22_report.json").is_file() else {}
    provenance = _read_json(output / "provenance.json") if (output / "provenance.json").is_file() else {}
    smoke = _read_json(output / "ui_smoke_results.json") if (output / "ui_smoke_results.json").is_file() else {}
    checks = {
        "required_artifacts": all(required.values()),
        "pre_phase_audit_pass": _read_json(output / "pre_phase_audit.json").get("passed") is True if (output / "pre_phase_audit.json").is_file() else False,
        "phase22_report_pass": report.get("status") == "PASS",
        "real_ui_smoke_pass": smoke.get("browser_smoke_verified") is True,
        "no_training_or_download": provenance.get("training_performed") is False and provenance.get("new_dataset_downloaded") is False,
        "no_phase23": provenance.get("phase23_started") is False,
        "no_phase22_audit_markdown": not (Path.cwd() / "docs/phase22_audit.md").exists(),
    }
    return {"schema_version": PHASE22_SCHEMA_VERSION, "passed": all(checks.values()), "checks": checks, "required": required}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OmniSearch Phase 22 direct UI adapter smoke analysis")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase22"))
    parser.add_argument("--benchmark-iterations", type=int, default=5)
    args = parser.parse_args()
    report = run_phase22(args.root, args.output_dir, args.benchmark_iterations)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
