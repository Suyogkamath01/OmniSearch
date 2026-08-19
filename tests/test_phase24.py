"""Focused tests for native deployment configuration and preflight behavior."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from omnisearch.clip_baseline import select_device
from omnisearch.deployment import (
    DeploymentConfig,
    api_command,
    build_deployment_manifest,
    port_available,
    run_preflight,
    runtime_environment,
    ui_command,
)
from omnisearch.phase24 import audit_phase23_dependency, validate_phase24_artifacts


@pytest.mark.local_data
def test_phase23_dependency_audit_passes_before_deployment() -> None:
    result = audit_phase23_dependency(Path.cwd())
    assert result["passed"] is True, result
    assert result["audit_result"] == "PRE-PHASE AUDIT: Phase 23 PASS"


def test_deployment_config_reads_environment_overrides(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OMNISEARCH_HOST", "127.0.0.1")
    monkeypatch.setenv("OMNISEARCH_API_PORT", "18000")
    monkeypatch.setenv("OMNISEARCH_UI_PORT", "18501")
    monkeypatch.setenv("OMNISEARCH_DEVICE", "cpu")
    monkeypatch.setenv("OMNISEARCH_MAX_IMAGE_PIXELS", "123456")
    monkeypatch.setenv("OMNISEARCH_OFFLINE", "1")
    config = DeploymentConfig.from_env(tmp_path)
    assert config.host == "127.0.0.1"
    assert config.api_port == 18000
    assert config.ui_port == 18501
    assert config.service.device == "cpu"
    assert config.service.max_image_pixels == 123456
    assert config.offline is True


def test_preflight_reports_actionable_missing_artifacts(tmp_path: Path) -> None:
    config = DeploymentConfig.from_env(tmp_path)
    result = run_preflight(config)
    assert result["passed"] is False
    assert result["checks"]["required_artifacts_present"] is False
    assert "Missing required local artifacts" in result["errors"][0]


def test_device_auto_falls_back_to_cpu() -> None:
    class FakeMPS:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        class backends:
            mps = FakeMPS()

    assert select_device("auto", FakeTorch()) == "cpu"


def test_runtime_environment_sets_offline_mode_without_secrets(tmp_path: Path) -> None:
    config = DeploymentConfig.from_env(tmp_path)
    config = config.__class__(config.root, config.service, config.host, config.api_port, config.ui_port, True, config.openmp_workaround)
    environment = runtime_environment(config)
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert "OPENAI_API_KEY" not in environment or environment["OPENAI_API_KEY"]


def test_launcher_commands_are_concise_and_configured(tmp_path: Path) -> None:
    config = DeploymentConfig.from_env(tmp_path)
    assert "omnisearch.api.app:create_app" in api_command(config)
    assert "18000" not in api_command(config)
    assert "src/omnisearch/ui/streamlit_app.py" in ui_command(config)
    assert str(config.ui_port) in ui_command(config)


def test_port_conflict_is_detected_without_killing_processes() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
        assert port_available("127.0.0.1", port) is False


def test_deployment_manifest_is_relative_and_schema_complete(tmp_path: Path) -> None:
    config = DeploymentConfig.from_env(tmp_path)
    manifest = build_deployment_manifest(config)
    assert manifest["deployment_mode"] == "native_uv"
    assert manifest["required_local_artifacts"]
    assert "/Users/" not in str(manifest)


def test_phase24_artifact_validator_rejects_incomplete_directory(tmp_path: Path) -> None:
    result = validate_phase24_artifacts(tmp_path)
    assert result["passed"] is False
    assert result["checks"]["required_artifacts"] is False
