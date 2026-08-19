"""Focused tests for Phase 23 validation and reproducibility contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnisearch.phase23 import (
    _read_json,
    audit_phase22,
    build_test_inventory,
    check_documentation_links,
    validate_ci_workflow,
    validate_critical_artifacts,
    validate_phase23_artifacts,
)


@pytest.mark.local_data
def test_phase22_dependency_audit_passes() -> None:
    result = audit_phase22(Path.cwd())
    assert result["passed"] is True, result
    assert result["phase24_started"] is False


def test_test_inventory_classifies_every_file() -> None:
    result = build_test_inventory(Path.cwd())
    assert result["all_test_files_classified"] is True
    assert "test_real_model_smoke.py" in result["categories"]["REAL-MODEL SMOKE"]
    assert result["marker_policy"]["default"] == "not real_model and not slow and not local_data"


def test_ci_workflow_has_required_fast_checks() -> None:
    result = validate_ci_workflow(Path(".github/workflows/ci.yml"))
    assert result["passed"] is True, result
    assert result["checks"]["no_large_data_or_model_download"] is True


def test_artifact_reader_rejects_nonfinite_json(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"value": NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        _read_json(invalid)


def test_critical_artifact_validator_reports_missing_contract(tmp_path: Path) -> None:
    result = validate_critical_artifacts(tmp_path)
    assert result["passed"] is False
    assert result["validated_artifact_count"] == 0


def test_documentation_links_are_current() -> None:
    result = check_documentation_links(Path.cwd())
    assert result["passed"] is True, result


def test_phase23_artifact_validator_rejects_incomplete_directory(tmp_path: Path) -> None:
    result = validate_phase23_artifacts(tmp_path)
    assert result["passed"] is False
    assert result["checks"]["required_artifacts"] is False
