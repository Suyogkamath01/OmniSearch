"""Focused Phase 27 documentation and release-evidence tests."""

from __future__ import annotations

import json
from pathlib import Path

from omnisearch.phase27 import (
    REQUIRED_ARTIFACTS,
    _claims_validation,
    _documentation_consistency,
    _validate_links,
    validate_phase27_artifacts,
)


def test_phase27_dependency_documentation_checks_pass_after_finalization() -> None:
    root = Path.cwd()
    assert _documentation_consistency(root)["passed"] is True
    assert _claims_validation(root)["passed"] is True
    assert _validate_links(root)["passed"] is True


def test_phase27_artifact_validator_rejects_incomplete_directory(tmp_path: Path) -> None:
    result = validate_phase27_artifacts(tmp_path)
    assert result["passed"] is False
    assert result["checks"]["required_artifacts"] is False
    assert set(result["required"]) == set(REQUIRED_ARTIFACTS)


def test_phase27_artifact_validator_accepts_minimal_valid_fixture(tmp_path: Path) -> None:
    for name in REQUIRED_ARTIFACTS:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "phase27_report.json").write_text(
        json.dumps({"status": "PASS", "quality_gate": {"complete": True}}),
        encoding="utf-8",
    )
    (tmp_path / "provenance.json").write_text(
        json.dumps({"training_performed": False, "git_index_modified": False, "committed": False, "pushed": False}),
        encoding="utf-8",
    )
    (tmp_path / "artifact_validation.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    result = validate_phase27_artifacts(tmp_path)
    assert result["passed"] is True
