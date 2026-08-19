"""Focused Phase 26 schema and decision tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnisearch.phase26 import (
    _ALLOWED_COMPONENT_STATUSES,
    _LIMITATION_CATEGORIES,
    REQUIRED_ARTIFACTS,
    audit_phase25_dependency,
    validate_phase26_artifacts,
)


@pytest.mark.local_data
def test_phase25_dependency_audit_is_passed() -> None:
    result = audit_phase25_dependency(Path.cwd())
    assert result["passed"] is True
    assert result["audit_result"] == "PRE-PHASE AUDIT: Phase 25 PASS"


def test_phase26_artifact_validator_rejects_incomplete_directory(tmp_path: Path) -> None:
    result = validate_phase26_artifacts(tmp_path)
    assert result["passed"] is False
    assert result["checks"]["required_artifacts"] is False
    assert set(result["required"]) == set(REQUIRED_ARTIFACTS)


def test_phase26_artifact_validator_accepts_minimal_valid_fixture(tmp_path: Path) -> None:
    for name in REQUIRED_ARTIFACTS:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "phase26_report.json").write_text(
        json.dumps({"status": "PASS", "quality_gate": {"complete": True}}),
        encoding="utf-8",
    )
    (tmp_path / "provenance.json").write_text(
        json.dumps({"training_performed": False, "new_dataset_downloaded": False, "phase27_started": False}),
        encoding="utf-8",
    )
    (tmp_path / "frozen_configuration.json").write_text(
        json.dumps({"retrieval": {"backend": "FAISS Flat exact inner-product search", "reranker_enabled": False}}),
        encoding="utf-8",
    )
    (tmp_path / "component_decisions.json").write_text(
        json.dumps({"decisions": [{"component": "Flat", "status": "KEEP", "reason": "exact", "evidence_ref": "fixture"}]}),
        encoding="utf-8",
    )
    (tmp_path / "limitation_consolidation.json").write_text(
        json.dumps({"categories": {"RESOLVED": [], "MUST_FIX_BEFORE_FINAL_RELEASE": []}}),
        encoding="utf-8",
    )
    (tmp_path / "final_scorecard.json").write_text(
        json.dumps({"status": "PASS", "checks": {"complete": True}}),
        encoding="utf-8",
    )
    result = validate_phase26_artifacts(tmp_path)
    assert result["passed"] is True


def test_phase26_status_and_limitation_vocabularies_are_explicit() -> None:
    assert {"KEEP", "OPTIONAL", "DISABLED_DEFAULT", "REMOVE"} <= _ALLOWED_COMPONENT_STATUSES
    assert {"RESOLVED", "ACCEPTED_SCOPE_LIMITATION", "OPTIONAL_FUTURE_EVALUATION", "MUST_FIX_BEFORE_FINAL_RELEASE"} == _LIMITATION_CATEGORIES
