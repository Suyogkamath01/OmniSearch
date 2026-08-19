"""Shared marker taxonomy for the fast and optional test suites."""

from __future__ import annotations

from pathlib import Path

import pytest

INTEGRATION_FILES = {"test_coco_dataset.py", "test_phase21.py", "test_phase22.py", "test_phase23.py", "test_phase24.py"}
ARTIFACT_VALIDATION_FILES = {
    "test_phase14.py",
    "test_phase16.py",
    "test_phase17.py",
    "test_phase18.py",
    "test_phase19.py",
    "test_phase20.py",
    "test_phase21.py",
    "test_phase22.py",
    "test_phase23.py",
    "test_phase24.py",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply stable categories without rewriting the established test modules."""

    for item in items:
        filename = Path(str(item.fspath)).name
        if filename == "test_real_model_smoke.py":
            item.add_marker(pytest.mark.real_model)
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.local_data)
        elif filename in INTEGRATION_FILES:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
        if filename in ARTIFACT_VALIDATION_FILES:
            item.add_marker(pytest.mark.artifact_validation)
