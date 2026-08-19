from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnisearch.phase20 import (
    aggregate_latency,
    bytes_to_mib,
    pareto_frontier,
    storage_calculation,
    validate_phase20_artifacts,
)


def test_storage_arithmetic_is_explicitly_calculated() -> None:
    result = storage_calculation(5_000, 512, 4)
    assert result["value"] == 10_240_000
    assert result["measurement_status"] == "CALCULATED"
    assert bytes_to_mib(result["value"]) == pytest.approx(9.765625)


def test_storage_arithmetic_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError):
        storage_calculation(0, 512, 4)
    with pytest.raises(ValueError):
        storage_calculation(5, 512, 0)


def test_latency_aggregation_preserves_components_and_total() -> None:
    result = aggregate_latency({"encode": 0.02, "search": 0.001, "optional": None})
    assert result["total_seconds"] == pytest.approx(0.021)
    assert result["components_seconds"] == {"encode": 0.02, "search": 0.001}
    assert result["measurement_status"] == "CONSOLIDATED_MEASURED_COMPONENTS"


def test_pareto_frontier_filters_dominated_configurations() -> None:
    rows = [
        {"configuration": "slow_best", "quality": 0.99, "cost": 10.0},
        {"configuration": "fast_good", "quality": 0.98, "cost": 2.0},
        {"configuration": "dominated", "quality": 0.97, "cost": 5.0},
    ]
    frontier = pareto_frontier(rows, quality_key="quality", cost_key="cost")
    assert [row["configuration"] for row in frontier] == ["fast_good", "slow_best"]
    assert all(row["pareto_status"] == "NON_DOMINATED" for row in frontier)


def test_phase20_artifact_validation_rejects_missing_directory(tmp_path: Path) -> None:
    result = validate_phase20_artifacts(tmp_path)
    assert result["passed"] is False
    assert result["checks"]["required_artifacts"] is False
    (tmp_path / "resource_inventory.json").write_text(json.dumps({"rows": []}))
    malformed = validate_phase20_artifacts(tmp_path)
    assert malformed["passed"] is False
