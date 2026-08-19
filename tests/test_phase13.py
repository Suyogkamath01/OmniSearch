from __future__ import annotations

import json

import pytest

from omnisearch.manifest import CaptionRecord, ImageRecord
from omnisearch.phase7 import _subset_records
from omnisearch.phase13 import (
    ALL_METRICS,
    PRIMARY_SEEDS,
    aggregate_seed_metrics,
    align_query_metrics,
    classify_seed_stability,
    holm_bonferroni,
    paired_bootstrap,
    paired_permutation_test,
    per_query_metrics,
    seed_everything,
    validate_seed_plan,
    win_loss_tie,
)


def _result(query_ids: list[str], task: str = "text_to_image") -> dict[str, object]:
    return {
        "task": task,
        "ranking_records": [
            {
                "query_id": query_id,
                "candidate_ids": ["positive", "other"],
                "relevant_ids": ["positive"],
            }
            for query_id in query_ids
        ],
    }


def test_seed_plan_is_exact_and_seed_setup_is_recorded() -> None:
    assert validate_seed_plan(PRIMARY_SEEDS) == (42, 123, 2026)
    with pytest.raises(ValueError):
        validate_seed_plan((42, 123))
    details = seed_everything(123)
    assert details["python_seeded"] is True
    assert details["numpy_seeded"] is True


def test_per_query_metrics_and_alignment_reject_query_mismatch() -> None:
    metrics = per_query_metrics(_result(["a", "b"]))
    assert set(metrics["a"]) == set(ALL_METRICS)
    assert metrics["a"]["recall_at_1"] == 1.0
    assert metrics["a"]["mrr"] == 1.0
    with pytest.raises(ValueError):
        align_query_metrics(metrics, {"a": metrics["a"]})


def test_aggregate_seed_metrics_uses_sample_standard_deviation() -> None:
    summary = aggregate_seed_metrics(
        [{"seed": 42, "value": 1.0}, {"seed": 123, "value": 2.0}, {"seed": 2026, "value": 3.0}]
    )
    assert summary["mean"] == 2.0
    assert summary["sample_std"] == 1.0
    assert summary["min"] == 1.0
    assert summary["max"] == 3.0


def test_paired_bootstrap_and_permutation_are_reproducible() -> None:
    baseline = [0.0, 0.0, 1.0, 1.0]
    comparison = [1.0, 1.0, 1.0, 1.0]
    first = paired_bootstrap(baseline, comparison, resamples=50, seed=9)
    second = paired_bootstrap(baseline, comparison, resamples=50, seed=9)
    assert first == second
    assert first["observed_delta"] == 0.5
    permutation = paired_permutation_test(baseline, comparison, permutations=100, seed=9)
    assert permutation["observed_delta"] == 0.5
    assert 0.0 <= permutation["p_value"] <= 1.0


def test_holm_bonferroni_and_win_loss_tie() -> None:
    correction = holm_bonferroni({"large": 0.04, "small": 0.001, "none": 0.8})
    assert correction["tests"]["small"]["adjusted_p_value"] == pytest.approx(0.003)
    counts = win_loss_tie([0.5, 0.5, 0.5], [0.6, 0.5, 0.4])
    assert counts["improved"] == 1
    assert counts["unchanged"] == 1
    assert counts["degraded"] == 1


def test_seed_stability_categories_cover_actual_directions() -> None:
    assert classify_seed_stability([0.1, 0.2, 0.3]) == "ROBUST"
    assert classify_seed_stability([0.1, 0.0, 0.2]) == "DIRECTIONALLY CONSISTENT BUT UNCERTAIN"
    assert classify_seed_stability([-0.1, 0.0, 0.1]) == "SEED-SENSITIVE"
    assert classify_seed_stability([0.0, 0.0, 0.0]) == "NO CLEAR DIFFERENCE"
    assert classify_seed_stability([-0.1, -0.2, -0.3]) == "NOT SUPPORTED"


def test_fixed_subset_seed_keeps_the_same_tier2_groups_across_training_seeds() -> None:
    records = tuple(
        ImageRecord(
            image_id=str(index),
            filename=f"{index}.jpg",
            captions=(CaptionRecord(caption_id=f"c{index}", text="caption"),),
            split="train",
        )
        for index in range(20)
    )
    seed42_subset = _subset_records(records, "train", 42, 8)
    seed123_with_fixed_subset = _subset_records(records, "train", 42, 8)
    seed123_subset = _subset_records(records, "train", 123, 8)
    assert tuple(item.image_id for item in seed42_subset) == tuple(
        item.image_id for item in seed123_with_fixed_subset
    )
    assert tuple(item.image_id for item in seed42_subset) != tuple(
        item.image_id for item in seed123_subset
    )


def test_malformed_result_is_rejected(tmp_path) -> None:
    from omnisearch.phase13 import _result_payload

    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"task": "text_to_image"}), encoding="utf-8")
    with pytest.raises(ValueError):
        _result_payload(path)
