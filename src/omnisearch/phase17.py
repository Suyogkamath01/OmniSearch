"""Phase 17 uncertainty, confidence, calibration, and selective retrieval.

This module is deliberately evaluation-only.  It uses the retained Phase 7
test rankings and performs inference on the matching validation tier so that
calibration parameters are fitted without test labels.  Raw CLIP similarity
is always treated as a retrieval score or confidence proxy, never as a
probability.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import platform
import random
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .manifest import ImageRecord, read_manifest

PHASE17_SCHEMA_VERSION = 1
PROTOCOL_VERSION = "retrieval_eval_v1"
SYSTEMS = ("zero_shot", "full_ft")
DIRECTIONS = ("text_to_image", "image_to_text")
RAW_PROXIES = (
    "top1_score",
    "top1_top2_margin",
    "softmax_top1_mass",
    "entropy_confidence",
)
TARGET_COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.5)
RELIABILITY_BIN_COUNT = 10
BOOTSTRAP_RESAMPLES = 200
TOP_K = 10
HIGH_CONFIDENCE_ERROR_THRESHOLD = 0.8
LOW_CONFIDENCE_CORRECT_THRESHOLD = 0.5
EXPLICIT_TAXONOMY_SOURCES = {"heuristic", "mechanical", "mechanical_context"}
REQUIRED_ARTIFACTS = (
    "confidence_definition.json",
    "confidence_records.json",
    "success_failure_distributions.json",
    "discrimination_metrics.json",
    "calibration_parameters.json",
    "calibration_metrics.json",
    "reliability_bins.json",
    "selective_retrieval.json",
    "risk_coverage.json",
    "high_confidence_errors.json",
    "taxonomy_confidence.json",
    "robustness_confidence.json",
    "qualitative_examples.json",
    "provenance.json",
    "phase17_report.json",
)


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hash_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: Sequence[float]) -> dict[str, Any]:
    usable = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(usable),
        "mean": _mean(usable),
        "median": _median(usable),
        "q25": _quantile(usable, 0.25),
        "q75": _quantile(usable, 0.75),
        "min": min(usable) if usable else None,
        "max": max(usable) if usable else None,
    }


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exponent = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(max(value, -700.0))
    return exponent / (1.0 + exponent)


def _logit(value: float) -> float:
    bounded = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(bounded / (1.0 - bounded))


def _validate_binary_inputs(scores: Sequence[float], labels: Sequence[bool | int]) -> None:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    if not scores:
        raise ValueError("scores and labels cannot be empty")
    if any(not math.isfinite(float(score)) for score in scores):
        raise ValueError("scores must be finite")
    if any(int(label) not in {0, 1} for label in labels):
        raise ValueError("labels must be binary")


def roc_auc(scores: Sequence[float], labels: Sequence[bool | int]) -> float | None:
    """Calculate tie-aware ROC-AUC; return None when one class is absent."""

    _validate_binary_inputs(scores, labels)
    positives = sum(int(label) for label in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(zip((float(score) for score in scores), labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(int(item[1]) for item in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def pr_auc(scores: Sequence[float], labels: Sequence[bool | int]) -> float | None:
    """Calculate average-precision-style PR-AUC for binary query targets."""

    _validate_binary_inputs(scores, labels)
    positives = sum(int(label) for label in labels)
    if positives == 0:
        return None
    ordered = sorted(
        zip((float(score) for score in scores), labels),
        key=lambda item: (-item[0],),
    )
    true_positives = 0
    area = 0.0
    for rank, (_, label) in enumerate(ordered, start=1):
        if int(label):
            true_positives += 1
            area += true_positives / rank
    return area / positives


def _bootstrap_metric(
    scores: Sequence[float],
    labels: Sequence[bool | int],
    metric: Any,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 42,
) -> dict[str, Any]:
    _validate_binary_inputs(scores, labels)
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    point = metric(scores, labels)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(resamples):
        indices = [rng.randrange(len(scores)) for _ in scores]
        sampled_scores = [scores[index] for index in indices]
        sampled_labels = [labels[index] for index in indices]
        value = metric(sampled_scores, sampled_labels)
        if value is not None:
            values.append(float(value))
    return {
        "estimate": point,
        "lower": _quantile(values, 0.025),
        "upper": _quantile(values, 0.975),
        "resamples": resamples,
        "usable_resamples": len(values),
        "seed": seed,
    }


def correctness_targets(
    candidate_ids: Sequence[str], relevant_ids: Sequence[str], top_k: int = TOP_K
) -> dict[str, Any]:
    """Return deterministic top-1/top-5 targets and observed relevant rank."""

    if not candidate_ids:
        raise ValueError("candidate_ids cannot be empty")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate_ids must be unique")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    relevant = {str(item) for item in relevant_ids}
    first_rank = next(
        (rank for rank, item in enumerate(candidate_ids, start=1) if item in relevant),
        None,
    )
    return {
        "top1_correct": bool(candidate_ids[0] in relevant),
        "top5_correct": bool(any(item in relevant for item in candidate_ids[:5])),
        "first_relevant_rank": first_rank,
        "rank_observed": first_rank is not None,
        "rank_lower_bound": first_rank if first_rank is not None else len(candidate_ids) + 1,
        "rank_severity": (
            "success"
            if first_rank == 1
            else "low"
            if first_rank is not None and first_rank <= 5
            else "medium"
            if first_rank is not None and first_rank <= top_k
            else "severe"
        ),
    }


def confidence_proxies(scores: Sequence[float], temperature: float = 1.0) -> dict[str, float]:
    """Return interpretable score-derived signals over the retained candidates.

    The softmax mass and entropy are experimental proxies over the retained
    top-10 list, not probabilities over the full corpus.
    """

    if len(scores) < 2:
        raise ValueError("at least two candidate scores are required")
    values = [float(score) for score in scores]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("candidate scores must be finite")
    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    top1 = values[0]
    top2 = values[1]
    margin = top1 - top2
    shifted = [(value - top1) / temperature for value in values]
    exponentials = [math.exp(max(-700.0, min(0.0, value))) for value in shifted]
    total = math.fsum(exponentials)
    probabilities = [value / total for value in exponentials]
    entropy = -math.fsum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )
    max_entropy = math.log(len(probabilities))
    return {
        "top1_score": top1,
        "top1_top2_margin": margin,
        "softmax_top1_mass": probabilities[0],
        "entropy_confidence": 1.0 - entropy / max_entropy if max_entropy else 1.0,
        "softmax_temperature": temperature,
        "softmax_candidate_scope": len(values),
    }


def _relevant_score_margin(
    candidate_ids: Sequence[str], scores: Sequence[float], relevant_ids: Sequence[str]
) -> tuple[float | None, bool]:
    relevant = {str(item) for item in relevant_ids}
    observed_scores = [float(score) for item, score in zip(candidate_ids, scores) if item in relevant]
    if not observed_scores:
        return None, False
    return float(scores[0]) - max(observed_scores), True


def _record_from_ranking(
    ranking: Mapping[str, Any],
    split: str,
    system: str,
    direction: str,
    context: str | None,
    taxonomy: Mapping[tuple[str, str, str], Any] | None = None,
) -> dict[str, Any]:
    candidate_ids = [str(item) for item in ranking.get("candidate_ids", [])]
    scores = [float(value) for value in ranking.get("scores", [])]
    if len(candidate_ids) != len(scores) or len(candidate_ids) < 2:
        raise ValueError("ranking must contain matching candidate IDs and at least two scores")
    if str(ranking.get("task")) != direction:
        raise ValueError("ranking direction does not match requested direction")
    relevant_ids = [str(item) for item in ranking.get("relevant_ids", [])]
    targets = correctness_targets(candidate_ids, relevant_ids)
    proxies = confidence_proxies(scores[:TOP_K])
    relevant_margin, relevant_observed = _relevant_score_margin(
        candidate_ids, scores, relevant_ids
    )
    key = (system, direction, str(ranking.get("query_id")))
    taxonomy_rows = taxonomy.get(key, []) if taxonomy is not None else []
    return {
        "record_schema_version": PHASE17_SCHEMA_VERSION,
        "split": split,
        "system": system,
        "direction": direction,
        "query_id": str(ranking.get("query_id")),
        "query_context": context,
        "candidate_ids": candidate_ids,
        "scores": scores,
        "relevant_ids": relevant_ids,
        "candidate_count": int(ranking.get("candidate_count", len(candidate_ids))),
        "candidate_corpus_id": str(ranking.get("candidate_corpus_id", "")),
        **targets,
        "top1_failure": not targets["top1_correct"],
        "top5_failure": not targets["top5_correct"],
        "confidence_proxies": proxies,
        "relevant_score_margin": relevant_margin,
        "relevant_score_observed": relevant_observed,
        "taxonomy_categories": taxonomy_rows,
        "taxonomy_label_source": (
            sorted({str(item.get("label_source")) for item in taxonomy_rows if isinstance(item, Mapping)})
            if taxonomy_rows
            else []
        ),
    }


def _standardize(value: float, mean: float, scale: float) -> float:
    return (float(value) - mean) / scale if scale else 0.0


def fit_logistic_calibrator(values: Sequence[float], labels: Sequence[bool | int]) -> dict[str, Any]:
    """Fit a two-parameter logistic calibration transform on validation only."""

    _validate_binary_inputs(values, labels)
    mean = statistics.fmean(float(value) for value in values)
    population_scale = statistics.pstdev(float(value) for value in values)
    scale = population_scale if population_scale > 1e-12 else 1.0
    standardized = [_standardize(float(value), mean, scale) for value in values]
    positives = sum(int(label) for label in labels)
    prior = positives / len(labels)
    if positives == 0 or positives == len(labels):
        return {
            "method": "prior_only",
            "fit_split": "validation",
            "intercept": _logit(prior),
            "slope": 0.0,
            "mean": mean,
            "scale": scale,
            "sample_count": len(labels),
            "positive_count": positives,
            "negative_count": len(labels) - positives,
            "degenerate_class": True,
            "test_labels_used_for_fit": False,
        }
    intercept = _logit(prior)
    slope = 0.0
    for _ in range(100):
        probabilities = [_sigmoid(intercept + slope * value) for value in standardized]
        weights = [probability * (1.0 - probability) for probability in probabilities]
        gradient_intercept = math.fsum(
            probability - int(label) for probability, label in zip(probabilities, labels)
        )
        gradient_slope = math.fsum(
            (probability - int(label)) * value
            for probability, label, value in zip(probabilities, labels, standardized)
        ) + 1e-3 * slope
        h00 = math.fsum(weights) + 1e-6
        h01 = math.fsum(weight * value for weight, value in zip(weights, standardized))
        h11 = math.fsum(weight * value * value for weight, value in zip(weights, standardized)) + 1e-3
        determinant = h00 * h11 - h01 * h01
        if abs(determinant) < 1e-12:
            break
        delta_intercept = (h11 * gradient_intercept - h01 * gradient_slope) / determinant
        delta_slope = (-h01 * gradient_intercept + h00 * gradient_slope) / determinant
        intercept -= delta_intercept
        slope -= delta_slope
        if abs(delta_intercept) + abs(delta_slope) < 1e-8:
            break
    return {
        "method": "logistic_platt",
        "fit_split": "validation",
        "intercept": intercept,
        "slope": slope,
        "mean": mean,
        "scale": scale,
        "sample_count": len(labels),
        "positive_count": positives,
        "negative_count": len(labels) - positives,
        "degenerate_class": False,
        "test_labels_used_for_fit": False,
    }


def apply_logistic_calibrator(parameters: Mapping[str, Any], value: float) -> float:
    if parameters.get("fit_split") != "validation":
        raise ValueError("calibration parameters must be fitted on validation")
    standardized = _standardize(
        float(value), float(parameters["mean"]), float(parameters["scale"])
    )
    return _sigmoid(float(parameters["intercept"]) + float(parameters["slope"]) * standardized)


def reliability_bins(
    confidences: Sequence[float], labels: Sequence[bool | int], bins: int = RELIABILITY_BIN_COUNT
) -> list[dict[str, Any]]:
    """Return non-empty reliability bins with confidence and empirical accuracy."""

    _validate_binary_inputs(confidences, labels)
    if bins <= 0:
        raise ValueError("bins must be positive")
    grouped: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for confidence, label in zip(confidences, labels):
        bounded = min(max(float(confidence), 0.0), 1.0)
        index = min(int(bounded * bins), bins - 1)
        grouped[index].append((bounded, int(label)))
    rows: list[dict[str, Any]] = []
    for index, values in enumerate(grouped):
        lower = index / bins
        upper = (index + 1) / bins
        if not values:
            continue
        rows.append(
            {
                "bin": index,
                "lower": lower,
                "upper": upper,
                "count": len(values),
                "mean_confidence": statistics.fmean(item[0] for item in values),
                "empirical_accuracy": statistics.fmean(item[1] for item in values),
                "absolute_gap": abs(
                    statistics.fmean(item[0] for item in values)
                    - statistics.fmean(item[1] for item in values)
                ),
            }
        )
    return rows


def expected_calibration_error(
    bins: Sequence[Mapping[str, Any]], total_count: int
) -> float:
    if total_count <= 0:
        raise ValueError("total_count must be positive")
    return math.fsum(
        (int(row["count"]) / total_count) * abs(float(row["absolute_gap"]))
        for row in bins
    )


def selective_metrics(
    confidences: Sequence[float], labels: Sequence[bool | int], threshold: float
) -> dict[str, Any]:
    _validate_binary_inputs(confidences, labels)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    accepted = [index for index, confidence in enumerate(confidences) if confidence >= threshold]
    correct = sum(int(labels[index]) for index in accepted)
    count = len(confidences)
    coverage = len(accepted) / count
    accuracy = correct / len(accepted) if accepted else None
    return {
        "threshold": threshold,
        "total_queries": count,
        "accepted_queries": len(accepted),
        "abstained_queries": count - len(accepted),
        "coverage": coverage,
        "selective_top1_accuracy": accuracy,
        "risk": 1.0 - accuracy if accuracy is not None else None,
    }


def area_under_risk_coverage(
    confidences: Sequence[float], labels: Sequence[bool | int]
) -> dict[str, Any]:
    """Calculate rectangle-rule AURC after descending-confidence acceptance."""

    _validate_binary_inputs(confidences, labels)
    ordered = sorted(
        zip((float(value) for value in confidences), labels),
        key=lambda item: -item[0],
    )
    incorrect = 0
    points: list[dict[str, Any]] = []
    for index, (_, label) in enumerate(ordered, start=1):
        incorrect += int(not bool(label))
        coverage = index / len(ordered)
        risk = incorrect / index
        points.append({"accepted_queries": index, "coverage": coverage, "risk": risk})
    aurc = math.fsum(float(point["risk"]) for point in points) / len(points)
    return {"aurc": aurc, "procedure": "mean risk over descending-confidence prefixes", "points": points}


def _distribution_overlap(left: Sequence[float], right: Sequence[float], bins: int = 10) -> float | None:
    if not left or not right:
        return None
    lower = min(min(left), min(right))
    upper = max(max(left), max(right))
    if lower == upper:
        return 1.0
    width = (upper - lower) / bins
    left_counts = [0] * bins
    right_counts = [0] * bins
    for value in left:
        left_counts[min(int((value - lower) / width), bins - 1)] += 1
    for value in right:
        right_counts[min(int((value - lower) / width), bins - 1)] += 1
    left_total = len(left)
    right_total = len(right)
    return sum(min(left_counts[index] / left_total, right_counts[index] / right_total) for index in range(bins))


def _phase16_taxonomy(path: Path) -> dict[tuple[str, str, str], Any]:
    rows = _read_json(path / "failure_records.json").get("rows", [])
    output: dict[tuple[str, str, str], Any] = {}
    for row in rows:
        key = (str(row["system"]), str(row["direction"]), str(row["query_id"]))
        categories = row.get("taxonomy_categories", [])
        if any(
            str(category.get("label_source")) not in EXPLICIT_TAXONOMY_SOURCES
            for category in categories
            if isinstance(category, Mapping)
        ):
            raise ValueError("Phase 16 taxonomy label source is not explicit")
        output[key] = categories
    return output


def audit_phase16(phase16_dir: Path | str, phase7_dir: Path | str) -> dict[str, Any]:
    """Audit only the direct Phase 16 dependencies required here."""

    phase16 = Path(phase16_dir)
    phase7 = Path(phase7_dir)
    report = _read_json(phase16 / "phase16_report.json")
    validation = _read_json(phase16 / "artifact_validation.json")
    failure_records = _read_json(phase16 / "failure_records.json")
    provenance = _read_json(phase16 / "provenance.json")
    checks: dict[str, bool] = {
        "phase16_report_pass": report.get("status") == "PASS",
        "phase16_artifact_validation_pass": validation.get("passed") is True,
        "failure_records_readable": isinstance(failure_records.get("rows"), list)
        and bool(failure_records.get("rows")),
        "directions_separated": {
            str(row.get("direction")) for row in failure_records.get("rows", [])
        }
        == set(DIRECTIONS),
        "systems_present": {
            str(row.get("system")) for row in failure_records.get("rows", [])
        }
        == set(SYSTEMS),
        "semantic_labels_explicit": all(
            str(category.get("label_source")) in EXPLICIT_TAXONOMY_SOURCES
            for row in failure_records.get("rows", [])
            for category in row.get("taxonomy_categories", [])
            if isinstance(category, Mapping)
        ),
        "phase16_no_phase17": provenance.get("phase17_started") is False,
    }
    ranking_paths = {
        "zero_shot_text_to_image": phase7 / "zero_shot_text_to_image.json",
        "zero_shot_image_to_text": phase7 / "zero_shot_image_to_text.json",
        "full_ft_text_to_image": phase7 / "fine_tuned_text_to_image.json",
        "full_ft_image_to_text": phase7 / "fine_tuned_image_to_text.json",
    }
    checks["rankings_available"] = all(path.exists() for path in ranking_paths.values())
    ranking_payloads: dict[str, Any] = {}
    for name, path in ranking_paths.items():
        if path.exists():
            payload = _read_json(path)
            ranking_payloads[name] = payload
            checks[f"{name}_is_test"] = payload.get("split") == "test"
            checks[f"{name}_has_scores"] = bool(payload.get("ranking_records")) and all(
                len(row.get("candidate_ids", [])) == len(row.get("scores", []))
                for row in payload.get("ranking_records", [])
            )
    if checks["rankings_available"]:
        checks["zero_full_query_alignment"] = all(
            [
                [row["query_id"] for row in ranking_payloads["zero_shot_text_to_image"]["ranking_records"]]
                == [row["query_id"] for row in ranking_payloads["full_ft_text_to_image"]["ranking_records"]],
                [row["query_id"] for row in ranking_payloads["zero_shot_image_to_text"]["ranking_records"]]
                == [row["query_id"] for row in ranking_payloads["full_ft_image_to_text"]["ranking_records"]],
            ]
        )
    else:
        checks["zero_full_query_alignment"] = False
    passed = all(checks.values())
    return {
        "schema_version": PHASE17_SCHEMA_VERSION,
        "phase": 17,
        "audit_scope": "direct Phase 16 dependencies only",
        "status": "PASS" if passed else "FAIL",
        "audit_result": "PRE-PHASE AUDIT: Phase 16 PASS" if passed else "PRE-PHASE AUDIT: Phase 16 BLOCKED",
        "checks": checks,
        "phase16_failure_record_count": len(failure_records.get("rows", [])),
        "phase16_report": str(phase16 / "phase16_report.json"),
        "phase7_ranking_inputs": {name: str(path) for name, path in ranking_paths.items()},
        "recorded_before_phase17_analysis": True,
    }


def _validation_records(
    manifest_path: Path,
    split: str = "validation",
    limit: int = 100,
    seed: int = 42,
) -> tuple[ImageRecord, ...]:
    from .phase7 import _subset_records

    manifest = read_manifest(manifest_path)
    return _subset_records(manifest.records, split, seed, limit)


def _load_inference_rankings(
    manifest_path: Path,
    image_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    model_id: str,
    device: str,
    batch_size: int,
    split: str,
    records: Sequence[ImageRecord],
    full_ft: bool,
) -> dict[str, list[dict[str, Any]]]:
    """Run inference only; no optimizer, gradients, or training are created."""

    from .phase7 import _load_checkpoint, _load_trainable_model, evaluate_model

    model, processor, torch, _ = _load_trainable_model(model_id, device)
    if full_ft:
        _load_checkpoint(checkpoint_path, model)
    model.eval()
    try:
        result = evaluate_model(
            model,
            processor,
            torch,
            records,
            image_root,
            batch_size,
            77,
            read_manifest(manifest_path),
            manifest_path,
            config_path,
            split,
            42,
            "phase17_full_ft" if full_ft else "phase17_zero_shot",
            "phase17_full_ft_validation" if full_ft else "phase17_zero_shot_validation",
            {"frozen": not full_ft, "inference_only": True},
            1,
            0,
            "fp32",
        )
        return {
            direction: [record.to_dict() for record in result["rankings"][direction]]
            for direction in DIRECTIONS
        }
    finally:
        del model
        gc.collect()
        try:
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except AttributeError:
            pass


def _context_maps(manifest_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    manifest = read_manifest(manifest_path)
    caption_context = {
        caption.caption_id: caption.text
        for record in manifest.records
        for caption in record.captions
    }
    image_context = {
        record.image_id: record.captions[0].text if record.captions else ""
        for record in manifest.records
    }
    return caption_context, image_context


def _build_split_records(
    rankings_by_system: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    split: str,
    contexts: tuple[Mapping[str, str], Mapping[str, str]],
    taxonomy: Mapping[tuple[str, str, str], Any] | None = None,
) -> list[dict[str, Any]]:
    caption_context, image_context = contexts
    rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        for direction in DIRECTIONS:
            query_context = caption_context if direction == "text_to_image" else image_context
            for ranking in rankings_by_system[system][direction]:
                query_id = str(ranking.get("query_id"))
                rows.append(
                    _record_from_ranking(
                        ranking,
                        split,
                        system,
                        direction,
                        query_context.get(query_id),
                        taxonomy,
                    )
                )
    expected = 2 * sum(len(rankings_by_system[SYSTEMS[0]][direction]) for direction in DIRECTIONS)
    if len(rows) != expected:
        raise AssertionError("confidence record count is not aligned across systems")
    return rows


def _select_proxy(validation_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = [bool(row["top1_correct"]) for row in validation_rows]
    metrics: dict[str, Any] = {}
    for proxy in RAW_PROXIES:
        values = [float(row["confidence_proxies"][proxy]) for row in validation_rows]
        metrics[proxy] = {
            "roc_auc": roc_auc(values, labels),
            "pr_auc": pr_auc(values, labels),
            "sample_count": len(values),
            "positive_count": sum(labels),
            "negative_count": len(labels) - sum(labels),
        }
    usable = [proxy for proxy in RAW_PROXIES if metrics[proxy]["roc_auc"] is not None]
    selected = max(
        usable,
        key=lambda proxy: (float(metrics[proxy]["roc_auc"]), -RAW_PROXIES.index(proxy)),
    ) if usable else RAW_PROXIES[0]
    return {
        "selected_proxy": selected,
        "selection_split": "validation",
        "selection_metric": "roc_auc",
        "candidate_proxy_metrics": metrics,
        "test_labels_used_for_selection": False,
    }


def _add_calibrated_confidence(
    rows: Sequence[Mapping[str, Any]],
    selected_proxy: str,
    parameters: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        raw = float(row["confidence_proxies"][selected_proxy])
        copied["selected_confidence_proxy"] = selected_proxy
        copied["selected_raw_confidence_proxy"] = raw
        copied["calibrated_confidence"] = apply_logistic_calibrator(parameters, raw)
        copied["confidence_interpretation"] = "bounded calibrated confidence estimate, not a guaranteed probability"
        output.append(copied)
    return output


def _fit_all_calibrators(
    validation_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in validation_rows:
        grouped[(str(row["system"]), str(row["direction"]))].append(row)
    parameters: dict[tuple[str, str], dict[str, Any]] = {}
    summary: list[dict[str, Any]] = []
    for system in SYSTEMS:
        for direction in DIRECTIONS:
            rows = grouped[(system, direction)]
            selection = _select_proxy(rows)
            selected = str(selection["selected_proxy"])
            fit = fit_logistic_calibrator(
                [float(row["confidence_proxies"][selected]) for row in rows],
                [bool(row["top1_correct"]) for row in rows],
            )
            combined = {
                "schema_version": PHASE17_SCHEMA_VERSION,
                "system": system,
                "direction": direction,
                **selection,
                **fit,
                "test_labels_used_for_fit": False,
            }
            parameters[(system, direction)] = combined
            summary.append(combined)
    return parameters, summary


def _test_discrimination(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for system in SYSTEMS:
        for direction in DIRECTIONS:
            subset = [
                row for row in rows if row["system"] == system and row["direction"] == direction
            ]
            labels = [bool(row["top1_correct"]) for row in subset]
            key = f"{system}:{direction}"
            metrics: dict[str, Any] = {}
            for proxy in RAW_PROXIES:
                scores = [float(row["confidence_proxies"][proxy]) for row in subset]
                metrics[proxy] = {
                    "roc_auc": _bootstrap_metric(
                    scores, labels, roc_auc, seed=_stable_seed(key, proxy, "roc")
                    ),
                    "pr_auc": _bootstrap_metric(
                    scores, labels, pr_auc, seed=_stable_seed(key, proxy, "pr")
                    ),
                }
            selected_scores = [float(row["calibrated_confidence"]) for row in subset]
            metrics["selected_calibrated_confidence"] = {
                "roc_auc": _bootstrap_metric(
                    selected_scores, labels, roc_auc, seed=_stable_seed(key, "calibrated", "roc")
                ),
                "pr_auc": _bootstrap_metric(
                    selected_scores, labels, pr_auc, seed=_stable_seed(key, "calibrated", "pr")
                ),
            }
            output[key] = {
                "system": system,
                "direction": direction,
                "query_count": len(subset),
                "positive_count": sum(labels),
                "negative_count": len(labels) - sum(labels),
                "metrics": metrics,
                "discrimination_is_not_calibration": True,
            }
    return output


def _calibration_outputs(
    rows: Sequence[Mapping[str, Any]],
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reliability: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for system in SYSTEMS:
        for direction in DIRECTIONS:
            subset = [
                row for row in rows
                if row["system"] == system and row["direction"] == direction and row["split"] == split
            ]
            confidences = [float(row["calibrated_confidence"]) for row in subset]
            labels = [bool(row["top1_correct"]) for row in subset]
            bins = reliability_bins(confidences, labels)
            reliability.append(
                {
                    "system": system,
                    "direction": direction,
                    "split": split,
                    "bin_count": RELIABILITY_BIN_COUNT,
                    "empty_bins": [
                        index for index in range(RELIABILITY_BIN_COUNT)
                        if index not in {int(row["bin"]) for row in bins}
                    ],
                    "bins": bins,
                }
            )
            ece = expected_calibration_error(bins, len(subset))
            brier = statistics.fmean(
                (confidence - int(label)) ** 2
                for confidence, label in zip(confidences, labels)
            )
            metrics.append(
                {
                    "system": system,
                    "direction": direction,
                    "split": split,
                    "query_count": len(subset),
                    "mean_confidence": statistics.fmean(confidences),
                    "empirical_top1_accuracy": statistics.fmean(int(label) for label in labels),
                    "ece": ece,
                    "brier_score": brier,
                    "confidence_is_bounded_transformation": True,
                    "test_labels_used_for_fit": False,
                    "interpretation": "validation metrics are apparent fit-split diagnostics; test metrics are final held-out evaluation",
                }
            )
    return reliability, metrics


def _threshold_for_coverage(confidences: Sequence[float], target: float) -> float:
    if not confidences:
        raise ValueError("cannot select a threshold from empty confidence values")
    if not 0.0 < target <= 1.0:
        raise ValueError("target coverage must be in (0, 1]")
    ordered = sorted(float(value) for value in confidences)
    index = max(0, min(len(ordered) - 1, math.ceil(target * len(ordered)) - 1))
    return ordered[index]


def _selective_outputs(
    validation_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    recommendations: list[dict[str, Any]] = []
    for system in SYSTEMS:
        for direction in DIRECTIONS:
            validation = [
                row for row in validation_rows
                if row["system"] == system and row["direction"] == direction
            ]
            test = [
                row for row in test_rows
                if row["system"] == system and row["direction"] == direction
            ]
            validation_confidence = [float(row["calibrated_confidence"]) for row in validation]
            validation_labels = [bool(row["top1_correct"]) for row in validation]
            test_confidence = [float(row["calibrated_confidence"]) for row in test]
            test_labels = [bool(row["top1_correct"]) for row in test]
            validation_baseline = statistics.fmean(int(label) for label in validation_labels)
            test_baseline = statistics.fmean(int(label) for label in test_labels)
            candidates: list[dict[str, Any]] = []
            for target in TARGET_COVERAGES:
                threshold = _threshold_for_coverage(validation_confidence, target)
                validation_metric = selective_metrics(validation_confidence, validation_labels, threshold)
                test_metric = selective_metrics(test_confidence, test_labels, threshold)
                row = {
                    "system": system,
                    "direction": direction,
                    "target_coverage": target,
                    "threshold_source": "validation_only",
                    "threshold": threshold,
                    "validation": validation_metric,
                    "test": test_metric,
                }
                rows.append(row)
                if target < 1.0 and validation_metric["coverage"] >= 0.5 and validation_metric["selective_top1_accuracy"] is not None and validation_metric["selective_top1_accuracy"] >= validation_baseline + 0.02:
                    candidates.append(row)
            selected = min(
                candidates,
                key=lambda item: (
                    float(item["validation"]["coverage"]),
                    -float(item["threshold"]),
                ),
                default=None,
            )
            test_selected = selected["test"] if selected is not None else None
            supported = bool(
                selected is not None
                and test_selected is not None
                and test_selected["selective_top1_accuracy"] is not None
                and float(test_selected["selective_top1_accuracy"]) >= test_baseline
                and float(test_selected["risk"]) < 1.0 - test_baseline
            )
            recommendations.append(
                {
                    "system": system,
                    "direction": direction,
                    "status": "supported_on_held_out_test" if supported else "no_supported_threshold",
                    "criterion": "validation selective accuracy at least baseline plus 0.02, coverage at least 0.50; choose lowest validation coverage; confirm test risk is lower",
                    "validation_baseline_accuracy": validation_baseline,
                    "test_baseline_accuracy": test_baseline,
                    "selected_target_coverage": selected["target_coverage"] if selected else None,
                    "selected_threshold": selected["threshold"] if selected else None,
                    "validation_result": selected["validation"] if selected else None,
                    "test_result": test_selected,
                    "test_labels_used_for_threshold_selection": False,
                }
            )
    return {"schema_version": PHASE17_SCHEMA_VERSION, "rows": rows}, {
        "schema_version": PHASE17_SCHEMA_VERSION,
        "recommendations": recommendations,
    }


def _risk_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    aurc_rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        for direction in DIRECTIONS:
            subset = [
                row for row in rows
                if row["system"] == system and row["direction"] == direction
            ]
            confidence = [float(row["calibrated_confidence"]) for row in subset]
            labels = [bool(row["top1_correct"]) for row in subset]
            aurc = area_under_risk_coverage(confidence, labels)
            output.append({"system": system, "direction": direction, **aurc})
            aurc_rows.append({"system": system, "direction": direction, "aurc": aurc["aurc"]})
    return {"schema_version": PHASE17_SCHEMA_VERSION, "rows": output, "summary": aurc_rows}


def _success_failure_distributions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    groups = ("top1_correct", "top1_failure", "success", "low", "medium", "severe")
    for system in SYSTEMS:
        for direction in DIRECTIONS:
            subset = [
                row for row in rows
                if row["system"] == system and row["direction"] == direction
            ]
            for group in groups:
                if group == "top1_correct":
                    selected = [row for row in subset if row["top1_correct"]]
                elif group == "top1_failure":
                    selected = [row for row in subset if not row["top1_correct"]]
                else:
                    selected = [row for row in subset if row["rank_severity"] == group]
                output.append(
                    {
                        "system": system,
                        "direction": direction,
                        "group": group,
                        "query_count": len(selected),
                        "proxies": {
                            proxy: _summary(
                                [float(row["confidence_proxies"][proxy]) for row in selected]
                            )
                            for proxy in RAW_PROXIES
                        },
                        "selected_calibrated_confidence": _summary(
                            [float(row["calibrated_confidence"]) for row in selected]
                        ),
                    }
                )
            correct = [float(row["calibrated_confidence"]) for row in subset if row["top1_correct"]]
            failed = [float(row["calibrated_confidence"]) for row in subset if not row["top1_correct"]]
            output.append(
                {
                    "system": system,
                    "direction": direction,
                    "group": "success_failure_overlap",
                    "query_count": len(subset),
                    "selected_confidence_overlap_coefficient": _distribution_overlap(correct, failed),
                    "success_mean_minus_failure_mean": (
                        statistics.fmean(correct) - statistics.fmean(failed)
                        if correct and failed else None
                    ),
                }
            )
    return {"schema_version": PHASE17_SCHEMA_VERSION, "rows": output}


def _taxonomy_confidence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row["top1_failure"]:
            continue
        categories = row.get("taxonomy_categories", []) or [{"category": "other_unclassified", "label_source": "mechanical_context"}]
        for category in categories:
            if isinstance(category, Mapping):
                grouped[(str(row["system"]), str(row["direction"]), str(category.get("category")))].append(row)
    output = []
    for (system, direction, category), selected in sorted(grouped.items()):
        output.append(
            {
                "system": system,
                "direction": direction,
                "category": category,
                "failure_count": len(selected),
                "label_source": "heuristic" if category != "other_unclassified" else "mechanical_context",
                "selected_calibrated_confidence": _summary(
                    [float(row["calibrated_confidence"]) for row in selected]
                ),
                "high_confidence_error_count": sum(
                    float(row["calibrated_confidence"]) >= HIGH_CONFIDENCE_ERROR_THRESHOLD
                    for row in selected
                ),
                "semantic_ground_truth": False,
            }
        )
    return {"schema_version": PHASE17_SCHEMA_VERSION, "rows": output}


def _high_confidence_errors(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    for system in SYSTEMS:
        for direction in DIRECTIONS:
            selected = sorted(
                [
                    row for row in rows
                    if row["system"] == system
                    and row["direction"] == direction
                    and row["split"] == "test"
                    and row["top1_failure"]
                    and float(row["calibrated_confidence"]) >= HIGH_CONFIDENCE_ERROR_THRESHOLD
                ],
                key=lambda row: (-float(row["calibrated_confidence"]), str(row["query_id"])),
            )
            output.extend(
                {
                    "system": row["system"],
                    "direction": row["direction"],
                    "query_id": row["query_id"],
                    "query_context": row["query_context"],
                    "calibrated_confidence": row["calibrated_confidence"],
                    "selected_proxy": row["selected_confidence_proxy"],
                    "top1_score": row["confidence_proxies"]["top1_score"],
                    "top1_correct": row["top1_correct"],
                    "first_relevant_rank": row["first_relevant_rank"],
                    "rank_severity": row["rank_severity"],
                    "taxonomy_categories": row.get("taxonomy_categories", []),
                }
                for row in selected[:5]
            )
    return {
        "schema_version": PHASE17_SCHEMA_VERSION,
        "threshold": HIGH_CONFIDENCE_ERROR_THRESHOLD,
        "rows": output,
        "definition": "wrong top-1 retrieval with calibrated confidence at or above the fixed 0.80 diagnostic threshold",
    }


def _qualitative_examples(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    test_rows = [row for row in rows if row["split"] == "test"]
    examples: list[dict[str, Any]] = []

    def add(example_type: str, row: Mapping[str, Any]) -> None:
        examples.append(
            {
                "example_type": example_type,
                "system": row["system"],
                "direction": row["direction"],
                "query_id": row["query_id"],
                "query_context": row["query_context"],
                "calibrated_confidence": row["calibrated_confidence"],
                "selected_proxy": row["selected_confidence_proxy"],
                "top1_score": row["confidence_proxies"]["top1_score"],
                "top1_top2_margin": row["confidence_proxies"]["top1_top2_margin"],
                "top1_correct": row["top1_correct"],
                "first_relevant_rank": row["first_relevant_rank"],
                "rank_severity": row["rank_severity"],
                "top_candidates": row["candidate_ids"][:5],
            }
        )

    def sorted_rows(predicate: Any, reverse: bool = False) -> list[Mapping[str, Any]]:
        return sorted(
            [row for row in test_rows if predicate(row)],
            key=lambda row: (float(row["calibrated_confidence"]), str(row["query_id"])),
            reverse=reverse,
        )

    for label, predicate, reverse in (
        ("high_confidence_correct", lambda row: bool(row["top1_correct"]), True),
        ("low_confidence_correct", lambda row: bool(row["top1_correct"]), False),
        ("low_confidence_error", lambda row: bool(row["top1_failure"]), False),
        ("high_confidence_error", lambda row: bool(row["top1_failure"]), True),
    ):
        for index, row in enumerate(sorted_rows(predicate, reverse)[:2], start=1):
            add(f"{label}_{index}", row)

    comparisons: list[tuple[float, Mapping[str, Any], Mapping[str, Any]]] = []
    for direction in DIRECTIONS:
        zero = {row["query_id"]: row for row in test_rows if row["system"] == "zero_shot" and row["direction"] == direction}
        full = {row["query_id"]: row for row in test_rows if row["system"] == "full_ft" and row["direction"] == direction}
        for query_id in sorted(set(zero) & set(full)):
            comparisons.append((float(full[query_id]["calibrated_confidence"]) - float(zero[query_id]["calibrated_confidence"]), zero[query_id], full[query_id]))
    if comparisons:
        improvement = max(comparisons, key=lambda item: (item[0], str(item[2]["query_id"])))
        regression = min(comparisons, key=lambda item: (item[0], str(item[2]["query_id"])))
        add("fine_tuning_confidence_improvement", improvement[2])
        add("fine_tuning_confidence_regression", regression[2])
    return {
        "schema_version": PHASE17_SCHEMA_VERSION,
        "count": len(examples),
        "selection": "deterministic sorted confidence/query-id rules; no favorable-only filtering",
        "examples": examples[:12],
    }


def _robustness_confidence(phase15_dir: Path) -> dict[str, Any]:
    metrics = _read_json(phase15_dir / "robustness_metrics.json")
    qualitative = _read_json(phase15_dir / "qualitative_examples.json")
    aligned_score_fields = any(
        "scores" in example or "top1_score" in example
        for row in qualitative.get("rows", [])
        for example in row.get("examples", [])
    )
    selected = [
        row for row in metrics.get("rows", [])
        if row.get("family") in {"shortened", "occlusion"}
        and row.get("severity") == "high"
    ]
    return {
        "schema_version": PHASE17_SCHEMA_VERSION,
        "source": [str(phase15_dir / "robustness_metrics.json"), str(phase15_dir / "qualitative_examples.json")],
        "aligned_scores_available": aligned_score_fields,
        "confidence_claim": "not evaluated: Phase 15 retained ranking metrics/examples but not aligned clean/corrupted score arrays",
        "focus_rows": selected,
        "no_unsupported_robustness_confidence_claim": not aligned_score_fields,
    }


def validate_phase17_artifacts(output_dir: Path | str) -> dict[str, Any]:
    output = Path(output_dir)
    checks: dict[str, bool] = {
        "required_artifacts": all((output / name).exists() for name in REQUIRED_ARTIFACTS),
    }
    if not checks["required_artifacts"]:
        return {"schema_version": PHASE17_SCHEMA_VERSION, "passed": False, "checks": checks, "required": list(REQUIRED_ARTIFACTS)}
    definition = _read_json(output / "confidence_definition.json")
    records = _read_json(output / "confidence_records.json")
    parameters = _read_json(output / "calibration_parameters.json")
    selective = _read_json(output / "selective_retrieval.json")
    risk = _read_json(output / "risk_coverage.json")
    robustness = _read_json(output / "robustness_confidence.json")
    provenance = _read_json(output / "provenance.json")
    rows = records.get("rows", [])
    checks.update(
        {
            "confidence_records_nonempty": bool(rows),
            "systems_present": {row.get("system") for row in rows} == set(SYSTEMS),
            "directions_separated": {row.get("direction") for row in rows} == set(DIRECTIONS),
            "splits_present": {row.get("split") for row in rows} == {"validation", "test"},
            "raw_scores_not_probabilities": definition.get("raw_scores_are_not_probabilities") is True,
            "proxies_explicit": set(definition.get("raw_proxies", [])) >= set(RAW_PROXIES),
            "calibration_validation_only": all(
                item.get("fit_split") == "validation" and item.get("test_labels_used_for_fit") is False
                for item in parameters.get("rows", [])
            ) and len(parameters.get("rows", [])) == 4,
            "selective_records_real": bool(selective.get("rows")) and bool(selective.get("recommendations")),
            "risk_coverage_present": len(risk.get("rows", [])) == 4 and all("aurc" in row for row in risk.get("rows", [])),
            "robustness_limit_explicit": robustness.get("no_unsupported_robustness_confidence_claim") is True,
            "no_training": provenance.get("training_performed") is False,
            "no_phase18": provenance.get("phase18_started") is False,
            "test_labels_not_used_for_threshold": all(
                row.get("test_labels_used_for_threshold_selection") is False
                for row in selective.get("recommendations", [])
            ),
        }
    )
    checks["confidence_bounds"] = all(
        0.0 <= float(row.get("calibrated_confidence", -1.0)) <= 1.0 for row in rows
    )
    passed = all(checks.values())
    return {
        "schema_version": PHASE17_SCHEMA_VERSION,
        "passed": passed,
        "checks": checks,
        "required": list(REQUIRED_ARTIFACTS),
    }


def run_phase17(
    output_dir: Path | str = "artifacts/phase17",
    manifest_path: Path | str = "data/processed/coco2017_val_split_manifest.json",
    image_root: Path | str = "data/raw/coco2017/val2017",
    config_path: Path | str = "configs/default.toml",
    phase7_dir: Path | str = "artifacts/phase7",
    phase15_dir: Path | str = "artifacts/phase15",
    phase16_dir: Path | str = "artifacts/phase16",
    model_id: str = "openai/clip-vit-base-patch32",
    device: str = "auto",
    batch_size: int = 16,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_file = Path(manifest_path)
    phase7_path = Path(phase7_dir)
    phase15_path = Path(phase15_dir)
    phase16_path = Path(phase16_dir)
    phase16_audit = audit_phase16(phase16_path, phase7_path)
    _write_json(phase16_audit, output / "pre_phase_audit.json")
    if phase16_audit["status"] != "PASS":
        raise ValueError("Phase 16 dependency audit failed")
    definition = {
        "schema_version": PHASE17_SCHEMA_VERSION,
        "phase": 17,
        "target_definitions": {
            "top1_correct": "the first returned candidate is in the existing relevance set",
            "top5_correct": "at least one returned candidate in the first five is in the existing relevance set",
            "primary_target": "top1_correct",
        },
        "raw_proxies": list(RAW_PROXIES),
        "proxy_definitions": {
            "top1_score": "retrieval score of the first returned candidate",
            "top1_top2_margin": "s1 - s2",
            "softmax_top1_mass": "softmax mass of top-1 over retained top-10 scores at temperature 1.0",
            "entropy_confidence": "1 - normalized entropy of that retained top-10 softmax distribution",
            "relevant_score_margin": "top-1 score minus best observed relevant score; oracle diagnostic only",
        },
        "raw_scores_are_not_probabilities": True,
        "score_comparison_scope": "within system and direction; raw scales are not compared across systems without transformation",
        "validation_scope": {"split": "validation", "selected_image_groups": 100, "seed": 42},
        "test_scope": {"split": "test", "source": "retained Phase 7 rankings", "candidate_depth": TOP_K},
        "calibration": {
            "method": "validation-fitted logistic transformation of validation-selected proxy",
            "test_labels_used_for_fit": False,
            "reliability_bins": RELIABILITY_BIN_COUNT,
            "empty_bin_handling": "omit empty bins and record them explicitly",
        },
        "selective_retrieval": {
            "target_coverages": list(TARGET_COVERAGES),
            "threshold_selection": "validation rank threshold among predetermined coverage points",
            "threshold_not_hard_coded_in_production": True,
        },
        "diagnostic_thresholds": {
            "high_confidence_error": HIGH_CONFIDENCE_ERROR_THRESHOLD,
            "low_confidence_correct": LOW_CONFIDENCE_CORRECT_THRESHOLD,
        },
        "phase17_is_evaluation_only": True,
    }
    _write_json(definition, output / "confidence_definition.json")
    contexts = _context_maps(manifest_file)
    taxonomy = _phase16_taxonomy(phase16_path)
    validation_records = _validation_records(manifest_file, "validation", 100, 42)
    validation_rankings: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for system, full_ft in (("zero_shot", False), ("full_ft", True)):
        validation_rankings[system] = _load_inference_rankings(
            manifest_file,
            Path(image_root),
            Path(config_path),
            phase7_path / "best_checkpoint.pt",
            model_id,
            device,
            batch_size,
            "validation",
            validation_records,
            full_ft,
        )
    test_rankings: dict[str, dict[str, list[dict[str, Any]]]] = {}
    test_files = {
        "zero_shot": {
            "text_to_image": phase7_path / "zero_shot_text_to_image.json",
            "image_to_text": phase7_path / "zero_shot_image_to_text.json",
        },
        "full_ft": {
            "text_to_image": phase7_path / "fine_tuned_text_to_image.json",
            "image_to_text": phase7_path / "fine_tuned_image_to_text.json",
        },
    }
    for system in SYSTEMS:
        test_rankings[system] = {
            direction: _read_json(test_files[system][direction])["ranking_records"]
            for direction in DIRECTIONS
        }
    validation_rows = _build_split_records(validation_rankings, "validation", contexts)
    test_rows = _build_split_records(test_rankings, "test", contexts, taxonomy)
    parameters, parameter_rows = _fit_all_calibrators(validation_rows)
    calibrated_validation: list[dict[str, Any]] = []
    calibrated_test: list[dict[str, Any]] = []
    for system in SYSTEMS:
        for direction in DIRECTIONS:
            selected = str(parameters[(system, direction)]["selected_proxy"])
            fit = parameters[(system, direction)]
            calibrated_validation.extend(
                _add_calibrated_confidence(
                    [row for row in validation_rows if row["system"] == system and row["direction"] == direction],
                    selected,
                    fit,
                )
            )
            calibrated_test.extend(
                _add_calibrated_confidence(
                    [row for row in test_rows if row["system"] == system and row["direction"] == direction],
                    selected,
                    fit,
                )
            )
    all_rows = calibrated_validation + calibrated_test
    _write_json({"schema_version": PHASE17_SCHEMA_VERSION, "row_count": len(all_rows), "rows": all_rows}, output / "confidence_records.json")
    _write_json({"schema_version": PHASE17_SCHEMA_VERSION, "rows": parameter_rows}, output / "calibration_parameters.json")
    _write_json(_success_failure_distributions(calibrated_test), output / "success_failure_distributions.json")
    _write_json({"schema_version": PHASE17_SCHEMA_VERSION, "rows": list(_test_discrimination(calibrated_test).values())}, output / "discrimination_metrics.json")
    reliability_validation, calibration_validation = _calibration_outputs(calibrated_validation, "validation")
    reliability_test, calibration_test = _calibration_outputs(calibrated_test, "test")
    _write_json({"schema_version": PHASE17_SCHEMA_VERSION, "rows": reliability_validation + reliability_test}, output / "reliability_bins.json")
    _write_json({"schema_version": PHASE17_SCHEMA_VERSION, "rows": calibration_validation + calibration_test}, output / "calibration_metrics.json")
    selective_rows, recommendations = _selective_outputs(calibrated_validation, calibrated_test)
    _write_json({"schema_version": PHASE17_SCHEMA_VERSION, **selective_rows, **recommendations}, output / "selective_retrieval.json")
    _write_json(_risk_coverage(calibrated_test), output / "risk_coverage.json")
    _write_json(_high_confidence_errors(calibrated_test), output / "high_confidence_errors.json")
    _write_json(_taxonomy_confidence(calibrated_test), output / "taxonomy_confidence.json")
    _write_json(_robustness_confidence(phase15_path), output / "robustness_confidence.json")
    _write_json(_qualitative_examples(calibrated_test), output / "qualitative_examples.json")
    provenance = {
        "schema_version": PHASE17_SCHEMA_VERSION,
        "phase": 17,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "manifest": str(manifest_file),
        "manifest_sha256": _hash_file(manifest_file),
        "phase16_report_sha256": _hash_file(phase16_path / "phase16_report.json"),
        "phase16_failure_records_sha256": _hash_file(phase16_path / "failure_records.json"),
        "phase7_inputs": {name: _hash_file(path) for system in test_files.values() for name, path in system.items()},
        "phase15_inputs": {
            "robustness_metrics": _hash_file(phase15_path / "robustness_metrics.json"),
            "qualitative_examples": _hash_file(phase15_path / "qualitative_examples.json"),
        },
        "validation_inference": {
            "selected_image_groups": len(validation_records),
            "caption_queries": sum(len(record.captions) for record in validation_records),
            "model_id": model_id,
            "device_requested": device,
            "batch_size": batch_size,
            "training_performed": False,
        },
        "calibration_fit_split": "validation",
        "final_evaluation_split": "test",
        "training_performed": False,
        "test_labels_used_for_fit": False,
        "test_labels_used_for_threshold_selection": False,
        "new_dataset_downloaded": False,
        "phase18_started": False,
        "raw_similarity_called_probability": False,
        "python": sys.version,
        "platform": platform.platform(),
    }
    _write_json(provenance, output / "provenance.json")
    _write_json({"phase": 17, "status": "PENDING"}, output / "phase17_report.json")
    validation = validate_phase17_artifacts(output)
    _write_json(validation, output / "artifact_validation.json")
    report = {
        "report_schema_version": PHASE17_SCHEMA_VERSION,
        "project": "OmniSearch",
        "phase": 17,
        "status": "PASS" if validation["passed"] else "FAIL",
        "pre_phase_audit": "Phase 16 PASS",
        "scope": {
            "systems": list(SYSTEMS),
            "directions": list(DIRECTIONS),
            "validation_image_groups": len(validation_records),
            "test_source": "retained Phase 7 test rankings",
            "training": False,
        },
        "quality_gate": {"status": "PASS" if validation["passed"] else "FAIL", "checks": validation["checks"]},
        "artifacts": sorted(path.name for path in output.glob("*.json")),
    }
    _write_json(report, output / "phase17_report.json")
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation-only Phase 17 uncertainty analysis")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase17"))
    parser.add_argument("--manifest", dest="manifest_path", type=Path, default=Path("data/processed/coco2017_val_split_manifest.json"))
    parser.add_argument("--image-root", type=Path, default=Path("data/raw/coco2017/val2017"))
    parser.add_argument("--config", dest="config_path", type=Path, default=Path("configs/default.toml"))
    parser.add_argument("--phase7-dir", type=Path, default=Path("artifacts/phase7"))
    parser.add_argument("--phase15-dir", type=Path, default=Path("artifacts/phase15"))
    parser.add_argument("--phase16-dir", type=Path, default=Path("artifacts/phase16"))
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    report = run_phase17(**vars(args))
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
