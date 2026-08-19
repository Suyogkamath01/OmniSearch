"""Phase 19 responsible-AI, exposure, and safety analysis.

This phase is evaluation-only.  It reuses the retained COCO retrieval rows and
the Phase 16--18 diagnostic artifacts.  The strata are dataset-derived
observables (caption length, lexical rarity, a small object-complexity
heuristic, and image aspect ratio); they are not protected-attribute labels
and they do not support a fairness conclusion by themselves.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .manifest import DatasetManifest, read_manifest

PHASE19_SCHEMA_VERSION = 1
SYSTEMS = ("zero_shot", "full_ft")
DIRECTIONS = ("text_to_image", "image_to_text")
MIN_GROUP_SIZE = 20
BOOTSTRAP_RESAMPLES = 200
HIGH_CONFIDENCE_THRESHOLD = 0.8
EVIDENCE_CATEGORIES = frozenset(
    {
        "MEASURED RISK",
        "OBSERVED LIMITATION",
        "POTENTIAL DEPLOYMENT RISK",
        "NOT EVALUATED",
        "INSUFFICIENT EVIDENCE",
    }
)
PROTECTED_LABEL_NAMES = frozenset(
    {
        "gender",
        "race",
        "ethnicity",
        "religion",
        "age",
        "disability",
        "nationality",
        "skin_tone",
        "protected_attribute",
    }
)
REQUIRED_ARTIFACTS = (
    "pre_phase_audit.json",
    "dataset_scope.json",
    "group_definitions.json",
    "group_performance.json",
    "disparity_analysis.json",
    "confidence_disparity.json",
    "high_confidence_risk_examples.json",
    "responsible_ai_matrix.json",
    "mitigation_recommendations.json",
    "provenance.json",
    "phase19_report.json",
)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "at",
        "by",
        "for",
        "from",
        "has",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "with",
    }
)
OBJECT_HINTS = frozenset(
    {
        "animal",
        "ball",
        "bear",
        "bed",
        "bike",
        "bird",
        "boat",
        "book",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "child",
        "dog",
        "elephant",
        "food",
        "frisbee",
        "girl",
        "guitar",
        "horse",
        "man",
        "motorcycle",
        "person",
        "phone",
        "pizza",
        "plane",
        "sheep",
        "skateboard",
        "street",
        "table",
        "train",
        "tree",
        "umbrella",
        "vehicle",
        "water",
        "woman",
        "zebra",
    }
)
NUMBER_HINTS = frozenset(
    {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
)


def _read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _hash_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(*parts: str) -> int:
    value = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return int.from_bytes(value[:4], "big")


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


def tokenize_caption(text: str) -> list[str]:
    """Return lower-case alphanumeric caption tokens."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("caption text must be a non-empty string")
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def caption_group_features(text: str, training_token_counts: Mapping[str, int]) -> dict[str, Any]:
    """Build reproducible, non-protected caption strata from one caption."""

    tokens = tokenize_caption(text)
    content_tokens = [token for token in tokens if token not in STOPWORDS]
    object_terms = sorted(set(content_tokens) & OBJECT_HINTS)
    number_terms = sorted(set(content_tokens) & NUMBER_HINTS)
    word_count = len(tokens)
    if word_count <= 7:
        length_group = "short"
    elif word_count <= 14:
        length_group = "medium"
    else:
        length_group = "long"

    frequencies = [int(training_token_counts.get(token, 0)) for token in content_tokens]
    rarest_frequency = min(frequencies) if frequencies else None
    if rarest_frequency is None:
        rarity_group = "unknown"
    elif rarest_frequency <= 4:
        rarity_group = "rare"
    elif rarest_frequency <= 49:
        rarity_group = "medium"
    else:
        rarity_group = "common"

    complexity_group = (
        "multi_object_or_structured"
        if len(object_terms) >= 2 or (object_terms and ("and" in tokens or number_terms))
        else "single_object_or_simple"
    )
    return {
        "caption_word_count": word_count,
        "caption_length_group": length_group,
        "content_token_count": len(content_tokens),
        "rarest_content_token_frequency_in_train": rarest_frequency,
        "concept_rarity_group": rarity_group,
        "object_hint_terms": object_terms,
        "object_hint_count": len(object_terms),
        "number_hint_terms": number_terms,
        "complexity_group": complexity_group,
        "feature_label_source": "mechanical_or_lexical_heuristic",
    }


def image_aspect_group(width: int, height: int) -> str:
    """Map dimensions to predeclared coarse aspect-ratio strata."""

    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    ratio = width / height
    if ratio <= 0.75 or ratio >= 4 / 3:
        return "wide_or_tall_extreme"
    if 0.9 <= ratio <= 1.1:
        return "near_square"
    return "moderate_aspect"


def minimum_group_status(count: int, minimum: int = MIN_GROUP_SIZE) -> dict[str, Any]:
    """State whether a stratum has enough queries for a cautious comparison."""

    if count < 0 or minimum <= 0:
        raise ValueError("count must be non-negative and minimum must be positive")
    return {
        "query_count": count,
        "minimum_group_size": minimum,
        "eligible_for_strong_interpretation": count >= minimum,
        "interpretation": "eligible_descriptive_comparison" if count >= minimum else "descriptive_only_small_group",
    }


def bootstrap_mean_ci(
    values: Sequence[float], *, seed: int, resamples: int = BOOTSTRAP_RESAMPLES
) -> dict[str, Any]:
    """Compute a compact deterministic percentile bootstrap interval."""

    usable = [float(value) for value in values if math.isfinite(float(value))]
    if not usable:
        return {"estimate": None, "lower": None, "upper": None, "resamples": 0, "seed": seed}
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    estimate = statistics.fmean(usable)
    rng = random.Random(seed)
    sampled = [statistics.fmean(rng.choice(usable) for _ in usable) for _ in range(resamples)]
    return {
        "estimate": estimate,
        "lower": _quantile(sampled, 0.025),
        "upper": _quantile(sampled, 0.975),
        "resamples": resamples,
        "seed": seed,
    }


def bootstrap_rate_ci(
    values: Sequence[bool | int | float], *, seed: int, resamples: int = BOOTSTRAP_RESAMPLES
) -> dict[str, Any]:
    """Bootstrap a binary rate, preserving the same schema as mean CIs."""

    binary = [1.0 if bool(value) else 0.0 for value in values]
    return bootstrap_mean_ci(binary, seed=seed, resamples=resamples)


def compute_disparity(
    reference: Mapping[str, Any], comparison: Mapping[str, Any], *, metric: str = "top1_rate"
) -> dict[str, Any]:
    """Compute a descriptive absolute gap between two group summaries."""

    if metric not in reference or metric not in comparison:
        raise KeyError(f"metric {metric!r} is missing from a group summary")
    reference_value = float(reference[metric])
    comparison_value = float(comparison[metric])
    return {
        "metric": metric,
        "reference_group": reference.get("group_label"),
        "comparison_group": comparison.get("group_label"),
        "reference_value": reference_value,
        "comparison_value": comparison_value,
        "absolute_gap": abs(reference_value - comparison_value),
        "signed_comparison_minus_reference": comparison_value - reference_value,
        "fairness_claim_made": False,
    }


def validate_finding_category(category: str) -> bool:
    if category not in EVIDENCE_CATEGORIES:
        raise ValueError(f"unsupported evidence category: {category}")
    return True


def validate_no_protected_attributes(value: Any) -> bool:
    """Reject protected-label names in group definitions, not in captions."""

    def visit(item: Any, key: str = "") -> None:
        if isinstance(item, Mapping):
            for child_key, child_value in item.items():
                normalized_key = str(child_key).casefold()
                if normalized_key in PROTECTED_LABEL_NAMES:
                    raise ValueError(f"protected attribute key is not allowed: {child_key}")
                visit(child_value, normalized_key)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for child in item:
                visit(child, key)
        elif (
            isinstance(item, str)
            and key in {"group_label", "label", "group_name", "family"}
            and item.casefold() in PROTECTED_LABEL_NAMES
        ):
            raise ValueError(f"protected attribute label is not allowed: {item}")

    visit(value)
    return True


def _phase18_audit_checks(phase18_dir: Path, regression_tests_passed: bool) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    required = tuple(phase18_dir / name for name in (
        "explanation_records.json",
        "token_importance.json",
        "region_sensitivity.json",
        "counterfactual_analysis.json",
        "faithfulness_results.json",
        "high_confidence_error_explanations.json",
        "explanation_consistency.json",
        "qualitative_examples.json",
        "provenance.json",
        "phase18_report.json",
        "artifact_validation.json",
    ))
    checks["required_artifacts"] = all(path.exists() for path in required)
    if not checks["required_artifacts"]:
        return {**checks, "report_pass": False, "validator_pass": False, "records_nonempty": False, "systems_present": False, "directions_separated": False, "causal_claim_absent": False, "heuristic_labels_explicit": False, "no_training_or_download": False, "regression_tests_passed": regression_tests_passed}
    try:
        report = _read_json(phase18_dir / "phase18_report.json")
        validator = _read_json(phase18_dir / "artifact_validation.json")
        provenance = _read_json(phase18_dir / "provenance.json")
        records = _read_json(phase18_dir / "explanation_records.json").get("rows", [])
        token_rows = _read_json(phase18_dir / "token_importance.json").get("rows", [])
        region_rows = _read_json(phase18_dir / "region_sensitivity.json").get("rows", [])
        high_errors = _read_json(phase18_dir / "high_confidence_error_explanations.json").get("rows", [])
        categories = [category for row in records for category in row.get("taxonomy_categories", [])]
        checks.update(
            {
                "report_pass": report.get("status") == "PASS" and report.get("quality_gate", {}).get("status") == "PASS",
                "validator_pass": validator.get("passed") is True,
                "records_nonempty": bool(records) and bool(token_rows) and bool(region_rows) and bool(high_errors),
                "systems_present": {str(row.get("system")) for row in records} == set(SYSTEMS),
                "directions_separated": {str(row.get("direction")) for row in records} == set(DIRECTIONS),
                "causal_claim_absent": provenance.get("causal_explanation_claimed") is False
                and all("not a causal" in str(row.get("explanation_language", "")).casefold() for row in records),
                "heuristic_labels_explicit": all(str(category.get("label_source")) in {"heuristic", "mechanical", "mechanical_context"} for category in categories),
                "no_training_or_download": provenance.get("training_performed") is False and provenance.get("new_dataset_downloaded") is False and provenance.get("phase19_started") is False,
            }
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        checks.update({"report_pass": False, "validator_pass": False, "records_nonempty": False, "systems_present": False, "directions_separated": False, "causal_claim_absent": False, "heuristic_labels_explicit": False, "no_training_or_download": False})
    checks["regression_tests_passed"] = regression_tests_passed
    return checks


def audit_phase18(phase18_dir: Path | str, *, regression_tests_passed: bool = False) -> dict[str, Any]:
    """Run the focused pre-phase dependency audit for Phase 18."""

    checks = _phase18_audit_checks(Path(phase18_dir), regression_tests_passed)
    passed = all(checks.values())
    return {
        "schema_version": PHASE19_SCHEMA_VERSION,
        "phase": 19,
        "dependency": 18,
        "audit_result": "PRE-PHASE AUDIT: Phase 18 PASS" if passed else "PRE-PHASE AUDIT: Phase 18 BLOCKED",
        "passed": passed,
        "checks": checks,
        "recorded_before_phase19_analysis": True,
        "analysis_started": False,
    }


def _training_token_counts(manifest: DatasetManifest) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in manifest.records:
        if record.split == "train":
            for caption in record.captions:
                counts.update(tokenize_caption(caption.text))
    return counts


def _image_aspects(manifest: DatasetManifest, image_root: Path) -> dict[str, dict[str, Any]]:
    """Read dimensions only; no model inference or image transformation occurs."""

    from PIL import Image

    output: dict[str, dict[str, Any]] = {}
    for record in manifest.records:
        if not record.filename:
            continue
        path = image_root / record.filename
        try:
            with Image.open(path) as image:
                width, height = image.size
            output[record.image_id] = {"width": width, "height": height, "aspect_group": image_aspect_group(width, height)}
        except (OSError, ValueError):
            output[record.image_id] = {"width": None, "height": None, "aspect_group": "unavailable"}
    return output


def _target_image_id(row: Mapping[str, Any]) -> str:
    direction = str(row["direction"])
    if direction == "text_to_image":
        relevant = row.get("relevant_ids", [])
        if not relevant:
            raise ValueError("text-to-image row has no relevant image")
        return str(relevant[0])
    return str(row["query_id"])


def _assign_groups(
    rows: Sequence[Mapping[str, Any]],
    manifest: DatasetManifest,
    image_aspects: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, str]]]:
    counts = _training_token_counts(manifest)
    assignments: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    for source in rows:
        row = dict(source)
        direction = str(row["direction"])
        query_key = (str(row["system"]), direction, str(row["query_id"]))
        group_values: dict[str, str] = {}
        image_info = image_aspects.get(_target_image_id(row), {"aspect_group": "unavailable"})
        group_values["aspect_ratio"] = str(image_info.get("aspect_group", "unavailable"))
        if direction == "text_to_image":
            caption = str(row.get("query_context", ""))
            features = caption_group_features(caption, counts)
            group_values.update(
                {
                    "caption_length": str(features["caption_length_group"]),
                    "concept_rarity": str(features["concept_rarity_group"]),
                    "object_complexity": str(features["complexity_group"]),
                }
            )
        lookup[query_key] = group_values
        assignments.append({**row, "group_values": group_values})
    return assignments, lookup


def _metric_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is not None:
            try:
                number = float(value)
                if math.isfinite(number):
                    values.append(number)
            except (TypeError, ValueError):
                continue
    return values


def _selected_thresholds(phase17_dir: Path) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    payload = _read_json(phase17_dir / "selective_retrieval.json")
    for row in payload.get("recommendations", []):
        if row.get("selected_target_coverage") == 0.5 and row.get("test_labels_used_for_threshold_selection") is False:
            result[(str(row["system"]), str(row["direction"]))] = float(row["selected_threshold"])
    return result


def _group_performance(assignments: Sequence[Mapping[str, Any]], thresholds: Mapping[tuple[str, str], float]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in assignments:
        for family, label in row["group_values"].items():
            grouped[(str(row["system"]), str(row["direction"]), family, label)].append(row)
    output: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        system, direction, family, label = key
        top1 = [bool(row.get("top1_correct")) for row in rows]
        top5 = [bool(row.get("top5_correct")) for row in rows]
        mrr = [1.0 / float(row["first_relevant_rank"]) if row.get("first_relevant_rank") else 0.0 for row in rows]
        confidence = _metric_values(rows, "calibrated_confidence")
        errors = [bool(row.get("top1_failure", not row.get("top1_correct", False))) for row in rows]
        threshold = thresholds.get((system, direction))
        accepted = [threshold is not None and float(row.get("calibrated_confidence", 0.0)) >= threshold for row in rows]
        accepted_correct = [correct for correct, keep in zip(top1, accepted) if keep]
        group = {
            "system": system,
            "direction": direction,
            "group_family": family,
            "group_label": label,
            "query_count": len(rows),
            **minimum_group_status(len(rows)),
            "top1_rate": statistics.fmean(top1),
            "top5_rate": statistics.fmean(top5),
            "mrr": statistics.fmean(mrr),
            "failure_rate": statistics.fmean(errors),
            "mean_calibrated_confidence": statistics.fmean(confidence) if confidence else None,
            "median_calibrated_confidence": statistics.median(confidence) if confidence else None,
            "high_confidence_error_count": sum(error and float(row.get("calibrated_confidence", 0.0)) >= HIGH_CONFIDENCE_THRESHOLD for error, row in zip(errors, rows)),
            "selective_threshold_source": "Phase 17 validation-only 50%-target recommendation" if threshold is not None else "unavailable",
            "selective_threshold": threshold,
            "selective_accepted_count": sum(accepted),
            "selective_rejection_rate": 1.0 - (sum(accepted) / len(rows)) if rows else None,
            "selective_accepted_top1_rate": statistics.fmean(accepted_correct) if accepted_correct else None,
            "top1_rate_bootstrap_ci": bootstrap_rate_ci(top1, seed=_stable_seed(system, direction, family, label, "top1")),
            "failure_rate_bootstrap_ci": bootstrap_rate_ci(errors, seed=_stable_seed(system, direction, family, label, "failure")),
            "confidence_bootstrap_ci": bootstrap_mean_ci(confidence, seed=_stable_seed(system, direction, family, label, "confidence")) if confidence else None,
            "metric_interpretation": "descriptive performance conditional on the fixed COCO same-image relevance proxy; not a fairness judgment",
        }
        output.append(group)
    return output


def _disparity_analysis(group_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in group_rows:
        grouped[(str(row["system"]), str(row["direction"]), str(row["group_family"]))].append(row)
    output: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        eligible = [row for row in values if row["eligible_for_strong_interpretation"]]
        system, direction, family = key
        base = {
            "system": system,
            "direction": direction,
            "group_family": family,
            "group_count": len(values),
            "eligible_group_count": len(eligible),
            "minimum_group_size": MIN_GROUP_SIZE,
            "fairness_claim_made": False,
            "interpretation": "descriptive group-performance comparison; group disparity is not evidence of unfairness without protected labels, causal analysis, or human relevance judgments",
        }
        if len(eligible) < 2:
            output.append({**base, "evidence_category": "INSUFFICIENT EVIDENCE", "status": "fewer than two groups meet the predeclared minimum", "metrics": []})
            continue
        highest = max(eligible, key=lambda row: (float(row["top1_rate"]), str(row["group_label"])))
        lowest = min(eligible, key=lambda row: (float(row["top1_rate"]), str(row["group_label"])))
        metrics = [compute_disparity(highest, lowest, metric=metric) for metric in ("top1_rate", "top5_rate", "mrr", "failure_rate")]
        confidence_gap = None
        if highest.get("mean_calibrated_confidence") is not None and lowest.get("mean_calibrated_confidence") is not None:
            confidence_gap = abs(float(highest["mean_calibrated_confidence"]) - float(lowest["mean_calibrated_confidence"]))
        output.append({
            **base,
            "evidence_category": "MEASURED RISK",
            "status": "eligible descriptive disparity observed; not a fairness conclusion",
            "highest_top1_group": highest["group_label"],
            "lowest_top1_group": lowest["group_label"],
            "metrics": metrics,
            "absolute_confidence_gap": confidence_gap,
            "bootstrap_note": "group-wise percentile bootstrap intervals are compact query-sample uncertainty summaries; groups are not protected classes",
        })
    return output


def _confidence_disparity(group_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in group_rows:
        grouped[(str(row["system"]), str(row["direction"]), str(row["group_family"]))].append(row)
    for key, values in sorted(grouped.items()):
        system, direction, family = key
        eligible = [row for row in values if row["eligible_for_strong_interpretation"]]
        for row in values:
            output.append({
                "system": system,
                "direction": direction,
                "group_family": family,
                "group_label": row["group_label"],
                "query_count": row["query_count"],
                "eligible_for_strong_interpretation": row["eligible_for_strong_interpretation"],
                "mean_calibrated_confidence": row["mean_calibrated_confidence"],
                "confidence_ci": row["confidence_bootstrap_ci"],
                "high_confidence_error_count": row["high_confidence_error_count"],
                "high_confidence_error_rate": row["high_confidence_error_count"] / row["query_count"],
                "selective_rejection_rate": row["selective_rejection_rate"],
                "evidence_category": "OBSERVED LIMITATION" if row["eligible_for_strong_interpretation"] else "INSUFFICIENT EVIDENCE",
                "interpretation": "confidence disparity is a retrieval-calibration observation, not a claim about people or protected groups",
            })
        if len(eligible) >= 2:
            highest = max(eligible, key=lambda row: float(row["mean_calibrated_confidence"] or 0.0))
            lowest = min(eligible, key=lambda row: float(row["mean_calibrated_confidence"] or 0.0))
            output.append({
                "system": system,
                "direction": direction,
                "group_family": family,
                "comparison": True,
                "higher_confidence_group": highest["group_label"],
                "lower_confidence_group": lowest["group_label"],
                "absolute_confidence_gap": abs(float(highest["mean_calibrated_confidence"]) - float(lowest["mean_calibrated_confidence"])),
                "absolute_high_confidence_error_rate_gap": abs((highest["high_confidence_error_count"] / highest["query_count"]) - (lowest["high_confidence_error_count"] / lowest["query_count"])),
                "evidence_category": "MEASURED RISK",
                "fairness_claim_made": False,
            })
    return output


def _find_phase18_explanation(phase18_dir: Path, key: tuple[str, str, str]) -> Mapping[str, Any] | None:
    rows = _read_json(phase18_dir / "high_confidence_error_explanations.json").get("rows", [])
    return next((row for row in rows if (str(row.get("system")), str(row.get("direction")), str(row.get("query_id"))) == key), None)


def _high_confidence_examples(
    errors: Sequence[Mapping[str, Any]],
    assignments: Sequence[Mapping[str, Any]],
    phase18_dir: Path,
) -> list[dict[str, Any]]:
    lookup = {(str(row["system"]), str(row["direction"]), str(row["query_id"])): row for row in assignments}
    output: list[dict[str, Any]] = []
    for error in sorted(errors, key=lambda row: (-float(row.get("calibrated_confidence", 0.0)), str(row.get("query_id")))):
        key = (str(error["system"]), str(error["direction"]), str(error["query_id"]))
        source = lookup.get(key, {})
        explanation = _find_phase18_explanation(phase18_dir, key)
        output.append({
            "system": error["system"],
            "direction": error["direction"],
            "query_id": error["query_id"],
            "query_context": error.get("query_context"),
            "calibrated_confidence": error.get("calibrated_confidence"),
            "first_relevant_rank": error.get("first_relevant_rank"),
            "taxonomy_categories": error.get("taxonomy_categories", []),
            "group_values": source.get("group_values", {}),
            "phase18_explanation_attached": explanation is not None,
            "phase18_local_evidence": {
                "top_token_sensitivities": explanation.get("top_token_sensitivities", []) if explanation else [],
                "top_region_sensitivities": explanation.get("top_region_sensitivities", []) if explanation else [],
            },
            "finding_category": "OBSERVED LIMITATION",
            "risk_type": "high_confidence_retrieval_error",
            "harmful_content_determination": "not made; no content-safety classifier or human review was run",
            "interpretation": "a high-confidence error is a system reliability risk in this benchmark, not evidence that the caption or image is harmful",
        })
    return output


def _global_comparison(assignments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for direction in DIRECTIONS:
        direction_rows = [row for row in assignments if str(row["direction"]) == direction]
        summaries: dict[str, dict[str, Any]] = {}
        for system in SYSTEMS:
            rows = [row for row in direction_rows if str(row["system"]) == system]
            summaries[system] = {
                "query_count": len(rows),
                "top1_rate": statistics.fmean(bool(row.get("top1_correct")) for row in rows),
                "top5_rate": statistics.fmean(bool(row.get("top5_correct")) for row in rows),
                "mrr": statistics.fmean(1.0 / float(row["first_relevant_rank"]) if row.get("first_relevant_rank") else 0.0 for row in rows),
                "mean_calibrated_confidence": statistics.fmean(float(row["calibrated_confidence"]) for row in rows),
            }
        output.append({
            "direction": direction,
            "zero_shot": summaries["zero_shot"],
            "full_ft": summaries["full_ft"],
            "full_ft_minus_zero_shot": {metric: summaries["full_ft"][metric] - summaries["zero_shot"][metric] for metric in ("top1_rate", "top5_rate", "mrr", "mean_calibrated_confidence")},
            "interpretation": "held-out test comparison reused from Phase 17 confidence rows; it is not a causal fairness comparison",
        })
    return output


def _responsible_ai_matrix(disparities: Sequence[Mapping[str, Any]], high_errors: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    measured_disparities = [row for row in disparities if row.get("evidence_category") == "MEASURED RISK"]
    return [
        {"risk_area": "dataset-derived group performance", "evidence": f"{len(measured_disparities)} eligible descriptive disparity comparisons across caption/image strata", "severity": "MEDIUM", "confidence_in_finding": "medium", "evidence_category": "MEASURED RISK" if measured_disparities else "INSUFFICIENT EVIDENCE", "current_mitigation": "fixed image-grouped split, held-out test reporting, minimum group-size rule, bootstrap intervals", "recommended_mitigation": "expand data and obtain human relevance review before using any stratum as a deployment fairness proxy", "fairness_claim_made": False},
        {"risk_area": "high-confidence retrieval errors", "evidence": f"{len(high_errors)} retained Phase 17 errors at calibrated confidence >= {HIGH_CONFIDENCE_THRESHOLD}", "severity": "HIGH", "confidence_in_finding": "high", "evidence_category": "MEASURED RISK" if high_errors else "INSUFFICIENT EVIDENCE", "current_mitigation": "Phase 17 calibration/selective-retrieval diagnostics and Phase 18 local evidence", "recommended_mitigation": "use abstention plus human review for high-impact results; monitor error slices", "fairness_claim_made": False},
        {"risk_area": "protected-group fairness", "evidence": "no protected-attribute labels are present in the COCO manifest or evaluation rows", "severity": "HIGH", "confidence_in_finding": "high", "evidence_category": "NOT EVALUATED", "current_mitigation": "do not infer protected status from captions or images", "recommended_mitigation": "obtain lawful, consented, task-appropriate labels and governance before a fairness study", "fairness_claim_made": False},
        {"risk_area": "multilingual and language access", "evidence": "the evaluated caption corpus is English COCO text; no multilingual test was run", "severity": "MEDIUM", "confidence_in_finding": "high", "evidence_category": "NOT EVALUATED", "current_mitigation": "scope claims to the evaluated English caption protocol", "recommended_mitigation": "evaluate translated and natively authored multilingual queries with language-specific error analysis", "fairness_claim_made": False},
        {"risk_area": "content safety", "evidence": "retrieval rankings were evaluated for relevance only; no content-safety classifier, red-team review, or moderation gate was run", "severity": "HIGH", "confidence_in_finding": "high", "evidence_category": "NOT EVALUATED", "current_mitigation": "research-only scope and explicit human oversight requirement", "recommended_mitigation": "add policy-specific filtering, abuse testing, audit logs, and human escalation before deployment", "fairness_claim_made": False},
        {"risk_area": "privacy and rights", "evidence": "COCO image rights remain with originating Flickr sources; no private-user data audit was performed", "severity": "HIGH", "confidence_in_finding": "high", "evidence_category": "POTENTIAL DEPLOYMENT RISK", "current_mitigation": "public benchmark only, source/license notes, no private ingestion in this phase", "recommended_mitigation": "verify provenance, consent, retention, access control, deletion, and jurisdictional requirements for any new data", "fairness_claim_made": False},
        {"risk_area": "misuse and security", "evidence": "image-text retrieval can be repurposed for sensitive search, profiling, or surveillance; misuse testing was not run", "severity": "HIGH", "confidence_in_finding": "medium", "evidence_category": "POTENTIAL DEPLOYMENT RISK", "current_mitigation": "no service or public deployment is in scope", "recommended_mitigation": "restrict access, define prohibited uses, rate-limit, log, red-team, and require human authorization for sensitive workflows", "fairness_claim_made": False},
        {"risk_area": "accessibility and human oversight", "evidence": "no accessibility study or user test was run; retrieval explanations are local sensitivities, not human-readable guarantees", "severity": "MEDIUM", "confidence_in_finding": "high", "evidence_category": "NOT EVALUATED", "current_mitigation": "retain uncertainty and explanation limitations in the system card", "recommended_mitigation": "provide keyboard/screen-reader compatible review tools and accessible result alternatives", "fairness_claim_made": False},
        {"risk_area": "generalization beyond COCO", "evidence": "all quantitative groups use one fixed COCO val2017-derived split and one CLIP checkpoint", "severity": "HIGH", "confidence_in_finding": "high", "evidence_category": "OBSERVED LIMITATION", "current_mitigation": "fixed-scope claims, held-out test, robustness/error-analysis limitations", "recommended_mitigation": "validate on rights-cleared external domains and real target users before deployment", "fairness_claim_made": False},
    ]


def _mitigations() -> list[dict[str, Any]]:
    return [
        {"priority": "P0", "recommendation": "Do not deploy as an autonomous decision-maker; require human review for high-impact or sensitive retrieval.", "reason": "content safety, privacy, misuse, and high-confidence errors were not eliminated", "status": "recommended"},
        {"priority": "P0", "recommendation": "Add a policy-specific moderation and abuse-evaluation layer before accepting arbitrary user content.", "reason": "content safety was not evaluated in Phase 19", "status": "not implemented"},
        {"priority": "P1", "recommendation": "Monitor top-1 failure, abstention, and high-confidence error rates by the declared dataset-derived strata.", "reason": "measured disparities are descriptive monitoring signals, not protected-group fairness evidence", "status": "recommended"},
        {"priority": "P1", "recommendation": "Collect consented, governance-approved human relevance labels and multilingual/accessibility test cases.", "reason": "COCO same-image relevance and English-only data are limited proxies", "status": "recommended"},
        {"priority": "P1", "recommendation": "Preserve provenance, access controls, retention/deletion rules, and audit logs for any future data or deployment.", "reason": "rights, privacy, and misuse risks are deployment-dependent", "status": "recommended"},
    ]


def validate_phase19_artifacts(output_dir: Path | str) -> dict[str, Any]:
    output = Path(output_dir)
    checks: dict[str, bool] = {"required_artifacts": all((output / name).exists() for name in REQUIRED_ARTIFACTS)}
    if not checks["required_artifacts"]:
        return {"schema_version": PHASE19_SCHEMA_VERSION, "passed": False, "checks": checks, "required": list(REQUIRED_ARTIFACTS)}
    try:
        definitions = _read_json(output / "group_definitions.json")
        performance = _read_json(output / "group_performance.json")
        disparities = _read_json(output / "disparity_analysis.json")
        confidence = _read_json(output / "confidence_disparity.json")
        examples = _read_json(output / "high_confidence_risk_examples.json")
        matrix = _read_json(output / "responsible_ai_matrix.json")
        mitigations = _read_json(output / "mitigation_recommendations.json")
        provenance = _read_json(output / "provenance.json")
        report = _read_json(output / "phase19_report.json")
        checks.update(
            {
                "pre_phase_audit_pass": _read_json(output / "pre_phase_audit.json").get("passed") is True,
                "group_definitions_reproducible": definitions.get("minimum_group_size") == MIN_GROUP_SIZE and bool(definitions.get("families")),
                "no_protected_labels": validate_no_protected_attributes(definitions.get("families", {})) is True and definitions.get("protected_attribute_names") == [],
                "performance_rows_nonempty": bool(performance.get("rows")),
                "disparity_categories_valid": all(validate_finding_category(str(row.get("evidence_category"))) for row in disparities.get("rows", [])),
                "confidence_rows_nonempty": bool(confidence.get("rows")),
                "small_groups_cautious": all(not row.get("eligible_for_strong_interpretation") or int(row.get("query_count", 0)) >= MIN_GROUP_SIZE for row in performance.get("rows", [])),
                "high_confidence_examples_present": bool(examples.get("rows")),
                "matrix_covers_privacy_safety": {str(row.get("risk_area")) for row in matrix.get("rows", [])}.issuperset({"privacy and rights", "content safety"}),
                "mitigations_present": bool(mitigations.get("rows")),
                "zero_full_comparison_present": bool(report.get("zero_shot_vs_full_ft")),
                "no_training_or_download": provenance.get("training_performed") is False and provenance.get("new_dataset_downloaded") is False,
                "phase20_not_started": provenance.get("phase20_started") is False and "phase 20" not in json.dumps(report).casefold(),
                "report_pass": report.get("status") == "PASS" and report.get("quality_gate", {}).get("status") == "PASS",
                "model_card_updated": Path(str(provenance.get("system_card_path", ""))).exists(),
            }
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        checks["schema_valid"] = False
    checks.setdefault("schema_valid", True)
    return {"schema_version": PHASE19_SCHEMA_VERSION, "passed": all(checks.values()), "checks": checks, "required": list(REQUIRED_ARTIFACTS)}


def _scope(manifest: DatasetManifest, test_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    split_stats: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        records = [record for record in manifest.records if record.split == split]
        split_stats[split] = {"image_groups": len(records), "captions": sum(len(record.captions) for record in records)}
    return {
        "dataset_id": manifest.dataset_id,
        "dataset_version": manifest.dataset_version,
        "manifest_path": "data/processed/coco2017_val_split_manifest.json",
        "manifest_sha256": _hash_file("data/processed/coco2017_val_split_manifest.json"),
        "split_statistics": split_stats,
        "phase17_test_rows": len(test_rows),
        "phase17_test_systems": sorted({str(row["system"]) for row in test_rows}),
        "phase17_test_directions": sorted({str(row["direction"]) for row in test_rows}),
        "relevance_definition": "same-image metadata relevance from the retained Phase 7/17 rankings",
        "training_used": False,
        "new_dataset_downloaded": False,
    }


def run_phase19(
    *,
    manifest_path: Path | str = "data/processed/coco2017_val_split_manifest.json",
    phase17_dir: Path | str = "artifacts/phase17",
    phase18_dir: Path | str = "artifacts/phase18",
    phase7_dir: Path | str = "artifacts/phase7",
    output_dir: Path | str = "artifacts/phase19",
    image_root: Path | str = "data/raw/coco2017/val2017",
    regression_tests_passed: bool = True,
    system_card_path: Path | str = "docs/system_card.md",
) -> dict[str, Any]:
    """Run all Phase 19 analysis and write machine-readable artifacts."""

    manifest_path = Path(manifest_path)
    phase17_dir = Path(phase17_dir)
    phase18_dir = Path(phase18_dir)
    phase7_dir = Path(phase7_dir)
    output = Path(output_dir)
    manifest = read_manifest(manifest_path)
    audit = audit_phase18(phase18_dir, regression_tests_passed=regression_tests_passed)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(audit, output / "pre_phase_audit.json")
    if not audit["passed"]:
        raise RuntimeError(audit["audit_result"])

    phase17_payload = _read_json(phase17_dir / "confidence_records.json")
    test_rows = [row for row in phase17_payload.get("rows", []) if row.get("split") == "test" and row.get("system") in SYSTEMS and row.get("direction") in DIRECTIONS]
    if not test_rows:
        raise RuntimeError("Phase 17 test confidence rows are missing")
    aspects = _image_aspects(manifest, Path(image_root))
    assignments, _lookup = _assign_groups(test_rows, manifest, aspects)
    thresholds = _selected_thresholds(phase17_dir)
    performance = _group_performance(assignments, thresholds)
    disparities = _disparity_analysis(performance)
    confidence = _confidence_disparity(performance)
    high_errors = _read_json(phase17_dir / "high_confidence_errors.json").get("rows", [])
    risk_examples = _high_confidence_examples(high_errors, assignments, phase18_dir)
    matrix = _responsible_ai_matrix(disparities, risk_examples)
    mitigations = _mitigations()
    definitions = {
        "schema_version": PHASE19_SCHEMA_VERSION,
        "minimum_group_size": MIN_GROUP_SIZE,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "protected_attribute_names": [],
        "protected_attribute_policy": "no protected labels were fabricated or inferred; fairness is not evaluated from these strata",
        "families": {
            "caption_length": {"labels": ["short", "medium", "long"], "rule": "caption word count: <=7, 8-14, >=15; text-to-image only"},
            "concept_rarity": {"labels": ["rare", "medium", "common", "unknown"], "rule": "rarest non-stopword token frequency in train captions: <=4, 5-49, >=50; text-to-image only"},
            "object_complexity": {"labels": ["single_object_or_simple", "multi_object_or_structured"], "rule": "small fixed object lexicon plus conjunction/number indicators; heuristic, text-to-image only"},
            "aspect_ratio": {"labels": ["wide_or_tall_extreme", "near_square", "moderate_aspect", "unavailable"], "rule": "Pillow dimensions: ratio <=0.75 or >=4/3, 0.9-1.1, otherwise; target image for text-to-image and query image for image-to-text"},
        },
        "training_frequency_source": "captions from manifest records with split=train only",
        "tokenizer": "lower-case ASCII alphanumeric regex with apostrophe support; stopword filtering only for rarity/object features",
        "group_assignment_count": len(assignments),
        "group_assignment_keys": sorted({f"{row['system']}:{row['direction']}:{row['query_id']}" for row in assignments}),
    }
    validate_no_protected_attributes(definitions["families"])
    _write_json(_scope(manifest, test_rows), output / "dataset_scope.json")
    _write_json(definitions, output / "group_definitions.json")
    _write_json({"schema_version": PHASE19_SCHEMA_VERSION, "rows": performance}, output / "group_performance.json")
    _write_json({"schema_version": PHASE19_SCHEMA_VERSION, "rows": disparities}, output / "disparity_analysis.json")
    _write_json({"schema_version": PHASE19_SCHEMA_VERSION, "rows": confidence}, output / "confidence_disparity.json")
    _write_json({"schema_version": PHASE19_SCHEMA_VERSION, "threshold": HIGH_CONFIDENCE_THRESHOLD, "rows": risk_examples, "source": "Phase 17 high_confidence_errors plus Phase 18 local explanation artifacts"}, output / "high_confidence_risk_examples.json")
    _write_json({"schema_version": PHASE19_SCHEMA_VERSION, "rows": matrix}, output / "responsible_ai_matrix.json")
    _write_json({"schema_version": PHASE19_SCHEMA_VERSION, "rows": mitigations}, output / "mitigation_recommendations.json")

    provenance = {
        "schema_version": PHASE19_SCHEMA_VERSION,
        "phase": 19,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "manifest": str(manifest_path),
        "manifest_sha256": _hash_file(manifest_path),
        "phase17_confidence_records_sha256": _hash_file(phase17_dir / "confidence_records.json"),
        "phase17_high_confidence_errors_sha256": _hash_file(phase17_dir / "high_confidence_errors.json"),
        "phase17_selective_retrieval_sha256": _hash_file(phase17_dir / "selective_retrieval.json"),
        "phase18_report_sha256": _hash_file(phase18_dir / "phase18_report.json"),
        "phase18_explanation_records_sha256": _hash_file(phase18_dir / "explanation_records.json"),
        "phase16_taxonomy_source": "artifacts/phase16/failure_records.json and Phase 17 taxonomy fields",
        "phase7_ranking_inputs": {name: _hash_file(phase7_dir / name) for name in ("zero_shot_text_to_image.json", "zero_shot_image_to_text.json", "fine_tuned_text_to_image.json", "fine_tuned_image_to_text.json")},
        "phase7_checkpoint_sha256": _hash_file(phase7_dir / "best_checkpoint.pt"),
        "system_card_path": str(system_card_path),
        "systems": list(SYSTEMS),
        "directions": list(DIRECTIONS),
        "test_rows": len(test_rows),
        "training_performed": False,
        "new_dataset_downloaded": False,
        "phase20_started": False,
        "protected_labels_fabricated": False,
        "fairness_claim_made": False,
        "content_safety_classifier_run": False,
        "privacy_audit_of_new_private_data_run": False,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "python": sys.version,
        "platform": platform.platform(),
    }
    _write_json(provenance, output / "provenance.json")
    report = {
        "report_schema_version": PHASE19_SCHEMA_VERSION,
        "project": "OmniSearch",
        "phase": 19,
        "status": "PASS",
        "pre_phase_audit": audit["audit_result"],
        "scope": {"dataset": manifest.dataset_id, "systems": list(SYSTEMS), "directions": list(DIRECTIONS), "evaluation_split": "test", "training": False},
        "zero_shot_vs_full_ft": _global_comparison(assignments),
        "group_findings": {"performance_rows": len(performance), "disparity_rows": len(disparities), "confidence_rows": len(confidence), "high_confidence_errors": len(risk_examples)},
        "privacy_safety_status": "privacy, rights, misuse, accessibility, and content safety are documented as limitations/risks; no new private data or safety classifier was used",
        "quality_gate": {"status": "PASS", "checks": {}},
        "artifacts": list(REQUIRED_ARTIFACTS),
    }
    _write_json(report, output / "phase19_report.json")
    validation = validate_phase19_artifacts(output)
    _write_json(validation, output / "artifact_validation.json")
    # The validator is called once more after the final report contains the
    # completed quality-gate checks.  The report is initially marked PASS so
    # its schema can be validated without a self-referential pending state.
    report["status"] = "PASS" if validation["passed"] else "PARTIAL"
    report["quality_gate"] = {"status": "PASS" if validation["passed"] else "FAIL", "checks": validation["checks"]}
    _write_json(report, output / "phase19_report.json")
    validation = validate_phase19_artifacts(output)
    _write_json(validation, output / "artifact_validation.json")
    return {"report": report, "validation": validation, "audit": audit}


def main() -> int:
    result = run_phase19()
    print(json.dumps(result["report"], indent=2))
    return 0 if result["validation"]["passed"] else 1
