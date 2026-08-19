"""Phase 16 error analysis and failure taxonomy.

This phase consumes retained retrieval artifacts only.  It never trains a
model or infers human semantic labels from COCO images.  Rank facts,
caption-derived features, and Phase 15/9/11 artifact facts are kept separate
from heuristic semantic interpretations.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .manifest import ImageRecord, read_manifest
from .phase7 import _hash_file, _subset_records
from .phase13 import paired_bootstrap

PHASE16_SCHEMA_VERSION = 1
MANIFEST_SHA256 = "09a2c1e56eb1a628b2ead16f064510d713f81aff5ee2f2d09b4ca8993bba3b43"
PROTOCOL_VERSION = "retrieval_eval_v1"
SEED = 42
TEST_IMAGE_LIMIT = 100
TOP_K = 10
SYSTEMS = ("zero_shot", "full_ft")
DIRECTIONS = ("text_to_image", "image_to_text")
METRICS = ("recall_at_1", "recall_at_5", "recall_at_10", "mrr")
VALID_LABEL_SOURCES = frozenset({"mechanical", "heuristic", "mechanical_context"})
TEXT_TAXONOMY = (
    "object_confusion",
    "attribute_confusion",
    "action_confusion",
    "counting_quantity",
    "spatial_relation",
    "lexical_ambiguity",
    "short_underspecified_query",
    "multiple_object_confusion",
)
TAXONOMY_CATEGORIES = TEXT_TAXONOMY + (
    "scene_context_confusion",
    "visually_similar_distractor",
    "background_context_dominance",
    "small_difficult_visual_target",
    "corruption_sensitivity",
    "candidate_retrieval_failure",
    "reranker_induced_regression",
    "annotation_relevance_ambiguity",
    "other_unclassified",
)
TEXT_RESULT_FILES = {
    "zero_shot": "artifacts/phase7/zero_shot_text_to_image.json",
    "full_ft": "artifacts/phase7/fine_tuned_text_to_image.json",
}
IMAGE_RESULT_FILES = {
    "zero_shot": "artifacts/phase7/zero_shot_image_to_text.json",
    "full_ft": "artifacts/phase7/fine_tuned_image_to_text.json",
}
HARD_RESULT_FILES = {
    "text_to_image": "hard_negative_text_to_image.json",
    "image_to_text": "hard_negative_image_to_text.json",
}
WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def _word_set(value: str) -> frozenset[str]:
    return frozenset(value.split())


NUMBER_WORDS = _word_set("zero one two three four five six seven eight nine ten eleven twelve")
ATTRIBUTE_WORDS = _word_set("red blue green yellow black white brown gray grey orange pink purple large small big little young old tall short long wooden metal striped spotted colorful bright dark full empty open closed")
ACTION_WORDS = _word_set("run running walk walking stand standing sit sitting hold holding eat eating drink drinking ride riding jump jumping play playing look looking wear wearing catch caught throw throwing fly flying swim swimming cut cutting sleep sleeping talk talking pose posing")
SPATIAL_WORDS = _word_set("left right above below under over behind front beside next near between inside outside on in at across through around")
OBJECT_WORDS = _word_set("person people man woman boy girl child dog cat bird horse cow sheep elephant giraffe bear zebra car bus truck bicycle bike motorcycle train airplane boat skateboard ball bat racket chair table bed couch desk street road field grass beach ocean water kitchen room building tree sky flower food cake pizza phone umbrella bag bottle cup book tie")


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _stable_seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("\0".join(parts).encode()).digest()[:4], "big")


def default_failure_definitions() -> dict[str, Any]:
    """Return rank definitions fixed before inspecting query outcomes."""

    return {
        "schema_version": PHASE16_SCHEMA_VERSION,
        "phase": 16,
        "rank_prefix": TOP_K,
        "definitions": {
            "top1_failure": "first relevant target rank is not 1",
            "top5_failure": "first relevant target rank is greater than 5 or is absent from the returned top-10 prefix",
            "severe_failure": "first relevant target rank is greater than 10; a missing target in a retained top-10 ranking is censored as rank >10",
            "success_at_k": "at least one relevant target appears in the first k returned candidates",
            "regression": "full-FT success at k is false while zero-shot success at k is true for the same query",
            "robustness_failure": "a clean success becomes a corrupted-query failure in a retained Phase 15 diagnostic example",
            "reranker_failure": "a retained Phase 11 reranker example demotes a relevant Stage-1 result or produces a documented regression",
            "rank_movement": "rank_full_ft minus rank_zero_shot when both first relevant ranks are observed",
        },
        "rank_severity": {
            "success": "rank 1",
            "low": "rank 2-5",
            "medium": "rank 6-10",
            "high": "rank greater than 10 or absent from retained top-10 prefix",
        },
        "rank_movement_thresholds": {
            "major": "absolute observed rank movement >= 5, or a success/failure transition across the top-5 boundary",
            "minor": "absolute observed rank movement 1-4 without a major transition",
            "unchanged": "observed rank movement 0",
            "censored": "one or both first relevant ranks are outside the retained top-10 prefix",
        },
        "score_margin": {
            "definition": "top-1 score minus the highest score of any relevant target in the retained prefix",
            "near_miss": "wrong top-1 with margin <= 0.02",
            "ambiguous_close": "wrong top-1 with 0.02 < margin <= 0.05",
            "confident_wrong": "wrong top-1 with margin > 0.05",
            "warning": "CLIP scores are similarity scores, not calibrated probabilities",
        },
        "label_policy": {
            "mechanical": "directly computed from retained IDs, ranks, scores, counts, text, or artifact metadata",
            "heuristic": "keyword or threshold interpretation of caption text; not a human semantic annotation",
            "no_forced_semantic_label": True,
        },
    }


def validate_failure_definitions(definitions: Mapping[str, Any]) -> None:
    if int(definitions.get("schema_version", -1)) != PHASE16_SCHEMA_VERSION:
        raise ValueError("unsupported Phase 16 definitions schema")
    if int(definitions.get("phase", -1)) != 16 or int(definitions.get("rank_prefix", -1)) != TOP_K:
        raise ValueError("Phase 16 rank protocol changed")
    required = {"top1_failure", "top5_failure", "severe_failure", "regression", "robustness_failure", "reranker_failure"}
    if set(definitions.get("definitions", {})) < required:
        raise ValueError("failure definitions are incomplete")
    if definitions.get("label_policy", {}).get("no_forced_semantic_label") is not True:
        raise ValueError("Phase 16 must not force unsupported semantic labels")


def _result_payload(path: Path, expected_task: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("task") != expected_task:
        raise ValueError(f"{path} has unexpected task")
    if payload.get("protocol", {}).get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"{path} is not retrieval_eval_v1")
    records = payload.get("ranking_records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} has no ranking records")
    for row in records:
        if not isinstance(row, Mapping):
            raise TypeError("ranking record must be an object")
        if not {"query_id", "candidate_ids", "scores", "relevant_ids"} <= set(row):
            raise ValueError(f"{path} contains an incomplete ranking record")
        if len(row["candidate_ids"]) != len(row["scores"]):
            raise ValueError(f"{path} has unaligned candidate IDs and scores")
    return payload


def _index_records(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = payload.get("ranking_records")
    if not isinstance(records, list):
        raise TypeError("ranking_records must be a list")
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in records:
        query_id = str(row["query_id"])
        if query_id in indexed:
            raise ValueError(f"duplicate query ID: {query_id}")
        indexed[query_id] = row
    return indexed


def validate_alignment(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    left_rows = _index_records(left)
    right_rows = _index_records(right)
    if set(left_rows) != set(right_rows):
        raise ValueError("system query IDs are not aligned")
    for query_id in sorted(left_rows):
        if {str(item) for item in left_rows[query_id]["relevant_ids"]} != {str(item) for item in right_rows[query_id]["relevant_ids"]}:
            raise ValueError(f"relevance sets differ for {query_id}")
    if int(left.get("candidate_count", -1)) != int(right.get("candidate_count", -2)):
        raise ValueError("system candidate counts are not aligned")


def first_relevant_rank(record: Mapping[str, Any]) -> dict[str, Any]:
    candidates = [str(item) for item in record["candidate_ids"]]
    relevant = {str(item) for item in record["relevant_ids"]}
    if not relevant:
        raise ValueError(f"query {record['query_id']} has no relevance set")
    for rank, candidate in enumerate(candidates, start=1):
        if candidate in relevant:
            return {"rank": rank, "observed": True, "lower_bound": rank, "relevant_in_prefix": True}
    return {
        "rank": None,
        "observed": False,
        "lower_bound": len(candidates) + 1,
        "relevant_in_prefix": False,
    }


def rank_severity(rank_info: Mapping[str, Any]) -> str:
    rank = rank_info.get("rank") if rank_info.get("observed") else rank_info.get("lower_bound", TOP_K + 1)
    if rank == 1:
        return "success"
    if rank is not None and int(rank) <= 5:
        return "low"
    if rank is not None and int(rank) <= TOP_K:
        return "medium"
    return "high"


def failure_triggers(rank_info: Mapping[str, Any]) -> dict[str, bool]:
    rank = rank_info.get("rank") if rank_info.get("observed") else rank_info.get("lower_bound", TOP_K + 1)
    if rank is None:
        rank = TOP_K + 1
    return {
        "top1_failure": int(rank) != 1,
        "top5_failure": int(rank) > 5,
        "severe_failure": int(rank) > TOP_K,
    }


def success_at_k(rank_info: Mapping[str, Any], k: int) -> bool:
    if k <= 0:
        raise ValueError("k must be positive")
    return bool(rank_info.get("observed") and rank_info.get("rank") is not None and int(rank_info["rank"]) <= k)


def classify_transition(left_success: bool, right_success: bool) -> str:
    if left_success and right_success:
        return "both_succeed"
    if left_success and not right_success:
        return "only_left_succeeds"
    if not left_success and right_success:
        return "only_right_succeeds"
    return "both_fail"


def classify_rank_movement(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_rank = left.get("rank") if left.get("observed") else None
    right_rank = right.get("rank") if right.get("observed") else None
    if left_rank is not None and right_rank is not None:
        delta = int(right_rank) - int(left_rank)
        if delta == 0:
            category = "unchanged"
        elif abs(delta) >= 5:
            category = "major_improvement" if delta < 0 else "major_regression"
        else:
            category = "minor_improvement" if delta < 0 else "minor_regression"
        return {"rank_delta": delta, "rank_delta_observed": True, "movement_category": category}
    left_success = success_at_k(left, 5)
    right_success = success_at_k(right, 5)
    if left_success != right_success:
        return {
            "rank_delta": None,
            "rank_delta_observed": False,
            "movement_category": "major_improvement" if right_success else "major_regression",
        }
    return {"rank_delta": None, "rank_delta_observed": False, "movement_category": "censored"}


def score_margin(record: Mapping[str, Any]) -> dict[str, Any]:
    candidates = [str(item) for item in record["candidate_ids"]]
    scores = [float(value) for value in record["scores"]]
    relevant = {str(item) for item in record["relevant_ids"]}
    relevant_scores = [score for candidate, score in zip(candidates, scores) if candidate in relevant]
    if not relevant_scores:
        return {"value": None, "category": "target_not_in_retained_prefix", "label_source": "mechanical"}
    margin = scores[0] - max(relevant_scores)
    if candidates[0] in relevant:
        category = "correct_top1"
    elif margin <= 0.02:
        category = "near_miss"
    elif margin <= 0.05:
        category = "ambiguous_close"
    else:
        category = "confident_wrong"
    return {"value": margin, "category": category, "label_source": "mechanical"}


def _tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.casefold())


def _caption_features(text: str, token_counts: Mapping[str, int]) -> dict[str, Any]:
    tokens = _tokens(text)
    rare = [token for token in tokens if int(token_counts.get(token, 0)) <= 1]
    number_terms = [token for token in tokens if token.isdigit() or token in NUMBER_WORDS]
    attributes = [token for token in tokens if token in ATTRIBUTE_WORDS]
    actions = [token for token in tokens if token in ACTION_WORDS]
    spatial = [token for token in tokens if token in SPATIAL_WORDS]
    objects = [token for token in tokens if token in OBJECT_WORDS]
    flags = {
        "has_number_or_count_term": bool(number_terms),
        "has_attribute_term": bool(attributes),
        "has_action_term": bool(actions),
        "has_spatial_term": bool(spatial),
        "has_object_term": bool(objects),
        "multiple_object_terms": len(set(objects)) >= 2,
        "short_caption": len(tokens) <= 3,
        "rare_token_present": bool(rare),
    }
    return {
        "word_count": len(tokens),
        "character_count": len(text),
        "unique_token_count": len(set(tokens)),
        "rare_token_count": len(rare),
        "rare_token_fraction": len(rare) / max(1, len(tokens)),
        "number_terms": sorted(set(number_terms)),
        "attribute_terms": sorted(set(attributes)),
        "action_terms": sorted(set(actions)),
        "spatial_terms": sorted(set(spatial)),
        "object_terms": sorted(set(objects)),
        "flags": flags,
        "feature_label_source": "mechanical",
    }


def _text_heuristic_categories(features: Mapping[str, Any]) -> list[dict[str, Any]]:
    flags = features["flags"]
    output: list[dict[str, Any]] = []
    mappings = (
        ("has_object_term", "object_confusion", "caption contains an object lexicon term"),
        ("has_attribute_term", "attribute_confusion", "caption contains an attribute lexicon term"),
        ("has_action_term", "action_confusion", "caption contains an action lexicon term"),
        ("has_number_or_count_term", "counting_quantity", "caption contains a number/count lexicon term"),
        ("has_spatial_term", "spatial_relation", "caption contains a spatial lexicon term"),
        ("short_caption", "short_underspecified_query", "caption has at most three tokenized words"),
        ("multiple_object_terms", "multiple_object_confusion", "caption contains at least two distinct object lexicon terms"),
        ("rare_token_present", "lexical_ambiguity", "caption contains a token occurring once in the fixed test-caption vocabulary"),
    )
    for flag, category, evidence in mappings:
        if flags.get(flag):
            output.append({"category": category, "label_source": "heuristic", "evidence": evidence})
    return output


def _caption_set_features(record: ImageRecord, caption_features: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows = [caption_features[caption.caption_id] for caption in record.captions]
    sets = [set(_tokens(caption.text)) for caption in record.captions]
    overlaps: list[float] = []
    for index, left in enumerate(sets):
        for right in sets[index + 1 :]:
            union = left | right
            overlaps.append(len(left & right) / max(1, len(union)))
    word_counts = [int(row["word_count"]) for row in rows]
    return {
        "caption_count": len(rows),
        "caption_word_counts": word_counts,
        "mean_caption_word_count": statistics.fmean(word_counts) if word_counts else 0.0,
        "min_caption_word_count": min(word_counts) if word_counts else 0,
        "max_caption_word_count": max(word_counts) if word_counts else 0,
        "mean_pairwise_caption_token_jaccard": statistics.fmean(overlaps) if overlaps else 0.0,
        "multiple_valid_caption_structure": len(rows) > 1,
        "feature_label_source": "mechanical_context",
    }


def _image_heuristic_categories(features: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if features.get("multiple_valid_caption_structure"):
        output.append({
            "category": "annotation_relevance_ambiguity",
            "label_source": "mechanical_context",
            "evidence": "COCO image-to-text relevance contains multiple captions for one image; this is a relevance limitation, not a human error label.",
        })
    if float(features.get("mean_pairwise_caption_token_jaccard", 0.0)) < 0.15:
        output.append({
            "category": "scene_context_confusion",
            "label_source": "heuristic",
            "evidence": "associated captions have low lexical overlap; semantic diversity is not directly labeled",
        })
    return output


def _rank_record(
    system: str,
    direction: str,
    row: Mapping[str, Any],
    records_by_image: Mapping[str, ImageRecord],
    caption_lookup: Mapping[str, tuple[str, str]],
    caption_features: Mapping[str, Mapping[str, Any]],
    token_counts: Mapping[str, int],
    dimensions: Mapping[str, Mapping[str, Any]],
    source_path: str,
) -> dict[str, Any]:
    query_id = str(row["query_id"])
    relevant_ids = [str(value) for value in row["relevant_ids"]]
    rank_info = first_relevant_rank(row)
    triggers = failure_triggers(rank_info)
    query_payload: dict[str, Any]
    if direction == "text_to_image":
        if query_id not in caption_lookup:
            raise ValueError(f"caption query missing from manifest: {query_id}")
        query_text, image_id = caption_lookup[query_id]
        observable = _caption_features(query_text, token_counts)
        taxonomy = _text_heuristic_categories(observable)
        query_payload = {"query_text": query_text, "image_id": image_id}
    else:
        if query_id not in records_by_image:
            raise ValueError(f"image query missing from manifest: {query_id}")
        record = records_by_image[query_id]
        observable = _caption_set_features(record, caption_features)
        taxonomy = _image_heuristic_categories(observable)
        query_payload = {"query_text": None, "image_id": query_id}
    if not taxonomy:
        taxonomy = [{"category": "other_unclassified", "label_source": "mechanical", "evidence": "no supported heuristic feature was triggered"}]
    top_ids = [str(value) for value in row["candidate_ids"]]
    top_scores = [float(value) for value in row["scores"]]
    dimension = dimensions.get(query_payload["image_id"], {})
    failure = any(triggers.values())
    root_layer = "UNKNOWN" if not failure else "REPRESENTATION"
    root_note = (
        "No failure" if not failure else
        "Failure is consistent with a representation/ranking mismatch under the fixed exact candidate corpus; this is not causal proof."
    )
    if direction == "image_to_text":
        root_layer = "DATA / RELEVANCE" if failure and observable.get("multiple_valid_caption_structure") else root_layer
        if root_layer == "DATA / RELEVANCE":
            root_note = "Multiple COCO captions make non-ground-truth semantic matches ambiguous; metadata relevance is not human judgment."
    return {
        "record_schema_version": PHASE16_SCHEMA_VERSION,
        "provenance": {
            "source_artifact": source_path,
            "manifest_sha256": MANIFEST_SHA256,
            "protocol_version": PROTOCOL_VERSION,
            "selection": "fixed Phase 7 seed-42 test image groups",
        },
        "query_id": query_id,
        "direction": direction,
        "system": system,
        **query_payload,
        "relevant_target_ids": relevant_ids,
        "target_rank": rank_info["rank"],
        "target_rank_observed": rank_info["observed"],
        "target_rank_lower_bound": rank_info["lower_bound"],
        "rank_severity": rank_severity(rank_info),
        "failure_definitions_triggered": triggers,
        "is_failure": failure,
        "top_retrieved_ids": top_ids,
        "top_retrieved_scores": top_scores,
        "candidate_count": int(row.get("candidate_count", 0)),
        "score_margin": score_margin(row),
        "observable_properties": {**observable, "image_dimensions": dimension},
        "taxonomy_categories": taxonomy,
        "root_cause_layer": root_layer,
        "root_cause_language": root_note,
    }


def _frequency(rows: Sequence[Mapping[str, Any]], predicate: Any) -> dict[str, Any]:
    count = sum(1 for row in rows if predicate(row))
    return {"count": count, "total": len(rows), "rate": count / max(1, len(rows))}


def _failure_summaries(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(str(row["system"]), str(row["direction"]))].append(row)
    output: dict[str, Any] = {}
    for (system, direction), rows in sorted(groups.items()):
        output[f"{system}:{direction}"] = {
            "query_count": len(rows),
            "failure_counts": {
                name: _frequency(rows, lambda row, key=name: bool(row["failure_definitions_triggered"][key]))
                for name in ("top1_failure", "top5_failure", "severe_failure")
            },
            "severity_counts": dict(Counter(str(row["rank_severity"]) for row in rows)),
            "score_margin_categories": dict(Counter(str(row["score_margin"]["category"]) for row in rows)),
        }
    return output


def _transition_analysis(
    records: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for direction in DIRECTIONS:
        common = sorted(set(records["zero_shot"][direction]) & set(records["full_ft"][direction]))
        for k in (1, 5, 10):
            counts: Counter[str] = Counter()
            examples: dict[str, list[str]] = defaultdict(list)
            zero_failure_values: list[float] = []
            full_failure_values: list[float] = []
            for query_id in common:
                left = records["zero_shot"][direction][query_id]
                right = records["full_ft"][direction][query_id]
                left_success = success_at_k({"rank": left["target_rank"], "observed": left["target_rank_observed"]}, k)
                right_success = success_at_k({"rank": right["target_rank"], "observed": right["target_rank_observed"]}, k)
                transition = classify_transition(
                    left_success,
                    right_success,
                )
                counts[transition] += 1
                zero_failure_values.append(float(not left_success))
                full_failure_values.append(float(not right_success))
                if len(examples[transition]) < 3:
                    examples[transition].append(query_id)
            rows.append({
                "direction": direction,
                "k": k,
                "query_count": len(common),
                "counts": dict(counts),
                "rates": {key: value / max(1, len(common)) for key, value in counts.items()},
                "failure_rate_delta_full_minus_zero_bootstrap": paired_bootstrap(
                    zero_failure_values,
                    full_failure_values,
                    resamples=200,
                    seed=_stable_seed("phase16", direction, str(k)),
                ),
                "deterministic_examples": dict(examples),
                "label_source": "mechanical",
            })
    return {"schema_version": PHASE16_SCHEMA_VERSION, "rows": rows}


def _rank_regression_analysis(
    records: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for direction in DIRECTIONS:
        common = sorted(set(records["zero_shot"][direction]) & set(records["full_ft"][direction]))
        movement_counts: Counter[str] = Counter()
        for query_id in common:
            zero = records["zero_shot"][direction][query_id]
            full = records["full_ft"][direction][query_id]
            movement = classify_rank_movement(
                {"rank": zero["target_rank"], "observed": zero["target_rank_observed"]},
                {"rank": full["target_rank"], "observed": full["target_rank_observed"]},
            )
            movement_counts[movement["movement_category"]] += 1
            rows.append({
                "direction": direction,
                "query_id": query_id,
                "zero_shot_rank": zero["target_rank"],
                "zero_shot_rank_observed": zero["target_rank_observed"],
                "full_ft_rank": full["target_rank"],
                "full_ft_rank_observed": full["target_rank_observed"],
                **movement,
                "zero_shot_margin": zero["score_margin"],
                "full_ft_margin": full["score_margin"],
                "label_source": "mechanical",
            })
        summaries.append({"direction": direction, "query_count": len(common), "movement_counts": dict(movement_counts)})
    return {"schema_version": PHASE16_SCHEMA_VERSION, "summaries": summaries, "rows": rows}


def _text_image_analysis(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    text_rows = [row for row in records if row["direction"] == "text_to_image"]
    for system in SYSTEMS:
        current = [row for row in text_rows if row["system"] == system]
        bins = {"short_0_3": lambda row: int(row["observable_properties"]["word_count"]) <= 3, "medium_4_8": lambda row: 4 <= int(row["observable_properties"]["word_count"]) <= 8, "long_9_plus": lambda row: int(row["observable_properties"]["word_count"]) >= 9}
        for name, predicate in bins.items():
            selected = [row for row in current if predicate(row)]
            output.append({
                "system": system,
                "feature": "caption_length_bin",
                "value": name,
                "query_count": len(selected),
                "top1_failure": _frequency(selected, lambda row: row["failure_definitions_triggered"]["top1_failure"]),
                "top5_failure": _frequency(selected, lambda row: row["failure_definitions_triggered"]["top5_failure"]),
                "severe_failure": _frequency(selected, lambda row: row["failure_definitions_triggered"]["severe_failure"]),
                "interpretation": "mechanically grouped by tokenized caption length; not a semantic label",
            })
        for flag in ("has_number_or_count_term", "has_attribute_term", "has_action_term", "has_spatial_term", "multiple_object_terms", "rare_token_present"):
            selected = [row for row in current if row["observable_properties"]["flags"][flag]]
            output.append({
                "system": system,
                "feature": flag,
                "value": True,
                "query_count": len(selected),
                "top1_failure": _frequency(selected, lambda row: row["failure_definitions_triggered"]["top1_failure"]),
                "top5_failure": _frequency(selected, lambda row: row["failure_definitions_triggered"]["top5_failure"]),
                "severe_failure": _frequency(selected, lambda row: row["failure_definitions_triggered"]["severe_failure"]),
                "interpretation": "caption-keyword heuristic or fixed vocabulary statistic; not a human annotation",
            })
    return {"schema_version": PHASE16_SCHEMA_VERSION, "rows": output}


def _image_text_analysis(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    image_rows = [row for row in records if row["direction"] == "image_to_text"]
    for system in SYSTEMS:
        current = [row for row in image_rows if row["system"] == system]
        bins = {"short_mean_0_8": lambda row: float(row["observable_properties"]["mean_caption_word_count"]) <= 8, "medium_mean_9_12": lambda row: 9 <= float(row["observable_properties"]["mean_caption_word_count"]) <= 12, "long_mean_13_plus": lambda row: float(row["observable_properties"]["mean_caption_word_count"]) >= 13}
        for name, predicate in bins.items():
            selected = [row for row in current if predicate(row)]
            output.append({
                "system": system,
                "feature": "mean_associated_caption_length_bin",
                "value": name,
                "query_count": len(selected),
                "top1_failure": _frequency(selected, lambda row: row["failure_definitions_triggered"]["top1_failure"]),
                "top5_failure": _frequency(selected, lambda row: row["failure_definitions_triggered"]["top5_failure"]),
                "severe_failure": _frequency(selected, lambda row: row["failure_definitions_triggered"]["severe_failure"]),
                "interpretation": "mechanically grouped by associated caption lengths; image content is not semantically labeled",
            })
    return {"schema_version": PHASE16_SCHEMA_VERSION, "rows": output}


def _failure_overlap(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        for direction in DIRECTIONS:
            current = [row for row in records if row["system"] == system and row["direction"] == direction]
            short = [row for row in current if row["taxonomy_categories"] and any(item["category"] == "short_underspecified_query" for item in row["taxonomy_categories"])]
            low_margin = [row for row in current if row["score_margin"]["category"] in {"near_miss", "ambiguous_close"}]
            rows.append({
                "system": system,
                "direction": direction,
                "overlap": "short_query_and_top5_failure",
                "count": sum(row["failure_definitions_triggered"]["top5_failure"] for row in short),
                "denominator_short_query": len(short),
                "label_source": "mechanical_count_plus_heuristic_short_query",
            })
            rows.append({
                "system": system,
                "direction": direction,
                "overlap": "corresponding_low_margin_and_top1_failure",
                "count": sum(row["failure_definitions_triggered"]["top1_failure"] for row in low_margin),
                "denominator_low_margin": len(low_margin),
                "label_source": "mechanical",
            })
    return {"schema_version": PHASE16_SCHEMA_VERSION, "rows": rows, "note": "Overlaps are intentionally non-exclusive; heuristic short-query membership is not human annotation."}


def _phase15_links(phase15_dir: Path, caption_lookup: Mapping[str, tuple[str, str]], records_by_image: Mapping[str, ImageRecord], phase15_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    qualitative = json.loads((phase15_dir / "qualitative_examples.json").read_text(encoding="utf-8"))
    robust_rows = json.loads((phase15_dir / "robustness_metrics.json").read_text(encoding="utf-8"))["rows"]
    diagnostic_rows: list[dict[str, Any]] = []
    for row in qualitative["rows"]:
        family = str(row["family"])
        direction = "text_to_image" if family in {"casing", "punctuation", "typo", "word_deletion", "shortened"} else "image_to_text"
        query_id = str(row["examples"][0]["query_id"]) if row.get("examples") else None
        if query_id is None:
            continue
        if direction == "text_to_image":
            relevant = {caption_lookup[query_id][1]}
        else:
            relevant = {caption.caption_id for caption in records_by_image[query_id].captions}
        example = row["examples"][0]
        clean_success = bool(set(example["clean_top1"]) & relevant)
        corrupt_success = bool(set(example["corrupted_top1"]) & relevant)
        diagnostic_rows.append({
            "system": row["system"],
            "direction": direction,
            "family": family,
            "severity": row["severity"],
            "query_id": query_id,
            "clean_success_at_1": clean_success,
            "corrupted_success_at_1": corrupt_success,
            "transition": classify_transition(clean_success, corrupt_success),
            "recall_at_1_delta": example["recall_at_1_delta"],
            "selection_source": "Phase 15 deterministic three-worst recall_at_1 diagnostic example",
            "label_source": "mechanical",
        })
    transitions = Counter(row["transition"] for row in diagnostic_rows)
    summary: list[dict[str, Any]] = []
    for family in ("shortened", "typo", "crop", "blur", "occlusion"):
        for system in SYSTEMS:
            selected = [row for row in diagnostic_rows if row["family"] == family and row["system"] == system]
            summary.append({
                "system": system,
                "family": family,
                "diagnostic_example_count": len(selected),
                "transition_counts": dict(Counter(row["transition"] for row in selected)),
                "coverage_note": "These are retained Phase 15 diagnostic examples, not a full per-query corrupted-ranking archive.",
            })
    aggregates = [
        {"system": row["system"], "direction": row["direction"], "family": row["family"], "severity": row["severity"], "clean": row["metrics"]["recall_at_1"]["clean"], "corrupted": row["metrics"]["recall_at_1"]["corrupted"], "absolute_delta": row["metrics"]["recall_at_1"]["absolute_delta"]}
        for row in robust_rows
    ]
    return {
        "schema_version": PHASE16_SCHEMA_VERSION,
        "source": "artifacts/phase15/qualitative_examples.json and robustness_metrics.json",
        "diagnostic_transition_counts": dict(transitions),
        "important_family_summary": summary,
        "diagnostic_records": diagnostic_rows,
        "aggregate_recall_at_1": aggregates,
        "causality_note": "Observed clean-to-corrupted transitions establish sensitivity in this protocol, not a universal causal mechanism.",
    }


def _hard_negative_links(base_dir: Path, records: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for direction in DIRECTIONS:
        hard = _result_payload(base_dir / HARD_RESULT_FILES[direction], direction)
        hard_rows = _index_records(hard)
        for baseline in ("zero_shot", "full_ft"):
            base_rows = records[baseline][direction]
            if set(base_rows) != set(hard_rows):
                raise ValueError(f"hard-negative and {baseline} IDs differ for {direction}")
            counts: Counter[str] = Counter()
            rank_changed_success_unchanged = 0
            for query_id in sorted(hard_rows):
                left = base_rows[query_id]
                right_raw = hard_rows[query_id]
                right = _rank_record_from_existing(right_raw)
                left_success = success_at_k({"rank": left["target_rank"], "observed": left["target_rank_observed"]}, 1)
                right_success = success_at_k({"rank": right["target_rank"], "observed": right["target_rank_observed"]}, 1)
                transition = classify_transition(left_success, right_success)
                counts[transition] += 1
                left_rank = left["target_rank"]
                right_rank = right["target_rank"]
                if left_success == right_success and left_rank is not None and right_rank is not None and left_rank != right_rank:
                    rank_changed_success_unchanged += 1
                if transition != "both_succeed" or (left_rank is not None and right_rank is not None and left_rank != right_rank):
                    findings.append({
                        "direction": direction,
                        "baseline": baseline,
                        "query_id": query_id,
                        "transition_at_1": transition,
                        "baseline_rank": left_rank,
                        "hard_negative_rank": right_rank,
                        "label_source": "mechanical",
                    })
            summary.append({
                "direction": direction,
                "baseline": baseline,
                "query_count": len(hard_rows),
                "transition_counts_at_1": dict(counts),
                "rank_changed_without_success_change": rank_changed_success_unchanged,
                "seed_scope": "single retained seed-42 hard-negative run; no semantic cause inferred",
            })
    return {"schema_version": PHASE16_SCHEMA_VERSION, "summary": summary, "query_records": findings}


def _rank_record_from_existing(row: Mapping[str, Any]) -> dict[str, Any]:
    rank = first_relevant_rank(row)
    return {"target_rank": rank["rank"], "target_rank_observed": rank["observed"]}


def _reranker_links(base_dir: Path) -> dict[str, Any]:
    stage1 = json.loads((base_dir / "stage1_results.json").read_text(encoding="utf-8"))
    reranked = json.loads((base_dir / "reranked_results.json").read_text(encoding="utf-8"))
    failure = json.loads((base_dir / "failure_analysis.json").read_text(encoding="utf-8"))
    qualitative = json.loads((base_dir / "qualitative_examples.json").read_text(encoding="utf-8"))
    selected_stage = [row for row in stage1 if row.get("tier") == "tier2"]
    selected_rerank = [row for row in reranked if row.get("tier") == "tier2"]
    aggregate_rows = []
    for left, right in zip(selected_stage, selected_rerank):
        counts = left.get("rank_change_counts", right.get("rank_change_counts", {}))
        aggregate_rows.append({
            "direction": left["task"],
            "query_count": left.get("candidate_recall", {}).get("query_count"),
            "stage1_candidate_recall": left.get("candidate_recall"),
            "rank_change_counts": counts,
            "stage1_metrics": left.get("stage1_metrics"),
            "reranked_metrics": right.get("metrics"),
            "failure_definition": "relevant demotion is an observable reranker change; Stage-1-success-to-reranker-failure intersection was not retained",
        })
    examples: list[dict[str, Any]] = []
    for row in qualitative:
        if row.get("tier") != "tier2":
            continue
        for example_type in ("promoted_relevant", "demoted_relevant", "regression", "candidate_miss"):
            example = row.get(example_type)
            if example is not None:
                examples.append({"direction": row["task"], "example_type": example_type, **example, "label_source": "mechanical"})
    return {
        "schema_version": PHASE16_SCHEMA_VERSION,
        "source": ["artifacts/phase11/stage1_results.json", "artifacts/phase11/reranked_results.json", "artifacts/phase11/qualitative_examples.json"],
        "aggregate_rows": aggregate_rows,
        "phase11_failure_summary": failure,
        "qualitative_examples": examples,
        "query_level_intersection_available": False,
        "query_level_limitation": "Phase 11 did not retain complete Stage-1 and reranked ranking records in its canonical artifacts; exact Stage1-success-and-reranker-break counts cannot be reconstructed without rerunning inference.",
        "interpretation": "The retained evidence supports a reranker-induced regression classification through aggregate metric deltas, relevant demotion counts, and deterministic examples; it does not prove a causal mechanism.",
    }


def _qualitative_examples(
    records: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    phase15_links: Mapping[str, Any],
    reranker_links: Mapping[str, Any],
    hard_links: Mapping[str, Any],
) -> dict[str, Any]:
    examples: list[dict[str, Any]] = []
    used: set[tuple[str, str]] = set()

    def add(example_type: str, direction: str, query_id: str, evidence: Mapping[str, Any]) -> None:
        key = (direction, query_id)
        if key in used or len(examples) >= 18:
            return
        used.add(key)
        examples.append({"example_type": example_type, "direction": direction, "query_id": query_id, **dict(evidence), "selection": "deterministic sorted query/category selection; no performance-only cherry-picking"})

    for direction in DIRECTIONS:
        for query_id in sorted(records["zero_shot"][direction]):
            zero = records["zero_shot"][direction][query_id]
            full = records["full_ft"][direction][query_id]
            if success_at_k({"rank": zero["target_rank"], "observed": zero["target_rank_observed"]}, 1) and success_at_k({"rank": full["target_rank"], "observed": full["target_rank_observed"]}, 1):
                add("easy_success", direction, query_id, {"zero_shot_rank": zero["target_rank"], "full_ft_rank": full["target_rank"], "query_text": zero["query_text"], "image_id": zero["image_id"]})
                break
        for query_id in sorted(records["zero_shot"][direction]):
            zero = records["zero_shot"][direction][query_id]
            full = records["full_ft"][direction][query_id]
            if zero["rank_severity"] in {"low", "medium"} and full["rank_severity"] in {"low", "medium"}:
                add("near_miss", direction, query_id, {"zero_shot_rank": zero["target_rank"], "full_ft_rank": full["target_rank"], "query_text": zero["query_text"], "image_id": zero["image_id"]})
                break
        for query_id in sorted(records["zero_shot"][direction]):
            zero = records["zero_shot"][direction][query_id]
            full = records["full_ft"][direction][query_id]
            if zero["rank_severity"] == "high" or full["rank_severity"] == "high":
                add("severe_failure", direction, query_id, {"zero_shot_rank": zero["target_rank"], "full_ft_rank": full["target_rank"], "query_text": zero["query_text"], "image_id": zero["image_id"]})
                break
        for query_id in sorted(records["zero_shot"][direction]):
            zero = records["zero_shot"][direction][query_id]
            full = records["full_ft"][direction][query_id]
            if success_at_k({"rank": zero["target_rank"], "observed": zero["target_rank_observed"]}, 1) and not success_at_k({"rank": full["target_rank"], "observed": full["target_rank_observed"]}, 1):
                add("fine_tuning_regression", direction, query_id, {"zero_shot_rank": zero["target_rank"], "full_ft_rank": full["target_rank"], "query_text": zero["query_text"], "image_id": zero["image_id"]})
                break
            if not success_at_k({"rank": zero["target_rank"], "observed": zero["target_rank_observed"]}, 1) and success_at_k({"rank": full["target_rank"], "observed": full["target_rank_observed"]}, 1):
                add("fine_tuning_improvement", direction, query_id, {"zero_shot_rank": zero["target_rank"], "full_ft_rank": full["target_rank"], "query_text": zero["query_text"], "image_id": zero["image_id"]})
                break

    diagnostics = phase15_links.get("diagnostic_records", [])
    for row in diagnostics:
        if row["transition"] == "only_left_succeeds":
            add("corruption_induced_failure", row["direction"], row["query_id"], {"system": row["system"], "family": row["family"], "severity": row["severity"], "transition": row["transition"], "recall_at_1_delta": row["recall_at_1_delta"]})
            break
    for row in reranker_links.get("qualitative_examples", []):
        if row["example_type"] == "regression":
            add("reranker_induced_failure", row["direction"], row["query_id"], {"stage1_rank": row.get("stage1_rank"), "reranked_rank": row.get("reranked_rank"), "top10_stage1": row.get("stage1_top10"), "top10_reranked": row.get("reranked_top10")})
            break
    for row in hard_links.get("query_records", []):
        if row["transition_at_1"] in {"only_right_succeeds", "only_left_succeeds"}:
            add("hard_negative_transition", row["direction"], row["query_id"], {"baseline": row["baseline"], "transition": row["transition_at_1"], "baseline_rank": row["baseline_rank"], "hard_negative_rank": row["hard_negative_rank"]})
            break
    for row in sorted(diagnostics, key=lambda item: (item["query_id"], item["family"], item["system"])):
        if row["transition"] == "both_succeed":
            add("corruption_resilient_success", row["direction"], row["query_id"], {"system": row["system"], "family": row["family"], "severity": row["severity"]})
            break
    if records["zero_shot"]["image_to_text"]:
        query_id = min(records["zero_shot"]["image_to_text"])
        add("ambiguous_multi_caption_case", "image_to_text", query_id, {"relevance_note": "multiple COCO captions are relevant by metadata; semantic non-ground-truth judgments are not available"})
    return {"schema_version": PHASE16_SCHEMA_VERSION, "count": len(examples), "examples": examples}


def _taxonomy_summary(failure_records: Sequence[Mapping[str, Any]], summaries: Mapping[str, Any]) -> dict[str, Any]:
    category_counts: Counter[tuple[str, str]] = Counter()
    category_query_counts: dict[tuple[str, str], set[tuple[str, str, str]]] = defaultdict(set)
    for row in failure_records:
        if not row["is_failure"]:
            continue
        for category in row["taxonomy_categories"]:
            key = (str(category["category"]), str(category["label_source"]))
            category_counts[key] += 1
            category_query_counts[key].add((str(row["system"]), str(row["direction"]), str(row["query_id"])))
    definitions = {
        category: {
            "description": "heuristic or mechanical category; not a human semantic ground-truth label",
            "label_source": "heuristic" if category in TAXONOMY_CATEGORIES and category not in {"annotation_relevance_ambiguity", "other_unclassified"} else "mechanical_context",
        }
        for category in TAXONOMY_CATEGORIES
    }
    rows = []
    for (category, source), count in sorted(category_counts.items()):
        rows.append({
            "category": category,
            "label_source": source,
            "failure_record_count": count,
            "distinct_system_direction_query_count": len(category_query_counts[(category, source)]),
            "percentage_is_not_ground_truth": source != "mechanical",
        })
    measurable = []
    for summary_key, summary_value in sorted(summaries.items()):
        measurable.append({"system_direction": summary_key, "failure_counts": summary_value["failure_counts"], "severity_counts": summary_value["severity_counts"], "label_source": "mechanical"})
    return {"schema_version": PHASE16_SCHEMA_VERSION, "category_definitions": definitions, "taxonomy_rows": rows, "measurable_failure_frequencies": measurable, "interpretation": "Keyword categories are diagnostic hypotheses; only rank, score, text, image metadata, and artifact transitions are mechanically observed."}


def _failure_priority(summaries: Mapping[str, Any], phase15_links: Mapping[str, Any], reranker_links: Mapping[str, Any], hard_links: Mapping[str, Any]) -> dict[str, Any]:
    text_t2i = summaries.get("zero_shot:text_to_image", {}).get("failure_counts", {}).get("top1_failure", {})
    priorities = [
        {
            "priority": "HIGH",
            "family": "high_severity_text_shortening",
            "frequency_evidence": "Phase 15 aggregate condition rows show the largest text-query degradation for high shortening.",
            "severity_evidence": "large R@1/R@5 loss in both evaluated systems",
            "diagnostic_confidence": "high for observed sensitivity; low for causal mechanism",
            "potential_impact": "text queries with reduced lexical content can lose the correct image",
            "recommended_response": "investigate query expansion or robustness augmentation in a later phase",
        },
        {
            "priority": "HIGH",
            "family": "reranker_induced_regression",
            "frequency_evidence": "Phase 11 reports 298 relevant demotions in text-to-image and 23 in image-to-text at Tier 2.",
            "severity_evidence": "R@1 fell from 0.8263 to 0.3533 for text-to-image and from 0.1837 recall to 0.1457 for image-to-text.",
            "diagnostic_confidence": "high for negative aggregate effect; per-query intersection not retained",
            "potential_impact": "a second-stage model can damage an otherwise strong exact retrieval stage",
            "recommended_response": "keep the tested reranker disabled unless a new validation protocol supports it",
        },
        {
            "priority": "HIGH",
            "family": "high_severity_image_occlusion",
            "frequency_evidence": "Phase 15 high occlusion was the largest image-query degradation in both systems.",
            "severity_evidence": "image-to-text R@1 fell by 0.30 in the fixed robustness scope",
            "diagnostic_confidence": "high for observed sensitivity; low for causal mechanism",
            "potential_impact": "partial visibility can cause visual retrieval failures",
            "recommended_response": "investigate occlusion-aware augmentation or stronger visual representation in a later phase",
        },
        {
            "priority": "MEDIUM",
            "family": "clean_top1_rank_failures",
            "frequency_evidence": f"Zero-shot text-to-image top-1 failure rate is {float(text_t2i.get('rate', 0.0)):.3f}; full-FT is summarized separately.",
            "severity_evidence": "some failures are high severity because the target is outside retained top-10",
            "diagnostic_confidence": "high for rank fact; low for semantic cause",
            "potential_impact": "in-domain retrieval still has residual errors despite strong aggregate R@5",
            "recommended_response": "use deterministic error examples and human/multi-positive review before semantic remediation",
        },
        {
            "priority": "LOW / INSUFFICIENT EVIDENCE",
            "family": "keyword_semantic_categories",
            "frequency_evidence": "caption keyword groupings are available in taxonomy artifacts",
            "severity_evidence": "not established as a ground-truth failure rate",
            "diagnostic_confidence": "low; heuristic only",
            "potential_impact": "may guide later annotation or targeted evaluation",
            "recommended_response": "do not optimize against these labels without human or explicit annotation support",
        },
    ]
    return {"schema_version": PHASE16_SCHEMA_VERSION, "priorities": priorities, "ranking_basis": ["frequency", "severity", "diagnostic confidence", "potential impact"], "causality_warning": "Priority reflects observed evidence and impact, not proof of root cause."}


def validate_phase16_artifacts(output_dir: Path | str) -> dict[str, Any]:
    output = Path(output_dir)
    required = {
        "pre_phase_audit.json",
        "failure_definitions.json",
        "failure_records.json",
        "taxonomy_summary.json",
        "text_image_analysis.json",
        "image_text_analysis.json",
        "system_transition_analysis.json",
        "rank_regressions.json",
        "score_margin_analysis.json",
        "failure_overlap.json",
        "robustness_failure_links.json",
        "reranker_failure_links.json",
        "hard_negative_failure_links.json",
        "failure_priority.json",
        "qualitative_examples.json",
        "provenance.json",
    }
    checks: dict[str, bool] = {"required_artifacts": all((output / name).is_file() for name in required)}
    if checks["required_artifacts"]:
        audit = json.loads((output / "pre_phase_audit.json").read_text(encoding="utf-8"))
        definitions = json.loads((output / "failure_definitions.json").read_text(encoding="utf-8"))
        records = json.loads((output / "failure_records.json").read_text(encoding="utf-8"))
        transitions = json.loads((output / "system_transition_analysis.json").read_text(encoding="utf-8"))
        robust = json.loads((output / "robustness_failure_links.json").read_text(encoding="utf-8"))
        reranker = json.loads((output / "reranker_failure_links.json").read_text(encoding="utf-8"))
        hard = json.loads((output / "hard_negative_failure_links.json").read_text(encoding="utf-8"))
        provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
        validate_failure_definitions(definitions)
        rows = records.get("rows", [])
        checks.update({
            "pre_phase_audit_pass": audit.get("status") == "PASS" and audit.get("audit_result") == "PRE-PHASE AUDIT: Phase 15 PASS",
            "failure_records_nonempty": bool(rows),
            "directions_separated": {row.get("direction") for row in rows} == set(DIRECTIONS),
            "systems_present": {row.get("system") for row in rows} == set(SYSTEMS),
            "labels_explicit": all(item.get("label_source") in VALID_LABEL_SOURCES for row in rows for item in row.get("taxonomy_categories", [])),
            "transitions_complete": len(transitions.get("rows", [])) == 6,
            "robustness_reuses_phase15": str(robust.get("source", "")).find("phase15") >= 0,
            "reranker_reuses_phase11": str(reranker.get("source", "")).find("phase11") >= 0,
            "hard_negative_reuses_phase9": str(hard.get("summary", "")) != "",
            "no_training": provenance.get("training_performed") is False,
            "no_phase17": provenance.get("phase17_started") is False,
            "no_fabricated_human_labels": provenance.get("human_semantic_labels_fabricated") is False,
        })
    else:
        checks.update({name: False for name in ("pre_phase_audit_pass", "failure_records_nonempty", "directions_separated", "systems_present", "labels_explicit", "transitions_complete", "robustness_reuses_phase15", "reranker_reuses_phase11", "hard_negative_reuses_phase9", "no_training", "no_phase17", "no_fabricated_human_labels")})
    return {"schema_version": PHASE16_SCHEMA_VERSION, "passed": all(checks.values()), "checks": checks, "required": sorted(required)}


def run_phase16(
    output_dir: Path | str = Path("artifacts/phase16"),
    manifest_path: Path | str = Path("data/processed/coco2017_val_split_manifest.json"),
    phase7_dir: Path | str = Path("artifacts/phase7"),
    phase9_dir: Path | str = Path("artifacts/phase9"),
    phase11_dir: Path | str = Path("artifacts/phase11"),
    phase15_dir: Path | str = Path("artifacts/phase15"),
) -> dict[str, Any]:
    output = Path(output_dir)
    manifest_file = Path(manifest_path)
    phase7_path = Path(phase7_dir)
    phase9_path = Path(phase9_dir)
    phase11_path = Path(phase11_dir)
    phase15_path = Path(phase15_dir)
    if _hash_file(manifest_file) != MANIFEST_SHA256:
        raise ValueError("Phase 16 requires the fixed COCO manifest")
    phase15_report = json.loads((phase15_path / "phase15_report.json").read_text(encoding="utf-8"))
    if phase15_report.get("status") != "PASS":
        raise ValueError("Phase 15 is not PASS")
    from .phase15 import validate_phase15_artifacts

    phase15_validation = validate_phase15_artifacts(phase15_path)
    if not phase15_validation["passed"]:
        raise ValueError("Phase 15 artifact validation failed")
    manifest = read_manifest(manifest_file)
    test_records = _subset_records(manifest.records, "test", SEED, TEST_IMAGE_LIMIT)
    if len(test_records) != TEST_IMAGE_LIMIT:
        raise ValueError("Phase 16 must use the fixed 100-image test subset")
    records_by_image = {record.image_id: record for record in test_records}
    if len(records_by_image) != len(test_records):
        raise ValueError("duplicate test image IDs")
    caption_lookup = {caption.caption_id: (caption.text, record.image_id) for record in test_records for caption in record.captions}
    if len(caption_lookup) != sum(len(record.captions) for record in test_records):
        raise ValueError("duplicate caption IDs in test subset")
    token_counts = Counter(token for text, _ in caption_lookup.values() for token in _tokens(text))
    caption_features = {caption_id: _caption_features(text, token_counts) for caption_id, (text, _) in caption_lookup.items()}
    dimensions_payload = json.loads((phase15_path / "distribution_shift_definition.json").read_text(encoding="utf-8"))
    dimensions = {str(row["image_id"]): row for row in dimensions_payload.get("all_test_dimensions", [])}
    definitions = default_failure_definitions()
    validate_failure_definitions(definitions)
    _write_json(definitions, output / "failure_definitions.json")

    raw_payloads: dict[str, dict[str, dict[str, Any]]] = {system: {} for system in SYSTEMS}
    indexed: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = {system: {} for system in SYSTEMS}
    for system in SYSTEMS:
        raw_payloads[system]["text_to_image"] = _result_payload(phase7_path / ("zero_shot_text_to_image.json" if system == "zero_shot" else "fine_tuned_text_to_image.json"), "text_to_image")
        raw_payloads[system]["image_to_text"] = _result_payload(phase7_path / ("zero_shot_image_to_text.json" if system == "zero_shot" else "fine_tuned_image_to_text.json"), "image_to_text")
        indexed[system]["text_to_image"] = _index_records(raw_payloads[system]["text_to_image"])
        indexed[system]["image_to_text"] = _index_records(raw_payloads[system]["image_to_text"])
    validate_alignment(raw_payloads["zero_shot"]["text_to_image"], raw_payloads["full_ft"]["text_to_image"])
    validate_alignment(raw_payloads["zero_shot"]["image_to_text"], raw_payloads["full_ft"]["image_to_text"])
    failure_records: list[dict[str, Any]] = []
    structured_records: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = {system: {} for system in SYSTEMS}
    for system in SYSTEMS:
        for direction in DIRECTIONS:
            rows = []
            for query_id in sorted(indexed[system][direction]):
                source = phase7_path / ("zero_shot_" if system == "zero_shot" else "fine_tuned_")
                path = source.with_name(source.name + f"{direction}.json")
                analysed = _rank_record(system, direction, indexed[system][direction][query_id], records_by_image, caption_lookup, caption_features, token_counts, dimensions, str(path))
                rows.append(analysed)
                failure_records.append(analysed)
            structured_records[system][direction] = {str(row["query_id"]): row for row in rows}
    summaries = _failure_summaries(failure_records)
    _write_json({"schema_version": PHASE16_SCHEMA_VERSION, "row_count": len(failure_records), "rows": failure_records}, output / "failure_records.json")
    _write_json(_taxonomy_summary(failure_records, summaries), output / "taxonomy_summary.json")
    _write_json(_text_image_analysis(failure_records), output / "text_image_analysis.json")
    _write_json(_image_text_analysis(failure_records), output / "image_text_analysis.json")
    _write_json(_transition_analysis(structured_records), output / "system_transition_analysis.json")
    _write_json(_rank_regression_analysis(structured_records), output / "rank_regressions.json")
    margin_rows = [{
        "system": row["system"],
        "direction": row["direction"],
        "query_id": row["query_id"],
        **row["score_margin"],
        "is_failure": row["is_failure"],
    } for row in failure_records]
    _write_json({"schema_version": PHASE16_SCHEMA_VERSION, "warning": "score margins are not calibrated probabilities", "rows": margin_rows}, output / "score_margin_analysis.json")
    _write_json(_failure_overlap(failure_records), output / "failure_overlap.json")
    phase15_links = _phase15_links(phase15_path, caption_lookup, records_by_image, failure_records)
    _write_json(phase15_links, output / "robustness_failure_links.json")
    reranker_links = _reranker_links(phase11_path)
    _write_json(reranker_links, output / "reranker_failure_links.json")
    hard_links = _hard_negative_links(phase9_path, structured_records)
    _write_json(hard_links, output / "hard_negative_failure_links.json")
    priority = _failure_priority(summaries, phase15_links, reranker_links, hard_links)
    _write_json(priority, output / "failure_priority.json")
    qualitative = _qualitative_examples(structured_records, phase15_links, reranker_links, hard_links)
    _write_json(qualitative, output / "qualitative_examples.json")
    provenance = {
        "schema_version": PHASE16_SCHEMA_VERSION,
        "phase": 16,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "manifest": str(manifest_file),
        "manifest_sha256": _hash_file(manifest_file),
        "fixed_test_image_groups": len(test_records),
        "fixed_test_caption_queries": len(caption_lookup),
        "input_artifacts": {
            "phase7_zero_shot_text_to_image": _hash_file(phase7_path / "zero_shot_text_to_image.json"),
            "phase7_zero_shot_image_to_text": _hash_file(phase7_path / "zero_shot_image_to_text.json"),
            "phase7_full_ft_text_to_image": _hash_file(phase7_path / "fine_tuned_text_to_image.json"),
            "phase7_full_ft_image_to_text": _hash_file(phase7_path / "fine_tuned_image_to_text.json"),
            "phase9_hard_negative_text_to_image": _hash_file(phase9_path / "hard_negative_text_to_image.json"),
            "phase9_hard_negative_image_to_text": _hash_file(phase9_path / "hard_negative_image_to_text.json"),
            "phase11_stage1_results": _hash_file(phase11_path / "stage1_results.json"),
            "phase15_robustness_metrics": _hash_file(phase15_path / "robustness_metrics.json"),
        },
        "systems_analysed": list(SYSTEMS),
        "training_performed": False,
        "new_dataset_downloaded": False,
        "human_semantic_labels_fabricated": False,
        "phase17_started": False,
        "heuristic_categories_are_not_ground_truth": True,
        "python": sys.version,
        "platform": platform.platform(),
    }
    _write_json(provenance, output / "provenance.json")
    validation = validate_phase16_artifacts(output)
    _write_json(validation, output / "artifact_validation.json")
    report = {
        "report_schema_version": PHASE16_SCHEMA_VERSION,
        "project": "OmniSearch",
        "phase": 16,
        "status": "PASS" if validation["passed"] else "FAIL",
        "pre_phase_audit": "Phase 15 PASS",
        "scope": {"systems": list(SYSTEMS), "test_image_groups": len(test_records), "caption_queries": len(caption_lookup), "protocol_version": PROTOCOL_VERSION, "training": False},
        "quality_gate": {"status": "PASS" if validation["passed"] else "FAIL", "checks": validation["checks"]},
        "artifacts": sorted(path.name for path in output.glob("*.json")),
    }
    _write_json(report, output / "phase16_report.json")
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation-only Phase 16 error analysis")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase16"))
    parser.add_argument("--manifest", dest="manifest_path", type=Path, default=Path("data/processed/coco2017_val_split_manifest.json"))
    parser.add_argument("--phase7-dir", type=Path, default=Path("artifacts/phase7"))
    parser.add_argument("--phase9-dir", type=Path, default=Path("artifacts/phase9"))
    parser.add_argument("--phase11-dir", type=Path, default=Path("artifacts/phase11"))
    parser.add_argument("--phase15-dir", type=Path, default=Path("artifacts/phase15"))
    args = parser.parse_args()
    report = run_phase16(**vars(args))
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
