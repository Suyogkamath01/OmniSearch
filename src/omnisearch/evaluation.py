"""Canonical retrieval evaluation for OmniSearch.

This module is the single metric/protocol implementation for future phases.
It evaluates already-produced rankings; it does not train models or build
retrieval indexes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import random
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG_PATH, load_config
from .manifest import ImageRecord, read_manifest
from .splitting import assert_no_split_leakage

PROTOCOL_VERSION = "retrieval_eval_v1"
RESULT_SCHEMA_VERSION = 1
TASKS = ("text_to_image", "image_to_text", "text_to_text", "image_to_image")
DEFAULT_KS = (1, 5, 10)

TASK_DEFINITIONS: dict[str, dict[str, Any]] = {
    "text_to_image": {
        "query_unit": "caption",
        "candidate_unit": "image_group",
        "relevance": "the image associated with the query caption is relevant",
        "self_match": "not_applicable",
        "metric_note": "Recall@K and rank statistics are primary; precision is optional and secondary",
    },
    "image_to_text": {
        "query_unit": "image_group",
        "candidate_unit": "caption",
        "relevance": "all legitimate captions associated with the query image are relevant",
        "self_match": "not_applicable",
        "metric_note": "Recall@K and first-relevant rank handle multiple relevant captions",
    },
    "text_to_text": {
        "query_unit": "caption",
        "candidate_unit": "caption",
        "relevance": "explicit task-defined related-caption set; for COCO, same-image captions only",
        "self_match": "excluded by producer when appropriate",
        "metric_note": "Use only when the related-caption definition is declared",
    },
    "image_to_image": {
        "query_unit": "image_group",
        "candidate_unit": "image_group",
        "relevance": "not provided by COCO caption metadata; no semantic labels are invented",
        "self_match": "excluded by producer when appropriate",
        "metric_note": "Ranking can be inspected, but relevance metrics are not applicable without labels",
    },
}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("K must be a positive integer")


def _normalize_ks(ks: Sequence[int]) -> tuple[int, ...]:
    """Validate and deterministically deduplicate requested cutoffs."""

    normalized: list[int] = []
    for k in ks:
        _validate_k(k)
        normalized.append(k)
    values = tuple(sorted(set(normalized)))
    if not values:
        raise ValueError("at least one K value is required")
    return values


def make_protocol(task: str, ks: Sequence[int] = DEFAULT_KS) -> dict[str, Any]:
    if task not in TASKS:
        raise ValueError(f"unknown retrieval task: {task}")
    normalized_ks = _normalize_ks(ks)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "task": task,
        "task_definition": TASK_DEFINITIONS[task],
        "k_values": list(normalized_ks),
        "score_direction": "higher_is_better",
        "tie_policy": "score_desc_then_candidate_id_asc",
        "aggregation": "macro_query_level",
        "candidate_filter": "declared candidate corpus; ranked list may be truncated",
        "precision_definition": "hits in the returned prefix divided by K; omitted candidates count as non-hits",
        "average_precision_definition": "AP uses the declared relevance set as denominator over the returned ranked list; report is truncated when the list is truncated",
        "query_filter": "queries with declared relevance are evaluated; no-relevance queries are reported separately",
        "rank_statistic": "first relevant rank; misses are counted separately and excluded from hit-rank mean/median",
    }


@dataclass(frozen=True)
class RankingRecord:
    query_id: str
    task: str
    candidate_ids: tuple[str, ...]
    scores: tuple[float, ...]
    relevant_ids: frozenset[str]
    system_id: str
    experiment_id: str
    candidate_count: int
    candidate_corpus_id: str
    relevance_definition: str

    def __post_init__(self) -> None:
        if not self.query_id:
            raise ValueError("ranking query_id is required")
        if self.task not in TASKS:
            raise ValueError(f"invalid ranking task: {self.task}")
        if len(self.candidate_ids) != len(self.scores):
            raise ValueError(
                f"candidate_ids and scores differ for query {self.query_id}"
            )
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError(f"duplicate candidate IDs for query {self.query_id}")
        if not all(_finite(score) for score in self.scores):
            raise ValueError(f"non-finite ranking score for query {self.query_id}")
        if self.candidate_count < len(self.candidate_ids) or self.candidate_count < 0:
            raise ValueError(f"invalid candidate_count for query {self.query_id}")
        for left, right, left_id, right_id in zip(
            self.scores, self.scores[1:], self.candidate_ids, self.candidate_ids[1:]
        ):
            if left < right or (left == right and left_id > right_id):
                raise ValueError(
                    f"ranking is not deterministic score/ID order for query {self.query_id}"
                )

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(range(1, len(self.candidate_ids) + 1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "task": self.task,
            "candidate_ids": list(self.candidate_ids),
            "scores": list(self.scores),
            "ranks": list(self.ranks),
            "relevant_ids": sorted(self.relevant_ids),
            "system_id": self.system_id,
            "experiment_id": self.experiment_id,
            "candidate_count": self.candidate_count,
            "candidate_corpus_id": self.candidate_corpus_id,
            "relevance_definition": self.relevance_definition,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RankingRecord:
        query_id = str(value.get("query_id", ""))
        task = str(value.get("task", ""))
        candidate_ids = tuple(str(item) for item in value.get("candidate_ids", []))
        scores = tuple(float(item) for item in value.get("scores", []))
        relevant_ids = frozenset(str(item) for item in value.get("relevant_ids", []))
        if not query_id:
            raise ValueError("ranking query_id is required")
        if task not in TASKS:
            raise ValueError(f"invalid ranking task: {task}")
        if len(candidate_ids) != len(scores):
            raise ValueError(f"candidate_ids and scores differ for query {query_id}")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"duplicate candidate IDs for query {query_id}")
        if not all(_finite(score) for score in scores):
            raise ValueError(f"non-finite ranking score for query {query_id}")
        candidate_count = int(value.get("candidate_count", len(candidate_ids)))
        if candidate_count < len(candidate_ids) or candidate_count < 0:
            raise ValueError(f"invalid candidate_count for query {query_id}")
        candidate_corpus_id = str(value.get("candidate_corpus_id", ""))
        relevance_definition = str(
            value.get("relevance_definition", TASK_DEFINITIONS[task]["relevance"])
        )
        # The input is a ranked list. Scores may tie, but ties must already be
        # ordered by candidate ID so repeated evaluations cannot drift.
        for left, right, left_id, right_id in zip(
            scores, scores[1:], candidate_ids, candidate_ids[1:]
        ):
            if left < right or (left == right and left_id > right_id):
                raise ValueError(
                    f"ranking is not deterministic score/ID order for query {query_id}"
                )
        return cls(
            query_id=query_id,
            task=task,
            candidate_ids=candidate_ids,
            scores=scores,
            relevant_ids=relevant_ids,
            system_id=str(value.get("system_id", "unknown_system")),
            experiment_id=str(value.get("experiment_id", "unknown_experiment")),
            candidate_count=candidate_count,
            candidate_corpus_id=candidate_corpus_id,
            relevance_definition=relevance_definition,
        )


def ranking_from_scores(
    query_id: str,
    task: str,
    candidates: Sequence[tuple[str, float]],
    relevant_ids: Iterable[str],
    system_id: str,
    experiment_id: str,
    candidate_count: int | None = None,
    candidate_corpus_id: str = "",
    relevance_definition: str | None = None,
) -> RankingRecord:
    if task not in TASKS:
        raise ValueError(f"invalid ranking task: {task}")
    if not query_id:
        raise ValueError("query_id is required")
    if len({candidate_id for candidate_id, _ in candidates}) != len(candidates):
        raise ValueError("candidate IDs must be unique")
    if not all(_finite(score) for _, score in candidates):
        raise ValueError("candidate scores must be finite")
    ordered = sorted(candidates, key=lambda item: (-float(item[1]), str(item[0])))
    return RankingRecord(
        query_id=str(query_id),
        task=task,
        candidate_ids=tuple(str(item[0]) for item in ordered),
        scores=tuple(float(item[1]) for item in ordered),
        relevant_ids=frozenset(str(item) for item in relevant_ids),
        system_id=system_id,
        experiment_id=experiment_id,
        candidate_count=len(candidates)
        if candidate_count is None
        else int(candidate_count),
        candidate_corpus_id=candidate_corpus_id,
        relevance_definition=relevance_definition
        or TASK_DEFINITIONS[task]["relevance"],
    )


def precision_at_k(ranking: RankingRecord, k: int) -> float | None:
    _validate_k(k)
    if not ranking.relevant_ids:
        return None
    return sum(item in ranking.relevant_ids for item in ranking.candidate_ids[:k]) / k


def recall_at_k(ranking: RankingRecord, k: int) -> float | None:
    _validate_k(k)
    if not ranking.relevant_ids:
        return None
    return sum(
        item in ranking.relevant_ids for item in ranking.candidate_ids[:k]
    ) / len(ranking.relevant_ids)


def hit_rate_at_k(ranking: RankingRecord, k: int) -> float | None:
    """Return whether at least one relevant candidate is in the top-K prefix."""

    _validate_k(k)
    if not ranking.relevant_ids:
        return None
    return float(
        any(item in ranking.relevant_ids for item in ranking.candidate_ids[:k])
    )


def reciprocal_rank(ranking: RankingRecord) -> float | None:
    if not ranking.relevant_ids:
        return None
    for rank, item in enumerate(ranking.candidate_ids, start=1):
        if item in ranking.relevant_ids:
            return 1.0 / rank
    return 0.0


def average_precision(ranking: RankingRecord) -> float | None:
    if not ranking.relevant_ids:
        return None
    hits = 0
    total = 0.0
    for rank, item in enumerate(ranking.candidate_ids, start=1):
        if item in ranking.relevant_ids:
            hits += 1
            total += hits / rank
    return total / len(ranking.relevant_ids)


def dcg_at_k(
    ranking: RankingRecord, k: int, graded_relevance: Mapping[str, float] | None = None
) -> float | None:
    _validate_k(k)
    if not ranking.relevant_ids and graded_relevance is None:
        return None
    gains = graded_relevance or {item: 1.0 for item in ranking.relevant_ids}
    return sum(
        float(gains.get(item, 0.0)) / math.log2(rank + 1)
        for rank, item in enumerate(ranking.candidate_ids[:k], start=1)
    )


def ndcg_at_k(
    ranking: RankingRecord, k: int, graded_relevance: Mapping[str, float] | None = None
) -> float | None:
    _validate_k(k)
    if not ranking.relevant_ids and graded_relevance is None:
        return None
    gains = graded_relevance or {item: 1.0 for item in ranking.relevant_ids}
    observed = dcg_at_k(ranking, k, gains) or 0.0
    ideal_gains = sorted((float(value) for value in gains.values()), reverse=True)[:k]
    ideal = sum(
        value / math.log2(rank + 1) for rank, value in enumerate(ideal_gains, start=1)
    )
    return observed / ideal if ideal else None


def _first_relevant_rank(ranking: RankingRecord) -> int | None:
    if not ranking.relevant_ids:
        return None
    return next(
        (
            rank
            for rank, item in enumerate(ranking.candidate_ids, start=1)
            if item in ranking.relevant_ids
        ),
        None,
    )


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _aggregate(values: Iterable[float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return _mean(usable)


def _per_query_metric(
    ranking: RankingRecord, metric: str, k: int | None = None
) -> float | None:
    if metric == "precision":
        if k is None:
            raise ValueError("precision requires k")
        return precision_at_k(ranking, k)
    if metric == "recall":
        if k is None:
            raise ValueError("recall requires k")
        return recall_at_k(ranking, k)
    if metric == "reciprocal_rank":
        return reciprocal_rank(ranking)
    if metric == "average_precision":
        return average_precision(ranking)
    if metric == "ndcg":
        if k is None:
            raise ValueError("ndcg requires k")
        return ndcg_at_k(ranking, k)
    raise ValueError(f"unknown metric: {metric}")


def evaluate_rankings(
    records: Iterable[RankingRecord], ks: Sequence[int] = DEFAULT_KS
) -> dict[str, Any]:
    records = tuple(records)
    if not records:
        raise ValueError("cannot evaluate an empty ranking set")
    if len({record.query_id for record in records}) != len(records):
        raise ValueError("duplicate query IDs in ranking set")
    tasks = {record.task for record in records}
    if len(tasks) != 1:
        raise ValueError("one evaluation result cannot mix retrieval tasks")
    normalized_ks = _normalize_ks(ks)
    relevance_queries = [record for record in records if record.relevant_ids]
    metrics: dict[str, Any] = {
        "queries_total": len(records),
        "queries_evaluated": len(relevance_queries),
        "queries_without_relevance": len(records) - len(relevance_queries),
    }
    for k in normalized_ks:
        metrics[f"precision_at_{k}"] = _aggregate(
            precision_at_k(record, k) for record in relevance_queries
        )
        metrics[f"recall_at_{k}"] = _aggregate(
            recall_at_k(record, k) for record in relevance_queries
        )
        metrics[f"ndcg_at_{k}"] = _aggregate(
            ndcg_at_k(record, k) for record in relevance_queries
        )
    metrics["mrr"] = _aggregate(reciprocal_rank(record) for record in relevance_queries)
    metrics["map"] = _aggregate(
        average_precision(record) for record in relevance_queries
    )
    max_k = max(normalized_ks)
    metrics[f"map_at_{max_k}"] = metrics["map"]
    metrics["map_scope"] = f"average precision over returned top-{max_k} results"
    hit_ranks = [
        rank
        for record in relevance_queries
        if (rank := _first_relevant_rank(record)) is not None
    ]
    metrics["rank_statistics"] = {
        "definition": "first relevant result; misses excluded from mean/median",
        "hit_queries": len(hit_ranks),
        "miss_queries": len(relevance_queries) - len(hit_ranks),
        "mean_first_relevant_rank": _mean([float(rank) for rank in hit_ranks]),
        "median_first_relevant_rank": _median([float(rank) for rank in hit_ranks]),
    }
    return metrics


def bootstrap_ci(
    records: Sequence[RankingRecord],
    metric: str,
    k: int | None = None,
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    usable = [record for record in records if record.relevant_ids]
    if not usable:
        return {
            "status": "not_evaluated_no_relevant_queries",
            "metric": metric,
            "k": k,
            "resamples": resamples,
            "confidence": confidence,
            "seed": seed,
            "unit": "query",
        }
    point_values = [
        value
        for record in usable
        if (value := _per_query_metric(record, metric, k)) is not None
    ]
    estimate = statistics.fmean(point_values)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(resamples):
        values = [point_values[rng.randrange(len(point_values))] for _ in point_values]
        samples.append(statistics.fmean(values))
    samples.sort()
    alpha = (1.0 - confidence) / 2.0
    lower = _quantile(samples, alpha)
    upper = _quantile(samples, 1.0 - alpha)
    return {
        "status": "completed",
        "metric": metric,
        "k": k,
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "resamples": resamples,
        "confidence": confidence,
        "seed": seed,
        "unit": "query",
    }


def _quantile(values: Sequence[float], q: float) -> float:
    position = (len(values) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def compare_systems(
    left_records: Sequence[RankingRecord],
    right_records: Sequence[RankingRecord],
    left_metadata: Mapping[str, Any],
    right_metadata: Mapping[str, Any],
    ks: Sequence[int] = DEFAULT_KS,
    bootstrap_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    left_ids = {record.query_id for record in left_records}
    right_ids = {record.query_id for record in right_records}
    if left_ids != right_ids:
        raise ValueError("paired comparison requires identical query IDs")
    for key in (
        "task",
        "dataset_id",
        "split",
        "protocol_version",
        "candidate_corpus_id",
        "relevance_definition",
    ):
        if left_metadata.get(key) != right_metadata.get(key):
            raise ValueError(f"paired comparison metadata mismatch: {key}")
    left_by_id = {record.query_id: record for record in left_records}
    right_by_id = {record.query_id: record for record in right_records}
    for query_id in left_ids:
        left_record = left_by_id[query_id]
        right_record = right_by_id[query_id]
        if left_record.task != right_record.task:
            raise ValueError("paired comparison requires identical task per query")
        if left_record.candidate_count != right_record.candidate_count:
            raise ValueError("paired comparison requires identical candidate counts")
        if left_record.candidate_corpus_id != right_record.candidate_corpus_id:
            raise ValueError(
                "paired comparison requires identical candidate corpus IDs"
            )
        if left_record.relevance_definition != right_record.relevance_definition:
            raise ValueError(
                "paired comparison requires identical relevance definitions"
            )
        if left_record.relevant_ids != right_record.relevant_ids:
            raise ValueError("paired comparison requires identical relevance sets")
    left_metrics = evaluate_rankings(left_records, ks)
    right_metrics = evaluate_rankings(right_records, ks)
    deltas: dict[str, float | None] = {}
    for key, value in left_metrics.items():
        if isinstance(value, (int, float)) and isinstance(
            right_metrics.get(key), (int, float)
        ):
            deltas[key] = float(right_metrics[key]) - float(value)
    paired: dict[str, Any] = {}
    normalized_ks = _normalize_ks(ks)
    metric_specs: list[tuple[str, int | None]] = [
        ("precision", k) for k in normalized_ks
    ]
    metric_specs += [("recall", k) for k in normalized_ks]
    metric_specs += [("ndcg", k) for k in normalized_ks]
    metric_specs += [("reciprocal_rank", None), ("average_precision", None)]
    for metric, k in metric_specs:
        values: list[float] = []
        for query_id in sorted(left_ids):
            left_value = _per_query_metric(left_by_id[query_id], metric, k)
            right_value = _per_query_metric(right_by_id[query_id], metric, k)
            if left_value is not None and right_value is not None:
                values.append(float(right_value) - float(left_value))
        if not values:
            continue
        key = metric if k is None else f"{metric}_at_{k}"
        paired[key] = {
            "query_count": len(values),
            "mean_delta_right_minus_left": statistics.fmean(values),
            "bootstrap_ci": _bootstrap_values_ci(
                values, bootstrap_resamples, 0.95, seed
            ),
        }
    return {
        "status": "comparable",
        "left_system": left_metadata.get("system_id"),
        "right_system": right_metadata.get("system_id"),
        "query_count": len(left_ids),
        "left_metrics": left_metrics,
        "right_metrics": right_metrics,
        "right_minus_left": deltas,
        "paired_query_deltas": paired,
    }


def _bootstrap_values_ci(
    values: Sequence[float],
    resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    rng = random.Random(seed)
    samples = sorted(
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(resamples)
    )
    alpha = (1.0 - confidence) / 2.0
    return {
        "status": "completed",
        "estimate": statistics.fmean(values),
        "lower": _quantile(samples, alpha),
        "upper": _quantile(samples, 1.0 - alpha),
        "resamples": resamples,
        "confidence": confidence,
        "seed": seed,
        "unit": "paired_query",
    }


def validate_result(result: Mapping[str, Any]) -> None:
    required = {
        "result_schema_version",
        "project",
        "experiment_id",
        "system_id",
        "dataset",
        "split",
        "protocol",
        "task",
        "query_count",
        "candidate_count",
        "seed",
        "metrics",
        "uncertainty",
        "runtime",
        "hardware",
        "provenance",
    }
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f"result schema missing fields: {', '.join(missing)}")
    if result["result_schema_version"] != RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported result schema version")
    if result["project"] != "OmniSearch":
        raise ValueError("result project must be OmniSearch")
    if result["task"] not in TASKS:
        raise ValueError("result task is invalid")
    protocol = result["protocol"]
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("protocol_version") != PROTOCOL_VERSION
    ):
        raise ValueError("result protocol version is invalid")
    if protocol.get("task") != result["task"]:
        raise ValueError("result task and protocol task differ")
    if not isinstance(result["metrics"], Mapping):
        raise TypeError("result metrics must be an object")
    k_values = protocol.get("k_values")
    if (
        not isinstance(k_values, list)
        or not k_values
        or any(
            isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in k_values
        )
    ):
        raise ValueError(
            "result protocol k_values must be a non-empty positive-integer list"
        )
    if len(set(k_values)) != len(k_values) or k_values != sorted(k_values):
        raise ValueError("result protocol k_values must be sorted and unique")
    for field in ("query_count", "candidate_count"):
        if not isinstance(result[field], int) or result[field] < 0:
            raise ValueError(f"result {field} must be a non-negative integer")
    for key, value in result["metrics"].items():
        if isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                raise ValueError(f"result metric {key} is not finite")
            if (
                key.startswith(("precision_at_", "recall_at_", "ndcg_at_"))
                or key in {"mrr", "map"}
            ) and not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"result metric {key} is outside [0, 1]")
    if result["metrics"].get("queries_total") != result["query_count"]:
        raise ValueError("result query_count does not match metrics queries_total")
    evaluated = result["metrics"].get("queries_evaluated")
    without_relevance = result["metrics"].get("queries_without_relevance")
    if (
        evaluated is not None
        and without_relevance is not None
        and evaluated + without_relevance != result["query_count"]
    ):
        raise ValueError("result query relevance counts do not sum to query_count")
    rank_statistics = result["metrics"].get("rank_statistics", {})
    if not isinstance(rank_statistics, Mapping):
        raise TypeError("rank_statistics must be an object")
    for key in ("hit_queries", "miss_queries"):
        if key in rank_statistics and (
            not isinstance(rank_statistics[key], int) or rank_statistics[key] < 0
        ):
            raise ValueError(f"rank statistic {key} must be a non-negative integer")
    for key in ("mean_first_relevant_rank", "median_first_relevant_rank"):
        value = rank_statistics.get(key)
        if value is not None and (not _finite(value) or float(value) < 1):
            raise ValueError(f"rank statistic {key} is invalid")


def build_result(
    records: Sequence[RankingRecord],
    protocol: Mapping[str, Any],
    dataset: Mapping[str, Any],
    split: str,
    experiment_id: str,
    system_id: str,
    model: Mapping[str, Any] | None,
    seed: int,
    uncertainty: Mapping[str, Any],
    runtime: Mapping[str, Any],
    hardware: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "project": "OmniSearch",
        "experiment_id": experiment_id,
        "system_id": system_id,
        "model": dict(model or {}),
        "dataset": dict(dataset),
        "split": split,
        "subset": dataset.get("subset"),
        "protocol": dict(protocol),
        "task": protocol["task"],
        "query_count": len(records),
        "candidate_count": max(
            (record.candidate_count for record in records), default=0
        ),
        "seed": seed,
        "metrics": evaluate_rankings(records, protocol["k_values"]),
        "uncertainty": dict(uncertainty),
        "runtime": dict(runtime),
        "hardware": dict(hardware),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "provenance": dict(provenance),
    }
    validate_result(result)
    return result


def _hash_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ranking_input(
    path: Path | str,
) -> tuple[dict[str, Any], tuple[RankingRecord, ...]]:
    with Path(path).open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict) or not isinstance(payload.get("queries"), list):
        raise TypeError("ranking input must contain a queries list")
    metadata = dict(payload)
    task = str(payload.get("task", ""))
    records = tuple(
        RankingRecord.from_mapping(
            {
                **query,
                "task": query.get("task", task),
                "system_id": query.get(
                    "system_id", payload.get("system_id", "unknown_system")
                ),
                "experiment_id": query.get(
                    "experiment_id", payload.get("experiment_id", "unknown_experiment")
                ),
                "candidate_corpus_id": query.get(
                    "candidate_corpus_id", payload.get("candidate_corpus_id", "")
                ),
                "relevance_definition": query.get(
                    "relevance_definition",
                    payload.get(
                        "relevance_definition",
                        TASK_DEFINITIONS.get(task, {}).get("relevance", ""),
                    ),
                ),
            }
        )
        for query in payload["queries"]
    )
    if task not in TASKS:
        raise ValueError("ranking input task is invalid")
    if any(record.task != task for record in records):
        raise ValueError("ranking input mixes tasks")
    return metadata, records


def _phase3_documents(records: Sequence[ImageRecord]) -> tuple[Any, ...]:
    from .baselines import TextDocument

    return tuple(
        TextDocument(caption.caption_id, record.image_id, caption.text)
        for record in records
        for caption in record.captions
    )


def _phase3_rankings(
    records: Sequence[ImageRecord],
    system_id: str,
    experiment_id: str,
    seed: int,
    top_k: int,
    config_path: Path,
    candidate_corpus_id: str,
) -> tuple[dict[str, Any], tuple[RankingRecord, ...]]:
    import tomllib

    from .baselines import BM25Index, TfidfIndex

    with config_path.open("rb") as file:
        phase3_config = dict(tomllib.load(file).get("phase3", {}))
    documents = _phase3_documents(records)
    if not documents:
        raise ValueError("Phase 3 migration found no documents")
    max_queries = int(phase3_config.get("max_text_queries", 256))
    ordered = sorted(
        documents,
        key=lambda doc: hashlib.sha256(f"{seed}\0{doc.doc_id}".encode()).hexdigest(),
    )
    queries = tuple(ordered[: min(max_queries, len(ordered))])
    relevance = {
        query.doc_id: {
            doc.doc_id
            for doc in documents
            if doc.group_id == query.group_id and doc.doc_id != query.doc_id
        }
        for query in queries
    }
    index: TfidfIndex | BM25Index
    if system_id == "tfidf_word_unigram_l2":
        index = TfidfIndex(
            ngram_range=(
                int(phase3_config.get("tfidf_ngram_min", 1)),
                int(phase3_config.get("tfidf_ngram_max", 1)),
            ),
            min_df=int(phase3_config.get("tfidf_min_df", 1)),
            max_df=phase3_config.get("tfidf_max_df", 1.0),
            sublinear_tf=bool(phase3_config.get("tfidf_sublinear_tf", True)),
        )
    elif system_id == "bm25_word":
        index = BM25Index(
            k1=float(phase3_config.get("bm25_k1", 1.5)),
            b=float(phase3_config.get("bm25_b", 0.75)),
        )
    else:
        raise ValueError(f"unsupported Phase 3 migration system: {system_id}")
    build_start = __import__("time").perf_counter()
    index.fit(documents)
    build_seconds = __import__("time").perf_counter() - build_start
    ranking_records: list[RankingRecord] = []
    for query in queries:
        results = index.search(query.text, top_k=top_k, exclude_doc_id=query.doc_id)
        ranking_records.append(
            ranking_from_scores(
                query_id=query.doc_id,
                task="text_to_text",
                candidates=[(result.item_id, result.score) for result in results],
                relevant_ids=relevance[query.doc_id],
                system_id=system_id,
                experiment_id=experiment_id,
                candidate_count=len(documents),
                candidate_corpus_id=candidate_corpus_id,
                relevance_definition="same-image captions in held-out split; query excluded",
            )
        )
    return {
        "index_build_seconds": build_seconds,
        "queries": len(queries),
        "candidate_count": len(documents),
    }, tuple(ranking_records)


def run_phase3_migration(
    manifest_path: Path | str,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    output_dir: Path | str = "artifacts/phase5",
    seed: int | None = None,
    bootstrap_resamples: int = 200,
) -> dict[str, Any]:
    """Rerun Phase 3 text rankings and evaluate them under retrieval_eval_v1."""

    import time

    manifest_path = Path(manifest_path)
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    manifest = read_manifest(manifest_path)
    assert_no_split_leakage(manifest.records)
    seed = config.seed if seed is None else seed
    split = "test"
    records = tuple(record for record in manifest.records if record.split == split)
    documents = _phase3_documents(records)
    if not documents:
        raise ValueError("Phase 3 migration found no documents in the test split")
    shared_corpus_id = f"{manifest.dataset_id}:{split}:{len(documents)}"
    protocol = make_protocol("text_to_text")
    system_results: list[dict[str, Any]] = []
    record_sets: dict[str, tuple[RankingRecord, ...]] = {}
    for system_id in ("tfidf_word_unigram_l2", "bm25_word"):
        experiment_id = f"phase5_{system_id}_canonical_seed{seed}"
        timing, ranking_records = _phase3_rankings(
            records,
            system_id,
            experiment_id,
            seed,
            max(protocol["k_values"]),
            config_path,
            shared_corpus_id,
        )
        record_sets[system_id] = ranking_records
        uncertainty = {
            "recall_at_1": bootstrap_ci(
                ranking_records, "recall", 1, bootstrap_resamples, 0.95, seed
            ),
            "recall_at_5": bootstrap_ci(
                ranking_records, "recall", 5, bootstrap_resamples, 0.95, seed
            ),
            "recall_at_10": bootstrap_ci(
                ranking_records, "recall", 10, bootstrap_resamples, 0.95, seed
            ),
        }
        dataset = {
            "dataset_id": manifest.dataset_id,
            "dataset_version": manifest.dataset_version,
            "manifest_sha256": _hash_file(manifest_path),
            "subset": f"deterministic_seed_{seed}_max_{timing['queries']}_queries",
        }
        evaluation_start = time.perf_counter()
        result = build_result(
            ranking_records,
            protocol,
            dataset,
            split,
            experiment_id,
            system_id,
            {"model": system_id, "learned": False},
            seed,
            uncertainty,
            {**timing, "evaluation_seconds": time.perf_counter() - evaluation_start},
            {"platform": platform.platform(), "python": sys.version, "device": "cpu"},
            {
                "config_sha256": _hash_file(config_path),
                "protocol_version": PROTOCOL_VERSION,
                "source_sha256": manifest.source_sha256,
            },
        )
        result["ranking_records"] = [record.to_dict() for record in ranking_records]
        result["candidate_corpus_id"] = (
            ranking_records[0].candidate_corpus_id if ranking_records else ""
        )
        result_path = output_dir / f"{system_id}.json"
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        system_results.append(result)
    left_meta = {
        "system_id": system_results[0]["system_id"],
        "task": "text_to_text",
        "dataset_id": manifest.dataset_id,
        "split": split,
        "protocol_version": PROTOCOL_VERSION,
        "candidate_corpus_id": shared_corpus_id,
        "relevance_definition": record_sets["tfidf_word_unigram_l2"][
            0
        ].relevance_definition,
    }
    right_meta = {**left_meta, "system_id": system_results[1]["system_id"]}
    comparison = compare_systems(
        record_sets["tfidf_word_unigram_l2"],
        record_sets["bm25_word"],
        left_meta,
        right_meta,
        protocol["k_values"],
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
    )
    comparison_path = output_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "system_id",
                "task",
                "queries",
                "recall_at_1",
                "recall_at_5",
                "recall_at_10",
                "mrr",
                "map",
                "ndcg_at_10",
                "median_first_relevant_rank",
                "mean_first_relevant_rank",
            ],
        )
        writer.writeheader()
        for result in system_results:
            metrics = result["metrics"]
            ranks = metrics["rank_statistics"]
            writer.writerow(
                {
                    "system_id": result["system_id"],
                    "task": result["task"],
                    "queries": result["query_count"],
                    "recall_at_1": metrics.get("recall_at_1"),
                    "recall_at_5": metrics.get("recall_at_5"),
                    "recall_at_10": metrics.get("recall_at_10"),
                    "mrr": metrics.get("mrr"),
                    "map": metrics.get("map"),
                    "ndcg_at_10": metrics.get("ndcg_at_10"),
                    "median_first_relevant_rank": ranks.get(
                        "median_first_relevant_rank"
                    ),
                    "mean_first_relevant_rank": ranks.get("mean_first_relevant_rank"),
                }
            )
    report = {
        "report_schema_version": 1,
        "project": "OmniSearch",
        "pre_phase_audit": {
            "phase4_audit": "PASS",
            "phase3_audit": "PASS",
            "naming_audit": "PASS_WITH_IMMUTABLE_HISTORICAL_EXCEPTION",
            "forbidden_later_phase_features": {
                "fine_tuning": False,
                "lora": False,
                "ann": False,
                "reranking": False,
            },
        },
        "protocol": protocol,
        "source": "fresh Phase 3 rerun under canonical Phase 5 evaluator; historical Phase 3 artifacts preserved",
        "dataset": {
            "dataset_id": manifest.dataset_id,
            "dataset_version": manifest.dataset_version,
            "split": split,
            "manifest_sha256": _hash_file(manifest_path),
        },
        "systems": [
            {
                "system_id": result["system_id"],
                "metrics": result["metrics"],
                "uncertainty": result["uncertainty"],
            }
            for result in system_results
        ],
        "paired_comparison": comparison,
        "real_dataset_image_tasks": "not_evaluated_no_image_relevance_labels",
        "real_flickr30k_image_tasks": "historical_key_retained_for_schema_compatibility",
    }
    (output_dir / "phase5_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# OmniSearch Phase 5 canonical evaluation report",
        "",
        f"Protocol: `{PROTOCOL_VERSION}`",
        "",
        "This report is a fresh Phase 3 text-baseline rerun under the canonical evaluator. Historical Phase 3 artifacts were not overwritten.",
        "",
        "| System | Task | Queries | R@1 | R@5 | R@10 | MRR | MAP | NDCG@10 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in system_results:
        metrics = result["metrics"]
        lines.append(
            "| {system_id} | {task} | {queries} | {r1:.4f} | {r5:.4f} | {r10:.4f} | {mrr:.4f} | {map:.4f} | {ndcg:.4f} |".format(
                system_id=result["system_id"],
                task=result["task"],
                queries=result["query_count"],
                r1=metrics["recall_at_1"],
                r5=metrics["recall_at_5"],
                r10=metrics["recall_at_10"],
                mrr=metrics["mrr"],
                map=metrics["map"],
                ndcg=metrics["ndcg_at_10"],
            )
        )
    lines.extend(
        [
            "",
            "Image-to-text and text-to-image canonical evaluation was not run in this text-baseline migration because this task has no image rankings; Phase 4 owns the real cross-modal run.",
            "",
        ]
    )
    (output_dir / "phase5_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def run_phase4_migration(
    phase4_report_path: Path | str = "artifacts/phase4/phase4_report.json",
    fixture_path: Path | str = "artifacts/phase4/fixture_smoke.json",
    output_dir: Path | str = "artifacts/phase5",
    bootstrap_resamples: int = 200,
    seed: int = 42,
    manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    """Migrate Phase 4 rankings when present, keeping fixture and real data separate."""

    phase4_report_path = Path(phase4_report_path)
    fixture_path = Path(fixture_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not fixture.get("fixture_only"):
        raise ValueError(
            "Phase 4 migration requires an explicitly fixture-only smoke artifact"
        )
    group_by_text = {"red-caption": "red", "blue-caption": "blue"}
    group_by_image = {"red-image": "red", "blue-image": "blue"}
    task_inputs = {
        "text_to_image": (
            fixture.get("text_to_image_rankings", {}),
            group_by_text,
            group_by_image,
            "fixture:images:2",
        ),
        "image_to_text": (
            fixture.get("image_to_text_rankings", {}),
            group_by_image,
            group_by_text,
            "fixture:captions:2",
        ),
    }
    results: list[dict[str, Any]] = []
    for task, (
        rankings,
        query_groups,
        candidate_groups,
        corpus_id,
    ) in task_inputs.items():
        if not rankings:
            continue
        records: list[RankingRecord] = []
        for query_id, candidates in rankings.items():
            relevant = {
                candidate_id
                for candidate_id, group in candidate_groups.items()
                if group == query_groups[query_id]
            }
            records.append(
                ranking_from_scores(
                    query_id=query_id,
                    task=task,
                    candidates=[
                        (str(item["id"]), float(item["score"])) for item in candidates
                    ],
                    relevant_ids=relevant,
                    system_id="clip_zero_shot_fixture",
                    experiment_id=f"phase5_phase4_fixture_{task}",
                    candidate_count=len(candidate_groups),
                    candidate_corpus_id=corpus_id,
                    relevance_definition=TASK_DEFINITIONS[task]["relevance"],
                )
            )
        protocol = make_protocol(task)
        uncertainty = {
            f"recall_at_{k}": bootstrap_ci(
                records, "recall", k, bootstrap_resamples, 0.95, seed
            )
            for k in protocol["k_values"]
        }
        result = build_result(
            records,
            protocol,
            {
                "dataset_id": "fixture_phase4",
                "dataset_version": "phase4_fixture_smoke",
                "subset": "fixture_only",
            },
            "fixture",
            f"phase5_phase4_fixture_{task}",
            "clip_zero_shot_fixture",
            {"model_id": fixture.get("model_id"), "learned": True, "frozen": True},
            seed,
            uncertainty,
            {"source": "phase4_fixture_smoke"},
            {"device": fixture.get("device")},
            {
                "phase4_report_sha256": _hash_file(phase4_report_path),
                "fixture_sha256": _hash_file(fixture_path),
                "protocol_version": PROTOCOL_VERSION,
            },
        )
        result["fixture_only"] = True
        result["ranking_records"] = [record.to_dict() for record in records]
        result_path = output / f"phase4_fixture_{task}.json"
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        results.append(
            {
                "task": task,
                "result_path": str(result_path),
                "metrics": result["metrics"],
            }
        )
    migration = {
        "project": "OmniSearch",
        "protocol_version": PROTOCOL_VERSION,
        "phase4_real_flickr30k_status": "not_migrated_no_authorized_image_root",
        "phase4_fixture_status": "migrated_fixture_only",
        "phase4_fixture_results": results,
        "source_phase4_report": str(phase4_report_path),
        "source_fixture": str(fixture_path),
        "note": "Fixture metrics verify schema compatibility only. Real results are included when the Phase 4 report contains a verified dataset run and a matching manifest is supplied.",
    }
    if manifest_path is not None:
        real = run_phase4_real_migration(
            phase4_report_path,
            manifest_path,
            output,
            bootstrap_resamples,
            seed,
        )
        migration["phase4_real_status"] = real["status"]
        migration["phase4_real_results"] = real.get("results", [])
    else:
        migration["phase4_real_status"] = "not_requested"
    (output / "phase4_migration.json").write_text(
        json.dumps(migration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return migration


def run_phase4_real_migration(
    phase4_report_path: Path | str,
    manifest_path: Path | str,
    output_dir: Path,
    bootstrap_resamples: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate real Phase 4 rankings under the canonical evaluator."""

    phase4_report_path = Path(phase4_report_path)
    manifest_path = Path(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(phase4_report_path.read_text(encoding="utf-8"))
    if not report.get("scope", {}).get("real_dataset_evaluation", False):
        return {"status": "not_available_in_phase4_report", "results": []}
    manifest = read_manifest(manifest_path)
    assert_no_split_leakage(manifest.records)
    split = str(report.get("evaluation", {}).get("split", "test"))
    selected = tuple(record for record in manifest.records if record.split == split)
    if not selected:
        return {"status": "manifest_split_not_found", "results": []}
    image_to_captions = {
        record.image_id: {caption.caption_id for caption in record.captions}
        for record in selected
    }
    caption_to_image = {
        caption.caption_id: {record.image_id}
        for record in selected
        for caption in record.captions
    }
    rankings_by_task = report.get("rankings", {})
    results: list[dict[str, Any]] = []
    for task, relevant_by_query, candidate_count, corpus_id in (
        (
            "text_to_image",
            caption_to_image,
            len(selected),
            f"{manifest.dataset_id}:{split}:images:{len(selected)}",
        ),
        (
            "image_to_text",
            image_to_captions,
            sum(len(record.captions) for record in selected),
            f"{manifest.dataset_id}:{split}:captions:{sum(len(record.captions) for record in selected)}",
        ),
    ):
        rankings = rankings_by_task.get(task, {})
        records: list[RankingRecord] = []
        for query_id, candidates in rankings.items():
            if query_id not in relevant_by_query:
                continue
            records.append(
                ranking_from_scores(
                    query_id=query_id,
                    task=task,
                    candidates=[
                        (str(item["id"]), float(item["score"])) for item in candidates
                    ],
                    relevant_ids=relevant_by_query[query_id],
                    system_id="clip_zero_shot_real",
                    experiment_id=f"phase5_clip_zero_shot_{task}",
                    candidate_count=candidate_count,
                    candidate_corpus_id=corpus_id,
                    relevance_definition=TASK_DEFINITIONS[task]["relevance"],
                )
            )
        if not records:
            continue
        protocol = make_protocol(task)
        uncertainty = {
            f"recall_at_{k}": bootstrap_ci(
                records, "recall", k, bootstrap_resamples, 0.95, seed
            )
            for k in protocol["k_values"]
        }
        result = build_result(
            records,
            protocol,
            {
                "dataset_id": manifest.dataset_id,
                "dataset_version": manifest.dataset_version,
                "manifest_sha256": _hash_file(manifest_path),
                "subset": f"{split}_all_verified_phase4_items",
            },
            split,
            f"phase5_clip_zero_shot_{task}",
            "clip_zero_shot_real",
            {"model_id": report.get("model", {}).get("model_id"), "frozen": True},
            seed,
            uncertainty,
            {"source_phase4_report": str(phase4_report_path)},
            {"device": report.get("model", {}).get("device")},
            {
                "phase4_report_sha256": _hash_file(phase4_report_path),
                "manifest_sha256": _hash_file(manifest_path),
                "protocol_version": PROTOCOL_VERSION,
            },
        )
        result["ranking_records"] = [record.to_dict() for record in records]
        result_path = output_dir / f"phase4_real_{task}.json"
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        results.append({"task": task, "result_path": str(result_path), "metrics": result["metrics"]})
    status = "migrated_real" if results else "no_real_rankings"
    migration = {
        "status": status,
        "dataset_id": manifest.dataset_id,
        "split": split,
        "results": results,
        "phase4_report": str(phase4_report_path),
        "manifest": str(manifest_path),
    }
    (output_dir / "phase4_real_migration.json").write_text(
        json.dumps(migration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return migration


def _write_single_result_artifacts(result: Mapping[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metrics = result["metrics"]
    lines = [
        f"# OmniSearch canonical retrieval result: {result['system_id']}",
        "",
        f"Protocol: `{result['protocol']['protocol_version']}`",
        "",
        f"Task: `{result['task']}`; split: `{result['split']}`; queries: `{result['query_count']}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in (
        "precision_at_1",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "map",
        "ndcg_at_10",
    ):
        if key in metrics:
            lines.append(f"| {key} | {metrics[key]} |")
    (output / "evaluation_result.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    with (output / "evaluation_result.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["system_id", "task", "metric", "value"])
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                writer.writerow([result["system_id"], result["task"], key, value])


def write_result_artifacts(result: Mapping[str, Any], output_dir: Path | str) -> None:
    """Write one canonical result as JSON, Markdown, and CSV."""

    _write_single_result_artifacts(result, Path(output_dir))


def run_ranking_file(
    rankings_path: Path | str,
    output_dir: Path | str = "artifacts/phase5",
    bootstrap_resamples: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    metadata, records = load_ranking_input(rankings_path)
    task = records[0].task
    protocol = make_protocol(task, metadata.get("k_values", DEFAULT_KS))
    uncertainty = {
        "recall_at_1": bootstrap_ci(
            records, "recall", 1, bootstrap_resamples, 0.95, seed
        ),
        "recall_at_5": bootstrap_ci(
            records, "recall", 5, bootstrap_resamples, 0.95, seed
        ),
    }
    result = build_result(
        records,
        protocol,
        metadata.get("dataset", {"dataset_id": "fixture"}),
        str(metadata.get("split", "not_declared")),
        str(metadata.get("experiment_id", "evaluation_file")),
        str(metadata.get("system_id", "unknown_system")),
        metadata.get("model"),
        seed,
        uncertainty,
        metadata.get("runtime", {}),
        metadata.get("hardware", {}),
        {
            "input_sha256": _hash_file(rankings_path),
            "protocol_version": PROTOCOL_VERSION,
        },
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_single_result_artifacts(result, output)
    (output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the canonical OmniSearch retrieval evaluator."
    )
    parser.add_argument(
        "--rankings", type=Path, default=None, help="canonical ranking JSON input"
    )
    parser.add_argument(
        "--source", choices=("rankings", "phase3", "phase4"), default="rankings"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/coco2017_val_split_manifest.json"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase5"))
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--phase4-report",
        type=Path,
        default=Path("artifacts/phase4/phase4_report.json"),
    )
    parser.add_argument(
        "--phase4-fixture",
        type=Path,
        default=Path("artifacts/phase4/fixture_smoke.json"),
    )
    args = parser.parse_args()
    if args.source == "rankings":
        if args.rankings is None:
            parser.error("--rankings is required when --source rankings")
        result = run_ranking_file(
            args.rankings, args.output_dir, args.bootstrap_resamples, args.seed
        )
        print(
            json.dumps(
                {
                    "output_dir": str(args.output_dir),
                    "task": result["task"],
                    "system_id": result["system_id"],
                },
                indent=2,
            )
        )
    elif args.source == "phase3":
        result = run_phase3_migration(
            args.manifest,
            args.config,
            args.output_dir,
            args.seed,
            args.bootstrap_resamples,
        )
        print(
            json.dumps(
                {
                    "output_dir": str(args.output_dir),
                    "protocol": PROTOCOL_VERSION,
                    "systems": [item["system_id"] for item in result["systems"]],
                },
                indent=2,
            )
        )
    else:
        result = run_phase4_migration(
            args.phase4_report,
            args.phase4_fixture,
            args.output_dir,
            args.bootstrap_resamples,
            args.seed,
            args.manifest,
        )
        print(
            json.dumps(
                {
                    "output_dir": str(args.output_dir),
                    "protocol": PROTOCOL_VERSION,
                    "fixture_tasks": [
                        item["task"] for item in result["phase4_fixture_results"]
                    ],
                    "real_status": result.get("phase4_real_status"),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
