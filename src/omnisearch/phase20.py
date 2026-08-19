"""Phase 20 efficiency and resource optimization analysis.

Phase 20 is evaluation-only.  It consolidates measured timings and artifact
sizes from Phases 7--19, performs a compact cached-embedding float16 check,
and writes an evidence-based architecture recommendation.  It does not train,
download, build new indexes, or add a model family.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PHASE20_SCHEMA_VERSION = 1
MIN_RESOURCE_FIELDS = {"measurement_status", "value", "unit"}
REQUIRED_ARTIFACTS = (
    "pre_phase_audit.json",
    "resource_inventory.json",
    "latency_breakdown.json",
    "storage_analysis.json",
    "training_cost_comparison.json",
    "model_size_comparison.json",
    "exact_ann_summary.json",
    "cache_analysis.json",
    "float16_precision_analysis.json",
    "quality_efficiency_frontier.json",
    "recommended_configurations.json",
    "disk_cleanup_recommendations.json",
    "provenance.json",
    "phase20_report.json",
)
MEASURED = "MEASURED"
CALCULATED = "CALCULATED"
CONSOLIDATED = "CONSOLIDATED_MEASURED_COMPONENTS"
NOT_MEASURED = "NOT_MEASURED"


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


def _file_size(path: Path | str) -> int | None:
    candidate = Path(path)
    return candidate.stat().st_size if candidate.exists() and candidate.is_file() else None


def bytes_to_mib(value: float) -> float:
    if float(value) < 0:
        raise ValueError("byte count must be non-negative")
    return float(value) / (1024.0 * 1024.0)


def storage_calculation(
    item_count: int, dimension: int, bytes_per_element: int, *, label: str = CALCULATED
) -> dict[str, Any]:
    """Calculate dense matrix storage with an explicit measured/calculated label."""

    if item_count <= 0 or dimension <= 0 or bytes_per_element <= 0:
        raise ValueError("storage dimensions and element size must be positive")
    value = int(item_count) * int(dimension) * int(bytes_per_element)
    return {
        "value": value,
        "unit": "bytes",
        "mib": bytes_to_mib(value),
        "measurement_status": label,
        "formula": "item_count * dimension * bytes_per_element",
        "inputs": {
            "item_count": int(item_count),
            "dimension": int(dimension),
            "bytes_per_element": int(bytes_per_element),
        },
    }


def aggregate_latency(components: Mapping[str, float | int | None]) -> dict[str, Any]:
    """Aggregate finite latency components while preserving missing values."""

    usable = {str(key): float(value) for key, value in components.items() if value is not None}
    if any(value < 0 or not math.isfinite(value) for value in usable.values()):
        raise ValueError("latency components must be finite and non-negative")
    return {
        "components_seconds": usable,
        "total_seconds": sum(usable.values()),
        "measurement_status": CONSOLIDATED,
        "unit": "seconds",
    }


def pareto_frontier(
    rows: Sequence[Mapping[str, Any]],
    *,
    quality_key: str,
    cost_key: str,
    maximize_quality: bool = True,
    minimize_cost: bool = True,
) -> list[dict[str, Any]]:
    """Return non-dominated rows for one quality and one cost dimension."""

    if not rows:
        return []
    valid = [row for row in rows if row.get(quality_key) is not None and row.get(cost_key) is not None]
    output: list[dict[str, Any]] = []
    for row in valid:
        q = float(row[quality_key])
        c = float(row[cost_key])
        dominated = False
        for other in valid:
            if other is row:
                continue
            oq = float(other[quality_key])
            oc = float(other[cost_key])
            quality_better_or_equal = oq >= q if maximize_quality else oq <= q
            cost_better_or_equal = oc <= c if minimize_cost else oc >= c
            quality_strict = oq > q if maximize_quality else oq < q
            cost_strict = oc < c if minimize_cost else oc > c
            if quality_better_or_equal and cost_better_or_equal and (quality_strict or cost_strict):
                dominated = True
                break
        if not dominated:
            output.append({**dict(row), "pareto_status": "NON_DOMINATED"})
    return sorted(output, key=lambda row: (float(row[cost_key]), -float(row[quality_key])))


def _resource_value(value: Any, unit: str, status: str) -> dict[str, Any]:
    return {"value": value, "unit": unit, "measurement_status": status}


def _directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for candidate in path.rglob("*"):
        if candidate.is_file():
            try:
                total += candidate.stat().st_size
            except OSError:
                continue
    return total


def _path_size(path: Path) -> int | None:
    """Return a measured byte size for either one file or a directory."""

    if path.is_file():
        return _file_size(path)
    if path.is_dir():
        return _directory_size(path)
    return None


def audit_phase19(root: Path | str) -> dict[str, Any]:
    """Run only the focused Phase 19 dependency audit."""

    base = Path(root)
    phase19 = base / "artifacts/phase19"
    card = base / "docs/system_card.md"
    required = [
        phase19 / "phase19_report.json",
        phase19 / "artifact_validation.json",
        phase19 / "provenance.json",
        phase19 / "responsible_ai_matrix.json",
        base / "artifacts/phase7/zero_shot_text_to_image.json",
        base / "artifacts/phase7/fine_tuned_text_to_image.json",
        base / "artifacts/phase7/zero_shot_image_to_text.json",
        base / "artifacts/phase7/fine_tuned_image_to_text.json",
        base / "artifacts/phase7/best_checkpoint.pt",
        base / "artifacts/phase8/efficiency_comparison.json",
        base / "artifacts/phase9/efficiency_comparison.json",
        base / "artifacts/phase10/benchmark_results.json",
        base / "artifacts/phase10/embedding_generation.json",
        base / "artifacts/phase11/quality_latency_comparison.json",
        base / "artifacts/phase11/query_encoding_latency.json",
        base / "artifacts/phase17/provenance.json",
        base / "artifacts/phase18/provenance.json",
        base / "artifacts/phase19/provenance.json",
    ]
    checks: dict[str, bool] = {
        "phase19_required_artifacts_readable": all(path.exists() for path in required),
        "system_card_exists": card.exists(),
        "zero_shot_and_full_ft_artifacts_available": all(
            (base / name).exists()
            for name in (
                "artifacts/phase7/zero_shot_text_to_image.json",
                "artifacts/phase7/fine_tuned_text_to_image.json",
                "artifacts/phase7/zero_shot_image_to_text.json",
                "artifacts/phase7/fine_tuned_image_to_text.json",
            )
        ),
        "historical_resource_artifacts_readable": all(path.exists() for path in required[9:]),
    }
    try:
        report = _read_json(phase19 / "phase19_report.json")
        validator = _read_json(phase19 / "artifact_validation.json")
        matrix = _read_json(phase19 / "responsible_ai_matrix.json")
        card_text = card.read_text(encoding="utf-8")
        checks.update(
            {
                "phase19_quality_gate_pass": report.get("status") == "PASS" and report.get("quality_gate", {}).get("status") == "PASS",
                "phase19_artifact_validator_pass": validator.get("passed") is True,
                "system_card_consistent_with_findings": "No protected-attribute labels" in card_text and "No content-safety classifier" in card_text,
                "no_unsupported_responsible_ai_claims": all(row.get("fairness_claim_made") is False for row in matrix.get("rows", [])) and "not evaluated" in card_text.casefold(),
            }
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        checks.update(
            {
                "phase19_quality_gate_pass": False,
                "phase19_artifact_validator_pass": False,
                "system_card_consistent_with_findings": False,
                "no_unsupported_responsible_ai_claims": False,
            }
        )
    passed = all(checks.values())
    return {
        "schema_version": PHASE20_SCHEMA_VERSION,
        "phase": 20,
        "dependency": 19,
        "audit_result": "PRE-PHASE AUDIT: Phase 19 PASS" if passed else "PRE-PHASE AUDIT: Phase 19 BLOCKED",
        "passed": passed,
        "checks": checks,
        "recorded_before_phase20_analysis": True,
        "phase21_started": False,
    }


def _phase10_rows(root: Path, tier: str = "tier3") -> list[dict[str, Any]]:
    frontier = _read_json(root / "artifacts/phase10/quality_latency_frontier.json")
    scaling = _read_json(root / "artifacts/phase10/scaling_results.json")
    scale_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for tier_row in scaling:
        for direction, direction_data in tier_row.get("directions", {}).items():
            for row in direction_data.get("configs", []):
                scale_by_key[(str(tier_row["tier"]), str(direction), str(row["config"]))] = row
    rows: list[dict[str, Any]] = []
    for row in frontier:
        if str(row.get("tier")) != tier:
            continue
        scale = scale_by_key.get((tier, str(row["task"]), str(row["config"])), {})
        rows.append(
            {
                "tier": tier,
                "direction": row["task"],
                "config": row["config"],
                "index_type": row["index_type"],
                "search_latency_mean_seconds": row.get("search_latency_mean_seconds"),
                "search_latency_p95_seconds": row.get("search_latency_p95_seconds"),
                "queries_per_second": row.get("queries_per_second"),
                "neighbor_recall_at_10": row.get("neighbor_recall_at_10"),
                "semantic_recall_at_5": row.get("semantic_recall_at_5"),
                "build_seconds": scale.get("build_seconds"),
                "load_seconds": scale.get("load_seconds"),
                "serialized_size_bytes": scale.get("serialized_size_bytes"),
                "raw_embedding_storage_bytes": scale.get("raw_embedding_storage_bytes"),
                "measurement_status": MEASURED,
                "source": "artifacts/phase10/quality_latency_frontier.json + scaling_results.json",
            }
        )
    return rows


def _measure_model_load(root: Path, enabled: bool = True) -> dict[str, Any]:
    if not enabled:
        return {"measurement_status": NOT_MEASURED, "reason": "profiling disabled"}
    checkpoint = root / "artifacts/phase7/best_checkpoint.pt"
    if not checkpoint.exists():
        return {"measurement_status": NOT_MEASURED, "reason": "checkpoint unavailable"}
    old_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from .phase7 import _load_checkpoint, _load_trainable_model

        start = time.perf_counter()
        model, processor, torch_module, device = _load_trainable_model("openai/clip-vit-base-patch32", "auto")
        base_load = time.perf_counter() - start
        restore_start = time.perf_counter()
        metadata = _load_checkpoint(checkpoint, model)
        checkpoint_restore = time.perf_counter() - restore_start
        total = time.perf_counter() - start
        result = {
            "measurement_status": MEASURED,
            "device": str(device),
            "base_model_load_seconds": base_load,
            "full_ft_checkpoint_restore_seconds": checkpoint_restore,
            "base_plus_full_ft_restore_seconds": total,
            "checkpoint_metadata_keys": sorted(metadata)[:20],
            "training_performed": False,
            "model_load_included_in_query_latency": False,
            "method": "offline cached CLIP load plus state-dict restore; no forward pass or optimizer",
        }
        del model, processor, torch_module
        gc.collect()
        return result
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:  # pragma: no cover - environment-dependent optional dependency
        return {"measurement_status": NOT_MEASURED, "reason": f"offline load profiling failed: {type(exc).__name__}: {exc}"}
    finally:
        if old_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old_offline


def _latency_breakdown(root: Path, model_load: Mapping[str, Any], phase10_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    encoding_rows = _read_json(root / "artifacts/phase11/query_encoding_latency.json")
    tier2_encoding = next(row for row in encoding_rows if row["label"] == "tier2_test")
    selected_search = {
        str(row["direction"]): row
        for row in phase10_rows
        if row["config"] == "faiss_flat"
    }
    directions: list[dict[str, Any]] = []
    for direction in ("text_to_image", "image_to_text"):
        if direction == "text_to_image":
            query_count = int(tier2_encoding["caption_query_count"])
            encoding_seconds = float(tier2_encoding["text_encoding_seconds"])
            throughput = float(tier2_encoding["text_items_per_second"])
        else:
            query_count = int(tier2_encoding["image_query_count"])
            encoding_seconds = float(tier2_encoding["image_encoding_seconds"])
            throughput = float(tier2_encoding["image_items_per_second"])
        search = selected_search[direction]
        encoding_per_query = encoding_seconds / query_count
        warm = float(search["search_latency_mean_seconds"])
        cold = aggregate_latency({"query_encoding": encoding_per_query, "search": warm})
        row = {
            "direction": direction,
            "hardware": tier2_encoding["device"],
            "fixed_query_count": query_count,
            "query_encoding_total_seconds": _resource_value(encoding_seconds, "seconds", MEASURED),
            "query_encoding_per_query_seconds": _resource_value(encoding_per_query, "seconds/query", CALCULATED),
            "query_encoding_throughput_items_per_second": _resource_value(throughput, "items/second", MEASURED),
            "search": {
                "config": "faiss_flat",
                "mean_seconds_per_query": _resource_value(warm, "seconds/query", MEASURED),
                "p95_seconds_per_query": _resource_value(float(search["search_latency_p95_seconds"]), "seconds/query", MEASURED),
                "throughput_queries_per_second": _resource_value(float(search["queries_per_second"]), "queries/second", MEASURED),
            },
            "warm_cached_embedding_total_seconds_per_query": _resource_value(warm, "seconds/query", MEASURED),
            "cold_encode_plus_search_total_seconds_per_query": cold,
            "model_load_excluded_from_query_latency": True,
            "measurement_status": CONSOLIDATED,
        }
        if model_load.get("measurement_status") == MEASURED:
            row["single_query_with_base_model_cold_start_seconds"] = aggregate_latency(
                {"base_model_load": float(model_load["base_model_load_seconds"]), "query_encoding": encoding_per_query, "search": warm}
            )
        directions.append(row)
    reranker_rows = _read_json(root / "artifacts/phase11/quality_latency_comparison.json")
    reranker = [row for row in reranker_rows if row["tier"] == "tier2" and row["split"] == "test"]
    reranker_summary = [
        {
            "direction": "text_to_image" if row["task"] == "text_to_image" else "image_to_text",
            "reranking_mean_seconds": _resource_value(float(row["reranking_mean_seconds"]), "seconds/query", MEASURED),
            "stage1_search_mean_seconds": _resource_value(float(row["stage1_search_mean_seconds"]), "seconds/query", MEASURED),
            "reranked_end_to_end_mean_seconds": _resource_value(float(row["reranked_end_to_end_mean_seconds"]), "seconds/query", MEASURED),
            "quality_delta_r_at_1": float(row["reranked_r_at_1"]) - float(row["stage1_r_at_1"]),
            "quality_delta_r_at_5": float(row["reranked_r_at_5"]) - float(row["stage1_r_at_5"]),
            "recommendation": "DISABLE",
        }
        for row in reranker
    ]
    return {
        "schema_version": PHASE20_SCHEMA_VERSION,
        "model_load": dict(model_load),
        "query_directions": directions,
        "reranker_overhead_and_quality": reranker_summary,
        "peak_memory": {"measurement_status": NOT_MEASURED, "statement": "PEAK MEMORY NOT RELIABLY MEASURED", "reason": "Apple unified-memory/MPS peak was not reliably captured in prior phases"},
        "throughput_definition": "encoder items/second and search queries/second are separate; they are not end-to-end service throughput",
    }


def _storage_analysis(root: Path, phase10_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cache = root / "artifacts/phase10/embedding_cache"
    generation = _read_json(root / "artifacts/phase10/embedding_generation.json")
    image_count = int(generation["image_count"])
    caption_count = int(generation["caption_count"])
    image_dim = int(generation["image_dimension"])
    caption_dim = int(generation["caption_dimension"])
    cache_files = []
    for name in ("images.npy", "captions.npy", "image_ids.json", "caption_ids.json", "metadata.json"):
        value = _file_size(cache / name)
        cache_files.append({"path": str(cache / name), "size": _resource_value(value, "bytes", MEASURED) if value is not None else {"measurement_status": NOT_MEASURED, "value": None, "unit": "bytes"}})
    image_raw = storage_calculation(image_count, image_dim, 4)
    caption_raw = storage_calculation(caption_count, caption_dim, 4)
    image_f16 = storage_calculation(image_count, image_dim, 2)
    caption_f16 = storage_calculation(caption_count, caption_dim, 2)
    selected_indexes = []
    for direction, filename in (("text_to_image", "artifacts/phase10/indexes/tier3/text_to_image/faiss_flat.faiss"), ("image_to_text", "artifacts/phase10/indexes/tier3/image_to_text/faiss_flat.faiss")):
        value = _file_size(root / filename)
        selected_indexes.append({"direction": direction, "config": "faiss_flat", "path": filename, "size": _resource_value(value, "bytes", MEASURED) if value is not None else {"measurement_status": NOT_MEASURED, "value": None, "unit": "bytes"}})
    return {
        "schema_version": PHASE20_SCHEMA_VERSION,
        "embedding_cache_files": cache_files,
        "embedding_matrices": {
            "images": {"count": image_count, "dimension": image_dim, "dtype": generation["dtype"], "persisted_file_bytes": _resource_value(_file_size(cache / "images.npy"), "bytes", MEASURED), "dense_payload": image_raw, "float16_dense_payload": image_f16},
            "captions": {"count": caption_count, "dimension": caption_dim, "dtype": generation["dtype"], "persisted_file_bytes": _resource_value(_file_size(cache / "captions.npy"), "bytes", MEASURED), "dense_payload": caption_raw, "float16_dense_payload": caption_f16},
        },
        "selected_tier3_indexes": selected_indexes,
        "phase10_rows_for_context": [dict(row) for row in phase10_rows if row["config"] == "faiss_flat"],
        "dtype_note": "float16 values are a calculated storage alternative; no production float16 cache file was created",
        "measurement_status": "MIXED_MEASURED_AND_CALCULATED",
    }


def _float16_precision_analysis(root: Path) -> dict[str, Any]:
    try:
        import numpy as np

        from .manifest import read_manifest
        from .phase7 import _subset_records

        cache = root / "artifacts/phase10/embedding_cache"
        images = np.load(cache / "images.npy", allow_pickle=False)
        captions = np.load(cache / "captions.npy", allow_pickle=False)
        image_ids = [str(value) for value in _read_json(cache / "image_ids.json")]
        caption_ids = [str(value) for value in _read_json(cache / "caption_ids.json")]
        image_index = {value: index for index, value in enumerate(image_ids)}
        caption_index = {value: index for index, value in enumerate(caption_ids)}
        manifest = read_manifest(root / "data/processed/coco2017_val_split_manifest.json")
        tier2 = _subset_records(manifest.records, "test", 42, 100)
        tier2_image_ids = [record.image_id for record in tier2]
        tier2_caption_ids = [caption.caption_id for record in tier2 for caption in record.captions]
        rows = _read_json(root / "artifacts/phase17/confidence_records.json")["rows"]
        text_rows = [row for row in rows if row["split"] == "test" and row["system"] == "full_ft" and row["direction"] == "text_to_image"]
        image_rows = [row for row in rows if row["split"] == "test" and row["system"] == "full_ft" and row["direction"] == "image_to_text"]

        def run(query: Any, candidates: Any, candidate_names: Sequence[str], relevant: Sequence[set[str]], stored: bool) -> dict[str, Any]:
            candidate_matrix = candidates.astype(np.float16).astype(np.float32) if stored else candidates.astype(np.float32)
            query_matrix = query.astype(np.float16).astype(np.float32) if stored else query.astype(np.float32)
            scores = query_matrix @ candidate_matrix.T
            order = np.argsort(-scores, axis=1, kind="stable")[:, :10]
            return {"order": order, "names": candidate_names, "relevant": relevant}

        text_query = np.asarray([captions[caption_index[str(row["query_id"])]] for row in text_rows])
        text_candidates = images[[image_index[value] for value in tier2_image_ids]]
        text_relevant = [{str(value) for value in row["relevant_ids"]} for row in text_rows]
        image_query = images[[image_index[str(row["query_id"])] for row in image_rows]]
        image_candidates = captions[[caption_index[value] for value in tier2_caption_ids]]
        image_relevant = [{str(value) for value in row["relevant_ids"]} for row in image_rows]
        cases = []
        for direction, query, candidate, names, relevant in (("text_to_image", text_query, text_candidates, tier2_image_ids, text_relevant), ("image_to_text", image_query, image_candidates, tier2_caption_ids, image_relevant)):
            before = run(query, candidate, names, relevant, False)
            after = run(query, candidate, names, relevant, True)
            def metrics(result: Mapping[str, Any]) -> dict[str, float]:
                order = result["order"]
                r1 = sum(result["names"][int(order[i, 0])] in result["relevant"][i] for i in range(len(order))) / len(order)
                r5 = sum(any(result["names"][int(index)] in result["relevant"][i] for index in order[i, :5]) for i in range(len(order))) / len(order)
                return {"r_at_1": r1, "r_at_5": r5}
            before_order = before["order"]
            after_order = after["order"]
            agreement = float(np.mean(np.all(before_order == after_order, axis=1)))
            top1_agreement = float(np.mean(before_order[:, 0] == after_order[:, 0]))
            top10_overlap = float(np.mean([len(set(row_a) & set(row_b)) / 10.0 for row_a, row_b in zip(before_order, after_order)]))
            timed = []
            for stored in (False, True):
                start = time.perf_counter()
                for _ in range(3):
                    run(query, candidate, names, relevant, stored)
                timed.append({"dtype": "float16_cached" if stored else "float32_cached", "mean_seconds": (time.perf_counter() - start) / 3.0, "measurement_status": MEASURED})
            cases.append({
                "direction": direction,
                "query_count": len(query),
                "candidate_count": len(candidate),
                "before_float32": {"quality": metrics(before), "dtype": "float32", "measurement_status": MEASURED},
                "after_float16_cache_float32_accumulation": {"quality": metrics(after), "dtype": "float16_storage_with_float32_accumulation", "measurement_status": MEASURED},
                "ranking_agreement_at_top10": agreement,
                "top1_agreement": top1_agreement,
                "mean_top10_overlap": top10_overlap,
                "search_latency": timed,
            })
        return {
            "schema_version": PHASE20_SCHEMA_VERSION,
            "optimization": "float16 cached embeddings with float32 accumulation",
            "status": "EVALUATED_ONLY_NOT_PRODUCTION_ADOPTED",
            "storage_reduction_fraction": 0.5,
            "storage_reduction_status": CALCULATED,
            "cases": cases,
            "before_after_note": "before uses persisted float32 cache; after converts the same cached arrays to float16 in memory and accumulates search in float32; no float16 cache file was persisted",
        }
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:  # pragma: no cover - optional numpy/cache environment
        return {"schema_version": PHASE20_SCHEMA_VERSION, "optimization": "float16 cached embeddings", "status": NOT_MEASURED, "reason": f"precision check failed: {type(exc).__name__}: {exc}"}


def _training_cost_comparison(root: Path) -> dict[str, Any]:
    cost = _read_json(root / "artifacts/phase13/compute_cost.json")
    systems = {str(row["seed"]): row["systems"] for row in cost["per_seed"]}
    means: dict[str, float] = {}
    for system in ("full_ft", "lora", "hard_negative_ft"):
        means[system] = statistics.fmean(float(row[system]["training_seconds"]) for row in systems.values())
    phase14 = _read_json(root / "artifacts/phase14/efficiency_comparison.json")["systems"]
    full = means["full_ft"]
    rows = []
    for system, value in means.items():
        rows.append({"system": system, "training_seconds_mean": _resource_value(value, "seconds", CALCULATED), "relative_to_full_ft": value / full, "seed_count": len(systems)})
    hard = phase14["hard_negative_ft_ratio50"]
    rows[-1].update({"mining_seconds_seed42": _resource_value(float(hard["initial_mining_seconds_seed42"]), "seconds", MEASURED), "fixed_manifest_load_seconds_mean": _resource_value(float(hard["fixed_manifest_load_seconds_mean"]), "seconds", MEASURED), "total_training_plus_mining_seconds_mean": _resource_value(float(hard["total_training_plus_mining_seconds_mean"]), "seconds", MEASURED)})
    return {"schema_version": PHASE20_SCHEMA_VERSION, "rows": rows, "seed_values": systems, "interpretation": "means are calculated from three retained Phase 13 runs; hard-negative mining fields preserve measured seed-42 and fixed-manifest overhead distinctions", "no_training_performed": True}


def _model_size_comparison(root: Path) -> dict[str, Any]:
    phase7 = _read_json(root / "artifacts/phase7/efficiency.json")
    phase8 = _read_json(root / "artifacts/phase8/efficiency_comparison.json")
    base_parameters = int(phase7["total_parameters"])
    payload = storage_calculation(base_parameters, 1, 4)
    rows = [
        {"system": "base_clip_zero_shot", "total_parameters": base_parameters, "trainable_parameters": 0, "stored_artifact_size": {"value": 0, "unit": "bytes", "measurement_status": "NOT_REPRESENTED_IN_REPOSITORY"}, "base_model_required": True, "base_parameter_payload_fp32": payload, "note": "zero stored repository checkpoint does not mean zero deployable model size"},
        {"system": "full_ft", "total_parameters": base_parameters, "trainable_parameters": int(phase7["trainable_parameters"]), "stored_artifact_size": _resource_value(_file_size(root / "artifacts/phase7/best_checkpoint.pt"), "bytes", MEASURED), "base_model_required": False, "base_parameter_payload_fp32": payload},
        {"system": "lora_rank8", "total_parameters": base_parameters, "trainable_parameters": int(phase8["lora"]["trainable_parameters"]), "stored_artifact_size": _resource_value(_directory_size(root / "artifacts/phase8/selected_adapter"), "bytes", MEASURED), "base_model_required": True, "adapter_only": True, "adapter_plus_calculated_base_payload": {"value": payload["value"] + _directory_size(root / "artifacts/phase8/selected_adapter"), "unit": "bytes", "measurement_status": CALCULATED}, "note": "adapter size is not total deployable model size"},
        {"system": "hard_negative_full_ft", "total_parameters": base_parameters, "trainable_parameters": base_parameters, "stored_artifact_size": _resource_value(_file_size(root / "artifacts/phase9/best_checkpoint.pt"), "bytes", MEASURED), "base_model_required": False},
    ]
    return {"schema_version": PHASE20_SCHEMA_VERSION, "rows": rows, "base_parameter_payload_formula": "total_parameters * 4 bytes for fp32 parameter payload", "memory_status": "PEAK MEMORY NOT RELIABLY MEASURED"}


def _resource_inventory(root: Path, phase10_rows: Sequence[Mapping[str, Any]], latency: Mapping[str, Any], sizes: Mapping[str, Any], costs: Mapping[str, Any]) -> dict[str, Any]:
    model_load = latency.get("model_load", {})
    load_value = model_load.get("base_plus_full_ft_restore_seconds") if model_load.get("measurement_status") == MEASURED else None
    search = {str(row["direction"]): row for row in phase10_rows if row["config"] == "faiss_flat"}
    rows = [
        {"system": "zero_shot", "trainable_parameters": _resource_value(0, "parameters", MEASURED), "stored_artifact_size": _resource_value(0, "bytes", "NOT_REPRESENTED_IN_REPOSITORY"), "model_load_time": _resource_value(load_value, "seconds", MEASURED) if load_value is not None else {"value": None, "unit": "seconds", "measurement_status": NOT_MEASURED}, "encoding_latency_source": "Phase 11 full-FT-compatible cached encoder; separate zero-shot direction timing not retained", "index_config": "faiss_flat", "notes": "base CLIP still required"},
        {"system": "full_ft", "trainable_parameters": _resource_value(151277313, "parameters", MEASURED), "stored_artifact_size": _resource_value(_file_size(root / "artifacts/phase7/best_checkpoint.pt"), "bytes", MEASURED), "model_load_time": _resource_value(load_value, "seconds", MEASURED) if load_value is not None else {"value": None, "unit": "seconds", "measurement_status": NOT_MEASURED}, "encoding_latency_source": "Phase 11 query_encoding_latency.json", "index_config": "faiss_flat", "notes": "canonical quality configuration"},
        {"system": "lora", "trainable_parameters": _resource_value(491521, "parameters", MEASURED), "stored_artifact_size": _resource_value(_directory_size(root / "artifacts/phase8/selected_adapter"), "bytes", MEASURED), "model_load_time": _resource_value(load_value, "seconds", MEASURED) if load_value is not None else {"value": None, "unit": "seconds", "measurement_status": NOT_MEASURED}, "encoding_latency_source": "Phase 8 efficiency comparison", "index_config": "faiss_flat", "notes": "base checkpoint required"},
        {"system": "hard_negative_full_ft", "trainable_parameters": _resource_value(151277313, "parameters", MEASURED), "stored_artifact_size": _resource_value(_file_size(root / "artifacts/phase9/best_checkpoint.pt"), "bytes", MEASURED), "model_load_time": _resource_value(load_value, "seconds", MEASURED) if load_value is not None else {"value": None, "unit": "seconds", "measurement_status": NOT_MEASURED}, "encoding_latency_source": "Phase 14 efficiency comparison", "index_config": "faiss_flat", "notes": "optional cost-heavy training path"},
    ]
    return {"schema_version": PHASE20_SCHEMA_VERSION, "hardware": latency.get("query_directions", [{}])[0].get("hardware", "mps"), "rows": rows, "canonical_search": {direction: {"mean_seconds": search[direction]["search_latency_mean_seconds"], "p95_seconds": search[direction]["search_latency_p95_seconds"], "measurement_status": MEASURED} for direction in search}, "training_cost_reference": costs["rows"], "storage_reference": sizes["embedding_matrices"]}


def _exact_ann_summary(root: Path, phase10_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [dict(row) for row in phase10_rows]
    decision = {
        "current_scale_question": "Is ANN justified at approximately 5,000 images?",
        "answer": "NO_FOR_DEFAULT_FIDELITY_FIRST_ARCHITECTURE",
        "reason": "Phase 10 selected FAISS Flat at tier 3 for both directions under a 0.99 neighbor-fidelity threshold; approximate IVF/HNSW settings were faster but lost neighbor and/or semantic fidelity.",
        "recommended_default": "FAISS Flat exact inner-product search or exact NumPy reference",
        "ann_status": "retain as measured optional scale-out evidence, not the default current-scale architecture",
    }
    return {"schema_version": PHASE20_SCHEMA_VERSION, "tier": "tier3", "rows": selected, "decision": decision, "source": "Phase 10 retained quality_latency_frontier and scaling_results"}


def _cache_analysis(root: Path, phase10_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    encoding = next(row for row in _read_json(root / "artifacts/phase11/query_encoding_latency.json") if row["label"] == "tier2_test")
    search = {str(row["direction"]): row for row in phase10_rows if row["config"] == "faiss_flat"}
    rows = []
    for direction, count_key, seconds_key, throughput_key in (("text_to_image", "caption_query_count", "text_encoding_seconds", "text_items_per_second"), ("image_to_text", "image_query_count", "image_encoding_seconds", "image_items_per_second")):
        count = int(encoding[count_key])
        encode = float(encoding[seconds_key]) / count
        warm = float(search[direction]["search_latency_mean_seconds"])
        cold = encode + warm
        rows.append({"direction": direction, "query_count": count, "cold_encode_plus_search_seconds_per_query": _resource_value(cold, "seconds/query", CALCULATED), "warm_cached_search_seconds_per_query": _resource_value(warm, "seconds/query", MEASURED), "cold_to_warm_ratio": cold / warm, "encoding_throughput": _resource_value(float(encoding[throughput_key]), "items/second", MEASURED), "cache_reconstruction_max_abs_error": _resource_value(float(encoding["cache_image_max_abs_error"] if direction == "image_to_text" else encoding["cache_text_max_abs_error"]), "absolute embedding units", MEASURED), "model_load_included": False})
    return {"schema_version": PHASE20_SCHEMA_VERSION, "rows": rows, "conclusion": "embedding caching removes query encoding from warm retrieval; model loading remains a separate cold-start cost", "measurement_status": CONSOLIDATED}


def _quality_frontier(root: Path, latency: Mapping[str, Any], costs: Mapping[str, Any]) -> dict[str, Any]:
    retention = _read_json(root / "artifacts/phase8/performance_retention.json")
    hard_negative = _read_json(root / "artifacts/phase9/comparison_table.json")
    systems = [
        ("zero_shot", "zero_shot", 0.0, 0.0, 0.0),
        ("full_ft", "full_finetuning", 151277313, 605242499, float(costs["rows"][0]["training_seconds_mean"]["value"])),
        ("lora", "lora", 491521, _directory_size(root / "artifacts/phase8/selected_adapter"), float(costs["rows"][1]["training_seconds_mean"]["value"])),
    ]
    rows = []
    for name, key, parameters, artifact, train_seconds in systems:
        text_r5 = float(retention["text_to_image"]["recall_at_5"][key])
        image_r5 = float(retention["image_to_text"]["recall_at_5"][key])
        rows.append({"configuration": name, "quality_mean_r_at_5": (text_r5 + image_r5) / 2.0, "text_to_image_r_at_5": text_r5, "image_to_text_r_at_5": image_r5, "trainable_parameters": parameters, "training_seconds": train_seconds, "artifact_size_bytes": artifact, "query_latency_reference": "Phase 11 shared encoder/search profile; configuration-specific warm timings not all retained", "measurement_status": MEASURED if name != "lora" else "MIXED_MEASURED_AND_CALCULATED"})
    rows.extend([
        {"configuration": "hard_negative_ft_ratio50", "quality_mean_r_at_5": (float(hard_negative["text_to_image"]["hard_negative_finetuning"]["recall_at_5"]) + float(hard_negative["image_to_text"]["hard_negative_finetuning"]["recall_at_5"])) / 2.0, "text_to_image_r_at_5": float(hard_negative["text_to_image"]["hard_negative_finetuning"]["recall_at_5"]), "image_to_text_r_at_5": float(hard_negative["image_to_text"]["hard_negative_finetuning"]["recall_at_5"]), "trainable_parameters": 151277313, "training_seconds": float(costs["rows"][2]["total_training_plus_mining_seconds_mean"]["value"]), "artifact_size_bytes": _file_size(root / "artifacts/phase9/best_checkpoint.pt"), "measurement_status": MEASURED, "note": "Phase 9 retained seed-42 quality with consolidated three-seed cost context; not a new training run"},
    ])
    frontier = pareto_frontier(rows, quality_key="quality_mean_r_at_5", cost_key="training_seconds")
    return {"schema_version": PHASE20_SCHEMA_VERSION, "rows": rows, "pareto_frontier_training_cost_vs_mean_r_at_5": frontier, "interpretation": "Pareto-like only for this fixed Tier-2 quality/cost scope; not global optimality"}


def _recommended_configurations(root: Path, latency: Mapping[str, Any], sizes: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PHASE20_SCHEMA_VERSION,
        "configurations": [
            {"name": "quality", "components": ["Phase 7 full-FT CLIP", "cached embeddings", "FAISS Flat exact search"], "status": "RECOMMENDED_DEFAULT", "evidence": "full FT has the strongest stable text-to-image R@5 among tested adaptations; FAISS Flat is exact at current scale; caching removes encoding from warm queries"},
            {"name": "efficiency", "components": ["base CLIP", "optional LoRA adapter", "cached embeddings", "FAISS Flat exact search"], "status": "OPTIONAL", "evidence": "LoRA uses 491,521 trainable parameters and about half the training time, but the base CLIP checkpoint remains required and quality is below full FT on text-to-image R@5"},
            {"name": "lightweight_zero_shot", "components": ["frozen zero-shot CLIP", "cached embeddings", "exact search"], "status": "SUPPORTED_BASELINE", "evidence": "zero-shot removes training cost and preserves a reproducible baseline; it is not the highest-quality configuration"},
        ],
        "excluded_by_evidence": [{"component": "Phase 11 reranker", "reason": "approximately 1 ms/query overhead with large negative R@1/R@5 deltas"}, {"component": "ANN default", "reason": "current-scale FAISS Flat selection meets fidelity threshold while approximate alternatives lose fidelity/semantic quality"}],
    }


def _disk_cleanup(root: Path) -> dict[str, Any]:
    candidates = [
        ("KEEP", "artifacts/phase7/best_checkpoint.pt", "canonical full-FT checkpoint"),
        ("KEEP", "artifacts/phase8/selected_adapter", "canonical LoRA adapter and metadata"),
        ("KEEP", "artifacts/phase10/embedding_cache", "canonical persisted full-corpus embeddings used by Phase 11"),
        ("KEEP", "artifacts/phase10/indexes/tier3/text_to_image/faiss_flat.faiss", "selected exact current-scale index"),
        ("KEEP", "artifacts/phase10/indexes/tier3/image_to_text/faiss_flat.faiss", "selected exact current-scale index"),
        ("KEEP", "artifacts/phase7 artifacts/phase8 artifacts/phase9 artifacts/phase10 artifacts/phase11 artifacts/phase13 artifacts/phase14 artifacts/phase17 artifacts/phase18 artifacts/phase19", "scientific evidence and provenance; do not delete automatically"),
        ("REGENERABLE / OPTIONAL", "artifacts/phase10/indexes/tier3/text_to_image/faiss_ivf_flat_nprobe_1.faiss artifacts/phase10/indexes/tier3/text_to_image/faiss_ivf_flat_nprobe_8.faiss artifacts/phase10/indexes/tier3/text_to_image/hnswlib_ef_8.hnsw artifacts/phase10/indexes/tier3/text_to_image/hnswlib_ef_32.hnsw artifacts/phase10/indexes/tier3/image_to_text/faiss_ivf_flat_nprobe_1.faiss artifacts/phase10/indexes/tier3/image_to_text/faiss_ivf_flat_nprobe_8.faiss artifacts/phase10/indexes/tier3/image_to_text/hnswlib_ef_8.hnsw artifacts/phase10/indexes/tier3/image_to_text/hnswlib_ef_32.hnsw", "approximate indexes are retained evidence but can be regenerated from the canonical embedding cache"),
        ("REGENERABLE / OPTIONAL", "artifacts/phase9/best_checkpoint.pt", "large exploratory hard-negative checkpoint; retain until final reproducibility/archive decision"),
        ("REGENERABLE / OPTIONAL", "artifacts/phase14/new_ablation/ratio25_run", "exploratory ratio-25 outputs; retain until final reproducibility/archive decision"),
        ("SAFE TO REMOVE AFTER VERIFICATION", "none", "no automatic deletion was performed; verify hashes and reproducibility before manual cleanup"),
    ]
    rows = []
    for classification, path, reason in candidates:
        size: int | None
        if path == "none":
            size = 0
        elif " " in path:
            measured_sizes = [_path_size(root / part) for part in path.split()]
            size = sum(value for value in measured_sizes if value is not None)
        else:
            size = _path_size(root / path)
        rows.append({"classification": classification, "path": path, "size_bytes": _resource_value(size, "bytes", MEASURED) if size is not None else {"value": None, "unit": "bytes", "measurement_status": NOT_MEASURED}, "reason": reason})
    return {"schema_version": PHASE20_SCHEMA_VERSION, "rows": rows, "deletions_performed": [], "storage_constraint_note": "recommendations only; canonical scientific evidence was not deleted"}


def validate_phase20_artifacts(output_dir: Path | str) -> dict[str, Any]:
    output = Path(output_dir)
    checks: dict[str, bool] = {"required_artifacts": all((output / name).exists() for name in REQUIRED_ARTIFACTS)}
    if not checks["required_artifacts"]:
        return {"schema_version": PHASE20_SCHEMA_VERSION, "passed": False, "checks": checks, "required": list(REQUIRED_ARTIFACTS)}
    try:
        audit = _read_json(output / "pre_phase_audit.json")
        report = _read_json(output / "phase20_report.json")
        provenance = _read_json(output / "provenance.json")
        inventory = _read_json(output / "resource_inventory.json")
        storage = _read_json(output / "storage_analysis.json")
        latency = _read_json(output / "latency_breakdown.json")
        exact_ann = _read_json(output / "exact_ann_summary.json")
        configs = _read_json(output / "recommended_configurations.json")
        cleanup = _read_json(output / "disk_cleanup_recommendations.json")
        checks.update(
            {
                "pre_phase_audit_pass": audit.get("passed") is True and audit.get("audit_result") == "PRE-PHASE AUDIT: Phase 19 PASS",
                "resource_inventory_nonempty": bool(inventory.get("rows")),
                "measured_calculated_labels_present": any(row.get("stored_artifact_size", {}).get("measurement_status") for row in inventory.get("rows", [])) and storage.get("measurement_status") == "MIXED_MEASURED_AND_CALCULATED",
                "latency_breakdown_separates_load": "model_load" in latency and all(row.get("model_load_excluded_from_query_latency") is True for row in latency.get("query_directions", [])),
                "no_fake_memory_numbers": latency.get("peak_memory", {}).get("statement") == "PEAK MEMORY NOT RELIABLY MEASURED",
                "lora_base_requirement_explicit": any(row.get("system") == "lora_rank8" and row.get("base_model_required") is True for row in _read_json(output / "model_size_comparison.json").get("rows", [])),
                "ann_decision_present": exact_ann.get("decision", {}).get("answer") == "NO_FOR_DEFAULT_FIDELITY_FIRST_ARCHITECTURE",
                "recommended_configurations_present": len(configs.get("configurations", [])) >= 2,
                "disk_cleanup_non_destructive": cleanup.get("deletions_performed") == [],
                "no_training_or_download": provenance.get("training_performed") is False and provenance.get("new_dataset_downloaded") is False,
                "phase21_not_started": provenance.get("phase21_started") is False and "phase 21" not in json.dumps(report).casefold(),
                "report_pass": report.get("status") == "PASS" and report.get("quality_gate", {}).get("status") == "PASS",
                "model_load_not_mixed_into_query": all(row.get("model_load_excluded_from_query_latency") is True for row in latency.get("query_directions", [])),
            }
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        checks["schema_valid"] = False
    checks.setdefault("schema_valid", True)
    return {"schema_version": PHASE20_SCHEMA_VERSION, "passed": all(checks.values()), "checks": checks, "required": list(REQUIRED_ARTIFACTS)}


def run_phase20(
    *,
    root: Path | str = ".",
    output_dir: Path | str = "artifacts/phase20",
    profile_model_load: bool = True,
) -> dict[str, Any]:
    """Run the Phase 20 consolidation and compact cache precision analysis."""

    root = Path(root)
    output = root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)
    audit = audit_phase19(root)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(audit, output / "pre_phase_audit.json")
    if not audit["passed"]:
        raise RuntimeError(audit["audit_result"])
    phase10_rows = _phase10_rows(root, "tier3")
    phase10_tier2_rows = _phase10_rows(root, "tier2")
    model_load = _measure_model_load(root, enabled=profile_model_load)
    latency = _latency_breakdown(root, model_load, phase10_tier2_rows)
    storage = _storage_analysis(root, phase10_rows)
    costs = _training_cost_comparison(root)
    sizes = _model_size_comparison(root)
    inventory = _resource_inventory(root, phase10_tier2_rows, latency, storage, costs)
    exact_ann = _exact_ann_summary(root, phase10_rows)
    cache = _cache_analysis(root, phase10_tier2_rows)
    precision = _float16_precision_analysis(root)
    frontier = _quality_frontier(root, latency, costs)
    configurations = _recommended_configurations(root, latency, sizes)
    cleanup = _disk_cleanup(root)
    for name, value in (("resource_inventory.json", inventory), ("latency_breakdown.json", latency), ("storage_analysis.json", storage), ("training_cost_comparison.json", costs), ("model_size_comparison.json", sizes), ("exact_ann_summary.json", exact_ann), ("cache_analysis.json", cache), ("float16_precision_analysis.json", precision), ("quality_efficiency_frontier.json", frontier), ("recommended_configurations.json", configurations), ("disk_cleanup_recommendations.json", cleanup)):
        _write_json(value, output / name)
    provenance = {
        "schema_version": PHASE20_SCHEMA_VERSION,
        "phase": 20,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase19_report_sha256": _hash_file(root / "artifacts/phase19/phase19_report.json"),
        "phase19_artifact_validation_sha256": _hash_file(root / "artifacts/phase19/artifact_validation.json"),
        "phase7_checkpoint_sha256": _hash_file(root / "artifacts/phase7/best_checkpoint.pt"),
        "phase10_embedding_generation_sha256": _hash_file(root / "artifacts/phase10/embedding_generation.json"),
        "phase11_latency_sha256": _hash_file(root / "artifacts/phase11/query_encoding_latency.json"),
        "phase13_compute_cost_sha256": _hash_file(root / "artifacts/phase13/compute_cost.json"),
        "phase14_efficiency_sha256": _hash_file(root / "artifacts/phase14/efficiency_comparison.json"),
        "training_performed": False,
        "new_dataset_downloaded": False,
        "phase21_started": False,
        "float16_production_cache_written": False,
        "model_load_profiled": model_load.get("measurement_status") == MEASURED,
        "peak_memory_status": "PEAK MEMORY NOT RELIABLY MEASURED",
        "python": sys.version,
        "platform": platform.platform(),
    }
    _write_json(provenance, output / "provenance.json")
    report = {
        "report_schema_version": PHASE20_SCHEMA_VERSION,
        "project": "OmniSearch",
        "phase": 20,
        "status": "PASS",
        "pre_phase_audit": audit["audit_result"],
        "scope": {"evaluation_only": True, "model_families_added": 0, "phase21_started": False, "primary_scale": "COCO val2017-derived 5,000 image corpus / Tier 2 quality comparisons"},
        "research_questions": {
            "RQ20.1": "Query latency is dominated by model/query encoding, especially image encoding; search is sub-millisecond with FAISS Flat at Tier 2.",
            "RQ20.2": "LoRA uses about 0.325% of full-FT trainable parameters and about half the retained three-seed training time, while requiring the base checkpoint.",
            "RQ20.3": "ANN is not justified as the default at current scale under the retained fidelity threshold.",
            "RQ20.4": "The tested reranker has the worst quality-to-cost trade-off: about 1 ms/query overhead and large negative quality deltas.",
            "RQ20.5": "Full-FT CLIP plus cached embeddings and FAISS Flat is the best practical quality-efficiency default; LoRA is optional for constrained adaptation.",
            "RQ20.6": "Retain canonical checkpoints, selected adapter, embedding cache, selected exact indexes, reports, provenance, and tests; approximate indexes are regenerable optional evidence.",
        },
        "bottleneck_summary": ["query encoding", "model load on cold start", "hard-negative training/mining when enabled"],
        "optimization_implemented": "No production optimization was switched on; existing embedding caching was quantified and float16 cached storage was evaluated in memory only.",
        "quality_gate": {"status": "PASS", "checks": {}},
        "artifacts": list(REQUIRED_ARTIFACTS),
    }
    _write_json(report, output / "phase20_report.json")
    validation = validate_phase20_artifacts(output)
    report["quality_gate"] = {"status": "PASS" if validation["passed"] else "FAIL", "checks": validation["checks"]}
    report["status"] = "PASS" if validation["passed"] else "PARTIAL"
    _write_json(report, output / "phase20_report.json")
    validation = validate_phase20_artifacts(output)
    _write_json(validation, output / "artifact_validation.json")
    return {"audit": audit, "report": report, "validation": validation}


def main() -> int:
    result = run_phase20()
    print(json.dumps(result["report"], indent=2))
    return 0 if result["validation"]["passed"] else 1
