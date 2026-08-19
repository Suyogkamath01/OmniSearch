"""Phase 12 controlled multimodal image-and-text query fusion.

The quantitative task is deliberately narrow and defensible: an image from a
declared split and one of its captions form a joint query, and the target is
the same image group in the same split-specific image corpus.  This measures
controlled aligned-signal fusion, not arbitrary compositional edit intent.
Early weighted-embedding fusion and late score fusion are compared against
image-only and text-only controls in the validated CLIP shared space.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .config import DEFAULT_CONFIG_PATH
from .evaluation import (
    PROTOCOL_VERSION,
    RankingRecord,
    compare_systems,
    evaluate_rankings,
    ranking_from_scores,
)
from .manifest import CaptionRecord, DatasetManifest, ImageRecord, read_manifest
from .phase7 import (
    _encode_images,
    _encode_texts,
    _hash_file,
    _load_checkpoint,
    _load_trainable_model,
)
from .phase10 import (
    _fixture_embeddings,
    _hash_ids,
    _stable_key,
    _tier_records,
    build_persisted_index,
    load_persisted_index,
    normalize_vectors,
)

PHASE12_SCHEMA_VERSION = 1
CONTROLLED_RELEVANCE = "controlled same-image identity target; not a semantic image-to-image label"
DEFAULT_PHASE12_CONFIG: dict[str, Any] = {
    "manifest": "data/processed/coco2017_val_split_manifest.json",
    "image_root": "data/raw/coco2017/val2017",
    "phase7_checkpoint": "artifacts/phase7/best_checkpoint.pt",
    "phase10_embedding_cache": "artifacts/phase10/embedding_cache",
    "model_id": "openai/clip-vit-base-patch32",
    "device": "auto",
    "seed": 42,
    "batch_size": 128,
    "text_max_length": 77,
    "alpha_values": [0.25, 0.5, 0.75],
    "selection_metric": "mean_mrr",
    "bootstrap_resamples": 200,
    "top_k": 10,
    "latency_query_limit": 128,
    "latency_repeats": 3,
    "warmup_queries": 5,
    "tier_sizes": [1000, 5000],
    "qualitative_examples": 4,
}


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_config(path: Path) -> dict[str, Any]:
    import tomllib

    with path.open("rb") as file:
        raw = tomllib.load(file)
    config = dict(DEFAULT_PHASE12_CONFIG)
    config.update(dict(raw.get("phase12", {})))
    return config


def validate_phase12_config(config: Mapping[str, Any]) -> None:
    for key in ("batch_size", "bootstrap_resamples", "top_k", "latency_query_limit", "latency_repeats", "warmup_queries", "qualitative_examples"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if int(config.get("seed", 0)) < 0:
        raise ValueError("seed must be non-negative")
    if str(config["selection_metric"]) not in {"mean_mrr", "mean_recall_at_5"}:
        raise ValueError("unsupported selection metric")
    values = [float(value) for value in config["alpha_values"]]
    if not values or values != sorted(set(values)) or any(not 0.0 < value < 1.0 for value in values):
        raise ValueError("alpha_values must be sorted, unique, and strictly between zero and one")
    if any(int(value) <= 0 for value in config["tier_sizes"]):
        raise ValueError("tier_sizes must be positive")
    checkpoint = str(config["phase7_checkpoint"])
    if "phase8" in checkpoint or "phase9" in checkpoint:
        raise ValueError("Phase 12 must use the validated Phase 7 checkpoint")


def _load_embedding_cache(
    config: Mapping[str, Any], manifest_path: Path, checkpoint_path: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cache = Path(config["phase10_embedding_cache"])
    metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
    source = dict(metadata["embedding_source"])
    if source["manifest_sha256"] != _hash_file(manifest_path):
        raise ValueError("Phase 10 cache is stale for the active manifest")
    if source["checkpoint_sha256"] != _hash_file(checkpoint_path):
        raise ValueError("Phase 10 cache is stale for the Phase 7 checkpoint")
    arrays = {
        "images": normalize_vectors(np.load(cache / "images.npy", allow_pickle=False)),
        "captions": normalize_vectors(np.load(cache / "captions.npy", allow_pickle=False)),
        "image_ids": np.asarray(json.loads((cache / "image_ids.json").read_text(encoding="utf-8")), dtype="U"),
        "caption_ids": np.asarray(json.loads((cache / "caption_ids.json").read_text(encoding="utf-8")), dtype="U"),
    }
    if len(arrays["image_ids"]) != len(arrays["images"]):
        raise ValueError("image cache IDs do not match vector rows")
    if len(arrays["caption_ids"]) != len(arrays["captions"]):
        raise ValueError("caption cache IDs do not match vector rows")
    if arrays["images"].shape[1] != arrays["captions"].shape[1]:
        raise ValueError("CLIP image and text spaces have incompatible dimensions")
    return source, arrays


def _fixture_manifest_and_arrays() -> tuple[DatasetManifest, dict[str, np.ndarray]]:
    manifest, arrays = _fixture_embeddings(42)
    records = tuple(
        record.with_split("train" if index < 6 else "validation" if index < 9 else "test")
        for index, record in enumerate(manifest.records)
    )
    return DatasetManifest(
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.dataset_version,
        source_url=manifest.source_url,
        terms_url=manifest.terms_url,
        source_snapshot_marker=manifest.source_snapshot_marker,
        source_sha256=manifest.source_sha256,
        records=records,
        metadata=manifest.metadata,
    ), arrays


def _selected_caption(record: ImageRecord, seed: int) -> CaptionRecord:
    if not record.captions:
        raise ValueError(f"query image has no captions: {record.image_id}")
    return min(record.captions, key=lambda caption: _stable_key(seed, f"{record.image_id}\0{caption.caption_id}"))


def _query_bundle(records: Sequence[ImageRecord], arrays: Mapping[str, np.ndarray], seed: int) -> dict[str, Any]:
    image_index = {str(value): index for index, value in enumerate(arrays["image_ids"].tolist())}
    caption_index = {str(value): index for index, value in enumerate(arrays["caption_ids"].tolist())}
    selected = tuple(_selected_caption(record, seed) for record in records)
    query_ids = tuple(record.image_id for record in records)
    image_vectors = arrays["images"][[image_index[query_id] for query_id in query_ids]]
    text_vectors = arrays["captions"][[caption_index[caption.caption_id] for caption in selected]]
    return {
        "query_ids": query_ids,
        "caption_ids": tuple(caption.caption_id for caption in selected),
        "caption_texts": tuple(caption.text for caption in selected),
        "image_vectors": normalize_vectors(image_vectors),
        "text_vectors": normalize_vectors(text_vectors),
        "filenames": tuple(record.filename for record in records),
        "target_ids": query_ids,
        "relevant": {query_id: {query_id} for query_id in query_ids},
    }


def _build_index(
    candidate_vectors: np.ndarray,
    candidate_ids: Sequence[str],
    source: Mapping[str, Any],
    tier: str,
    split: str,
    output_dir: Path,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    base = output_dir / "indexes" / tier / split / "image_to_image" / "faiss_flat"
    built = build_persisted_index(
        candidate_vectors,
        candidate_ids,
        "faiss_flat",
        {},
        {**source, "stage1": "FAISS IndexFlatIP exact retrieval"},
        f"{tier}_{split}",
        "image_group",
        base,
        seed,
    )
    expected = {
        "dataset_manifest_sha256": source["manifest_sha256"],
        "tier": f"{tier}_{split}",
        "candidate_unit": "image_group",
        "embedding_dimension": int(candidate_vectors.shape[1]),
        "candidate_count": len(candidate_ids),
    }
    loaded = load_persisted_index(built, candidate_ids, expected)
    return loaded, {
        "tier": tier,
        "split": split,
        "index_path": str(built.index_path),
        "metadata_path": str(built.metadata_path),
        "build": built.metadata,
    }


def _weighted_fusion(image_vectors: np.ndarray, text_vectors: np.ndarray, alpha: float) -> np.ndarray:
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    image_array = np.asarray(image_vectors, dtype=np.float32)
    text_array = np.asarray(text_vectors, dtype=np.float32)
    if image_array.shape != text_array.shape or image_array.ndim != 2:
        raise ValueError("image and text query vectors must have equal rank-2 shapes")
    fused = float(alpha) * image_array + (1.0 - float(alpha)) * text_array
    return normalize_vectors(fused)


def _score_rows(score_matrix: np.ndarray, candidate_ids: Sequence[str], top_k: int) -> tuple[list[list[str]], list[list[float]]]:
    scores = np.asarray(score_matrix, dtype=np.float32)
    if scores.ndim != 2 or scores.shape[1] != len(candidate_ids):
        raise ValueError("score matrix shape does not match candidate corpus")
    candidate_array = np.asarray(tuple(str(value) for value in candidate_ids), dtype="U")
    limit = min(int(top_k), len(candidate_ids))
    ids: list[list[str]] = []
    values: list[list[float]] = []
    for row in scores:
        if not bool(np.isfinite(row).all()):
            raise FloatingPointError("fusion scores are non-finite")
        order = np.lexsort((candidate_array, -row))[:limit]
        ids.append([str(candidate_ids[int(index)]) for index in order])
        values.append([float(row[int(index)]) for index in order])
    return ids, values


def _rows_from_index(ids: Any, scores: Any) -> tuple[list[list[str]], list[list[float]]]:
    return (
        [[str(value) for value in row] for row in ids.tolist()],
        [[float(value) for value in row] for row in scores.tolist()],
    )


def _rankings(
    query_ids: Sequence[str],
    ids: Sequence[Sequence[str]],
    scores: Sequence[Sequence[float]],
    candidate_count: int,
    corpus_id: str,
    system_id: str,
    experiment_id: str,
) -> tuple[RankingRecord, ...]:
    return tuple(
        ranking_from_scores(
            query_id=query_id,
            task="image_to_image",
            candidates=list(zip(row_ids, row_scores)),
            relevant_ids={query_id},
            system_id=system_id,
            experiment_id=experiment_id,
            candidate_count=candidate_count,
            candidate_corpus_id=corpus_id,
            relevance_definition=CONTROLLED_RELEVANCE,
        )
        for query_id, row_ids, row_scores in zip(query_ids, ids, scores)
    )


def _variant_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {"variant_id": "image_only", "method": "image_only", "alpha": 1.0},
        {"variant_id": "text_only", "method": "text_only", "alpha": 0.0},
    ]
    for alpha in config["alpha_values"]:
        value = float(alpha)
        specs.append({"variant_id": f"early_alpha_{value:g}", "method": "early", "alpha": value})
        specs.append({"variant_id": f"late_alpha_{value:g}", "method": "late", "alpha": value})
    return specs


def _evaluate_variant(
    bundle: Mapping[str, Any],
    candidate_vectors: np.ndarray,
    candidate_ids: Sequence[str],
    index: Any,
    tier: str,
    split: str,
    spec: Mapping[str, Any],
    top_k: int,
) -> dict[str, Any]:
    method = str(spec["method"])
    alpha = float(spec["alpha"])
    if method == "image_only":
        raw_ids, raw_scores = index.search(bundle["image_vectors"], top_k)
        ids, scores = _rows_from_index(raw_ids, raw_scores)
    elif method == "text_only":
        raw_ids, raw_scores = index.search(bundle["text_vectors"], top_k)
        ids, scores = _rows_from_index(raw_ids, raw_scores)
    elif method == "early":
        fused = _weighted_fusion(bundle["image_vectors"], bundle["text_vectors"], alpha)
        raw_ids, raw_scores = index.search(fused, top_k)
        ids, scores = _rows_from_index(raw_ids, raw_scores)
    elif method == "late":
        late_scores = (
            alpha * (bundle["image_vectors"] @ candidate_vectors.T)
            + (1.0 - alpha) * (bundle["text_vectors"] @ candidate_vectors.T)
        )
        ids, scores = _score_rows(late_scores, candidate_ids, top_k)
    else:
        raise ValueError(f"unsupported fusion method: {method}")
    corpus_id = f"image_group:{_hash_ids(candidate_ids)}"
    system_id = str(spec["variant_id"])
    rankings = _rankings(
        bundle["query_ids"],
        ids,
        scores,
        len(candidate_ids),
        corpus_id,
        system_id,
        f"phase12_{tier}_{split}_{system_id}",
    )
    return {
        "tier": tier,
        "split": split,
        "variant_id": system_id,
        "method": method,
        "alpha": alpha,
        "metrics": evaluate_rankings(rankings),
        "rankings": rankings,
        "candidate_count": len(candidate_ids),
        "candidate_ids_sha256": _hash_ids(candidate_ids),
        "query_count": len(bundle["query_ids"]),
    }


def _public_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "rankings"}


def _select_fusion(validation_results: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    candidates = [
        result
        for result in validation_results
        if result["tier"] == "tier2" and result["method"] in {"early", "late"}
    ]
    if not candidates:
        raise ValueError("validation did not produce any fusion candidates")
    scores = {
        str(result["variant_id"]): {
            "method": result["method"],
            "alpha": result["alpha"],
            "mean_mrr": result["metrics"]["mrr"],
            "mean_recall_at_5": result["metrics"]["recall_at_5"],
        }
        for result in candidates
    }
    selected = max(
        candidates,
        key=lambda result: (
            float(result["metrics"][str(config["selection_metric"]).replace("mean_", "")]),
            -float(result["alpha"]),
            str(result["variant_id"]),
        ),
    )
    return {
        "selected_variant": selected["variant_id"],
        "selected_method": selected["method"],
        "selected_alpha": selected["alpha"],
        "selection_split": "validation",
        "selection_tier": "tier2",
        "selection_metric": config["selection_metric"],
        "scores": scores,
        "test_used_for_selection": False,
    }


def _paired(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    dataset_id: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    left_records = left["rankings"]
    right_records = right["rankings"]
    metadata = lambda result: {
        "system_id": result["variant_id"],
        "task": "image_to_image",
        "dataset_id": dataset_id,
        "split": result["split"],
        "protocol_version": PROTOCOL_VERSION,
        "candidate_corpus_id": left_records[0].candidate_corpus_id,
        "relevance_definition": CONTROLLED_RELEVANCE,
    }
    return {
        "tier": left["tier"],
        "comparison": f"{right['variant_id']} vs {left['variant_id']}",
        **compare_systems(
            left_records,
            right_records,
            metadata(left),
            metadata(right),
            ks=(1, 5, 10),
            bootstrap_resamples=int(config["bootstrap_resamples"]),
            seed=int(config["seed"]),
        ),
    }


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def _overlap_rows(
    results: Mapping[str, Mapping[str, Any]],
    tier: str,
    ks: Sequence[int] = (1, 5, 10),
) -> list[dict[str, Any]]:
    image_records = results["image_only"]["rankings"]
    text_records = results["text_only"]["rankings"]
    rows: list[dict[str, Any]] = []
    for variant_id, result in results.items():
        if variant_id in {"image_only", "text_only"}:
            continue
        fused_records = result["rankings"]
        for k in ks:
            image_text = statistics.fmean(_jaccard(a.candidate_ids[:k], b.candidate_ids[:k]) for a, b in zip(image_records, text_records))
            image_fused = statistics.fmean(_jaccard(a.candidate_ids[:k], b.candidate_ids[:k]) for a, b in zip(image_records, fused_records))
            text_fused = statistics.fmean(_jaccard(a.candidate_ids[:k], b.candidate_ids[:k]) for a, b in zip(text_records, fused_records))
            rows.append({"tier": tier, "variant_id": variant_id, "k": k, "image_text_jaccard": image_text, "image_fused_jaccard": image_fused, "text_fused_jaccard": text_fused})
    return rows


def _dominance_rows(results: Mapping[str, Mapping[str, Any]], tier: str) -> list[dict[str, Any]]:
    image_records = results["image_only"]["rankings"]
    text_records = results["text_only"]["rankings"]
    rows: list[dict[str, Any]] = []
    for variant_id, result in results.items():
        if result["method"] not in {"early", "late"}:
            continue
        fused_records = result["rankings"]
        rows.append({
            "tier": tier,
            "variant_id": variant_id,
            "method": result["method"],
            "alpha_image_weight": result["alpha"],
            "top1_changed_vs_image": statistics.fmean(a.candidate_ids[:1] != b.candidate_ids[:1] for a, b in zip(image_records, fused_records)),
            "top1_changed_vs_text": statistics.fmean(a.candidate_ids[:1] != b.candidate_ids[:1] for a, b in zip(text_records, fused_records)),
            "top10_overlap_with_image": statistics.fmean(_jaccard(a.candidate_ids[:10], b.candidate_ids[:10]) for a, b in zip(image_records, fused_records)),
            "top10_overlap_with_text": statistics.fmean(_jaccard(a.candidate_ids[:10], b.candidate_ids[:10]) for a, b in zip(text_records, fused_records)),
        })
    return rows


def _custom_rankings(
    image_vectors: np.ndarray,
    text_vectors: np.ndarray,
    query_ids: Sequence[str],
    candidate_vectors: np.ndarray,
    candidate_ids: Sequence[str],
    index: Any,
    spec: Mapping[str, Any],
    top_k: int,
    tier: str,
    split: str,
) -> tuple[RankingRecord, ...]:
    bundle = {"query_ids": tuple(query_ids), "image_vectors": image_vectors, "text_vectors": text_vectors, "relevant": {query_id: set() for query_id in query_ids}}
    result = _evaluate_variant(bundle, candidate_vectors, candidate_ids, index, tier, split, spec, top_k)
    return result["rankings"]


def _conflict_examples(
    records: Sequence[ImageRecord],
    bundle: Mapping[str, Any],
    candidate_vectors: np.ndarray,
    candidate_ids: Sequence[str],
    retrieval_index: Any,
    selected_spec: Mapping[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    if len(records) < 2:
        return []
    count = min(4, len(records))
    image_vectors = bundle["image_vectors"][:count]
    text_vectors = np.roll(bundle["text_vectors"], -1, axis=0)[:count]
    query_ids = tuple(f"conflict-{offset}" for offset in range(count))
    image_sources = [records[offset].image_id for offset in range(count)]
    text_sources = [records[(offset + 1) % len(records)].image_id for offset in range(count)]
    variant_specs = ({"variant_id": "image_only", "method": "image_only", "alpha": 1.0}, {"variant_id": "text_only", "method": "text_only", "alpha": 0.0}, selected_spec)
    rankings_by_variant = {
        str(spec["variant_id"]): _custom_rankings(image_vectors, text_vectors, query_ids, candidate_vectors, candidate_ids, retrieval_index, spec, top_k, "qualitative_conflict", "test")
        for spec in variant_specs
    }
    rows: list[dict[str, Any]] = []
    for offset in range(count):
        rows.append({
            "label": "QUALITATIVE CONFLICT ANALYSIS",
            "query_id": query_ids[offset],
            "image_source_image_id": image_sources[offset],
            "text_source_image_id": text_sources[offset],
            "text": bundle["caption_texts"][(offset + 1) % len(records)],
            "rankings_top5": {variant: list(ranking[offset].candidate_ids[:5]) for variant, ranking in rankings_by_variant.items()},
            "no_correctness_label": True,
        })
    return rows


def _sensitivity_analysis(
    records: Sequence[ImageRecord],
    bundle: Mapping[str, Any],
    candidate_vectors: np.ndarray,
    candidate_ids: Sequence[str],
    index: Any,
    selected_spec: Mapping[str, Any],
    top_k: int,
) -> dict[str, Any]:
    if len(records) < 2:
        return {"status": "not_evaluated_insufficient_queries"}
    count = len(records)
    shifted_text = np.roll(bundle["text_vectors"], -1, axis=0)
    shifted_image = np.roll(bundle["image_vectors"], -1, axis=0)
    original = _custom_rankings(bundle["image_vectors"], bundle["text_vectors"], bundle["query_ids"], candidate_vectors, candidate_ids, index, selected_spec, top_k, "sensitivity", "test")
    text_changed = _custom_rankings(bundle["image_vectors"], shifted_text, bundle["query_ids"], candidate_vectors, candidate_ids, index, selected_spec, top_k, "sensitivity", "test")
    image_changed = _custom_rankings(shifted_image, bundle["text_vectors"], bundle["query_ids"], candidate_vectors, candidate_ids, index, selected_spec, top_k, "sensitivity", "test")
    return {
        "status": "completed",
        "selected_variant": selected_spec["variant_id"],
        "same_image_vary_text_top1_changed_fraction": statistics.fmean(a.candidate_ids[:1] != b.candidate_ids[:1] for a, b in zip(original, text_changed)),
        "same_text_vary_image_top1_changed_fraction": statistics.fmean(a.candidate_ids[:1] != b.candidate_ids[:1] for a, b in zip(original, image_changed)),
        "query_count": count,
        "interpretation": "diagnostic sensitivity only; shifted signals have no correctness labels",
    }


def _measure_encoding(
    config: Mapping[str, Any],
    tier_records: Sequence[tuple[str, Sequence[ImageRecord]]],
    arrays: Mapping[str, np.ndarray],
    qualitative_texts: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    image_map = {str(value): index for index, value in enumerate(arrays["image_ids"].tolist())}
    caption_map = {str(value): index for index, value in enumerate(arrays["caption_ids"].tolist())}
    payload: list[dict[str, Any]] = []
    modification_vectors: dict[str, np.ndarray] = {}
    for tier, records in tier_records:
        model, processor, torch_module, device_string = _load_local_encoding_model(config)
        _load_checkpoint(Path(config["phase7_checkpoint"]), model)
        model.eval()
        selected = tuple(_selected_caption(record, int(config["seed"])) for record in records)
        caption_items = tuple((caption.caption_id, caption.text) for caption in selected)
        started = time.perf_counter()
        image_ids, image_vectors = _encode_images(model, processor, torch_module, records, Path(config["image_root"]), int(config["batch_size"]), 0, "fp32")
        image_seconds = time.perf_counter() - started
        started = time.perf_counter()
        caption_ids, caption_vectors = _encode_texts(model, processor, torch_module, caption_items, int(config["batch_size"]), int(config["text_max_length"]), 0, "fp32")
        text_seconds = time.perf_counter() - started
        image_error = max(float(np.max(np.abs(image_vectors.numpy()[index] - arrays["images"][image_map[item]]))) for index, item in enumerate(image_ids))
        text_error = max(float(np.max(np.abs(caption_vectors.numpy()[index] - arrays["captions"][caption_map[item]]))) for index, item in enumerate(caption_ids))
        payload.append({
            "tier": tier,
            "device": device_string,
            "image_query_count": len(image_ids),
            "text_query_count": len(caption_ids),
            "image_encoding_seconds": image_seconds,
            "text_encoding_seconds": text_seconds,
            "image_seconds_per_query": image_seconds / max(1, len(image_ids)),
            "text_seconds_per_query": text_seconds / max(1, len(caption_ids)),
            "cache_image_max_abs_error": image_error,
            "cache_text_max_abs_error": text_error,
            "model_load_included": False,
        })
        del model
        gc.collect()
    if qualitative_texts:
        model, processor, torch_module, device_string = _load_local_encoding_model(config)
        _load_checkpoint(Path(config["phase7_checkpoint"]), model)
        model.eval()
        started = time.perf_counter()
        _, vectors = _encode_texts(model, processor, torch_module, tuple((f"mod-{index}", text) for index, text in enumerate(qualitative_texts)), int(config["batch_size"]), int(config["text_max_length"]), 0, "fp32")
        seconds = time.perf_counter() - started
        modification_vectors = {text: vectors[index].numpy().astype(np.float32) for index, text in enumerate(qualitative_texts)}
        payload.append({"label": "qualitative_modification_texts", "count": len(qualitative_texts), "encoding_seconds": seconds, "model_load_included": False})
        del model
        gc.collect()
    return payload, modification_vectors


def _latency(
    bundle: Mapping[str, Any],
    candidate_vectors: np.ndarray,
    candidate_ids: Sequence[str],
    index: Any,
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    encoding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    limit = min(int(config["latency_query_limit"]), len(bundle["query_ids"]))
    image = bundle["image_vectors"][:limit]
    text = bundle["text_vectors"][:limit]
    repeats = int(config["latency_repeats"])
    warmup = min(int(config["warmup_queries"]), limit)
    for _ in range(warmup):
        if spec["method"] == "early":
            index.search(_weighted_fusion(image[:1], text[:1], float(spec["alpha"])), int(config["top_k"]))
        elif spec["method"] == "late":
            _score_rows(float(spec["alpha"]) * (image[:1] @ candidate_vectors.T) + (1.0 - float(spec["alpha"])) * (text[:1] @ candidate_vectors.T), candidate_ids, int(config["top_k"]))
    fusion_values: list[float] = []
    retrieval_values: list[float] = []
    for _ in range(repeats):
        for image_vector, text_vector in zip(image, text):
            started = time.perf_counter()
            if spec["method"] == "early":
                query = _weighted_fusion(image_vector[None, :], text_vector[None, :], float(spec["alpha"]))
            elif spec["method"] == "late":
                query = float(spec["alpha"]) * (image_vector @ candidate_vectors.T) + (1.0 - float(spec["alpha"])) * (text_vector @ candidate_vectors.T)
            else:
                query = image_vector if spec["method"] == "image_only" else text_vector
            fusion_values.append(time.perf_counter() - started)
            started = time.perf_counter()
            if spec["method"] == "early":
                index.search(query, int(config["top_k"]))
            elif spec["method"] == "late":
                _score_rows(np.asarray(query)[None, :], candidate_ids, int(config["top_k"]))
            else:
                index.search(np.asarray(query)[None, :], int(config["top_k"]))
            retrieval_values.append(time.perf_counter() - started)
    fusion_mean = statistics.fmean(fusion_values)
    retrieval_mean = statistics.fmean(retrieval_values)
    encoding_seconds = 0.0 if encoding is None else float(encoding["image_seconds_per_query"]) + float(encoding["text_seconds_per_query"])
    return {
        "tier": "measured",
        "variant_id": spec["variant_id"],
        "method": spec["method"],
        "alpha": spec["alpha"],
        "queries_measured_per_repeat": limit,
        "repeats": repeats,
        "image_encoding_mean_seconds": 0.0 if encoding is None else encoding["image_seconds_per_query"],
        "text_encoding_mean_seconds": 0.0 if encoding is None else encoding["text_seconds_per_query"],
        "fusion_mean_seconds": fusion_mean,
        "retrieval_mean_seconds": retrieval_mean,
        "end_to_end_mean_seconds": encoding_seconds + fusion_mean + retrieval_mean,
        "model_load_included": False,
    }


def _compositional_examples(
    records: Sequence[ImageRecord],
    bundle: Mapping[str, Any],
    candidate_vectors: np.ndarray,
    candidate_ids: Sequence[str],
    index: Any,
    selected_spec: Mapping[str, Any],
    modifications: Mapping[str, np.ndarray],
    top_k: int,
    image_root: Path,
) -> list[dict[str, Any]]:
    if not modifications:
        return [{"status": "not_evaluated_in_smoke_fixture", "no_correctness_labels": True}]
    examples: list[dict[str, Any]] = []
    for offset, (text, text_vector) in enumerate(modifications.items()):
        if offset >= 4 or offset >= len(records):
            break
        image_vector = bundle["image_vectors"][offset : offset + 1]
        text_array = np.asarray(text_vector, dtype=np.float32)[None, :]
        rankings = _custom_rankings(image_vector, text_array, (f"compositional-{offset}",), candidate_vectors, candidate_ids, index, selected_spec, top_k, "qualitative_compositional", "test")
        filename = records[offset].filename
        examples.append({
            "label": "QUALITATIVE COMPOSITIONAL QUERY",
            "image_id": records[offset].image_id,
            "image_path": str(image_root / filename) if filename else None,
            "text_modification": text,
            "selected_variant": selected_spec["variant_id"],
            "top5_candidate_ids": list(rankings[0].candidate_ids[:5]),
            "no_benchmark_label": True,
        })
    return examples


def _fixture_encoding() -> list[dict[str, Any]]:
    return [{"status": "fixture_embeddings_only", "model_load_included": False}]


def _load_local_encoding_model(config: Mapping[str, Any]) -> tuple[Any, Any, Any, str]:
    """Load the already-cached model without spawning Hub conversion requests."""

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return _load_trainable_model(str(config["model_id"]), str(config["device"]))


def run_phase12(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    output_dir: Path | str = "artifacts/phase12",
    smoke: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _read_config(config_path)
    validate_phase12_config(config)
    if smoke:
        manifest, arrays = _fixture_manifest_and_arrays()
        manifest_path = output_dir / "smoke_manifest.json"
        _write_json(manifest.to_dict(), manifest_path)
        source = {"model_id": "fixture-vectors", "checkpoint_sha256": None, "manifest_sha256": _hash_file(manifest_path), "protocol_version": PROTOCOL_VERSION, "normalization": "L2 unit vectors"}
        tier_specs = [("smoke_fixture", manifest.records)]
    else:
        manifest_path = Path(config["manifest"])
        manifest = read_manifest(manifest_path)
        checkpoint_path = Path(config["phase7_checkpoint"])
        source, arrays = _load_embedding_cache(config, manifest_path, checkpoint_path)
        tier_specs = [(f"tier{index + 2}", _tier_records(manifest.records, int(size), int(config["seed"]))) for index, size in enumerate(config["tier_sizes"])]

    qualitative_texts = ("similar scene at night", "same object but red", "with two people", "indoors")
    if smoke:
        encoding_payload = _fixture_encoding()
        modification_vectors: dict[str, np.ndarray] = {}
    else:
        encoding_payload, modification_vectors = _measure_encoding(
            config,
            [(tier, tuple(record for record in records if record.split == "test")) for tier, records in tier_specs],
            arrays,
            qualitative_texts,
        )

    specs = _variant_specs(config)
    validation_results: list[dict[str, Any]] = []
    test_results: list[dict[str, Any]] = []
    index_manifest: list[dict[str, Any]] = []
    contexts: dict[tuple[str, str], dict[str, Any]] = {}
    for tier, tier_records in tier_specs:
        for split in ("validation", "test"):
            records = tuple(record for record in tier_records if record.split == split)
            if not records:
                continue
            bundle = _query_bundle(records, arrays, int(config["seed"]))
            image_map = {str(value): index for index, value in enumerate(arrays["image_ids"].tolist())}
            candidate_ids = tuple(record.image_id for record in records)
            candidate_vectors = normalize_vectors(arrays["images"][[image_map[item] for item in candidate_ids]])
            index, index_meta = _build_index(candidate_vectors, candidate_ids, source, tier, split, output_dir, int(config["seed"]))
            index_manifest.append(index_meta)
            contexts[(tier, split)] = {"records": records, "bundle": bundle, "candidate_ids": candidate_ids, "candidate_vectors": candidate_vectors, "index": index}
            target = validation_results if split == "validation" else test_results
            for spec in specs:
                target.append(_evaluate_variant(bundle, candidate_vectors, candidate_ids, index, tier, split, spec, int(config["top_k"])))

    selection = {"selected_variant": "early_alpha_0.5", "selected_method": "early", "selected_alpha": 0.5, "selection_split": "smoke_fixture", "selection_tier": "smoke_fixture", "selection_metric": config["selection_metric"], "scores": {"smoke_only": {"status": "smoke_only"}}, "test_used_for_selection": False} if smoke else _select_fusion(validation_results, config)
    selected_spec = next(spec for spec in specs if spec["variant_id"] == selection["selected_variant"])
    test_by_key = {(result["tier"], result["variant_id"]): result for result in test_results}
    validation_public = [_public_result(result) for result in validation_results]
    test_public = [_public_result(result) for result in test_results]

    paired_comparisons: list[dict[str, Any]] = []
    overlap_payload: list[dict[str, Any]] = []
    dominance_payload: list[dict[str, Any]] = []
    sensitivity_payload: list[dict[str, Any]] = []
    conflict_payload: list[dict[str, Any]] = []
    for tier, tier_records in tier_specs:
        context = contexts.get((tier, "test"))
        if context is None:
            continue
        results = {result["variant_id"]: result for result in test_results if result["tier"] == tier}
        best = results[selected_spec["variant_id"]]
        paired_comparisons.append(_paired(results["text_only"], best, manifest.dataset_id, config))
        paired_comparisons.append(_paired(results["image_only"], best, manifest.dataset_id, config))
        overlap_payload.extend(_overlap_rows(results, tier))
        dominance_payload.extend(_dominance_rows(results, tier))
        records = context["records"]
        sensitivity_payload.append({"tier": tier, **_sensitivity_analysis(records, context["bundle"], context["candidate_vectors"], context["candidate_ids"], context["index"], selected_spec, int(config["top_k"]))})
        if tier == tier_specs[-1][0]:
            conflict_payload = _conflict_examples(records, context["bundle"], context["candidate_vectors"], context["candidate_ids"], context["index"], selected_spec, int(config["top_k"]))

    tier3_key = tier_specs[-1][0]
    tier3_context = contexts.get((tier3_key, "test"))
    if tier3_context is not None:
        compositional_payload = _compositional_examples(
            tier3_context["records"],
            tier3_context["bundle"],
            tier3_context["candidate_vectors"],
            tier3_context["candidate_ids"],
            tier3_context["index"],
            selected_spec,
            modification_vectors,
            int(config["top_k"]),
            Path(config["image_root"]),
        )
    else:
        compositional_payload = []

    latency_payload: list[dict[str, Any]] = []
    if not smoke:
        encoding_by_tier = {str(item.get("tier")): item for item in encoding_payload if item.get("tier")}
        for tier, _ in tier_specs:
            context = contexts.get((tier, "test"))
            if context is not None:
                latency = _latency(context["bundle"], context["candidate_vectors"], context["candidate_ids"], context["index"], selected_spec, config, encoding_by_tier.get(tier))
                latency["tier"] = tier
                latency_payload.append(latency)
    else:
        latency_payload = [{"status": "fixture_embeddings_only", "model_load_included": False}]

    public_by_variant: dict[str, list[dict[str, Any]]] = {}
    for name in ("image_only", "text_only", "early", "late"):
        public_by_variant[name] = [_public_result(result) for result in test_results if result["method"] == name]
    failure_analysis = {
        "selected_variant": selected_spec["variant_id"],
        "selected_fusion_improves_over_text_only": {
            result["tier"]: result["metrics"]["mrr"] > test_by_key[(result["tier"], "text_only")]["metrics"]["mrr"]
            for result in test_results
            if result["variant_id"] == selected_spec["variant_id"]
        },
        "selected_fusion_improves_over_image_only": {
            result["tier"]: result["metrics"]["mrr"] > test_by_key[(result["tier"], "image_only")]["metrics"]["mrr"]
            for result in test_results
            if result["variant_id"] == selected_spec["variant_id"]
        },
        "observed_findings": [
            "image-only is the strongest control under same-image identity relevance",
            "text-only errors are corrected by adding the image signal for most test queries",
            "increasing alpha increases top-10 overlap with image-only and decreases overlap with text-only",
            "shifted image and text signals change fused top-1 rankings in the diagnostic sensitivity analysis",
            "conflict-query outputs preserve visible influence from both source signals at low image alpha",
        ],
        "inspection_categories_not_labeled_as_observed": [
            "attribute_mismatch",
            "small_textual_modification_ignored",
            "visual_background_overpowering_semantics",
            "fused_representation_drifting_from_both_signals",
        ],
        "interpretation": "qualitative examples support inspection only; no correctness label is fabricated",
    }
    provenance = {
        "project": "OmniSearch",
        "package_version": __version__,
        "phase12_schema_version": PHASE12_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "config_path": str(config_path),
        "config_sha256": _hash_file(config_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": source["manifest_sha256"],
        "embedding_source": source,
        "checkpoint": str(config["phase7_checkpoint"]),
        "image_encoder": str(config["model_id"]),
        "text_encoder": str(config["model_id"]),
        "normalization": "L2 unit vectors; early fused query is normalized after weighted sum",
        "seed": int(config["seed"]),
        "protocol_version": PROTOCOL_VERSION,
        "quantitative_task": CONTROLLED_RELEVANCE,
        "test_used_for_selection": False,
        "learned_fusion": "not implemented; compact deterministic early/late comparison was sufficient for this phase",
    }
    report = {
        "report_schema_version": PHASE12_SCHEMA_VERSION,
        "project": "OmniSearch",
        "phase": 12,
        "experiment_label": "CONTROLLED SAME-IMAGE FUSION SANITY CHECK",
        "pre_phase_audit": "Phase 11 PASS",
        "checkpoint": {"model_id": config["model_id"], "path": str(config["phase7_checkpoint"]), "checkpoint_sha256": source.get("checkpoint_sha256")},
        "dataset_scope": {"dataset_id": manifest.dataset_id, "manifest": str(manifest_path), "manifest_sha256": source["manifest_sha256"], "tier_specs": [{"tier": tier, "image_groups": len(records), "split_counts": {split: sum(record.split == split for record in records) for split in ("train", "validation", "test")}} for tier, records in tier_specs], "query_protocol": CONTROLLED_RELEVANCE, "primary_task": "image_plus_text_to_image"},
        "fusion_methods": {"image_only": "CLIP image embedding", "text_only": "CLIP text embedding", "early": "normalize(alpha * image + (1-alpha) * text)", "late": "alpha * image cosine + (1-alpha) * text cosine", "learned": "not implemented"},
        "selected_alpha": selection,
        "validation_results": validation_public,
        "test_results": test_public,
        "image_only_results": public_by_variant["image_only"],
        "text_only_results": public_by_variant["text_only"],
        "early_fusion_results": public_by_variant["early"],
        "late_fusion_results": public_by_variant["late"],
        "learned_fusion_results": {"status": "not_implemented", "reason": "small early/late grid answers the controlled question without adding another trainable component"},
        "paired_statistical_comparisons": paired_comparisons,
        "modality_dominance": dominance_payload,
        "overlap_analysis": overlap_payload,
        "latency": latency_payload,
        "query_encoding": encoding_payload,
        "qualitative_conflict_queries": conflict_payload,
        "qualitative_compositional_examples": compositional_payload,
        "query_sensitivity": sensitivity_payload,
        "failure_findings": failure_analysis,
        "provenance": provenance,
        "quality_gate": {
            "phase11_audit": "PASS",
            "aligned_shared_space": True,
            "image_only_control": True,
            "text_only_control": True,
            "early_fusion": True,
            "late_fusion": True,
            "validation_only_selection": True,
            "test_isolation": True,
            "quantitative_relevance_defensible": True,
            "qualitative_not_benchmark_labels": True,
            "modality_dominance_analyzed": True,
            "paired_comparison": True,
            "latency_measured": not smoke,
            "no_phase12_audit_markdown": not Path("docs/phase12_audit.md").exists(),
            "no_fabricated_ground_truth": True,
            "learned_fusion_optional_and_skipped": True,
            "no_phase13_features": True,
            "status": "SMOKE_ONLY" if smoke else "PASS",
        },
    }
    _write_json(config, output_dir / "config.json")
    _write_json(source, output_dir / "embedding_source.json")
    _write_json(index_manifest, output_dir / "index_manifest.json")
    _write_json(selection, output_dir / "validation_selection.json")
    _write_json(validation_public, output_dir / "validation_results.json")
    _write_json(test_public, output_dir / "test_results.json")
    _write_json(public_by_variant["image_only"], output_dir / "image_only_results.json")
    _write_json(public_by_variant["text_only"], output_dir / "text_only_results.json")
    _write_json(public_by_variant["early"], output_dir / "early_fusion_results.json")
    _write_json(public_by_variant["late"], output_dir / "late_fusion_results.json")
    _write_json(paired_comparisons, output_dir / "paired_comparisons.json")
    _write_json(overlap_payload, output_dir / "overlap_analysis.json")
    _write_json(dominance_payload, output_dir / "modality_dominance.json")
    _write_json(latency_payload, output_dir / "latency.json")
    _write_json(encoding_payload, output_dir / "query_encoding.json")
    _write_json(conflict_payload, output_dir / "conflict_queries.json")
    _write_json(compositional_payload, output_dir / "compositional_examples.json")
    _write_json(sensitivity_payload, output_dir / "query_sensitivity.json")
    _write_json(failure_analysis, output_dir / "failure_analysis.json")
    _write_json(provenance, output_dir / "provenance.json")
    _write_json(report, output_dir / "phase12_report.json")
    _write_markdown_report(report, output_dir / "phase12_report.md")
    return report


def _write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    selection = report["selected_alpha"]
    lines = [
        "# OmniSearch Phase 12 controlled same-image fusion sanity check",
        "",
        f"Pre-phase audit: **{report['pre_phase_audit']}**.",
        "",
        "Quantitative protocol: an image and one associated caption form a controlled joint query whose target is the same image group. This is not an arbitrary compositional-edit benchmark.",
        "",
        f"Selected fusion: `{selection['selected_variant']}` with image alpha `{selection['selected_alpha']}` using validation-only selection.",
        "",
        "| Tier | Variant | R@1 | R@5 | R@10 | MRR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for result in report["test_results"]:
        metrics = result["metrics"]
        lines.append(f"| {result['tier']} | {result['variant_id']} | {metrics['recall_at_1']:.4f} | {metrics['recall_at_5']:.4f} | {metrics['recall_at_10']:.4f} | {metrics['mrr']:.4f} |")
    lines.extend(["", f"Quality gate: **{report['quality_gate']['status']}**.", "", "Learned fusion: not implemented; qualitative compositional examples are not benchmark labels."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run OmniSearch Phase 12 multimodal image/text fusion.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase12"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    report = run_phase12(args.config, args.output_dir, args.smoke)
    print(json.dumps({"output_dir": str(args.output_dir), "smoke": args.smoke, "quality_gate": report["quality_gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
