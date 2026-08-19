"""Phase 6 frozen text and vision representation experiments.

This runner compares compact pretrained unimodal representations with the
Phase 3 lexical baselines and keeps frozen CLIP as the only shared-space
cross-modal reference. It never trains, aligns, indexes, reranks, or compares
unrelated text and image vectors.
"""

from __future__ import annotations

import gc
import hashlib
import json
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .baselines import ImageHistogramIndex, TextDocument, records_to_documents, tokenize
from .clip_baseline import EmbeddingBatch as ClipEmbeddingBatch
from .clip_baseline import encode_images as encode_clip_images
from .clip_baseline import encode_texts as encode_clip_texts
from .clip_baseline import load_clip_runtime
from .clip_baseline import model_metadata as clip_model_metadata
from .config import DEFAULT_CONFIG_PATH, load_config
from .evaluation import (
    PROTOCOL_VERSION,
    RankingRecord,
    bootstrap_ci,
    build_result,
    compare_systems,
    ranking_from_scores,
    write_result_artifacts,
)
from .manifest import CaptionRecord, ImageRecord, read_manifest
from .representations import (
    TEXT_SPECS,
    VISION_SPECS,
    RepresentationBatch,
    RepresentationSpec,
    encode_images,
    encode_texts,
    ids_sha256,
    load_text_runtime,
    load_vision_runtime,
    rank_embedding_batches,
    read_embedding_cache,
    runtime_metadata,
    write_embedding_cache,
)
from .splitting import assert_no_split_leakage

PHASE6_SCHEMA_VERSION = 1
DEFAULT_TEXT_MODELS = ("minilm_mean", "distilbert_mean")
DEFAULT_VISION_MODELS = ("resnet18_native", "vit_base_native")

MODEL_JUSTIFICATIONS = {
    "minilm_mean": "Compact sentence-level MiniLM embedding; chosen for semantic sentence retrieval on an 8 GB machine.",
    "distilbert_mean": "General pretrained DistilBERT mean-pooled control; separates generic encoder transfer from sentence-similarity tuning.",
    "resnet18_native": "Small CNN with local convolutional inductive bias and residual connections; feasible on the target device.",
    "vit_base_native": "Patch-based transformer representation for a CNN-versus-global-attention comparison; model-native pooler retained.",
    "clip_text": "Existing frozen CLIP text encoder; included as the canonical multimodal model's unimodal component.",
    "clip_vision": "Existing frozen CLIP vision encoder; included as the canonical shared-space model's visual component.",
}


def _hash_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_phase6_config(config_path: Path) -> dict[str, Any]:
    import tomllib

    with config_path.open("rb") as file:
        return dict(tomllib.load(file).get("phase6", {}))


def _stable_query_documents(
    documents: Sequence[TextDocument], seed: int, limit: int
) -> tuple[TextDocument, ...]:
    ordered = sorted(
        documents,
        key=lambda document: hashlib.sha256(
            f"{seed}\0{document.doc_id}".encode()
        ).hexdigest(),
    )
    return tuple(ordered[: min(limit, len(ordered))])


def _relevance(
    queries: Sequence[TextDocument], documents: Sequence[TextDocument]
) -> dict[str, set[str]]:
    return {
        query.doc_id: {
            document.doc_id
            for document in documents
            if document.group_id == query.group_id and document.doc_id != query.doc_id
        }
        for query in queries
    }


def _cache_metadata(
    spec: RepresentationSpec,
    runtime_metadata_value: Mapping[str, Any],
    manifest_sha256: str,
    dataset_id: str,
    split: str,
    ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "representation_schema_version": 1,
        "model_key": spec.key,
        "model_id": spec.model_id,
        "model_revision": runtime_metadata_value.get("model_revision"),
        "dataset_id": dataset_id,
        "manifest_sha256": manifest_sha256,
        "split": split,
        "ids_sha256": ids_sha256(ids),
        "pooling": spec.pooling,
        "normalization": spec.normalization,
        "embedding_dimension": runtime_metadata_value["embedding_dimension"],
        "device": runtime_metadata_value["device"],
    }


def _cached_encode(
    cache_path: Path,
    metadata: Mapping[str, Any],
    encoder: Any,
) -> tuple[RepresentationBatch, dict[str, Any]]:
    if cache_path.exists():
        try:
            return read_embedding_cache(cache_path, metadata), {
                "status": "hit",
                "path": str(cache_path),
            }
        except (OSError, ValueError) as exc:
            rejected = {
                "status": "rejected_recompute",
                "path": str(cache_path),
                "reason": str(exc),
            }
    else:
        rejected = {"status": "miss", "path": str(cache_path)}
    batch = encoder()
    write_embedding_cache(cache_path, batch, metadata)
    return batch, rejected


def _clip_to_representation(batch: ClipEmbeddingBatch) -> RepresentationBatch:
    return RepresentationBatch(batch.ids, batch.embeddings, batch.skipped)


def _text_result(
    spec: RepresentationSpec,
    runtime_info: Mapping[str, Any],
    rankings: Sequence[RankingRecord],
    manifest: Any,
    manifest_path: Path,
    config_path: Path,
    split: str,
    seed: int,
    query_count: int,
    candidate_count: int,
    runtime: Mapping[str, Any],
    output_dir: Path,
    cache_info: Mapping[str, Any],
    bootstrap_resamples: int,
) -> dict[str, Any]:
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "task": "text_to_text",
        "representation_similarity": "cosine; L2 normalization makes dot product equivalent",
        "pooling": spec.pooling,
        "normalization": spec.normalization,
        "k_values": [1, 5, 10],
        "candidate_corpus": "full held-out test caption corpus",
        "relevance_definition": "same-image captions in held-out split; query excluded",
        "score_direction": "higher_is_better",
        "tie_policy": "score_desc_then_candidate_id_asc",
    }
    dataset = {
        "dataset_id": manifest.dataset_id,
        "dataset_version": manifest.dataset_version,
        "manifest_sha256": _hash_file(manifest_path),
        "subset": f"deterministic_seed_{seed}_max_{query_count}_queries",
    }
    system_id = f"phase6_text_{spec.key}"
    result = build_result(
        rankings,
        protocol,
        dataset,
        split,
        f"phase6_{spec.key}_seed{seed}",
        system_id,
        {
            **dict(runtime_info),
            "frozen": True,
            "trainable_parameter_count": 0,
            "choice_justification": MODEL_JUSTIFICATIONS[spec.key],
        },
        seed,
        {
            f"recall_at_{k}": bootstrap_ci(
                rankings, "recall", k, bootstrap_resamples, 0.95, seed
            )
            for k in (1, 5, 10)
        },
        {**dict(runtime), "cache": dict(cache_info)},
        {
            "platform": platform.platform(),
            "python": sys.version,
            "device": runtime_info["device"],
        },
        {
            "project_version": __version__,
            "config_sha256": _hash_file(config_path),
            "manifest_sha256": _hash_file(manifest_path),
            "source_sha256": manifest.source_sha256,
            "protocol_version": PROTOCOL_VERSION,
            "candidate_count": candidate_count,
        },
    )
    result["ranking_records"] = [record.to_dict() for record in rankings]
    result["representation_space_id"] = system_id
    write_result_artifacts(result, output_dir / system_id)
    return result


def _build_text_rankings(
    query_batch: RepresentationBatch,
    candidate_batch: RepresentationBatch,
    queries: Sequence[TextDocument],
    relevance: Mapping[str, set[str]],
    system_id: str,
    candidate_corpus_id: str,
    top_k: int,
) -> tuple[RankingRecord, ...]:
    ranked = rank_embedding_batches(
        query_batch,
        candidate_batch,
        space_id=system_id,
        candidate_space_id=system_id,
        top_k=top_k,
        exclude_self=True,
    )
    query_by_id = {query.doc_id: query for query in queries}
    return tuple(
        ranking_from_scores(
            query_id=query_id,
            task="text_to_text",
            candidates=[
                (str(item["id"]), float(item["score"])) for item in ranked[query_id]
            ],
            relevant_ids=relevance[query_id],
            system_id=system_id,
            experiment_id=f"phase6_{system_id}",
            candidate_count=len(candidate_batch.ids),
            candidate_corpus_id=candidate_corpus_id,
            relevance_definition="same-image captions in held-out split; query excluded",
        )
        for query_id in (query.doc_id for query in queries)
        if query_id in ranked and query_id in query_by_id
    )


def _baseline_records(path: Path) -> tuple[RankingRecord, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        RankingRecord.from_mapping(item) for item in payload["ranking_records"]
    )


def _comparison_metadata(system_id: str, candidate_corpus_id: str) -> dict[str, Any]:
    return {
        "system_id": system_id,
        "task": "text_to_text",
        "dataset_id": candidate_corpus_id.split(":", 1)[0],
        "split": "test",
        "protocol_version": PROTOCOL_VERSION,
        "candidate_corpus_id": candidate_corpus_id,
        "relevance_definition": "same-image captions in held-out split; query excluded",
    }


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set and not right_set:
        return 1.0
    return (
        len(left_set & right_set) / len(left_set | right_set)
        if left_set | right_set
        else 0.0
    )


def _text_qualitative(
    documents: Sequence[TextDocument],
    queries: Sequence[TextDocument],
    relevance: Mapping[str, set[str]],
    baseline_records: Mapping[str, Sequence[RankingRecord]],
    model_records: Mapping[str, Sequence[RankingRecord]],
    output_path: Path,
) -> dict[str, Any]:
    document_text = {document.doc_id: document.text for document in documents}
    query_text = {query.doc_id: query.text for query in queries}
    all_rows: list[dict[str, Any]] = []
    categories: dict[str, dict[str, int]] = {}
    examples: dict[str, list[dict[str, Any]]] = {}
    for model_id, records in model_records.items():
        model_by_query = {record.query_id: record for record in records}
        categories[model_id] = {
            "transformer_wins_over_bm25": 0,
            "bm25_wins_over_transformer": 0,
            "ties": 0,
            "low_lexical_overlap_queries": 0,
        }
        examples[model_id] = []
        bm25_by_query = {
            record.query_id: record for record in baseline_records["bm25_word"]
        }
        for query_id in sorted(model_by_query):
            model_record = model_by_query[query_id]
            bm25_record = bm25_by_query[query_id]
            model_hit = bool(set(model_record.candidate_ids[:10]) & relevance[query_id])
            bm25_hit = bool(set(bm25_record.candidate_ids[:10]) & relevance[query_id])
            if model_hit and not bm25_hit:
                category = "transformer_wins_over_bm25"
            elif bm25_hit and not model_hit:
                category = "bm25_wins_over_transformer"
            else:
                category = "ties"
            categories[model_id][category] += 1
            overlap = max(
                (
                    _jaccard(
                        tokenize(query_text[query_id]), tokenize(document_text[item])
                    )
                    for item in relevance[query_id]
                ),
                default=0.0,
            )
            if overlap <= 0.2:
                categories[model_id]["low_lexical_overlap_queries"] += 1
            if len(examples[model_id]) < 3 and category != "ties":
                examples[model_id].append(
                    {
                        "category": category,
                        "query_id": query_id,
                        "query_text": query_text[query_id],
                        "max_relevant_lexical_jaccard": overlap,
                        "relevant_ids": sorted(relevance[query_id]),
                        "bm25_top5": list(bm25_record.candidate_ids[:5]),
                        "transformer_top5": list(model_record.candidate_ids[:5]),
                    }
                )
            if len(all_rows) < 6:
                all_rows.append(
                    {
                        "model_id": model_id,
                        "query_id": query_id,
                        "query_text": query_text[query_id],
                        "category": category,
                        "bm25_top5": list(bm25_record.candidate_ids[:5]),
                        "transformer_top5": list(model_record.candidate_ids[:5]),
                    }
                )
    result = {
        "selection": "first deterministic non-tie examples in sorted query-ID order; counts use all evaluated queries",
        "categories": categories,
        "examples": examples,
        "rows": all_rows,
        "limitation": "same-image caption relevance and lexical overlap are diagnostic proxies, not human semantic judgments",
    }
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _write_fixture_images(output_dir: Path) -> tuple[Path, Path]:
    fixture_dir = output_dir / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    paths = (fixture_dir / "red.ppm", fixture_dir / "blue.ppm")
    payloads = (
        bytes((220, 20, 20)) * (16 * 16),
        bytes((20, 40, 220)) * (16 * 16),
    )
    for path, payload in zip(paths, payloads):
        if not path.exists():
            path.write_bytes(b"P6\n16 16\n255\n" + payload)
    return paths


def _fixture_records(paths: Sequence[Path]) -> tuple[ImageRecord, ...]:
    groups = ("red", "blue")
    return tuple(
        ImageRecord(
            image_id=f"fixture-{group}",
            filename=path.name,
            captions=tuple(
                CaptionRecord(f"fixture-{group}#{index}", f"a {group} square")
                for index in range(5)
            ),
            split="test",
        )
        for group, path in zip(groups, paths)
    )


def _vision_qualitative(
    output_dir: Path,
    requested_device: str,
    batch_size: int,
    run_clip: bool,
    vision_keys: Sequence[str],
    real_records: Sequence[ImageRecord] = (),
    image_root: Path | str | None = None,
    dataset_id: str = "fixture",
    split: str = "test",
) -> dict[str, Any]:
    paths = _write_fixture_images(output_dir)
    items = [(f"fixture-{path.stem}", path) for path in paths]
    records = _fixture_records(paths)
    histogram = ImageHistogramIndex().fit(records, output_dir / "fixtures")
    classical = {
        image_id: [
            {"id": result.item_id, "score": result.score}
            for result in histogram.search(image_id, top_k=1)
        ]
        for image_id, _, _ in histogram.items
    }
    models: dict[str, Any] = {}
    model_keys = list(vision_keys)
    for key in model_keys:
        spec = VISION_SPECS[key]
        load_start = time.perf_counter()
        runtime = load_vision_runtime(spec, requested_device)
        load_seconds = time.perf_counter() - load_start
        encode_start = time.perf_counter()
        batch = encode_images(items, runtime, batch_size)
        encode_seconds = time.perf_counter() - encode_start
        ranked = rank_embedding_batches(
            batch, batch, key, key, top_k=1, exclude_self=True
        )
        metadata = runtime_metadata(runtime)
        models[key] = {
            "model": metadata,
            "runtime": {
                "model_load_seconds": load_seconds,
                "fixture_encode_seconds": encode_seconds,
                "fixture_items": len(batch.ids),
                "throughput_per_second": len(batch.ids) / encode_seconds
                if encode_seconds
                else None,
                "device": runtime.device,
            },
            "nearest_neighbors": ranked,
            "skipped": list(batch.skipped),
            "comparison_scope": "qualitative fixture-only; the dataset supplies no cross-image relevance labels",
        }
        del runtime
        gc.collect()
    real_result: dict[str, Any] | None = None
    if image_root is not None and real_records:
        real_root = Path(image_root)
        real_items = [
            (record.image_id, real_root / record.filename)
            for record in real_records
            if record.filename is not None
        ]
        real_histogram = ImageHistogramIndex().fit(real_records, real_root)
        real_models: dict[str, Any] = {
            "classical_descriptor": real_histogram.stats(),
            "nearest_neighbors": {
                image_id: [
                    {"id": result.item_id, "score": result.score}
                    for result in real_histogram.search(image_id, top_k=1)
                ]
                for image_id, _, _ in real_histogram.items
            },
        }
        for key in vision_keys:
            spec = VISION_SPECS[key]
            load_start = time.perf_counter()
            runtime = load_vision_runtime(spec, requested_device)
            load_seconds = time.perf_counter() - load_start
            encode_start = time.perf_counter()
            batch = encode_images(real_items, runtime, batch_size)
            encode_seconds = time.perf_counter() - encode_start
            real_models[key] = {
                "model": runtime_metadata(runtime),
                "runtime": {
                    "model_load_seconds": load_seconds,
                    "encode_seconds": encode_seconds,
                    "items_requested": len(real_items),
                    "items_encoded": len(batch.ids),
                    "throughput_per_second": len(batch.ids) / encode_seconds
                    if encode_seconds
                    else None,
                    "device": runtime.device,
                },
                "nearest_neighbors": rank_embedding_batches(
                    batch, batch, key, key, top_k=1, exclude_self=True
                ),
                "skipped": list(batch.skipped),
                "comparison_scope": (
                    "real image representation nearest-neighbor evidence; "
                    "no formal cross-image relevance labels"
                ),
            }
            del runtime
            gc.collect()
        if run_clip:
            load_start = time.perf_counter()
            clip_runtime = load_clip_runtime(requested_device=requested_device)
            load_seconds = time.perf_counter() - load_start
            encode_start = time.perf_counter()
            clip_batch = encode_clip_images(real_items, clip_runtime, batch_size)
            encode_seconds = time.perf_counter() - encode_start
            clip_representation = _clip_to_representation(clip_batch)
            real_models["clip_vision"] = {
                "model": clip_model_metadata(clip_runtime),
                "runtime": {
                    "model_load_seconds": load_seconds,
                    "encode_seconds": encode_seconds,
                    "items_requested": len(real_items),
                    "items_encoded": len(clip_batch.ids),
                    "throughput_per_second": len(clip_batch.ids) / encode_seconds
                    if encode_seconds
                    else None,
                    "device": clip_runtime.device,
                },
                "nearest_neighbors": rank_embedding_batches(
                    clip_representation,
                    clip_representation,
                    "clip_vision",
                    "clip_vision",
                    top_k=1,
                    exclude_self=True,
                ),
                "skipped": list(clip_batch.skipped),
                "comparison_scope": (
                    "real CLIP image representation nearest-neighbor evidence; "
                    "no formal cross-image relevance labels"
                ),
            }
        real_result = {
            "dataset_id": dataset_id,
            "split": split,
            "image_root": str(real_root),
            "items_requested": len(real_items),
            "items_encoded": len(real_histogram.items),
            "formal_metrics": "not_evaluated_no_cross_image_relevance_labels",
            "models": real_models,
        }
    if run_clip:
        load_start = time.perf_counter()
        clip_runtime = load_clip_runtime(requested_device=requested_device)
        load_seconds = time.perf_counter() - load_start
        encode_start = time.perf_counter()
        clip_batch = encode_clip_images(items, clip_runtime, batch_size)
        encode_seconds = time.perf_counter() - encode_start
        clip_representation = _clip_to_representation(clip_batch)
        models["clip_vision"] = {
            "model": {
                **clip_model_metadata(clip_runtime),
                "choice_justification": MODEL_JUSTIFICATIONS["clip_vision"],
            },
            "runtime": {
                "model_load_seconds": load_seconds,
                "fixture_encode_seconds": encode_seconds,
                "fixture_items": len(clip_batch.ids),
                "throughput_per_second": len(clip_batch.ids) / encode_seconds
                if encode_seconds
                else None,
                "device": clip_runtime.device,
            },
            "nearest_neighbors": rank_embedding_batches(
                clip_representation,
                clip_representation,
                "clip_vision",
                "clip_vision",
                top_k=1,
                exclude_self=True,
            ),
            "skipped": list(clip_batch.skipped),
            "comparison_scope": "qualitative fixture-only; CLIP remains the shared-space baseline elsewhere",
        }
    result = {
        "fixture_only": real_result is None,
        "fixture_evidence_present": True,
        "fixture_paths": [str(path) for path in paths],
        "classical_descriptor": histogram.stats(),
        "classical_nearest_neighbors": classical,
        "models": models,
        "real_dataset_image_status": (
            "completed_representation_evidence"
            if real_result is not None
            else "not_evaluated_no_image_root"
        ),
        "real_image_evaluation": real_result,
        "cross_modal_warning": "No unrelated text/image vectors were compared; all rankings are within one declared image space.",
    }
    (output_dir / "vision_qualitative.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _clip_text_experiment(
    documents: Sequence[TextDocument],
    queries: Sequence[TextDocument],
    relevance: Mapping[str, set[str]],
    manifest: Any,
    manifest_path: Path,
    config_path: Path,
    output_dir: Path,
    cache_dir: Path,
    device: str,
    batch_size: int,
    seed: int,
    split: str,
    candidate_corpus_id: str,
    bootstrap_resamples: int,
) -> tuple[dict[str, Any], tuple[RankingRecord, ...]]:
    from .representations import ids_sha256 as representation_ids_sha256

    spec = RepresentationSpec(
        key="clip_text",
        modality="text",
        model_id="openai/clip-vit-base-patch32",
        architecture="CLIP ViT-B/32 text encoder",
        pooling="model_native_projection",
        normalization="l2",
        max_length=77,
        role=MODEL_JUSTIFICATIONS["clip_text"],
    )
    load_start = time.perf_counter()
    runtime = load_clip_runtime(requested_device=device)
    load_seconds = time.perf_counter() - load_start
    all_items = [(document.doc_id, document.text) for document in documents]
    query_items = [(query.doc_id, query.text) for query in queries]
    metadata_base = {
        "representation_schema_version": 1,
        "model_key": spec.key,
        "model_id": spec.model_id,
        "dataset_id": manifest.dataset_id,
        "manifest_sha256": _hash_file(manifest_path),
        "split": split,
        "pooling": spec.pooling,
        "normalization": spec.normalization,
        "embedding_dimension": runtime.embedding_dimension,
        "device": runtime.device,
    }
    candidate_metadata = {
        **metadata_base,
        "ids_sha256": representation_ids_sha256([item[0] for item in all_items]),
    }
    query_metadata = {
        **metadata_base,
        "ids_sha256": representation_ids_sha256([item[0] for item in query_items]),
    }
    candidate_start = time.perf_counter()
    candidate_batch, candidate_cache = _cached_encode(
        cache_dir / "clip_text_candidates.json",
        candidate_metadata,
        lambda items=all_items, encoder_runtime=runtime: _clip_to_representation(
            encode_clip_texts(items, encoder_runtime, batch_size)
        ),
    )
    candidate_seconds = time.perf_counter() - candidate_start
    query_start = time.perf_counter()
    query_batch, query_cache = _cached_encode(
        cache_dir / "clip_text_queries.json",
        query_metadata,
        lambda items=query_items, encoder_runtime=runtime: _clip_to_representation(
            encode_clip_texts(items, encoder_runtime, batch_size)
        ),
    )
    query_seconds = time.perf_counter() - query_start
    runtime_info = clip_model_metadata(runtime)
    system_id = "phase6_text_clip_text"
    rankings = _build_text_rankings(
        query_batch,
        candidate_batch,
        queries,
        relevance,
        system_id,
        candidate_corpus_id,
        10,
    )
    result = _text_result(
        spec,
        {**runtime_info, "choice_justification": MODEL_JUSTIFICATIONS["clip_text"]},
        rankings,
        manifest,
        manifest_path,
        config_path,
        split,
        seed,
        len(queries),
        len(documents),
        {
            "model_load_seconds": load_seconds,
            "candidate_encode_seconds": candidate_seconds,
            "query_encode_seconds": query_seconds,
        },
        output_dir,
        {"candidates": candidate_cache, "queries": query_cache},
        bootstrap_resamples,
    )
    del runtime
    gc.collect()
    return result, rankings


def run_phase6(
    manifest_path: Path | str,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    output_dir: Path | str = "artifacts/phase6",
    image_root: Path | str | None = None,
) -> dict[str, Any]:
    """Run frozen Phase 6 representation experiments."""

    manifest_path = Path(manifest_path)
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    config = load_config(config_path)
    phase6_config = _read_phase6_config(config_path)
    manifest = read_manifest(manifest_path)
    assert_no_split_leakage(manifest.records)
    split = str(phase6_config.get("split", "test"))
    records = tuple(record for record in manifest.records if record.split == split)
    documents = records_to_documents(records)
    query_limit = int(phase6_config.get("text_max_queries", 256))
    queries = _stable_query_documents(documents, config.seed, query_limit)
    relevance = _relevance(queries, documents)
    batch_size = int(phase6_config.get("batch_size", 8))
    requested_device = str(phase6_config.get("device", "auto"))
    top_k = int(phase6_config.get("top_k", 10))
    bootstrap_resamples = int(phase6_config.get("bootstrap_resamples", 200))
    selected_text_models = tuple(
        str(value) for value in phase6_config.get("text_models", DEFAULT_TEXT_MODELS)
    )
    selected_vision_models = tuple(
        str(value)
        for value in phase6_config.get("vision_models", DEFAULT_VISION_MODELS)
    )
    if any(key not in TEXT_SPECS for key in selected_text_models):
        raise ValueError(
            f"unknown Phase 6 text model; expected one of {sorted(TEXT_SPECS)}"
        )
    if any(key not in VISION_SPECS for key in selected_vision_models):
        raise ValueError(
            f"unknown Phase 6 vision model; expected one of {sorted(VISION_SPECS)}"
        )
    if top_k <= 0 or batch_size <= 0:
        raise ValueError("Phase 6 top_k and batch_size must be positive")
    candidate_corpus_id = f"{manifest.dataset_id}:{split}:{len(documents)}"
    text_records: dict[str, tuple[RankingRecord, ...]] = {}
    text_model_runs: list[dict[str, Any]] = []
    for key in selected_text_models:
        spec = TEXT_SPECS[key]
        load_start = time.perf_counter()
        runtime = load_text_runtime(spec, requested_device)
        load_seconds = time.perf_counter() - load_start
        runtime_info = runtime_metadata(runtime)
        all_items = [(document.doc_id, document.text) for document in documents]
        query_items = [(query.doc_id, query.text) for query in queries]
        candidate_metadata = _cache_metadata(
            spec,
            runtime_info,
            _hash_file(manifest_path),
            manifest.dataset_id,
            split,
            [item[0] for item in all_items],
        )
        query_metadata = _cache_metadata(
            spec,
            runtime_info,
            _hash_file(manifest_path),
            manifest.dataset_id,
            split,
            [item[0] for item in query_items],
        )
        candidate_start = time.perf_counter()
        candidate_batch, candidate_cache = _cached_encode(
            cache_dir / f"{key}_candidates.json",
            candidate_metadata,
            lambda items=all_items, encoder_runtime=runtime: encode_texts(
                items, encoder_runtime, batch_size
            ),
        )
        candidate_seconds = time.perf_counter() - candidate_start
        query_start = time.perf_counter()
        query_batch, query_cache = _cached_encode(
            cache_dir / f"{key}_queries.json",
            query_metadata,
            lambda items=query_items, encoder_runtime=runtime: encode_texts(
                items, encoder_runtime, batch_size
            ),
        )
        query_seconds = time.perf_counter() - query_start
        system_id = f"phase6_text_{key}"
        rankings = _build_text_rankings(
            query_batch,
            candidate_batch,
            queries,
            relevance,
            system_id,
            candidate_corpus_id,
            top_k,
        )
        result = _text_result(
            spec,
            {**runtime_info, "choice_justification": MODEL_JUSTIFICATIONS[key]},
            rankings,
            manifest,
            manifest_path,
            config_path,
            split,
            config.seed,
            len(queries),
            len(documents),
            {
                "model_load_seconds": load_seconds,
                "candidate_encode_seconds": candidate_seconds,
                "query_encode_seconds": query_seconds,
                "candidate_throughput_per_second": len(candidate_batch.ids)
                / candidate_seconds
                if candidate_seconds
                else None,
                "query_throughput_per_second": len(query_batch.ids) / query_seconds
                if query_seconds
                else None,
            },
            output_dir,
            {"candidates": candidate_cache, "queries": query_cache},
            bootstrap_resamples,
        )
        text_records[key] = rankings
        text_model_runs.append(
            {
                "key": key,
                "runtime": result["runtime"],
                "model": result["model"],
                "metrics": result["metrics"],
            }
        )
        del runtime
        gc.collect()
        try:
            import torch  # type: ignore[import-not-found]

            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except (ImportError, AttributeError):
            pass
    clip_result, clip_records = _clip_text_experiment(
        documents,
        queries,
        relevance,
        manifest,
        manifest_path,
        config_path,
        output_dir,
        cache_dir,
        requested_device,
        batch_size,
        config.seed,
        split,
        candidate_corpus_id,
        bootstrap_resamples,
    )
    text_records["clip_text"] = clip_records
    text_model_runs.append(
        {
            "key": "clip_text",
            "runtime": clip_result["runtime"],
            "model": clip_result["model"],
            "metrics": clip_result["metrics"],
        }
    )

    import tomllib

    with config_path.open("rb") as file:
        raw_config = tomllib.load(file)
    phase5_dir = Path(
        raw_config.get("phase6", {}).get("phase5_artifact_dir", "artifacts/phase5")
    )
    baseline_records = {
        "tfidf_word_unigram_l2": _baseline_records(
            phase5_dir / "tfidf_word_unigram_l2.json"
        ),
        "bm25_word": _baseline_records(phase5_dir / "bm25_word.json"),
    }
    comparisons: dict[str, Any] = {}
    for model_key, records_for_model in text_records.items():
        comparisons[model_key] = {}
        for baseline_key, baseline in baseline_records.items():
            comparisons[model_key][baseline_key] = compare_systems(
                baseline,
                records_for_model,
                _comparison_metadata(baseline_key, candidate_corpus_id),
                _comparison_metadata(f"phase6_text_{model_key}", candidate_corpus_id),
                ks=(1, 5, 10),
                bootstrap_resamples=bootstrap_resamples,
                seed=config.seed,
            )
    (output_dir / "baseline_comparisons.json").write_text(
        json.dumps(comparisons, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    qualitative = _text_qualitative(
        documents,
        queries,
        relevance,
        baseline_records,
        text_records,
        output_dir / "text_qualitative.json",
    )
    vision = _vision_qualitative(
        output_dir,
        requested_device,
        batch_size,
        run_clip=bool(phase6_config.get("run_clip_components", True)),
        vision_keys=selected_vision_models,
        real_records=records,
        image_root=image_root,
        dataset_id=manifest.dataset_id,
        split=split,
    )
    model_matrix = {
        "text": [
            {
                "key": key,
                "spec": TEXT_SPECS[key].to_dict(),
                "justification": MODEL_JUSTIFICATIONS[key],
            }
            for key in selected_text_models
        ]
        + [
            {
                "key": "clip_text",
                "justification": MODEL_JUSTIFICATIONS["clip_text"],
                "canonical_reference": True,
            }
        ],
        "vision": [
            {
                "key": key,
                "spec": VISION_SPECS[key].to_dict(),
                "justification": MODEL_JUSTIFICATIONS[key],
            }
            for key in selected_vision_models
        ]
        + [
            {
                "key": "clip_vision",
                "justification": MODEL_JUSTIFICATIONS["clip_vision"],
                "canonical_reference": True,
            }
        ],
        "cross_modal_rule": "Only CLIP's own shared representation is cross-modal; unimodal text/image spaces are never compared directly.",
    }
    (output_dir / "model_matrix.json").write_text(
        json.dumps(model_matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    provenance = {
        "project": "OmniSearch",
        "package": "omnisearch",
        "project_version": __version__,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "config_path": str(config_path),
        "config_sha256": _hash_file(config_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _hash_file(manifest_path),
        "dataset_id": manifest.dataset_id,
        "dataset_version": manifest.dataset_version,
        "source_sha256": manifest.source_sha256,
        "split": split,
        "seed": config.seed,
        "batch_size": batch_size,
        "requested_device": requested_device,
        "image_root": str(image_root) if image_root is not None else None,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "report_schema_version": PHASE6_SCHEMA_VERSION,
        "project": "OmniSearch",
        "phase": 6,
        "pre_phase_audit": {
            "phase5_audit": "PASS",
            "protocol_version": PROTOCOL_VERSION,
        },
        "scope": {
            "frozen_representation_experiments": True,
            "fine_tuning": False,
            "cross_modal_alignment_training": False,
            "real_text_evaluation": True,
            "real_image_evaluation": vision.get("real_image_evaluation") is not None,
            "image_root_supplied": image_root is not None,
            "real_image_image_formal_metrics": False,
        },
        "model_matrix": model_matrix,
        "dataset": {
            "dataset_id": manifest.dataset_id,
            "dataset_version": manifest.dataset_version,
            "split": split,
            "text_queries": len(queries),
            "text_candidate_captions": len(documents),
            "candidate_corpus_id": candidate_corpus_id,
        },
        "text_models": text_model_runs,
        "baseline_comparisons": comparisons,
        "text_qualitative": qualitative,
        "vision": vision,
        "limitations": [
            "Image-side formal cross-image metrics are not reported because caption metadata does not label relevance between different images; real representation evidence is retained when an image root is supplied.",
            "Same-image caption relevance is a limited metadata proxy, not human semantic judgment.",
            "Unimodal text and image vectors are not cross-modal compatible and were never compared directly.",
            "Observed qualitative categories describe this deterministic sample only.",
        ],
        "forbidden_later_phase_features": {
            "fine_tuning": False,
            "lora": False,
            "adapters": False,
            "ann_indexes": False,
            "reranking": False,
            "query_expansion": False,
            "multimodal_fusion": False,
            "api_or_dashboard": False,
        },
        "provenance": provenance,
    }
    (output_dir / "phase6_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# OmniSearch Phase 6 frozen representation report",
        "",
        "Protocol: `retrieval_eval_v1`",
        "",
        "All Phase 6 models are frozen pretrained encoders. No cross-modal similarity was computed between unrelated unimodal spaces.",
        "",
        "| Representation | Dimension | R@1 | R@5 | R@10 | MRR | MAP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in text_model_runs:
        metrics = row["metrics"]
        lines.append(
            f"| {row['key']} | {row['model'].get('embedding_dimension', 'n/e')} | {metrics.get('recall_at_1', 'n/e')} | {metrics.get('recall_at_5', 'n/e')} | {metrics.get('recall_at_10', 'n/e')} | {metrics.get('mrr', 'n/e')} | {metrics.get('map', 'n/e')} |"
        )
    lines.extend(
        [
            "",
            "Vision results include real image representation evidence when supplied; formal image-to-image relevance metrics remain unevaluated.",
            "",
            "See `baseline_comparisons.json`, `text_qualitative.json`, and `vision_qualitative.json` for actual outputs.",
            "",
        ]
    )
    (output_dir / "phase6_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run OmniSearch Phase 6 frozen representation experiments."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/coco2017_val_split_manifest.json"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase6"))
    parser.add_argument("--image-root", type=Path, default=None)
    args = parser.parse_args()
    report = run_phase6(args.manifest, args.config, args.output_dir, args.image_root)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "text_models": [item["key"] for item in report["text_models"]],
                "vision_fixture_only": report["vision"]["fixture_only"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
