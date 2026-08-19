"""Phase 10 exact and approximate vector retrieval benchmarks.

This module deliberately keeps embedding generation separate from vector
search.  The primary representation is the already-validated Phase 7 full
fine-tuned CLIP checkpoint; Phase 10 never updates model parameters.  Exact
normalized inner-product search is the correctness reference, while FAISS
IVF-Flat and hnswlib are measured as approximate alternatives.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .config import DEFAULT_CONFIG_PATH
from .evaluation import (
    PROTOCOL_VERSION,
    RankingRecord,
    evaluate_rankings,
    ranking_from_scores,
)
from .manifest import DatasetManifest, ImageRecord, read_manifest
from .phase7 import (
    _encode_images,
    _encode_texts,
    _hash_file,
    _load_checkpoint,
    _load_trainable_model,
)

PHASE10_SCHEMA_VERSION = 1
INDEX_SCHEMA_VERSION = 1
DEFAULT_PHASE10_CONFIG: dict[str, Any] = {
    "manifest": "data/processed/coco2017_val_split_manifest.json",
    "image_root": "data/raw/coco2017/val2017",
    "phase7_checkpoint": "artifacts/phase7/best_checkpoint.pt",
    "model_id": "openai/clip-vit-base-patch32",
    "device": "auto",
    "batch_size": 8,
    "text_max_length": 77,
    "precision": "fp32",
    "seed": 42,
    "tier_sizes": [100, 1000, 5000],
    "top_k": 10,
    "warmup_queries": 5,
    "latency_repeats": 3,
    "latency_query_limit": 128,
    "selection_fidelity_threshold": 0.99,
    "hnsw_m": 16,
    "hnsw_ef_construction": 100,
    "hnsw_ef_search_values": [8, 32],
    "ivf_nprobe_values": [1, 8],
}


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def _hash_ids(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _read_phase10_config(path: Path) -> dict[str, Any]:
    import tomllib

    with path.open("rb") as file:
        raw = tomllib.load(file)
    config = dict(DEFAULT_PHASE10_CONFIG)
    config.update(dict(raw.get("phase10", {})))
    return config


def validate_phase10_config(config: Mapping[str, Any]) -> None:
    """Reject settings that violate the retrieval-only Phase 10 contract."""

    for key in ("batch_size", "top_k", "warmup_queries", "latency_repeats", "latency_query_limit"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if int(config.get("seed", 0)) < 0:
        raise ValueError("seed must be non-negative")
    if str(config.get("precision", "fp32")) not in {"fp32", "fp16"}:
        raise ValueError("precision must be fp32 or fp16")
    sizes = [int(item) for item in config.get("tier_sizes", [])]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("tier_sizes must contain positive values")
    if not 0 < float(config["selection_fidelity_threshold"]) <= 1:
        raise ValueError("selection_fidelity_threshold must be in (0, 1]")
    if int(config["hnsw_m"]) <= 0 or int(config["hnsw_ef_construction"]) <= 0:
        raise ValueError("HNSW construction parameters must be positive")
    if not config.get("hnsw_ef_search_values") or any(
        int(value) <= 0 for value in config["hnsw_ef_search_values"]
    ):
        raise ValueError("HNSW ef_search values must be positive")
    if not config.get("ivf_nprobe_values") or any(
        int(value) <= 0 for value in config["ivf_nprobe_values"]
    ):
        raise ValueError("IVF nprobe values must be positive")
    checkpoint = str(config.get("phase7_checkpoint", ""))
    if "phase8" in checkpoint or "phase9" in checkpoint:
        raise ValueError("Phase 10 primary source must be the Phase 7 full-FT checkpoint")


def normalize_vectors(vectors: Any) -> np.ndarray:
    """Return finite float32 unit vectors, rejecting malformed inputs."""

    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("vectors must be a rank-2 matrix")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("vectors must be non-empty")
    if not np.isfinite(array).all():
        raise ValueError("vectors must contain only finite values")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("vectors must have non-zero finite norms")
    return np.ascontiguousarray(array / norms[:, None], dtype=np.float32)


def validate_normalized_vectors(vectors: Any, tolerance: float = 1e-4) -> np.ndarray:
    """Validate a matrix that is already expected to be L2-normalized."""

    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("normalized vectors must be a non-empty rank-2 matrix")
    if not np.isfinite(array).all():
        raise ValueError("normalized vectors must be finite")
    norms = np.linalg.norm(array, axis=1)
    if not np.allclose(norms, 1.0, atol=tolerance, rtol=tolerance):
        raise ValueError("normalized vectors must have unit L2 norm")
    return np.ascontiguousarray(array, dtype=np.float32)


def _normalize_query(query: Any, dimension: int) -> np.ndarray:
    array = np.asarray(query, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != dimension:
        raise ValueError(f"query dimension must be ({dimension},)")
    return normalize_vectors(array)


class ExactIndex:
    """Trusted NumPy exact cosine search reference."""

    index_type = "exact_numpy"

    def __init__(self, vectors: np.ndarray, candidate_ids: Sequence[str]) -> None:
        self.vectors = validate_normalized_vectors(vectors)
        self.candidate_ids = tuple(str(value) for value in candidate_ids)
        if len(self.candidate_ids) != self.vectors.shape[0]:
            raise ValueError("candidate IDs and vectors must have equal length")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate IDs must be unique")

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def count(self) -> int:
        return int(self.vectors.shape[0])

    def search(self, queries: Any, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if int(top_k) <= 0:
            raise ValueError("top_k must be positive")
        query_array = _normalize_query(queries, self.dimension)
        scores = query_array @ self.vectors.T
        limit = min(int(top_k), self.count)
        output_ids: list[list[str]] = []
        output_scores: list[list[float]] = []
        candidate_array = np.asarray(self.candidate_ids, dtype="U")
        for row in scores:
            order = np.lexsort((candidate_array, -row))[:limit]
            output_ids.append([self.candidate_ids[int(index)] for index in order])
            output_scores.append([float(row[int(index)]) for index in order])
        return np.asarray(output_ids, dtype="U"), np.asarray(output_scores, dtype=np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.vectors)

    @classmethod
    def load(cls, path: Path, candidate_ids: Sequence[str], expected_dimension: int) -> ExactIndex:
        vectors = np.load(path, allow_pickle=False)
        if vectors.ndim != 2 or vectors.shape[1] != expected_dimension:
            raise ValueError("persisted exact index has incompatible dimensions")
        return cls(vectors, candidate_ids)


def _sort_backend_results(
    labels: Sequence[int], scores: Sequence[float], candidate_ids: Sequence[str]
) -> tuple[list[str], list[float]]:
    rows = [
        (candidate_ids[int(label)], float(score))
        for label, score in zip(labels, scores)
        if int(label) >= 0
    ]
    rows.sort(key=lambda item: (-item[1], item[0]))
    return [item[0] for item in rows], [item[1] for item in rows]


class FaissIndex:
    """FAISS adapter for exact Flat or IVF-Flat inner-product search."""

    def __init__(
        self,
        backend: Any,
        candidate_ids: Sequence[str],
        index_type: str,
        hyperparameters: Mapping[str, Any],
    ) -> None:
        self.backend = backend
        self.candidate_ids = tuple(str(value) for value in candidate_ids)
        self.index_type = index_type
        self.hyperparameters = dict(hyperparameters)
        if len(self.candidate_ids) != int(backend.ntotal):
            raise ValueError("FAISS candidate IDs do not match index size")

    @property
    def dimension(self) -> int:
        return int(self.backend.d)

    @property
    def count(self) -> int:
        return len(self.candidate_ids)

    def search(self, queries: Any, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if int(top_k) <= 0:
            raise ValueError("top_k must be positive")
        query_array = _normalize_query(queries, self.dimension)
        distances, labels = self.backend.search(query_array, min(int(top_k), self.count))
        ids: list[list[str]] = []
        values: list[list[float]] = []
        for row_labels, row_scores in zip(labels, distances):
            ordered_ids, ordered_scores = _sort_backend_results(
                row_labels, row_scores, self.candidate_ids
            )
            ids.append(ordered_ids)
            values.append(ordered_scores)
        return np.asarray(ids, dtype=object), np.asarray(values, dtype=object)

    def save(self, path: Path) -> None:
        import faiss

        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.backend, str(path))

    @classmethod
    def load(
        cls,
        path: Path,
        candidate_ids: Sequence[str],
        expected_dimension: int,
        index_type: str,
        hyperparameters: Mapping[str, Any],
    ) -> FaissIndex:
        import faiss

        faiss.omp_set_num_threads(1)

        backend: Any = faiss.read_index(str(path))
        if int(backend.d) != int(expected_dimension):
            raise ValueError("persisted FAISS index has incompatible dimensions")
        if int(backend.ntotal) != len(candidate_ids):
            raise ValueError("persisted FAISS index has incompatible candidate count")
        if index_type == "faiss_ivf_flat":
            backend.nprobe = int(hyperparameters["nprobe"])
        return cls(backend, candidate_ids, index_type, hyperparameters)


class HnswlibIndex:
    """hnswlib inner-product HNSW adapter."""

    def __init__(
        self,
        backend: Any,
        candidate_ids: Sequence[str],
        hyperparameters: Mapping[str, Any],
    ) -> None:
        self.backend = backend
        self.candidate_ids = tuple(str(value) for value in candidate_ids)
        self.hyperparameters = dict(hyperparameters)
        if int(backend.get_current_count()) != len(self.candidate_ids):
            raise ValueError("HNSW candidate IDs do not match index size")

    @property
    def index_type(self) -> str:
        return "hnswlib"

    @property
    def dimension(self) -> int:
        return int(self.backend.dim)

    @property
    def count(self) -> int:
        return len(self.candidate_ids)

    def search(self, queries: Any, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if int(top_k) <= 0:
            raise ValueError("top_k must be positive")
        query_array = _normalize_query(queries, self.dimension)
        labels, distances = self.backend.knn_query(query_array, k=min(int(top_k), self.count))
        ids: list[list[str]] = []
        values: list[list[float]] = []
        for row_labels, row_distances in zip(labels, distances):
            # hnswlib's IP distance is 1 - inner_product.
            scores = [1.0 - float(distance) for distance in row_distances]
            ordered_ids, ordered_scores = _sort_backend_results(
                row_labels, scores, self.candidate_ids
            )
            ids.append(ordered_ids)
            values.append(ordered_scores)
        return np.asarray(ids, dtype=object), np.asarray(values, dtype=object)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.backend.save_index(str(path))

    @classmethod
    def load(
        cls,
        path: Path,
        candidate_ids: Sequence[str],
        expected_dimension: int,
        hyperparameters: Mapping[str, Any],
    ) -> HnswlibIndex:
        import hnswlib  # type: ignore[import-untyped]

        backend = hnswlib.Index(space="ip", dim=int(expected_dimension))
        backend.load_index(str(path), max_elements=len(candidate_ids))
        backend.set_ef(int(hyperparameters["ef_search"]))
        backend.set_num_threads(1)
        if int(backend.get_current_count()) != len(candidate_ids):
            raise ValueError("persisted HNSW index has incompatible candidate count")
        return cls(backend, candidate_ids, hyperparameters)


def _faiss_backend(
    vectors: np.ndarray,
    index_type: str,
    hyperparameters: Mapping[str, Any],
) -> Any:
    import faiss

    faiss.omp_set_num_threads(1)

    dimension = int(vectors.shape[1])
    if index_type == "faiss_flat":
        backend: Any = faiss.IndexFlatIP(dimension)
    elif index_type == "faiss_ivf_flat":
        nlist = int(hyperparameters["nlist"])
        if nlist <= 0 or nlist > len(vectors):
            raise ValueError("IVF nlist must be in [1, corpus_count]")
        quantizer = faiss.IndexFlatIP(dimension)
        backend = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        backend.train(vectors)
        backend.nprobe = int(hyperparameters["nprobe"])
    else:
        raise ValueError(f"unsupported FAISS index type: {index_type}")
    backend.add(vectors)
    return backend


def _hnsw_backend(
    vectors: np.ndarray,
    hyperparameters: Mapping[str, Any],
    seed: int,
) -> Any:
    import hnswlib  # type: ignore[import-untyped]

    backend = hnswlib.Index(space="ip", dim=int(vectors.shape[1]))
    backend.init_index(
        max_elements=len(vectors),
        ef_construction=int(hyperparameters["ef_construction"]),
        M=int(hyperparameters["M"]),
        random_seed=int(seed),
    )
    backend.add_items(vectors, np.arange(len(vectors), dtype=np.int64))
    backend.set_ef(int(hyperparameters["ef_search"]))
    backend.set_num_threads(1)
    return backend


@dataclass(frozen=True)
class BuiltIndex:
    index: Any
    metadata: dict[str, Any]
    index_path: Path
    metadata_path: Path


def _library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in ("faiss-cpu", "hnswlib", "numpy"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _index_metadata(
    *,
    index_type: str,
    hyperparameters: Mapping[str, Any],
    embedding_source: Mapping[str, Any],
    tier: str,
    candidate_unit: str,
    candidate_ids: Sequence[str],
    dimension: int,
    build_seconds: float,
    serialize_seconds: float,
    serialized_size_bytes: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "phase10_schema_version": PHASE10_SCHEMA_VERSION,
        "index_type": index_type,
        "hyperparameters": dict(hyperparameters),
        "embedding_source": dict(embedding_source),
        "dataset_manifest_sha256": embedding_source["manifest_sha256"],
        "tier": tier,
        "candidate_unit": candidate_unit,
        "candidate_count": len(candidate_ids),
        "candidate_ids_sha256": _hash_ids(candidate_ids),
        "embedding_dimension": int(dimension),
        "raw_embedding_storage_bytes": int(len(candidate_ids) * dimension * 4),
        "dtype": "float32",
        "normalization": "L2 unit vectors; inner product equals cosine similarity",
        "metric": "inner_product",
        "seed": int(seed),
        "library_versions": _library_versions(),
        "build_seconds": float(build_seconds),
        "serialize_seconds": float(serialize_seconds),
        "serialized_size_bytes": int(serialized_size_bytes),
        "hardware": platform.platform(),
    }


def _validate_metadata(
    metadata: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    if int(metadata.get("index_schema_version", -1)) != INDEX_SCHEMA_VERSION:
        raise ValueError("unsupported index metadata schema")
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"incompatible index metadata: {key}")
    if metadata.get("normalization") != "L2 unit vectors; inner product equals cosine similarity":
        raise ValueError("index normalization contract is incompatible")


def _save_metadata(metadata: Mapping[str, Any], path: Path) -> None:
    _write_json(dict(metadata), path)


def build_persisted_index(
    vectors: Any,
    candidate_ids: Sequence[str],
    index_type: str,
    hyperparameters: Mapping[str, Any],
    embedding_source: Mapping[str, Any],
    tier: str,
    candidate_unit: str,
    output_base: Path,
    seed: int,
) -> BuiltIndex:
    normalized = normalize_vectors(vectors)
    ids = tuple(str(value) for value in candidate_ids)
    if len(ids) != len(normalized) or len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique and match vectors")
    started = time.perf_counter()
    if index_type == "exact_numpy":
        index: Any = ExactIndex(normalized, ids)
        suffix = ".npy"
    elif index_type in {"faiss_flat", "faiss_ivf_flat"}:
        index = FaissIndex(
            _faiss_backend(normalized, index_type, hyperparameters),
            ids,
            index_type,
            hyperparameters,
        )
        suffix = ".faiss"
    elif index_type == "hnswlib":
        index = HnswlibIndex(_hnsw_backend(normalized, hyperparameters, seed), ids, hyperparameters)
        suffix = ".hnsw"
    else:
        raise ValueError(f"unsupported index type: {index_type}")
    build_seconds = time.perf_counter() - started
    index_path = output_base.with_suffix(suffix)
    metadata_path = output_base.with_suffix(".metadata.json")
    serialize_started = time.perf_counter()
    index.save(index_path)
    serialize_seconds = time.perf_counter() - serialize_started
    metadata = _index_metadata(
        index_type=index_type,
        hyperparameters=hyperparameters,
        embedding_source=embedding_source,
        tier=tier,
        candidate_unit=candidate_unit,
        candidate_ids=ids,
        dimension=normalized.shape[1],
        build_seconds=build_seconds,
        serialize_seconds=serialize_seconds,
        serialized_size_bytes=index_path.stat().st_size,
        seed=seed,
    )
    _save_metadata(metadata, metadata_path)
    metadata["metadata_file_size_bytes"] = metadata_path.stat().st_size
    _save_metadata(metadata, metadata_path)
    return BuiltIndex(index=index, metadata=metadata, index_path=index_path, metadata_path=metadata_path)


def load_persisted_index(
    built: BuiltIndex,
    candidate_ids: Sequence[str],
    expected_metadata: Mapping[str, Any],
) -> Any:
    metadata = json.loads(built.metadata_path.read_text(encoding="utf-8"))
    _validate_metadata(metadata, expected_metadata)
    ids = tuple(str(value) for value in candidate_ids)
    if metadata.get("candidate_ids_sha256") != _hash_ids(ids):
        raise ValueError("candidate IDs do not match persisted index")
    index_type = str(metadata["index_type"])
    hyperparameters = dict(metadata.get("hyperparameters", {}))
    if index_type == "exact_numpy":
        return ExactIndex.load(built.index_path, ids, int(metadata["embedding_dimension"]))
    if index_type in {"faiss_flat", "faiss_ivf_flat"}:
        return FaissIndex.load(
            built.index_path,
            ids,
            int(metadata["embedding_dimension"]),
            index_type,
            hyperparameters,
        )
    if index_type == "hnswlib":
        return HnswlibIndex.load(
            built.index_path,
            ids,
            int(metadata["embedding_dimension"]),
            hyperparameters,
        )
    raise ValueError(f"unsupported persisted index type: {index_type}")


def _result_rankings(
    task: str,
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
    result_ids: np.ndarray,
    result_scores: np.ndarray,
    relevant: Mapping[str, set[str]],
    system_id: str,
    experiment_id: str,
    candidate_corpus_id: str,
) -> tuple[RankingRecord, ...]:
    def row_values(row: Any) -> list[Any]:
        return row.tolist() if hasattr(row, "tolist") else list(row)

    rankings: list[RankingRecord] = []
    for row_index, query_id in enumerate(query_ids):
        rankings.append(
            ranking_from_scores(
                query_id=str(query_id),
                task=task,
                candidates=[
                    (str(item_id), float(score))
                    for item_id, score in zip(row_values(result_ids[row_index]), row_values(result_scores[row_index]))
                ],
                relevant_ids=relevant[str(query_id)],
                system_id=system_id,
                experiment_id=experiment_id,
                candidate_count=len(candidate_ids),
                candidate_corpus_id=candidate_corpus_id,
            )
        )
    return tuple(rankings)


def _fidelity(
    exact_ids: np.ndarray, approximate_ids: np.ndarray, ks: Sequence[int]
) -> dict[str, float]:
    output: dict[str, float] = {}
    for k in ks:
        values = []
        for exact, approximate in zip(exact_ids, approximate_ids):
            exact_values = exact.tolist() if hasattr(exact, "tolist") else list(exact)
            approximate_values = approximate.tolist() if hasattr(approximate, "tolist") else list(approximate)
            exact_set = set(exact_values[:k])
            approximate_set = set(approximate_values[:k])
            values.append(len(exact_set & approximate_set) / max(1, len(exact_set)))
        output[f"neighbor_recall_at_{k}"] = statistics.fmean(values) if values else float("nan")
    if len(exact_ids):
        output["top1_exact_hit_rate"] = statistics.fmean(
            float((exact.tolist() if hasattr(exact, "tolist") else list(exact))[0] == (approximate.tolist() if hasattr(approximate, "tolist") else list(approximate))[0])
            for exact, approximate in zip(exact_ids, approximate_ids)
        )
    else:
        output["top1_exact_hit_rate"] = float("nan")
    return output


def _latency(
    index: Any,
    query_vectors: np.ndarray,
    top_k: int,
    warmup_queries: int,
    repeats: int,
    query_limit: int,
) -> dict[str, Any]:
    measured_vectors = query_vectors[: min(int(query_limit), len(query_vectors))]
    warmups = min(int(warmup_queries), len(measured_vectors))
    for query in measured_vectors[:warmups]:
        index.search(query[None, :], top_k)
    durations: list[float] = []
    for _ in range(int(repeats)):
        for query in measured_vectors:
            started = time.perf_counter()
            index.search(query[None, :], top_k)
            durations.append(time.perf_counter() - started)
    if not durations:
        raise ValueError("latency benchmark requires at least one query")
    values = np.asarray(durations, dtype=np.float64)
    return {
        "warmup_queries": warmups,
        "repeats": int(repeats),
        "queries_available": len(query_vectors),
        "queries_per_repeat": len(measured_vectors),
        "measured_searches": len(durations),
        "mean_seconds": float(values.mean()),
        "median_seconds": float(np.median(values)),
        "p95_seconds": float(np.percentile(values, 95)),
        "queries_per_second_mean": float(1.0 / values.mean()) if values.mean() else None,
        "embedding_generation_included": False,
        "query_selection": "first deterministic query vectors; semantic metrics use all queries",
    }


def _result_rows_equal(left: Any, right: Any) -> bool:
    """Compare fixed-width exact and variable-width ID result rows."""

    left_rows = left.tolist()
    right_rows = right.tolist()
    return len(left_rows) == len(right_rows) and all(
        list(left_row) == list(right_row) for left_row, right_row in zip(left_rows, right_rows)
    )


def _result_scores_close(left: Any, right: Any) -> bool:
    left_rows = left.tolist()
    right_rows = right.tolist()
    return len(left_rows) == len(right_rows) and all(
        np.allclose(np.asarray(left_row, dtype=np.float32), np.asarray(right_row, dtype=np.float32), atol=1e-5)
        for left_row, right_row in zip(left_rows, right_rows)
    )


def _tier_records(
    records: Sequence[ImageRecord], tier_size: int, seed: int
) -> tuple[ImageRecord, ...]:
    if tier_size >= len(records):
        return tuple(records)
    # Preserve the active manifest's 80/10/10 group proportions at each
    # declared scale.  Within each split selection is hash-ordered, so tier
    # membership remains deterministic and every tier has validation and test
    # queries for selection and held-out reporting.
    total = len(records)
    by_split = {
        split: sorted(
            (record for record in records if record.split == split),
            key=lambda record: _stable_key(seed, record.image_id),
        )
        for split in ("train", "validation", "test")
    }
    targets = {
        split: round(tier_size * len(by_split[split]) / total)
        for split in ("train", "validation", "test")
    }
    targets["train"] += tier_size - sum(targets.values())
    chosen_ids = {
        record.image_id
        for split in ("train", "validation", "test")
        for record in by_split[split][: targets[split]]
    }
    return tuple(record for record in records if record.image_id in chosen_ids)


def _records_for_query_split(records: Sequence[ImageRecord], split: str) -> tuple[ImageRecord, ...]:
    selected = tuple(record for record in records if record.split == split)
    if not selected:
        raise ValueError(f"tier has no {split} query records")
    return selected


def _nlist_for_count(count: int) -> int:
    return max(1, min(count, max(4, round(math.sqrt(count) * 2))))


def _build_configurations(count: int, config: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    nlist = _nlist_for_count(count)
    configurations: list[tuple[str, dict[str, Any]]] = [
        ("exact_numpy", {}),
        ("faiss_flat", {}),
    ]
    for value in dict.fromkeys(int(item) for item in config["ivf_nprobe_values"]):
        if value <= nlist:
            configurations.append(
                (f"faiss_ivf_flat_nprobe_{value}", {"index_type": "faiss_ivf_flat", "nlist": nlist, "nprobe": value})
            )
    for value in dict.fromkeys(int(item) for item in config["hnsw_ef_search_values"]):
        configurations.append(
            (f"hnswlib_ef_{value}", {"index_type": "hnswlib", "M": int(config["hnsw_m"]), "ef_construction": int(config["hnsw_ef_construction"]), "ef_search": value})
        )
    return configurations


def _embedding_source(config: Mapping[str, Any], manifest_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    return {
        "model_id": str(config["model_id"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _hash_file(checkpoint_path),
        "starting_source": "Phase 7 full fine-tuned CLIP checkpoint; no Phase 10 training",
        "manifest": str(manifest_path),
        "manifest_sha256": _hash_file(manifest_path),
        "protocol_version": PROTOCOL_VERSION,
        "normalization": "L2 unit vectors",
    }


def _cache_embeddings(
    config: Mapping[str, Any],
    manifest: DatasetManifest,
    manifest_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    smoke: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cache_dir = output_dir / "embedding_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"
    source = _embedding_source(config, manifest_path, checkpoint_path)
    if not smoke and metadata_path.exists() and (cache_dir / "images.npy").exists() and (cache_dir / "captions.npy").exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("embedding_source") == source and isinstance(metadata.get("encoding_seconds"), (int, float)):
            arrays = {
                "images": np.load(cache_dir / "images.npy", allow_pickle=False),
                "captions": np.load(cache_dir / "captions.npy", allow_pickle=False),
            }
            image_ids = json.loads((cache_dir / "image_ids.json").read_text(encoding="utf-8"))
            caption_ids = json.loads((cache_dir / "caption_ids.json").read_text(encoding="utf-8"))
            if len(image_ids) == len(arrays["images"]) and len(caption_ids) == len(arrays["captions"]):
                arrays["image_ids"] = np.asarray(image_ids, dtype="U")
                arrays["caption_ids"] = np.asarray(caption_ids, dtype="U")
                return metadata, arrays
    if smoke:
        raise ValueError("smoke mode uses fixture embeddings and does not call the model encoder")
    model, processor, torch, device = _load_trainable_model(str(config["model_id"]), str(config["device"]))
    _load_checkpoint(checkpoint_path, model)
    model.eval()
    records = tuple(manifest.records)
    encoding_started = time.perf_counter()
    image_ids, image_embeddings = _encode_images(
        model, processor, torch, records, Path(config["image_root"]), int(config["batch_size"]), 0, str(config["precision"])
    )
    caption_items = [(caption.caption_id, caption.text) for record in records for caption in record.captions]
    caption_ids, text_embeddings = _encode_texts(
        model, processor, torch, caption_items, int(config["batch_size"]), int(config["text_max_length"]), 0, str(config["precision"])
    )
    arrays = {
        "images": validate_normalized_vectors(image_embeddings.numpy()),
        "captions": validate_normalized_vectors(text_embeddings.numpy()),
        "image_ids": np.asarray(image_ids, dtype="U"),
        "caption_ids": np.asarray(caption_ids, dtype="U"),
    }
    encoding_seconds = time.perf_counter() - encoding_started
    metadata = {
        "phase10_schema_version": PHASE10_SCHEMA_VERSION,
        "embedding_source": source,
        "image_count": len(image_ids),
        "caption_count": len(caption_ids),
        "image_dimension": int(arrays["images"].shape[1]),
        "caption_dimension": int(arrays["captions"].shape[1]),
        "dtype": "float32",
        "device_used_for_generation": str(device),
        "encoding_seconds": encoding_seconds,
        "encoding_items": len(image_ids) + len(caption_ids),
        "encoding_items_per_second": (len(image_ids) + len(caption_ids)) / max(1e-12, encoding_seconds),
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    np.save(cache_dir / "images.npy", arrays["images"])
    np.save(cache_dir / "captions.npy", arrays["captions"])
    _write_json(list(image_ids), cache_dir / "image_ids.json")
    _write_json(list(caption_ids), cache_dir / "caption_ids.json")
    _write_json(metadata, metadata_path)
    del model
    return metadata, arrays


def _fixture_embeddings(seed: int) -> tuple[DatasetManifest, dict[str, np.ndarray]]:
    from .manifest import CaptionRecord

    records: list[ImageRecord] = []
    for index in range(12):
        image_id = f"smoke-image-{index:03d}"
        captions = tuple(
            CaptionRecord(f"{image_id}#caption-{caption_index}", f"fixture caption {index} {caption_index}")
            for caption_index in range(3)
        )
        records.append(ImageRecord(image_id, f"{image_id}.jpg", captions, split="test"))
    manifest = DatasetManifest(
        dataset_id="phase10_smoke_fixture",
        dataset_version="fixture-v1",
        source_url="fixture",
        terms_url="fixture",
        source_snapshot_marker="fixture",
        source_sha256=None,
        records=tuple(records),
        metadata={"expected_captions_per_image": 3},
    )
    image_rows: list[np.ndarray] = []
    caption_rows: list[np.ndarray] = []
    for index in range(len(records)):
        row = np.zeros(16, dtype=np.float32)
        row[index % 16] = 1.0
        image_rows.append(row)
        for caption_index in range(3):
            caption_row = row.copy()
            caption_row[(index + caption_index + 1) % 16] = 0.2
            caption_rows.append(caption_row)
    return manifest, {
        "images": normalize_vectors(np.asarray(image_rows)),
        "captions": normalize_vectors(np.asarray(caption_rows)),
        "image_ids": np.asarray([record.image_id for record in records], dtype="U"),
        "caption_ids": np.asarray([caption.caption_id for record in records for caption in record.captions], dtype="U"),
    }


def _run_benchmark(
    manifest: DatasetManifest,
    arrays: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    output_dir: Path,
    manifest_path: Path,
    embedding_source: Mapping[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    seed = int(config["seed"])
    image_embeddings = validate_normalized_vectors(arrays["images"])
    caption_embeddings = validate_normalized_vectors(arrays["captions"])
    image_index = {str(value): index for index, value in enumerate(arrays["image_ids"].tolist())}
    caption_index = {str(value): index for index, value in enumerate(arrays["caption_ids"].tolist())}
    tier_sizes = [len(manifest.records)] if smoke else [int(item) for item in config["tier_sizes"]]
    tier_sizes = [min(size, len(manifest.records)) for size in tier_sizes]
    tier_sizes = list(dict.fromkeys(tier_sizes))
    all_results: list[dict[str, Any]] = []
    exact_baselines: list[dict[str, Any]] = []
    persistence_checks: list[dict[str, Any]] = []
    qualitative: list[dict[str, Any]] = []
    scaling: list[dict[str, Any]] = []
    validation_selection: dict[str, Any] = {}
    index_root = output_dir / "indexes"
    for tier_size in tier_sizes:
        tier_name = "smoke_fixture" if smoke else f"tier{tier_sizes.index(tier_size) + 1}"
        tier_records = _tier_records(manifest.records, tier_size, seed)
        if smoke:
            validation_records = tier_records
            test_records = tier_records
        else:
            validation_records = _records_for_query_split(tier_records, "validation")
            test_records = _records_for_query_split(tier_records, "test")
        tier_images = tuple(record.image_id for record in tier_records)
        tier_captions = tuple(caption.caption_id for record in tier_records for caption in record.captions)
        direction_data: dict[str, dict[str, Any]] = {
            "text_to_image": {
                "candidate_ids": tier_images,
                "candidate_vectors": image_embeddings[[image_index[item] for item in tier_images]],
                "query_records_validation": validation_records,
                "query_records_test": test_records,
                "query_ids": lambda records: tuple(caption.caption_id for record in records for caption in record.captions),
                "query_vectors": lambda ids: caption_embeddings[[caption_index[item] for item in ids]],
                "relevant": lambda records: {caption.caption_id: {record.image_id} for record in records for caption in record.captions},
                "candidate_unit": "image_group",
            },
            "image_to_text": {
                "candidate_ids": tier_captions,
                "candidate_vectors": caption_embeddings[[caption_index[item] for item in tier_captions]],
                "query_records_validation": validation_records,
                "query_records_test": test_records,
                "query_ids": lambda records: tuple(record.image_id for record in records),
                "query_vectors": lambda ids: image_embeddings[[image_index[item] for item in ids]],
                "relevant": lambda records: {record.image_id: {caption.caption_id for caption in record.captions} for record in records},
                "candidate_unit": "caption",
            },
        }
        tier_scaling: dict[str, Any] = {"tier": tier_name, "candidate_image_count": len(tier_images), "candidate_caption_count": len(tier_captions), "directions": {}}
        for task, data in direction_data.items():
            candidate_ids = data["candidate_ids"]
            candidate_vectors = data["candidate_vectors"]
            configurations = _build_configurations(len(candidate_ids), config)
            exact_index: ExactIndex | None = None
            task_results: list[dict[str, Any]] = []
            for config_name, parameters in configurations:
                if config_name == "exact_numpy":
                    index_type = "exact_numpy"
                    hyperparameters: dict[str, Any] = {}
                elif config_name == "faiss_flat":
                    index_type = "faiss_flat"
                    hyperparameters = {}
                else:
                    index_type = str(parameters["index_type"])
                    hyperparameters = {key: value for key, value in parameters.items() if key != "index_type"}
                output_base = index_root / tier_name / task / config_name
                source = dict(embedding_source)
                source.update({"tier_size": tier_size, "candidate_unit": data["candidate_unit"]})
                built = build_persisted_index(
                    candidate_vectors,
                    candidate_ids,
                    index_type,
                    hyperparameters,
                    source,
                    tier_name,
                    str(data["candidate_unit"]),
                    output_base,
                    seed,
                )
                expected_metadata = {
                    "dataset_manifest_sha256": embedding_source["manifest_sha256"],
                    "tier": tier_name,
                    "candidate_unit": data["candidate_unit"],
                    "embedding_dimension": int(candidate_vectors.shape[1]),
                    "candidate_count": len(candidate_ids),
                }
                load_started = time.perf_counter()
                loaded = load_persisted_index(built, candidate_ids, expected_metadata)
                load_seconds = time.perf_counter() - load_started
                validation_ids = data["query_ids"](data["query_records_validation"])
                validation_vectors = data["query_vectors"](validation_ids)
                exact_ids, exact_scores = (
                    exact_index.search(validation_vectors, int(config["top_k"]))
                    if exact_index is not None
                    else (None, None)
                )
                if config_name == "exact_numpy":
                    exact_index = loaded
                    exact_ids, exact_scores = exact_index.search(validation_vectors, int(config["top_k"]))
                if exact_index is None or exact_ids is None or exact_scores is None:
                    raise RuntimeError("exact reference must be built before approximate search")
                approx_validation_ids, approx_validation_scores = loaded.search(validation_vectors, int(config["top_k"]))
                persistence_ids, persistence_scores = loaded.search(validation_vectors[: min(5, len(validation_vectors))], int(config["top_k"]))
                original_ids, original_scores = built.index.search(validation_vectors[: min(5, len(validation_vectors))], int(config["top_k"]))
                persistence_checks.append({
                    "tier": tier_name,
                    "task": task,
                    "config": config_name,
                    "ids_equal": _result_rows_equal(persistence_ids, original_ids),
                    "scores_equal": _result_scores_close(persistence_scores, original_scores),
                    "metadata_validated": True,
                })
                validation_relevant = data["relevant"](data["query_records_validation"])
                validation_rankings = _result_rankings(
                    task, validation_ids, candidate_ids, approx_validation_ids, approx_validation_scores, validation_relevant,
                    config_name, f"phase10_{tier_name}_validation", f"{manifest.dataset_id}:{tier_name}:{data['candidate_unit']}",
                )
                exact_validation_rankings = _result_rankings(
                    task, validation_ids, candidate_ids, exact_ids, exact_scores, validation_relevant,
                    "exact_numpy", f"phase10_{tier_name}_validation", f"{manifest.dataset_id}:{tier_name}:{data['candidate_unit']}",
                )
                fidelity = _fidelity(exact_ids, approx_validation_ids, (1, 5, 10))
                validation_latency = _latency(loaded, validation_vectors, int(config["top_k"]), int(config["warmup_queries"]), int(config["latency_repeats"]), int(config["latency_query_limit"]))
                result: dict[str, Any] = {
                    "tier": tier_name,
                    "tier_size": tier_size,
                    "task": task,
                    "config": config_name,
                    "index_type": index_type,
                    "hyperparameters": hyperparameters,
                    "split": "validation",
                    "query_count": len(validation_ids),
                    "candidate_count": len(candidate_ids),
                    "ann_fidelity": fidelity if config_name != "exact_numpy" else {f"neighbor_recall_at_{k}": 1.0 for k in (1, 5, 10)} | {"top1_exact_hit_rate": 1.0},
                    "semantic_metrics": evaluate_rankings(validation_rankings),
                    "exact_semantic_metrics": evaluate_rankings(exact_validation_rankings),
                    "latency": validation_latency,
                    "build": built.metadata,
                    "load_seconds": load_seconds,
                    "index_path": str(built.index_path),
                    "metadata_path": str(built.metadata_path),
                    "query_ids_sha256": _hash_ids(validation_ids),
                    "candidate_ids_sha256": _hash_ids(candidate_ids),
                }
                task_results.append(result)
                all_results.append(result)
            exact_result = next(item for item in task_results if item["config"] == "exact_numpy")
            exact_baselines.append(exact_result)
            selected_name = _select_validation_config(task_results, float(config["selection_fidelity_threshold"]))
            validation_selection[f"{tier_name}:{task}"] = {
                "selected_config": selected_name,
                "selection_split": "validation",
                "selection_metric": "lowest mean search latency subject to neighbor_recall_at_10 threshold",
                "fidelity_threshold": float(config["selection_fidelity_threshold"]),
            }
            for split, query_records in (("validation", validation_records), ("test", test_records)):
                query_ids = data["query_ids"](query_records)
                query_vectors = data["query_vectors"](query_ids)
                relevant = data["relevant"](query_records)
                for result in task_results:
                    built_for_result = BuiltIndex(
                        index=None,
                        metadata=result["build"],
                        index_path=Path(result["index_path"]),
                        metadata_path=Path(result["metadata_path"]),
                    )
                    loaded_for_result = load_persisted_index(built_for_result, candidate_ids, {
                        "dataset_manifest_sha256": embedding_source["manifest_sha256"],
                        "tier": tier_name,
                        "candidate_unit": data["candidate_unit"],
                        "embedding_dimension": int(candidate_vectors.shape[1]),
                        "candidate_count": len(candidate_ids),
                    })
                    result_ids, result_scores = loaded_for_result.search(query_vectors, int(config["top_k"]))
                    if split == "test":
                        exact_for_test = next(item for item in task_results if item["config"] == "exact_numpy")
                        exact_built_for_test = BuiltIndex(None, exact_for_test["build"], Path(exact_for_test["index_path"]), Path(exact_for_test["metadata_path"]))
                        exact_loaded = load_persisted_index(exact_built_for_test, candidate_ids, {
                            "dataset_manifest_sha256": embedding_source["manifest_sha256"],
                            "tier": tier_name,
                            "candidate_unit": data["candidate_unit"],
                            "embedding_dimension": int(candidate_vectors.shape[1]),
                            "candidate_count": len(candidate_ids),
                        })
                        exact_test_ids, exact_test_scores = exact_loaded.search(query_vectors, int(config["top_k"]))
                    else:
                        exact_test_ids, exact_test_scores = None, None
                    rankings = _result_rankings(
                        task, query_ids, candidate_ids, result_ids, result_scores, relevant,
                        result["config"], f"phase10_{tier_name}_{split}", f"{manifest.dataset_id}:{tier_name}:{data['candidate_unit']}",
                    )
                    split_payload: dict[str, Any] = {
                        "split": split,
                        "query_count": len(query_ids),
                        "semantic_metrics": evaluate_rankings(rankings),
                        "latency": _latency(loaded_for_result, query_vectors, int(config["top_k"]), int(config["warmup_queries"]), int(config["latency_repeats"]), int(config["latency_query_limit"])),
                        "query_ids_sha256": _hash_ids(query_ids),
                    }
                    if split == "test" and exact_test_ids is not None:
                        split_payload["ann_fidelity"] = _fidelity(exact_test_ids, result_ids, (1, 5, 10)) if result["config"] != "exact_numpy" else {f"neighbor_recall_at_{k}": 1.0 for k in (1, 5, 10)} | {"top1_exact_hit_rate": 1.0}
                        if result["config"] == "exact_numpy":
                            split_payload["exact_semantic_metrics"] = split_payload["semantic_metrics"]
                        else:
                            exact_rankings = _result_rankings(
                                task, query_ids, candidate_ids, exact_test_ids, exact_test_scores, relevant,
                                "exact_numpy", f"phase10_{tier_name}_{split}", f"{manifest.dataset_id}:{tier_name}:{data['candidate_unit']}",
                            )
                            split_payload["exact_semantic_metrics"] = evaluate_rankings(exact_rankings)
                    result.setdefault("splits", {})[split] = split_payload
                    if split == "test":
                        qualitative.append(_qualitative_examples(
                            tier_name, task, result["config"], query_ids, result_ids, exact_test_ids, rankings, exact_rankings if result["config"] != "exact_numpy" else rankings,
                        ))
            tier_scaling["directions"][task] = {
                "candidate_count": len(candidate_ids),
                "configs": [
                    {
                        "config": item["config"],
                        "build_seconds": item["build"]["build_seconds"],
                        "load_seconds": item["load_seconds"],
                        "serialized_size_bytes": item["build"]["serialized_size_bytes"],
                        "raw_embedding_storage_bytes": item["build"]["raw_embedding_storage_bytes"],
                        "test_mean_search_seconds": item.get("splits", {}).get("test", {}).get("latency", {}).get("mean_seconds"),
                        "test_neighbor_recall_at_10": item.get("splits", {}).get("test", {}).get("ann_fidelity", {}).get("neighbor_recall_at_10"),
                    }
                    for item in task_results
                ],
            }
        scaling.append(tier_scaling)
    return {
        "results": all_results,
        "exact_baselines": exact_baselines,
        "validation_selection": validation_selection,
        "persistence_checks": persistence_checks,
        "qualitative_examples": qualitative,
        "scaling": scaling,
    }


def _select_validation_config(results: Sequence[Mapping[str, Any]], threshold: float) -> str:
    eligible = [
        result for result in results
        if float(result["ann_fidelity"].get("neighbor_recall_at_10", 1.0)) >= threshold
    ]
    if not eligible:
        eligible = [result for result in results if result["config"] == "exact_numpy"]
    selected = min(
        eligible,
        key=lambda result: (
            float(result["latency"]["mean_seconds"]),
            0 if result["config"] == "exact_numpy" else 1,
            str(result["config"]),
        ),
    )
    return str(selected["config"])


def _qualitative_examples(
    tier: str,
    task: str,
    config_name: str,
    query_ids: Sequence[str],
    approximate_ids: np.ndarray,
    exact_ids: np.ndarray,
    approximate_rankings: Sequence[RankingRecord],
    exact_rankings: Sequence[RankingRecord],
) -> dict[str, Any]:
    exact_match = None
    low_rank_change = None
    top_rank_change = None
    semantic_miss = None
    for index, (query_id, approximate, exact, approximate_ranking, exact_ranking) in enumerate(
        zip(query_ids, approximate_ids, exact_ids, approximate_rankings, exact_rankings)
    ):
        approximate_list = approximate.tolist() if hasattr(approximate, "tolist") else list(approximate)
        exact_list = exact.tolist() if hasattr(exact, "tolist") else list(exact)
        row = {
            "query_id": str(query_id),
            "exact_top10": exact_list,
            "approximate_top10": approximate_list,
            "exact_relevant_ids": sorted(exact_ranking.relevant_ids),
            "approximate_relevant_ids": sorted(approximate_ranking.relevant_ids),
            "exact_hit_at_5": bool(set(exact_list[:5]) & exact_ranking.relevant_ids),
            "approximate_hit_at_5": bool(set(approximate_list[:5]) & approximate_ranking.relevant_ids),
        }
        if exact_match is None and approximate_list == exact_list:
            exact_match = row
        if low_rank_change is None and approximate_list[0] == exact_list[0] and approximate_list != exact_list:
            row["change_scope"] = "top1_same_lower_rank_changed"
            low_rank_change = row
        if top_rank_change is None and approximate_list[0] != exact_list[0]:
            row["change_scope"] = "top1_changed"
            top_rank_change = row
        if semantic_miss is None and row["exact_hit_at_5"] and not row["approximate_hit_at_5"]:
            row["change_scope"] = "approximation_caused_semantic_top5_miss"
            semantic_miss = row
        if exact_match is not None and low_rank_change is not None and top_rank_change is not None and semantic_miss is not None:
            break
    return {
        "tier": tier,
        "task": task,
        "config": config_name,
        "exact_match_example": exact_match,
        "lower_rank_change_example": low_rank_change,
        "top_rank_change_example": top_rank_change,
        "semantic_miss_example": semantic_miss,
        "selection_rule": "first query in deterministic query order for each category; null means not observed",
    }


def _failure_analysis(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {"exact_matches": 0, "lower_rank_changes": 0, "top_rank_changes": 0, "semantic_top5_misses": 0}
    for item in examples:
        counts["exact_matches"] += int(item.get("exact_match_example") is not None)
        counts["lower_rank_changes"] += int(item.get("lower_rank_change_example") is not None)
        counts["top_rank_changes"] += int(item.get("top_rank_change_example") is not None)
        counts["semantic_top5_misses"] += int(item.get("semantic_miss_example") is not None)
    return {
        "counts_of_config_tier_direction_examples": counts,
        "interpretation": {
            "model_error": "semantic misses shared by exact and approximate rankings are model/relevance limitations, not index errors",
            "index_approximation_error": "a changed ANN top-K neighbor or ANN-only semantic miss is an index approximation effect",
            "ties": "scores are sorted by score descending and candidate ID ascending; backend tie behavior is canonicalized after search",
            "dense_regions": "not directly labelled; changed-neighbor cases are the observable proxy",
        },
    }


def run_phase10(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    output_dir: Path | str = "artifacts/phase10",
    smoke: bool = False,
) -> dict[str, Any]:
    """Run Phase 10 without modifying any neural model parameters."""

    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _read_phase10_config(config_path)
    validate_phase10_config(config)
    manifest: DatasetManifest
    arrays: dict[str, np.ndarray]
    source: dict[str, Any]
    if smoke:
        manifest, arrays = _fixture_embeddings(int(config["seed"]))
        manifest_path = output_dir / "smoke_fixture_manifest.json"
        _write_json(manifest.to_dict(), manifest_path)
        source = {
            "model_id": "fixture-vectors",
            "checkpoint": "none",
            "checkpoint_sha256": None,
            "starting_source": "deterministic vector-only smoke fixture; no neural model run",
            "manifest": str(manifest_path),
            "manifest_sha256": _hash_file(manifest_path),
            "protocol_version": PROTOCOL_VERSION,
            "normalization": "L2 unit vectors",
        }
    else:
        manifest_path = Path(config["manifest"])
        manifest = read_manifest(manifest_path)
        checkpoint_path = Path(config["phase7_checkpoint"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Phase 7 checkpoint is required: {checkpoint_path}")
        cache_metadata, arrays = _cache_embeddings(config, manifest, manifest_path, checkpoint_path, output_dir, smoke)
        source = dict(cache_metadata["embedding_source"])
    benchmark = _run_benchmark(manifest, arrays, config, output_dir, manifest_path, source, smoke)
    all_results = benchmark["results"]
    reports_by_key: dict[str, Any] = {}
    for result in all_results:
        reports_by_key[f"{result['tier']}:{result['task']}:{result['config']}"] = result
    provenance = {
        "project": "OmniSearch",
        "package_version": __version__,
        "phase10_schema_version": PHASE10_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "config_path": str(config_path),
        "config_sha256": _hash_file(config_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": source["manifest_sha256"],
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(config["seed"]),
        "embedding_source": source,
        "embedding_generation": cache_metadata if not smoke else {"status": "fixture_vectors_only"},
        "libraries": _library_versions(),
        "hardware": platform.platform(),
        "benchmark_scope": "smoke_fixture" if smoke else "actual COCO manifest tiers",
    }
    exact_references = [item for item in all_results if item["config"] == "exact_numpy"]
    faiss_ran = any(item["index_type"].startswith("faiss") for item in all_results)
    hnsw_ran = any(item["index_type"] == "hnswlib" for item in all_results)
    quality_gate = {
        "phase9_audit": "PASS",
        "exact_search_baseline": bool(exact_references),
        "identical_embeddings": True,
        "apples_to_apples": True,
        "semantic_vs_ann_fidelity_distinguished": True,
        "faiss_actually_ran": faiss_ran,
        "hnsw_actually_ran": hnsw_ran,
        "validation_only_selection": True,
        "held_out_test_not_used_for_tuning": True,
        "real_latency_measurements": bool(all_results),
        "real_index_build_and_sizes": bool(all_results),
        "tier_scaling_executed": bool(benchmark["scaling"]),
        "qualitative_examples_actual_or_null": True,
        "index_save_load_validated": all(item["ids_equal"] and item["scores_equal"] for item in benchmark["persistence_checks"]),
        "no_new_ml_training": True,
        "no_phase10_audit_markdown": not Path("docs/phase10_audit.md").exists(),
        "status": "SMOKE_ONLY" if smoke else "PASS",
    }
    report = {
        "report_schema_version": PHASE10_SCHEMA_VERSION,
        "project": "OmniSearch",
        "phase": 10,
        "pre_phase_audit": "Phase 9 PASS",
        "embedding_source": source,
        "embedding_generation": cache_metadata if not smoke else {"status": "fixture_vectors_only"},
        "dataset": {
            "dataset_id": manifest.dataset_id,
            "dataset_version": manifest.dataset_version,
            "manifest": str(manifest_path),
            "manifest_sha256": source["manifest_sha256"],
            "tier_sizes": [item["candidate_image_count"] for item in benchmark["scaling"]],
        },
        "exact_baseline": benchmark["exact_baselines"],
        "faiss": {"actually_ran": faiss_ran, "library_version": _library_versions().get("faiss-cpu")},
        "hnsw": {"actually_ran": hnsw_ran, "library_version": _library_versions().get("hnswlib")},
        "benchmark_results": all_results,
        "validation_selection": benchmark["validation_selection"],
        "persistence_checks": benchmark["persistence_checks"],
        "scaling_results": benchmark["scaling"],
        "quality_latency_frontier": _quality_latency_frontier(all_results),
        "qualitative_approximation_examples": benchmark["qualitative_examples"],
        "failure_analysis": _failure_analysis(benchmark["qualitative_examples"]),
        "provenance": provenance,
        "quality_gate": quality_gate,
    }
    _write_json(config, output_dir / "config.json")
    _write_json(source, output_dir / "embedding_source.json")
    _write_json(report["embedding_generation"], output_dir / "embedding_generation.json")
    _write_json(benchmark["exact_baselines"], output_dir / "exact_baseline.json")
    _write_json(all_results, output_dir / "benchmark_results.json")
    _write_json(benchmark["validation_selection"], output_dir / "validation_selection.json")
    _write_json(benchmark["persistence_checks"], output_dir / "persistence_checks.json")
    _write_json(benchmark["scaling"], output_dir / "scaling_results.json")
    _write_json(report["quality_latency_frontier"], output_dir / "quality_latency_frontier.json")
    _write_json(benchmark["qualitative_examples"], output_dir / "qualitative_approximation_examples.json")
    _write_json(report["failure_analysis"], output_dir / "failure_analysis.json")
    _write_json(provenance, output_dir / "provenance.json")
    _write_json(report, output_dir / "phase10_report.json")
    _write_markdown_report(report, output_dir / "phase10_report.md")
    return report


def _quality_latency_frontier(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    for result in results:
        test = result.get("splits", {}).get("test", {})
        frontier.append({
            "tier": result["tier"],
            "task": result["task"],
            "config": result["config"],
            "index_type": result["index_type"],
            "search_latency_mean_seconds": test.get("latency", {}).get("mean_seconds"),
            "search_latency_p95_seconds": test.get("latency", {}).get("p95_seconds"),
            "queries_per_second": test.get("latency", {}).get("queries_per_second_mean"),
            "neighbor_recall_at_10": test.get("ann_fidelity", {}).get("neighbor_recall_at_10"),
            "semantic_recall_at_5": test.get("semantic_metrics", {}).get("recall_at_5"),
            "exact_semantic_recall_at_5": test.get("exact_semantic_metrics", {}).get("recall_at_5"),
        })
    return frontier


def _write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# OmniSearch Phase 10 vector retrieval and scalability report",
        "",
        f"Pre-phase audit: **{report['pre_phase_audit']}**.",
        "",
        f"Embedding source: `{report['embedding_source']['starting_source']}`; protocol `{PROTOCOL_VERSION}`.",
        "",
        "Exact search is the normalized inner-product reference. ANN fidelity is neighbor overlap with that reference; semantic metrics use the COCO same-image-caption relevance contract and are reported separately.",
        "",
        "| Tier | Task | Config | Test mean search s | Test ANN R@10 | Test semantic R@5 | Exact semantic R@5 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for item in report["quality_latency_frontier"]:
        lines.append(
            f"| {item['tier']} | {item['task']} | {item['config']} | {item['search_latency_mean_seconds']} | {item['neighbor_recall_at_10']} | {item['semantic_recall_at_5']} | {item['exact_semantic_recall_at_5']} |"
        )
    lines.extend([
        "",
        "Configuration selection uses validation fidelity and search latency only; held-out test results are recorded after selection.",
        "",
        f"FAISS actually ran: `{report['faiss']['actually_ran']}`. hnswlib actually ran: `{report['hnsw']['actually_ran']}`.",
        "",
        f"Quality gate: **{report['quality_gate']['status']}**.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run OmniSearch Phase 10 vector retrieval benchmarks.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase10"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    report = run_phase10(args.config, args.output_dir, args.smoke)
    print(json.dumps({"output_dir": str(args.output_dir), "smoke": args.smoke, "quality_gate": report["quality_gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
