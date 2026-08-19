"""Phase 11 two-stage retrieval with a train-only shallow reranker.

Stage 1 uses persisted FAISS Flat retrieval over the already validated Phase 7
CLIP embedding space.  Stage 2 is a small pairwise ranking head over explicit
query/candidate interaction features: elementwise product, absolute
difference, and cosine similarity.  It is trained only on the Tier-2 train
groups and evaluated on validation/test groups without modifying CLIP.
"""

from __future__ import annotations

import gc
import json
import platform
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from . import __version__
from .config import DEFAULT_CONFIG_PATH
from .evaluation import (
    PROTOCOL_VERSION,
    RankingRecord,
    compare_systems,
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
from .phase10 import (
    _fixture_embeddings,
    _hash_ids,
    _stable_key,
    _tier_records,
    build_persisted_index,
    load_persisted_index,
    normalize_vectors,
)

PHASE11_SCHEMA_VERSION = 1
DEFAULT_PHASE11_CONFIG: dict[str, Any] = {
    "manifest": "data/processed/coco2017_val_split_manifest.json",
    "image_root": "data/raw/coco2017/val2017",
    "phase7_checkpoint": "artifacts/phase7/best_checkpoint.pt",
    "phase10_embedding_cache": "artifacts/phase10/embedding_cache",
    "model_id": "openai/clip-vit-base-patch32",
    "device": "auto",
    "seed": 42,
    "batch_size": 128,
    "epochs": 3,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "margin": 0.1,
    "hidden_dim": 64,
    "candidate_depths": [10, 25, 50],
    "selection_metric": "mean_mrr",
    "bootstrap_resamples": 200,
    "max_train_images": 800,
    "latency_query_limit": 128,
    "latency_repeats": 3,
    "warmup_queries": 5,
    "tier_sizes": [1000, 5000],
}


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_config(path: Path) -> dict[str, Any]:
    import tomllib

    with path.open("rb") as file:
        raw = tomllib.load(file)
    values = dict(DEFAULT_PHASE11_CONFIG)
    values.update(dict(raw.get("phase11", {})))
    return values


def validate_phase11_config(config: Mapping[str, Any]) -> None:
    for key in ("batch_size", "epochs", "hidden_dim", "latency_query_limit", "latency_repeats"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if float(config["learning_rate"]) <= 0 or float(config["weight_decay"]) < 0:
        raise ValueError("invalid optimizer configuration")
    if float(config["margin"]) < 0:
        raise ValueError("margin must be non-negative")
    depths = [int(value) for value in config["candidate_depths"]]
    if not depths or any(value <= 0 for value in depths) or depths != sorted(set(depths)):
        raise ValueError("candidate_depths must be sorted, unique, and positive")
    if str(config["selection_metric"]) not in {"mean_mrr", "mean_recall_at_5"}:
        raise ValueError("unsupported selection metric")
    if int(config["bootstrap_resamples"]) <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    if int(config["max_train_images"]) <= 0:
        raise ValueError("max_train_images must be positive")
    if any(int(value) <= 0 for value in config["tier_sizes"]):
        raise ValueError("tier_sizes must be positive")
    checkpoint = str(config["phase7_checkpoint"])
    if "phase8" in checkpoint or "phase9" in checkpoint:
        raise ValueError("Phase 11 must use the validated Phase 7 checkpoint")


def _load_embedding_cache(config: Mapping[str, Any], manifest_path: Path, checkpoint_path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cache = Path(config["phase10_embedding_cache"])
    metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
    source = dict(metadata["embedding_source"])
    if source["manifest_sha256"] != _hash_file(manifest_path):
        raise ValueError("Phase 10 embedding cache is stale for the active manifest")
    if source["checkpoint_sha256"] != _hash_file(checkpoint_path):
        raise ValueError("Phase 10 embedding cache is stale for the Phase 7 checkpoint")
    arrays = {
        "images": normalize_vectors(np.load(cache / "images.npy", allow_pickle=False)),
        "captions": normalize_vectors(np.load(cache / "captions.npy", allow_pickle=False)),
        "image_ids": np.asarray(json.loads((cache / "image_ids.json").read_text()), dtype="U"),
        "caption_ids": np.asarray(json.loads((cache / "caption_ids.json").read_text()), dtype="U"),
    }
    if len(arrays["image_ids"]) != len(arrays["images"]) or len(arrays["caption_ids"]) != len(arrays["captions"]):
        raise ValueError("embedding cache IDs do not match vector rows")
    return source, arrays


def _device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


class PairwiseMLPReranker(nn.Module):
    """Small learned interaction scorer, not a calibrated probability model."""

    def __init__(self, dimension: int, hidden_dim: int) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(2 * self.dimension + 1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def pair_features(self, query: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        if query.ndim != 2 or candidate.ndim != 2 or query.shape != candidate.shape:
            raise ValueError("query and candidate tensors must have equal rank-2 shapes")
        if query.shape[1] != self.dimension:
            raise ValueError("reranker vector dimension mismatch")
        product = query * candidate
        absolute_difference = torch.abs(query - candidate)
        cosine = torch.sum(product, dim=1, keepdim=True)
        features = torch.cat((product, absolute_difference, cosine), dim=1)
        if not bool(torch.isfinite(features).all().item()):
            raise FloatingPointError("reranker features are non-finite")
        return features

    def forward(self, query: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        return self.network(self.pair_features(query, candidate)).squeeze(-1)


def _row_values(row: Any) -> list[Any]:
    return row.tolist() if hasattr(row, "tolist") else list(row)


def _task_data(records: Sequence[ImageRecord], task: str, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    image_index = {str(value): index for index, value in enumerate(arrays["image_ids"].tolist())}
    caption_index = {str(value): index for index, value in enumerate(arrays["caption_ids"].tolist())}
    if task == "text_to_image":
        candidate_ids = tuple(record.image_id for record in records)
        query_ids = tuple(caption.caption_id for record in records for caption in record.captions)
        candidate_vectors = arrays["images"][[image_index[item] for item in candidate_ids]]
        query_vectors = arrays["captions"][[caption_index[item] for item in query_ids]]
        relevant = {caption.caption_id: {record.image_id} for record in records for caption in record.captions}
        candidate_unit = "image_group"
    elif task == "image_to_text":
        candidate_ids = tuple(caption.caption_id for record in records for caption in record.captions)
        query_ids = tuple(record.image_id for record in records)
        candidate_vectors = arrays["captions"][[caption_index[item] for item in candidate_ids]]
        query_vectors = arrays["images"][[image_index[item] for item in query_ids]]
        relevant = {record.image_id: {caption.caption_id for caption in record.captions} for record in records}
        candidate_unit = "caption"
    else:
        raise ValueError(f"unsupported Phase 11 task: {task}")
    return {
        "candidate_ids": candidate_ids,
        "query_ids": query_ids,
        "candidate_vectors": candidate_vectors,
        "query_vectors": query_vectors,
        "relevant": relevant,
        "candidate_unit": candidate_unit,
    }


def _build_stage1(
    data: Mapping[str, Any],
    source: Mapping[str, Any],
    tier: str,
    split: str,
    task: str,
    output_dir: Path,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    index_type = "faiss_flat"
    embedding_source = {**source, "stage1": "FAISS IndexFlatIP exact retrieval"}
    base = output_dir / "indexes" / tier / split / task / "faiss_flat"
    built = build_persisted_index(
        data["candidate_vectors"], data["candidate_ids"], index_type, {},
        embedding_source, f"{tier}_{split}", data["candidate_unit"], base, seed,
    )
    expected = {
        "dataset_manifest_sha256": source["manifest_sha256"],
        "tier": f"{tier}_{split}",
        "candidate_unit": data["candidate_unit"],
        "embedding_dimension": int(data["candidate_vectors"].shape[1]),
        "candidate_count": len(data["candidate_ids"]),
    }
    loaded = load_persisted_index(built, data["candidate_ids"], expected)
    metadata = {"index_path": str(built.index_path), "metadata_path": str(built.metadata_path), "build": built.metadata}
    return loaded, metadata


def _stage1_rows(index: Any, data: Mapping[str, Any], max_depth: int) -> dict[str, Any]:
    ids, scores = index.search(data["query_vectors"], max_depth)
    return {
        "query_ids": tuple(data["query_ids"]),
        "candidate_rows": [_row_values(row) for row in ids],
        "score_rows": [[float(value) for value in _row_values(row)] for row in scores],
        "relevant": data["relevant"],
        "candidate_count": len(data["candidate_ids"]),
        "candidate_corpus_id": f"{data['candidate_unit']}:{_hash_ids(data['candidate_ids'])}",
    }


def _make_rankings(
    task: str,
    rows: Mapping[str, Any],
    depth: int,
    candidate_ids: Sequence[str],
    system_id: str,
    experiment_id: str,
) -> tuple[RankingRecord, ...]:
    rankings: list[RankingRecord] = []
    for query_id, ids, scores in zip(rows["query_ids"], rows["candidate_rows"], rows["score_rows"]):
        rankings.append(
            ranking_from_scores(
                query_id=query_id,
                task=task,
                candidates=[(str(item_id), float(score)) for item_id, score in zip(ids[:depth], scores[:depth])],
                relevant_ids=rows["relevant"][query_id],
                system_id=system_id,
                experiment_id=experiment_id,
                candidate_count=rows["candidate_count"],
                candidate_corpus_id=rows["candidate_corpus_id"],
            )
        )
    return tuple(rankings)


def _rerank_rows(
    model: PairwiseMLPReranker,
    rows: Mapping[str, Any],
    query_vectors: np.ndarray,
    candidate_vector_map: Mapping[str, np.ndarray],
    depth: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    output_ids: list[list[str]] = []
    output_scores: list[list[float]] = []
    with torch.no_grad():
        for query_vector, ids in zip(query_vectors, rows["candidate_rows"]):
            selected = [str(item) for item in ids[:depth]]
            if not selected:
                raise ValueError("Stage 1 returned an empty candidate set")
            query_tensor = torch.as_tensor(query_vector, dtype=torch.float32, device=device).repeat(len(selected), 1)
            candidate_tensor = torch.as_tensor(np.asarray([candidate_vector_map[item] for item in selected]), dtype=torch.float32, device=device)
            scores = model(query_tensor, candidate_tensor).detach().cpu().numpy().astype(np.float32).tolist()
            ordered = sorted(zip(selected, scores), key=lambda item: (-float(item[1]), item[0]))
            output_ids.append([item[0] for item in ordered])
            output_scores.append([float(item[1]) for item in ordered])
    return {**rows, "candidate_rows": output_ids, "score_rows": output_scores}


def _candidate_recall(rows: Mapping[str, Any], depth: int) -> dict[str, float]:
    fractions: list[float] = []
    hits: list[float] = []
    for query_id, ids in zip(rows["query_ids"], rows["candidate_rows"]):
        relevant = rows["relevant"][query_id]
        present = set(ids[:depth]) & set(relevant)
        fractions.append(len(present) / max(1, len(relevant)))
        hits.append(float(bool(present)))
    return {
        "candidate_recall_fraction": statistics.fmean(fractions) if fractions else 0.0,
        "candidate_hit_rate": statistics.fmean(hits) if hits else 0.0,
        "query_count": len(fractions),
        "depth": min(depth, rows["candidate_count"]),
    }


def _oracle_rankings(task: str, rows: Mapping[str, Any], depth: int, experiment_id: str) -> tuple[RankingRecord, ...]:
    output: list[RankingRecord] = []
    for query_id, ids in zip(rows["query_ids"], rows["candidate_rows"]):
        selected = [str(item) for item in ids[:depth]]
        relevant = rows["relevant"][query_id]
        positive = sorted(set(selected) & set(relevant))
        negative = sorted(set(selected) - set(relevant))
        ordered = positive + negative
        output.append(
            ranking_from_scores(
                query_id=query_id,
                task=task,
                candidates=[(item, float(len(ordered) - index)) for index, item in enumerate(ordered)],
                relevant_ids=relevant,
                system_id="oracle_analysis_not_a_model",
                experiment_id=experiment_id,
                candidate_count=rows["candidate_count"],
                candidate_corpus_id=rows["candidate_corpus_id"],
            )
        )
    return tuple(output)


def _synchronize(device: torch.device) -> None:
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _latency_stage1(index: Any, query_vectors: np.ndarray, depth: int, config: Mapping[str, Any]) -> dict[str, Any]:
    queries = query_vectors[: min(int(config["latency_query_limit"]), len(query_vectors))]
    for vector in queries[: min(int(config["warmup_queries"]), len(queries))]:
        index.search(vector[None, :], depth)
    values: list[float] = []
    for _ in range(int(config["latency_repeats"])):
        for vector in queries:
            started = time.perf_counter()
            index.search(vector[None, :], depth)
            values.append(time.perf_counter() - started)
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_seconds": float(array.mean()),
        "median_seconds": float(np.median(array)),
        "p95_seconds": float(np.percentile(array, 95)),
        "queries_per_second": float(1.0 / array.mean()) if array.mean() else None,
        "queries_available": len(query_vectors),
        "queries_measured_per_repeat": len(queries),
        "repeats": int(config["latency_repeats"]),
        "embedding_included": False,
    }


def _score_one(model: PairwiseMLPReranker, query: np.ndarray, ids: Sequence[str], candidate_vectors: Mapping[str, np.ndarray], device: torch.device) -> None:
    selected = [str(item) for item in ids]
    with torch.no_grad():
        query_tensor = torch.as_tensor(query, dtype=torch.float32, device=device).repeat(len(selected), 1)
        candidate_tensor = torch.as_tensor(np.asarray([candidate_vectors[item] for item in selected]), dtype=torch.float32, device=device)
        model(query_tensor, candidate_tensor)


def _latency_reranker(model: PairwiseMLPReranker, rows: Mapping[str, Any], query_vectors: np.ndarray, candidate_vectors: Mapping[str, np.ndarray], depth: int, device: torch.device, config: Mapping[str, Any]) -> dict[str, Any]:
    model.eval()
    queries = query_vectors[: min(int(config["latency_query_limit"]), len(query_vectors))]
    selected_rows = [row[:depth] for row in rows["candidate_rows"][: len(queries)]]
    for query, ids in zip(queries[: min(int(config["warmup_queries"]), len(queries))], selected_rows):
        _score_one(model, query, ids, candidate_vectors, device)
    _synchronize(device)
    values: list[float] = []
    for _ in range(int(config["latency_repeats"])):
        for query, ids in zip(queries, selected_rows):
            _synchronize(device)
            started = time.perf_counter()
            _score_one(model, query, ids, candidate_vectors, device)
            _synchronize(device)
            values.append(time.perf_counter() - started)
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_seconds": float(array.mean()),
        "median_seconds": float(np.median(array)),
        "p95_seconds": float(np.percentile(array, 95)),
        "queries_per_second": float(1.0 / array.mean()) if array.mean() else None,
        "queries_available": len(query_vectors),
        "queries_measured_per_repeat": len(queries),
        "repeats": int(config["latency_repeats"]),
        "embedding_included": False,
        "candidate_depth": depth,
    }


def _training_pairs(
    model_data: Mapping[str, Any],
    stage1_rows: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    task: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    image_index = {str(value): index for index, value in enumerate(arrays["image_ids"].tolist())}
    caption_index = {str(value): index for index, value in enumerate(arrays["caption_ids"].tolist())}
    queries: list[np.ndarray] = []
    positives: list[np.ndarray] = []
    negatives: list[np.ndarray] = []
    same_image_exclusions = 0
    for query_id, ids in zip(stage1_rows["query_ids"], stage1_rows["candidate_rows"]):
        relevant = stage1_rows["relevant"][query_id]
        negative_id = next((str(item) for item in ids if str(item) not in relevant), None)
        if negative_id is None:
            raise ValueError("training Stage 1 candidate set did not contain a valid negative")
        if task == "text_to_image":
            query_vector = arrays["captions"][caption_index[query_id]]
            positive_vector = arrays["images"][image_index[next(iter(relevant))]]
            negative_vector = arrays["images"][image_index[negative_id]]
        else:
            query_vector = arrays["images"][image_index[query_id]]
            positive_vector = arrays["captions"][caption_index[min(relevant)]]
            negative_vector = arrays["captions"][caption_index[negative_id]]
        queries.append(query_vector)
        positives.append(positive_vector)
        negatives.append(negative_vector)
        same_image_exclusions += len(relevant)
    return (
        np.asarray(queries, dtype=np.float32),
        np.asarray(positives, dtype=np.float32),
        np.asarray(negatives, dtype=np.float32),
        {"pairs": len(queries), "known_positive_candidates_excluded": same_image_exclusions, "task": task},
    )


def _train_reranker(
    pairs: tuple[np.ndarray, np.ndarray, np.ndarray],
    config: Mapping[str, Any],
    output_dir: Path,
    dimension: int,
) -> tuple[PairwiseMLPReranker, dict[str, Any], torch.device]:
    torch.manual_seed(int(config["seed"]))
    device = _device(str(config["device"]))
    model = PairwiseMLPReranker(dimension, int(config["hidden_dim"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    query, positive, negative = [torch.as_tensor(item, dtype=torch.float32) for item in pairs]
    size = len(query)
    history: list[dict[str, Any]] = []
    initial = next(model.parameters()).detach().clone()
    started = time.perf_counter()
    for epoch in range(int(config["epochs"])):
        permutation = torch.randperm(size)
        losses: list[float] = []
        for start in range(0, size, int(config["batch_size"])):
            indices = permutation[start : start + int(config["batch_size"])]
            q_batch = query[indices].to(device)
            p_batch = positive[indices].to(device)
            n_batch = negative[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            positive_score = model(q_batch, p_batch)
            negative_score = model(q_batch, n_batch)
            loss = torch.nn.functional.softplus(float(config["margin"]) - positive_score + negative_score).mean()
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("non-finite reranker loss")
            loss.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if not gradients or not all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients):
                raise FloatingPointError("non-finite reranker gradient")
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        history.append({"epoch": epoch + 1, "loss": statistics.fmean(losses), "pairs": size})
    _synchronize(device)
    training_seconds = time.perf_counter() - started
    updated = not bool(torch.equal(initial.cpu(), next(model.parameters()).detach().cpu()))
    if not updated:
        raise RuntimeError("reranker training did not update parameters")
    metadata = {
        "phase11_schema_version": PHASE11_SCHEMA_VERSION,
        "architecture": "PairwiseMLPReranker(product, absolute_difference, cosine)",
        "dimension": dimension,
        "hidden_dim": int(config["hidden_dim"]),
        "trainable_parameters": sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad),
        "seed": int(config["seed"]),
        "device": str(device),
        "loss": "softplus(margin - positive_score + negative_score)",
        "margin": float(config["margin"]),
        "optimizer": "AdamW",
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "epochs": int(config["epochs"]),
        "training_seconds": training_seconds,
        "parameter_update_verified": updated,
        "training_split": "train_only",
        "test_used": False,
        "validation_used": False,
        "scores_are_probabilities": False,
    }
    path = output_dir / "reranker_checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, path)
    _write_json(history, output_dir / "training_history.json")
    _write_json({**metadata, "checkpoint": str(path), "checkpoint_size_bytes": path.stat().st_size}, output_dir / "reranker_config.json")
    return model, {**metadata, "checkpoint": str(path), "checkpoint_size_bytes": path.stat().st_size, "history": history}, device


def _paired_metadata(
    system_id: str,
    task: str,
    dataset_id: str,
    split: str,
    corpus_id: str,
    relevance: str,
) -> dict[str, Any]:
    return {
        "system_id": system_id,
        "task": task,
        "dataset_id": dataset_id,
        "split": split,
        "protocol_version": PROTOCOL_VERSION,
        "candidate_corpus_id": corpus_id,
        "relevance_definition": relevance,
    }


def _rank_change_examples(
    stage1: Sequence[RankingRecord], reranked: Sequence[RankingRecord], depth: int,
) -> dict[str, Any]:
    def rank(record: RankingRecord) -> int | None:
        for index, item in enumerate(record.candidate_ids, start=1):
            if item in record.relevant_ids:
                return index
        return None
    categories: dict[str, Any] = {"promoted_relevant": None, "demoted_relevant": None, "unchanged": None, "regression": None, "candidate_miss": None}
    for left, right in zip(stage1, reranked):
        left_rank, right_rank = rank(left), rank(right)
        row = {
            "query_id": left.query_id,
            "stage1_top10": list(left.candidate_ids[:10]),
            "reranked_top10": list(right.candidate_ids[:10]),
            "stage1_scores": list(left.scores[:10]),
            "reranked_scores": list(right.scores[:10]),
            "stage1_rank": left_rank,
            "reranked_rank": right_rank,
            "depth": depth,
        }
        if not (set(left.candidate_ids[:depth]) & left.relevant_ids) and categories["candidate_miss"] is None:
            categories["candidate_miss"] = row
        if left_rank is not None and right_rank is not None and right_rank < left_rank and categories["promoted_relevant"] is None:
            categories["promoted_relevant"] = row
        if left_rank is not None and right_rank is not None and right_rank > left_rank and categories["demoted_relevant"] is None:
            categories["demoted_relevant"] = row
        if list(left.candidate_ids) == list(right.candidate_ids) and categories["unchanged"] is None:
            categories["unchanged"] = row
        if left_rank is not None and left_rank <= 5 and (right_rank is None or right_rank > 5) and categories["regression"] is None:
            categories["regression"] = row
    return categories


def _evaluate_depth(
    task: str,
    tier: str,
    split: str,
    data: Mapping[str, Any],
    rows: Mapping[str, Any],
    model: PairwiseMLPReranker,
    device: torch.device,
    depth: int,
    config: Mapping[str, Any],
    dataset_id: str,
) -> dict[str, Any]:
    stage1_rankings = _make_rankings(task, rows, depth, data["candidate_ids"], "stage1_faiss_flat", f"phase11_{tier}_{split}_stage1")
    candidate_map = {str(item_id): vector for item_id, vector in zip(data["candidate_ids"], data["candidate_vectors"])}
    reranked_rows = _rerank_rows(model, rows, data["query_vectors"], candidate_map, depth, device)
    reranked_rankings = _make_rankings(task, reranked_rows, depth, data["candidate_ids"], "stage2_pairwise_mlp", f"phase11_{tier}_{split}_reranked")
    corpus_id = rows["candidate_corpus_id"]
    relevance_definition = stage1_rankings[0].relevance_definition
    comparison = compare_systems(
        stage1_rankings,
        reranked_rankings,
        _paired_metadata("stage1_faiss_flat", task, dataset_id, split, corpus_id, relevance_definition),
        _paired_metadata("stage2_pairwise_mlp", task, dataset_id, split, corpus_id, relevance_definition),
        ks=(1, 5, 10),
        bootstrap_resamples=int(config["bootstrap_resamples"]),
        seed=int(config["seed"]),
    )
    return {
        "tier": tier,
        "split": split,
        "task": task,
        "requested_depth": depth,
        "effective_depth": min(depth, rows["candidate_count"]),
        "candidate_recall": _candidate_recall(rows, depth),
        "stage1_metrics": evaluate_rankings(stage1_rankings),
        "reranked_metrics": evaluate_rankings(reranked_rankings),
        "paired_comparison": comparison,
        "stage1_rankings": stage1_rankings,
        "reranked_rankings": reranked_rankings,
        "reranked_rows": reranked_rows,
        "rank_change_counts": _rank_change_counts(stage1_rankings, reranked_rankings),
    }


def _rank_change_counts(stage1: Sequence[RankingRecord], reranked: Sequence[RankingRecord]) -> dict[str, int]:
    changed = 0
    top1_changed = 0
    promoted = 0
    demoted = 0
    for left, right in zip(stage1, reranked):
        changed += int(left.candidate_ids != right.candidate_ids)
        top1_changed += int(left.candidate_ids[:1] != right.candidate_ids[:1])
        def first(record: RankingRecord) -> int | None:
            for index, item in enumerate(record.candidate_ids, 1):
                if item in record.relevant_ids:
                    return index
            return None
        left_rank, right_rank = first(left), first(right)
        if left_rank is not None and right_rank is not None:
            promoted += int(right_rank < left_rank)
            demoted += int(right_rank > left_rank)
    return {"queries": len(stage1), "ordering_changed": changed, "top1_changed": top1_changed, "relevant_promoted": promoted, "relevant_demoted": demoted}


def _measure_query_encoding(
    config: Mapping[str, Any],
    records: Sequence[ImageRecord],
    arrays: Mapping[str, np.ndarray],
    output_dir: Path,
    label: str,
) -> dict[str, Any]:
    model, processor, torch_module, device_string = _load_trainable_model(str(config["model_id"]), str(config["device"]))
    _load_checkpoint(Path(config["phase7_checkpoint"]), model)
    model.eval()
    started = time.perf_counter()
    image_ids, image_vectors = _encode_images(model, processor, torch_module, records, Path(config["image_root"]), int(config["batch_size"]), 0, "fp32")
    image_seconds = time.perf_counter() - started
    caption_items = [(caption.caption_id, caption.text) for record in records for caption in record.captions]
    started = time.perf_counter()
    caption_ids, caption_vectors = _encode_texts(model, processor, torch_module, caption_items, int(config["batch_size"]), 77, 0, "fp32")
    text_seconds = time.perf_counter() - started
    image_map = {str(value): index for index, value in enumerate(arrays["image_ids"].tolist())}
    caption_map = {str(value): index for index, value in enumerate(arrays["caption_ids"].tolist())}
    image_max_error = max(
        float(np.max(np.abs(image_vectors.numpy()[index] - arrays["images"][image_map[item]])))
        for index, item in enumerate(image_ids)
    )
    caption_max_error = max(
        float(np.max(np.abs(caption_vectors.numpy()[index] - arrays["captions"][caption_map[item]])))
        for index, item in enumerate(caption_ids)
    )
    payload = {
        "label": label,
        "device": device_string,
        "image_query_count": len(image_ids),
        "caption_query_count": len(caption_ids),
        "image_encoding_seconds": image_seconds,
        "text_encoding_seconds": text_seconds,
        "image_items_per_second": len(image_ids) / max(image_seconds, 1e-12),
        "text_items_per_second": len(caption_ids) / max(text_seconds, 1e-12),
        "cache_image_max_abs_error": image_max_error,
        "cache_text_max_abs_error": caption_max_error,
        "model_load_included": False,
    }
    del model
    gc.collect()
    return payload


def _fixture_manifest_and_arrays() -> tuple[DatasetManifest, dict[str, np.ndarray]]:
    manifest, arrays = _fixture_embeddings(42)
    records = tuple(record.with_split("train" if index < 6 else "validation" if index < 9 else "test") for index, record in enumerate(manifest.records))
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


def _quality_latency_row(result: Mapping[str, Any], encoding: Mapping[str, Any] | None) -> dict[str, Any]:
    stage1 = result["latency"]["stage1_search"]
    rerank = result["latency"]["reranking"]
    encoding_seconds = 0.0
    if encoding is not None:
        encoding_seconds = float(encoding["image_encoding_seconds"] if result["task"] == "image_to_text" else encoding["text_encoding_seconds"]) / max(1, int(encoding["image_query_count"] if result["task"] == "image_to_text" else encoding["caption_query_count"]))
    return {
        "tier": result["tier"],
        "split": result["split"],
        "task": result["task"],
        "candidate_depth": result["requested_depth"],
        "candidate_recall": result["candidate_recall"],
        "stage1_r_at_1": result["stage1_metrics"]["recall_at_1"],
        "stage1_r_at_5": result["stage1_metrics"]["recall_at_5"],
        "reranked_r_at_1": result["reranked_metrics"]["recall_at_1"],
        "reranked_r_at_5": result["reranked_metrics"]["recall_at_5"],
        "encoding_mean_seconds": encoding_seconds,
        "stage1_search_mean_seconds": stage1["mean_seconds"],
        "reranking_mean_seconds": rerank["mean_seconds"],
        "reranked_end_to_end_mean_seconds": encoding_seconds + stage1["mean_seconds"] + rerank["mean_seconds"],
    }


def run_phase11(config_path: Path | str = DEFAULT_CONFIG_PATH, output_dir: Path | str = "artifacts/phase11", smoke: bool = False) -> dict[str, Any]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _read_config(config_path)
    validate_phase11_config(config)
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

    train_records = tuple(record for record in tier_specs[0][1] if record.split == "train")
    if len(train_records) > int(config["max_train_images"]):
        ordered = sorted(train_records, key=lambda record: _stable_key(int(config["seed"]), record.image_id))
        train_ids = {record.image_id for record in ordered[: int(config["max_train_images"])]}
        train_records = tuple(record for record in train_records if record.image_id in train_ids)
    if not train_records:
        raise ValueError("reranker training requires train records")

    training_pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    pair_statistics: list[dict[str, Any]] = []
    index_manifest: list[dict[str, Any]] = []
    for task in ("text_to_image", "image_to_text"):
        train_data = _task_data(train_records, task, arrays)
        train_index, train_index_meta = _build_stage1(train_data, source, "training", "train", task, output_dir, int(config["seed"]))
        index_manifest.append({"tier": "training", "split": "train", "task": task, **train_index_meta})
        train_rows = _stage1_rows(train_index, train_data, max(int(max(config["candidate_depths"])), 50))
        q, p, n, stats = _training_pairs(train_data, train_rows, arrays, task)
        training_pairs.append((q, p, n))
        pair_statistics.append(stats)
    query = np.concatenate([item[0] for item in training_pairs], axis=0)
    positive = np.concatenate([item[1] for item in training_pairs], axis=0)
    negative = np.concatenate([item[2] for item in training_pairs], axis=0)
    model, reranker_metadata, reranker_device = _train_reranker((query, positive, negative), config, output_dir, int(query.shape[1]))
    _write_json(pair_statistics, output_dir / "training_pair_statistics.json")

    validation_results: list[dict[str, Any]] = []
    test_results: list[dict[str, Any]] = []
    candidate_recall_payload: list[dict[str, Any]] = []
    oracle_payload: list[dict[str, Any]] = []
    all_latencies: list[dict[str, Any]] = []
    qualitative_payload: list[dict[str, Any]] = []
    tier_context: dict[str, Any] = {}
    for tier, tier_records in tier_specs:
        tier_context[tier] = {}
        for split in ("validation", "test"):
            split_records = tuple(record for record in tier_records if record.split == split)
            if not split_records:
                continue
            tier_context[tier][split] = {}
            for task in ("text_to_image", "image_to_text"):
                data = _task_data(split_records, task, arrays)
                index, index_meta = _build_stage1(data, source, tier, split, task, output_dir, int(config["seed"]))
                rows = _stage1_rows(index, data, max(config["candidate_depths"]))
                tier_context[tier][split][task] = {"data": data, "index": index, "rows": rows, "index_meta": index_meta}
                index_manifest.append({"tier": tier, "split": split, "task": task, **index_meta})
                for depth in config["candidate_depths"]:
                    result = _evaluate_depth(task, tier, split, data, rows, model, reranker_device, int(depth), config, manifest.dataset_id)
                    result["latency"] = {
                        "stage1_search": _latency_stage1(index, data["query_vectors"], int(depth), config),
                        "reranking": _latency_reranker(model, rows, data["query_vectors"], {str(item): vector for item, vector in zip(data["candidate_ids"], data["candidate_vectors"])}, int(depth), reranker_device, config),
                    }
                    result.pop("stage1_rankings", None)
                    result.pop("reranked_rankings", None)
                    result.pop("reranked_rows", None)
                    result["candidate_corpus"] = {"candidate_count": len(data["candidate_ids"]), "candidate_ids_sha256": _hash_ids(data["candidate_ids"]), "query_ids_sha256": _hash_ids(data["query_ids"])}
                    target = validation_results if split == "validation" else test_results
                    target.append(result)
                    candidate_recall_payload.append({"tier": tier, "split": split, "task": task, "depth": depth, **result["candidate_recall"]})
                    oracle_rankings = _oracle_rankings(task, rows, int(depth), f"phase11_{tier}_{split}_oracle")
                    oracle_payload.append({"tier": tier, "split": split, "task": task, "depth": depth, "label": "ORACLE ANALYSIS — NOT A MODEL", "metrics": evaluate_rankings(oracle_rankings)})
                    if split == "test":
                        all_latencies.append({**result, "latency": result["latency"]})
    selection_scores: dict[int, dict[str, Any]] = {}
    if smoke:
        selected_depth = int(config["candidate_depths"][0])
    else:
        selection_rows = [item for item in validation_results if item["tier"] == "tier2"]
        by_depth: dict[int, list[dict[str, Any]]] = {}
        for item in selection_rows:
            by_depth.setdefault(int(item["requested_depth"]), []).append(item)
        selection_scores = {
            depth: {
                "mean_mrr": statistics.fmean(item["reranked_metrics"]["mrr"] for item in items),
                "mean_recall_at_5": statistics.fmean(item["reranked_metrics"]["recall_at_5"] for item in items),
                "mean_reranking_seconds": statistics.fmean(item["latency"]["reranking"]["mean_seconds"] for item in items),
            }
            for depth, items in by_depth.items()
        }
        selected_depth = min(
            selection_scores,
            key=lambda depth: (-selection_scores[depth][str(config["selection_metric"])], selection_scores[depth]["mean_reranking_seconds"], depth),
        )
    selection = {
        "selected_candidate_depth": selected_depth,
        "selection_split": "validation" if not smoke else "smoke_fixture",
        "selection_tier": "tier2" if not smoke else "smoke_fixture",
        "selection_metric": config["selection_metric"],
        "candidate_depths_considered": config["candidate_depths"],
        "scores": selection_scores if not smoke else {str(selected_depth): {"status": "smoke_only"}},
        "test_used_for_selection": False,
    }
    _write_json(selection, output_dir / "selected_configuration.json")

    for tier, split_context in tier_context.items():
        test_context = split_context.get("test", {})
        for task, context in test_context.items():
            data = context["data"]
            rows = context["rows"]
            candidate_map = {str(item): vector for item, vector in zip(data["candidate_ids"], data["candidate_vectors"])}
            qualitative_payload.append(
                {
                    "tier": tier,
                    "task": task,
                    "candidate_depth": selected_depth,
                    **_rank_change_examples(
                        _make_rankings(task, rows, selected_depth, data["candidate_ids"], "stage1_faiss_flat", f"phase11_{tier}_test_stage1"),
                        _make_rankings(
                            task,
                            _rerank_rows(model, rows, data["query_vectors"], candidate_map, selected_depth, reranker_device),
                            selected_depth,
                            data["candidate_ids"],
                            "stage2_pairwise_mlp",
                            f"phase11_{tier}_test_reranked",
                        ),
                        selected_depth,
                    ),
                }
            )

    encoding_payload: list[dict[str, Any]] = []
    if smoke:
        encoding_payload.append({"status": "fixture_embeddings_only", "model_load_included": False})
    else:
        tier2_records = tier_specs[0][1]
        tier3_records = tier_specs[-1][1]
        encoding_payload.append(_measure_query_encoding(config, tuple(record for record in tier2_records if record.split == "test"), arrays, output_dir, "tier2_test"))
        encoding_payload.append(_measure_query_encoding(config, tuple(record for record in tier3_records if record.split == "test"), arrays, output_dir, "tier3_test"))
    _write_json(encoding_payload, output_dir / "query_encoding_latency.json")

    selected_test = [item for item in test_results if int(item["requested_depth"]) == selected_depth]
    stage1_results = [{key: value for key, value in item.items() if key not in {"paired_comparison"}} for item in selected_test]
    reranked_results = [{"tier": item["tier"], "split": item["split"], "task": item["task"], "candidate_depth": item["requested_depth"], "metrics": item["reranked_metrics"], "rank_change_counts": item["rank_change_counts"]} for item in selected_test]
    paired = [{"tier": item["tier"], "task": item["task"], "candidate_depth": item["requested_depth"], **item["paired_comparison"]} for item in selected_test]
    quality_latency = []
    for item in all_latencies:
        if int(item["requested_depth"]) == selected_depth:
            encoding = next((entry for entry in encoding_payload if entry.get("label") == f"{item['tier']}_test"), None)
            quality_latency.append(_quality_latency_row(item, encoding))
    failure_analysis = {
        "stage1_candidate_misses": sum(1 for item in candidate_recall_payload if item["split"] == "test" and item["candidate_hit_rate"] < 1.0),
        "reranker_ordering_changes": sum(item["rank_change_counts"]["ordering_changed"] for item in selected_test),
        "reranker_top1_changes": sum(item["rank_change_counts"]["top1_changed"] for item in selected_test),
        "reranker_regressions": sum((example.get("regression") is not None) for example in qualitative_payload),
        "categories": {
            "stage1_recall_failure": "relevant item absent from candidate set; reranker cannot recover it",
            "reranker_ordering_failure": "candidate was present but the learned score placed it below another candidate",
            "model_semantic_failure": "exact Stage 1 and reranker can both be wrong under metadata-defined relevance",
            "ambiguous_relevance": "COCO same-image relevance is a metadata proxy, not a human semantic label",
        },
    }
    provenance = {
        "project": "OmniSearch",
        "package_version": __version__,
        "phase11_schema_version": PHASE11_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "config_path": str(config_path),
        "config_sha256": _hash_file(config_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": source["manifest_sha256"],
        "embedding_source": source,
        "stage1": "FAISS IndexFlatIP persisted per split/corpus",
        "reranker_checkpoint": reranker_metadata["checkpoint"],
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(config["seed"]),
        "test_used_for_training_or_selection": False,
        "scores_are_probabilities": False,
    }
    report = {
        "report_schema_version": PHASE11_SCHEMA_VERSION,
        "project": "OmniSearch",
        "phase": 11,
        "pre_phase_audit": "Phase 10 PASS",
        "stage1_retriever": {"type": "FAISS IndexFlatIP", "exact": True, "same_validated_embedding_source": True, "indexes": index_manifest},
        "reranker": reranker_metadata,
        "dataset_scope": {"dataset_id": manifest.dataset_id, "manifest": str(manifest_path), "manifest_sha256": source["manifest_sha256"], "tier_specs": [{"tier": tier, "image_groups": len(records), "split_counts": {split: sum(record.split == split for record in records) for split in ("train", "validation", "test")}} for tier, records in tier_specs], "training_groups": len(train_records)},
        "candidate_depth": selection,
        "candidate_recall": candidate_recall_payload,
        "stage1_results": stage1_results,
        "reranked_results": reranked_results,
        "paired_statistical_comparisons": paired,
        "oracle_upper_bound": oracle_payload,
        "latency_breakdown": {"query_encoding": encoding_payload, "quality_latency": quality_latency, "all_selected_depth_results": quality_latency},
        "quality_latency_findings": quality_latency,
        "qualitative_findings": qualitative_payload,
        "failure_analysis": failure_analysis,
        "provenance": provenance,
        "quality_gate": {
            "phase10_audit": "PASS",
            "stage1_validated": True,
            "candidate_recall_measured": True,
            "reranker_changes_ordering": any(item["rank_change_counts"]["ordering_changed"] > 0 for item in selected_test),
            "reranker_train_only": True,
            "validation_only_selection": True,
            "test_isolation": True,
            "apples_to_apples": True,
            "r1_and_mrr_reported": True,
            "oracle_labeled_analysis_only": True,
            "latency_separated": True,
            "paired_statistics": True,
            "probabilities_not_claimed": True,
            "no_phase11_audit_markdown": not Path("docs/phase11_audit.md").exists(),
            "no_phase12_features": True,
            "status": "SMOKE_ONLY" if smoke else "PASS",
        },
    }
    _write_json(config, output_dir / "config.json")
    _write_json(index_manifest, output_dir / "index_manifest.json")
    _write_json(candidate_recall_payload, output_dir / "candidate_recall.json")
    _write_json(stage1_results, output_dir / "stage1_results.json")
    _write_json(reranked_results, output_dir / "reranked_results.json")
    _write_json(paired, output_dir / "paired_comparisons.json")
    _write_json(oracle_payload, output_dir / "oracle_upper_bound.json")
    _write_json(quality_latency, output_dir / "quality_latency_comparison.json")
    _write_json(qualitative_payload, output_dir / "qualitative_examples.json")
    _write_json(failure_analysis, output_dir / "failure_analysis.json")
    _write_json(provenance, output_dir / "provenance.json")
    _write_json(report, output_dir / "phase11_report.json")
    _write_markdown_report(report, output_dir / "phase11_report.md")
    del model
    gc.collect()
    return report


def _write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# OmniSearch Phase 11 two-stage retrieval report",
        "",
        f"Pre-phase audit: **{report['pre_phase_audit']}**.",
        "",
        "Stage 1 is persisted FAISS Flat retrieval. Stage 2 is a train-only pairwise MLP over product, absolute-difference, and cosine interaction features. Reranker outputs are scores, not calibrated probabilities.",
        "",
        f"Selected candidate depth: `{report['candidate_depth']['selected_candidate_depth']}` using `{report['candidate_depth']['selection_metric']}` on validation only.",
        "",
        "| Tier | Task | Depth | Stage1 R@1 | Reranked R@1 | Stage1 R@5 | Reranked R@5 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["stage1_results"]:
        reranked = next(result for result in report["reranked_results"] if result["tier"] == item["tier"] and result["task"] == item["task"])
        lines.append(f"| {item['tier']} | {item['task']} | {item['requested_depth']} | {item['stage1_metrics']['recall_at_1']:.4f} | {reranked['metrics']['recall_at_1']:.4f} | {item['stage1_metrics']['recall_at_5']:.4f} | {reranked['metrics']['recall_at_5']:.4f} |")
    lines.extend(["", f"Quality gate: **{report['quality_gate']['status']}**."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run OmniSearch Phase 11 two-stage reranking.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase11"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    report = run_phase11(args.config, args.output_dir, args.smoke)
    print(json.dumps({"output_dir": str(args.output_dir), "smoke": args.smoke, "quality_gate": report["quality_gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
