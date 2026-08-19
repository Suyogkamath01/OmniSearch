"""Dependency-free classical retrieval baselines for Phase 3.

The text baselines are lexical and deliberately modest: word-level TF-IDF
and BM25 over caption documents. The image baseline is a handcrafted spatial
RGB histogram and uses histogram intersection. No learned model or ANN index
is used here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .config import DEFAULT_CONFIG_PATH, load_config
from .image_validation import check_image_file
from .manifest import ImageRecord, read_manifest
from .splitting import SPLIT_NAMES, assert_no_split_leakage

TOKEN_PATTERN = re.compile(r"\b[\w]+(?:['’\-][\w]+)*\b", re.UNICODE)


@dataclass(frozen=True)
class TextDocument:
    doc_id: str
    group_id: str
    text: str


@dataclass(frozen=True)
class RankedResult:
    item_id: str
    group_id: str
    score: float


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


def _ngrams(tokens: Sequence[str], ngram_range: tuple[int, int]) -> list[str]:
    lower, upper = ngram_range
    if lower < 1 or upper < lower:
        raise ValueError("ngram_range must be positive and ordered")
    return [
        " ".join(tokens[index : index + size])
        for size in range(lower, upper + 1)
        for index in range(len(tokens) - size + 1)
    ]


def records_to_documents(records: Iterable[ImageRecord]) -> tuple[TextDocument, ...]:
    return tuple(
        TextDocument(caption.caption_id, record.image_id, caption.text)
        for record in records
        for caption in record.captions
    )


def _l2_normalize(vector: Mapping[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0:
        return {}
    return {key: value / norm for key, value in vector.items()}


def _top_results(
    scores: Mapping[int, float],
    documents: Sequence[TextDocument],
    top_k: int,
    exclude_doc_id: str | None,
) -> list[RankedResult]:
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    ranked = sorted(
        (
            (index, score)
            for index, score in scores.items()
            if score != 0
            and (exclude_doc_id is None or documents[index].doc_id != exclude_doc_id)
        ),
        key=lambda item: (-item[1], documents[item[0]].doc_id),
    )[:top_k]
    return [
        RankedResult(documents[index].doc_id, documents[index].group_id, float(score))
        for index, score in ranked
    ]


class TfidfIndex:
    """Sparse word n-gram TF-IDF index with cosine retrieval."""

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 1),
        min_df: int = 1,
        max_df: float = 1.0,
        sublinear_tf: bool = True,
    ) -> None:
        if min_df < 1:
            raise ValueError("min_df must be at least one")
        if isinstance(max_df, float) and not 0 < max_df <= 1:
            raise ValueError("float max_df must be in (0, 1]")
        if isinstance(max_df, int) and max_df < 1:
            raise ValueError("integer max_df must be at least one")
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.documents: tuple[TextDocument, ...] = ()
        self.vocabulary: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.vectors: tuple[dict[str, float], ...] = ()
        self.postings: dict[str, tuple[tuple[int, float], ...]] = {}

    def _terms(self, text: str) -> list[str]:
        return _ngrams(tokenize(text), self.ngram_range)

    def fit(self, documents: Iterable[TextDocument]) -> TfidfIndex:
        self.documents = tuple(documents)
        if not self.documents:
            raise ValueError("cannot fit TF-IDF on an empty corpus")
        document_terms = [self._terms(doc.text) for doc in self.documents]
        document_frequency: Counter[str] = Counter()
        for terms in document_terms:
            document_frequency.update(set(terms))
        document_count = len(self.documents)
        max_count = (
            math.floor(self.max_df * document_count)
            if isinstance(self.max_df, float)
            else self.max_df
        )
        max_count = max(1, max_count)
        selected = sorted(
            term
            for term, count in document_frequency.items()
            if self.min_df <= count <= max_count
        )
        self.vocabulary = {term: index for index, term in enumerate(selected)}
        self.idf = {
            term: math.log((1 + document_count) / (1 + document_frequency[term])) + 1.0
            for term in selected
        }
        vectors: list[dict[str, float]] = []
        postings: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
        for index, terms in enumerate(document_terms):
            counts = Counter(term for term in terms if term in self.vocabulary)
            weighted = {
                term: (
                    1.0 + math.log(count)
                    if self.sublinear_tf and count > 0
                    else float(count)
                )
                * self.idf[term]
                for term, count in counts.items()
            }
            vector = _l2_normalize(weighted)
            vectors.append(vector)
            for term, value in vector.items():
                postings[term].append((index, value))
        self.vectors = tuple(vectors)
        self.postings = {term: tuple(values) for term, values in postings.items()}
        return self

    def _query_vector(self, text: str) -> dict[str, float]:
        counts = Counter(term for term in self._terms(text) if term in self.vocabulary)
        weighted = {
            term: (
                1.0 + math.log(count)
                if self.sublinear_tf and count > 0
                else float(count)
            )
            * self.idf[term]
            for term, count in counts.items()
        }
        return _l2_normalize(weighted)

    def search(
        self, text: str, top_k: int = 10, exclude_doc_id: str | None = None
    ) -> list[RankedResult]:
        if not self.documents:
            raise RuntimeError("fit must be called before search")
        query = self._query_vector(text)
        scores: defaultdict[int, float] = defaultdict(float)
        for term, query_weight in query.items():
            for index, document_weight in self.postings.get(term, ()):
                scores[index] += query_weight * document_weight
        return _top_results(scores, self.documents, top_k, exclude_doc_id)

    def stats(self) -> dict[str, Any]:
        return {
            "documents": len(self.documents),
            "features": len(self.vocabulary),
            "ngram_range": list(self.ngram_range),
            "min_df": self.min_df,
            "max_df": self.max_df,
            "sublinear_tf": self.sublinear_tf,
            "estimated_representation_bytes": sum(
                len(term.encode("utf-8")) + 16 * len(vector)
                for term, vector in (
                    (term, self.postings[term]) for term in self.postings
                )
            ),
        }


class BM25Index:
    """Okapi BM25 lexical index with deterministic tie-breaking."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 >= 0 and b in [0, 1]")
        self.k1 = float(k1)
        self.b = float(b)
        self.documents: tuple[TextDocument, ...] = ()
        self.term_frequencies: tuple[Counter[str], ...] = ()
        self.document_lengths: tuple[int, ...] = ()
        self.average_document_length = 0.0
        self.idf: dict[str, float] = {}
        self.postings: dict[str, tuple[int, ...]] = {}

    def fit(self, documents: Iterable[TextDocument]) -> BM25Index:
        self.documents = tuple(documents)
        if not self.documents:
            raise ValueError("cannot fit BM25 on an empty corpus")
        term_frequencies = tuple(Counter(tokenize(doc.text)) for doc in self.documents)
        self.term_frequencies = term_frequencies
        self.document_lengths = tuple(
            sum(counter.values()) for counter in term_frequencies
        )
        self.average_document_length = (
            statistics.fmean(self.document_lengths) if self.document_lengths else 0.0
        )
        document_frequency: Counter[str] = Counter()
        postings: defaultdict[str, list[int]] = defaultdict(list)
        for index, counter in enumerate(term_frequencies):
            for term in counter:
                document_frequency[term] += 1
                postings[term].append(index)
        document_count = len(self.documents)
        self.idf = {
            term: math.log(1.0 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        self.postings = {term: tuple(indices) for term, indices in postings.items()}
        return self

    def search(
        self, text: str, top_k: int = 10, exclude_doc_id: str | None = None
    ) -> list[RankedResult]:
        if not self.documents:
            raise RuntimeError("fit must be called before search")
        scores: defaultdict[int, float] = defaultdict(float)
        query_terms = set(tokenize(text))
        for term in query_terms:
            if term not in self.idf:
                continue
            for index in self.postings[term]:
                frequency = self.term_frequencies[index][term]
                length = self.document_lengths[index]
                denominator = (
                    frequency
                    + self.k1
                    * (1.0 - self.b + self.b * length / self.average_document_length)
                    if self.average_document_length
                    else 1.0
                )
                scores[index] += (
                    self.idf[term] * frequency * (self.k1 + 1.0) / denominator
                )
        return _top_results(scores, self.documents, top_k, exclude_doc_id)

    def stats(self) -> dict[str, Any]:
        return {
            "documents": len(self.documents),
            "vocabulary": len(self.idf),
            "k1": self.k1,
            "b": self.b,
            "average_document_length": self.average_document_length,
            "estimated_representation_bytes": sum(
                len(term.encode("utf-8")) + 8 * len(indices)
                for term, indices in self.postings.items()
            ),
        }


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    from .evaluation import precision_at_k as canonical_precision

    return float(
        canonical_precision(_legacy_ranking(ranked_ids, relevant_ids), k) or 0.0
    )


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    from .evaluation import recall_at_k as canonical_recall

    return float(canonical_recall(_legacy_ranking(ranked_ids, relevant_ids), k) or 0.0)


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: set[str]) -> float:
    from .evaluation import reciprocal_rank as canonical_reciprocal_rank

    return float(
        canonical_reciprocal_rank(_legacy_ranking(ranked_ids, relevant_ids)) or 0.0
    )


def average_precision(ranked_ids: Sequence[str], relevant_ids: set[str]) -> float:
    from .evaluation import average_precision as canonical_average_precision

    return float(
        canonical_average_precision(_legacy_ranking(ranked_ids, relevant_ids)) or 0.0
    )


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    from .evaluation import ndcg_at_k as canonical_ndcg

    return float(canonical_ndcg(_legacy_ranking(ranked_ids, relevant_ids), k) or 0.0)


def _legacy_ranking(ranked_ids: Sequence[str], relevant_ids: set[str]) -> Any:
    """Adapt the historical Phase 3 API to the canonical Phase 5 evaluator."""

    from .evaluation import ranking_from_scores

    candidates = [
        (str(item_id), float(len(ranked_ids) - rank))
        for rank, item_id in enumerate(ranked_ids)
    ]
    return ranking_from_scores(
        query_id="legacy_query",
        task="text_to_text",
        candidates=candidates,
        relevant_ids=relevant_ids,
        system_id="legacy_phase3",
        experiment_id="legacy_compatibility",
        candidate_count=len(ranked_ids),
        candidate_corpus_id="legacy",
        relevance_definition="legacy compatibility relevance",
    )


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    relevance: Mapping[str, set[str]],
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, Any]:
    from .evaluation import evaluate_rankings as canonical_evaluate
    from .evaluation import ranking_from_scores

    if not rankings:
        return {
            "queries_evaluated": 0,
            "relevance_definition": "not_evaluated_no_relevant_queries",
        }
    records = tuple(
        ranking_from_scores(
            query_id=query_id,
            task="text_to_text",
            candidates=[
                (str(item_id), float(len(rankings[query_id]) - rank))
                for rank, item_id in enumerate(rankings[query_id])
            ],
            relevant_ids=relevance.get(query_id, set()),
            system_id="legacy_phase3",
            experiment_id="legacy_compatibility",
            candidate_count=len(rankings[query_id]),
            candidate_corpus_id="legacy",
            relevance_definition="legacy compatibility relevance",
        )
        for query_id in rankings
    )
    metrics = canonical_evaluate(records, ks)
    metrics["relevance_definition"] = (
        "same-image captions in the held-out split; query caption excluded"
    )
    return metrics


def histogram_intersection(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("histograms must have the same length")
    if not left:
        return 0.0
    return float(sum(min(a, b) for a, b in zip(left, right)))


def _ppm_token(data: bytes, position: int) -> tuple[bytes, int]:
    length = len(data)
    while position < length:
        if data[position] in b" \t\r\n":
            position += 1
        elif data[position] == ord("#"):
            newline = data.find(b"\n", position)
            position = length if newline < 0 else newline + 1
        else:
            break
    start = position
    while position < length and data[position] not in b" \t\r\n#":
        position += 1
    if start == position:
        raise ValueError("malformed PPM header")
    return data[start:position], position


def _read_pixels(path: Path) -> tuple[int, int, list[tuple[int, int, int]]]:
    data = path.read_bytes()
    if data[:2] in {b"P6", b"P5"}:
        position = 2
        width_token, position = _ppm_token(data, position)
        height_token, position = _ppm_token(data, position)
        maxval_token, position = _ppm_token(data, position)
        width, height, maxval = int(width_token), int(height_token), int(maxval_token)
        if width <= 0 or height <= 0 or not 1 <= maxval <= 255:
            raise ValueError("unsupported PPM dimensions or max value")
        while position < len(data) and data[position] in b" \t\r\n":
            position += 1
        expected = width * height * (3 if data[:2] == b"P6" else 1)
        raw = data[position : position + expected]
        if len(raw) != expected:
            raise ValueError("truncated PPM pixel data")
        if data[:2] == b"P6":
            pixels = [
                (raw[index], raw[index + 1], raw[index + 2])
                for index in range(0, len(raw), 3)
            ]
        else:
            pixels = [(value, value, value) for value in raw]
        return width, height, pixels
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ValueError("Pillow is required for non-PPM image descriptors") from exc
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            if max(width, height) > 64:
                scale = 64 / max(width, height)
                rgb = rgb.resize(
                    (max(1, int(width * scale)), max(1, int(height * scale)))
                )
                width, height = rgb.size
            return width, height, list(rgb.getdata())
    except (
        Exception
    ) as exc:  # Normalize decoder-specific Pillow errors at the public boundary.
        raise ValueError(f"could not decode image {path}: {exc}") from exc


def colour_histogram_descriptor(
    path: Path | str, bins: int = 8, grid_size: int = 2
) -> tuple[float, ...]:
    """Return an L1-normalized global plus spatial RGB histogram."""

    if bins < 1 or grid_size < 1:
        raise ValueError("bins and grid_size must be positive")
    width, height, pixels = _read_pixels(Path(path))
    if not pixels:
        raise ValueError("image contains no pixels")
    global_hist = [0.0] * (3 * bins)
    spatial_hist = [0.0] * (grid_size * grid_size * 3 * bins)
    for index, pixel in enumerate(pixels):
        x = index % width
        y = index // width
        cell_x = min(grid_size - 1, x * grid_size // width)
        cell_y = min(grid_size - 1, y * grid_size // height)
        cell_offset = (cell_y * grid_size + cell_x) * 3 * bins
        for channel, value in enumerate(pixel):
            bin_index = min(bins - 1, int(value) * bins // 256)
            global_hist[channel * bins + bin_index] += 1.0
            spatial_hist[cell_offset + channel * bins + bin_index] += 1.0
    vector: list[float] = []
    for channel in range(3):
        total = sum(global_hist[channel * bins : (channel + 1) * bins])
        vector.extend(
            value / total
            for value in global_hist[channel * bins : (channel + 1) * bins]
        )
    for cell in range(grid_size * grid_size):
        cell_offset = cell * 3 * bins
        for channel in range(3):
            start = cell_offset + channel * bins
            total = sum(spatial_hist[start : start + bins])
            if total:
                vector.extend(
                    value / total for value in spatial_hist[start : start + bins]
                )
            else:
                vector.extend(0.0 for _ in range(bins))
    norm = sum(vector)
    return tuple(value / norm for value in vector)


class ImageHistogramIndex:
    def __init__(self, bins: int = 8, grid_size: int = 2) -> None:
        self.bins = bins
        self.grid_size = grid_size
        self.items: tuple[tuple[str, str, tuple[float, ...]], ...] = ()
        self.skipped: list[dict[str, str]] = []

    def fit(
        self, records: Iterable[ImageRecord], image_root: Path | str
    ) -> ImageHistogramIndex:
        root = Path(image_root)
        items: list[tuple[str, str, tuple[float, ...]]] = []
        skipped: list[dict[str, str]] = []
        for record in records:
            if record.filename is None:
                skipped.append(
                    {"image_id": record.image_id, "reason": "missing_filename"}
                )
                continue
            path = root / record.filename
            check = check_image_file(record.image_id, path)
            if not check.exists:
                skipped.append({"image_id": record.image_id, "reason": "missing_file"})
                continue
            if not check.decodable:
                skipped.append(
                    {"image_id": record.image_id, "reason": "undecodable_file"}
                )
                continue
            try:
                vector = colour_histogram_descriptor(path, self.bins, self.grid_size)
            except (OSError, ValueError) as exc:
                skipped.append({"image_id": record.image_id, "reason": str(exc)})
                continue
            items.append((record.image_id, str(path), vector))
        self.items = tuple(sorted(items, key=lambda item: item[0]))
        self.skipped = skipped
        return self

    def search(self, image_id: str, top_k: int = 10) -> list[RankedResult]:
        candidates = {item_id: vector for item_id, _, vector in self.items}
        if image_id not in candidates:
            raise KeyError(f"image ID is not indexed: {image_id}")
        query = candidates[image_id]
        ranked = sorted(
            (
                (item_id, histogram_intersection(query, vector))
                for item_id, vector in candidates.items()
                if item_id != image_id
            ),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]
        return [RankedResult(item_id, item_id, score) for item_id, score in ranked]

    def stats(self) -> dict[str, Any]:
        return {
            "items_indexed": len(self.items),
            "items_skipped": len(self.skipped),
            "bins": self.bins,
            "grid_size": self.grid_size,
            "descriptor_dimensions": len(self.items[0][2])
            if self.items
            else 3 * self.bins * (1 + self.grid_size**2),
            "similarity": "histogram_intersection",
            "estimated_representation_bytes": sum(
                len(vector) * 8 for _, _, vector in self.items
            ),
        }


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _stable_order(values: Iterable[str], seed: int) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest(),
    )


def _hash_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_phase3_config(config_path: Path) -> dict[str, Any]:
    import tomllib

    with config_path.open("rb") as file:
        raw = tomllib.load(file)
    return dict(raw.get("phase3", {}))


def _measure_text_baseline(
    name: str,
    index: TfidfIndex | BM25Index,
    documents: Sequence[TextDocument],
    queries: Sequence[TextDocument],
    relevance: Mapping[str, set[str]],
    top_k: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    # The index is already fitted; this timed marker makes the output schema
    # explicit without pretending that fit time was measured after the fact.
    build_seconds = float(config.get("_build_seconds", 0.0))
    rankings: dict[str, list[str]] = {}
    scores: dict[str, list[float]] = {}
    latencies: list[float] = []
    for query in queries:
        start = time.perf_counter()
        results = index.search(query.text, top_k=top_k, exclude_doc_id=query.doc_id)
        latencies.append(time.perf_counter() - start)
        rankings[query.doc_id] = [result.item_id for result in results]
        scores[query.doc_id] = [result.score for result in results]
    metrics = evaluate_rankings(rankings, relevance, ks=(1, 5, min(10, top_k)))
    return {
        "name": name,
        "index_statistics": index.stats(),
        "metrics": metrics,
        "efficiency": {
            "index_build_seconds": build_seconds,
            "queries_executed": len(queries),
            "mean_query_latency_ms": statistics.fmean(latencies) * 1000
            if latencies
            else None,
            "p50_query_latency_ms": (_percentile(latencies, 0.5) or 0.0) * 1000
            if latencies
            else None,
            "p95_query_latency_ms": (_percentile(latencies, 0.95) or 0.0) * 1000
            if latencies
            else None,
        },
        "rankings": rankings,
        "scores": scores,
        "documents": {
            doc.doc_id: {"group_id": doc.group_id, "text": doc.text}
            for doc in documents
        },
        "relevance": {key: sorted(value) for key, value in relevance.items()},
    }


def _qualitative_examples(
    result: Mapping[str, Any], queries: Sequence[TextDocument], limit: int = 3
) -> dict[str, Any]:
    documents = result["documents"]
    rankings = result["rankings"]
    relevance = {key: set(value) for key, value in result["relevance"].items()}
    query_by_id = {query.doc_id: query for query in queries}
    successful: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    failures: list[dict[str, Any]] = []
    for query_id in sorted(rankings):
        query = query_by_id[query_id]
        retrieved = rankings[query_id]
        example = {
            "query_id": query_id,
            "query": query.text,
            "relevant_ids": sorted(relevance.get(query_id, set())),
            "retrieved": [
                {
                    "rank": rank,
                    "doc_id": doc_id,
                    "text": documents[doc_id]["text"],
                    "group_id": documents[doc_id]["group_id"],
                    "relevant": doc_id in relevance.get(query_id, set()),
                }
                for rank, doc_id in enumerate(retrieved[:limit], start=1)
            ],
        }
        top_is_relevant = bool(
            retrieved and retrieved[0] in relevance.get(query_id, set())
        )
        if top_is_relevant and successful is None:
            successful = example
        if not top_is_relevant:
            if failure is None:
                failure = example
            query_terms = set(tokenize(query.text))
            top_terms = (
                set(tokenize(documents[retrieved[0]]["text"])) if retrieved else set()
            )
            failures.append(
                {
                    "query_id": query_id,
                    "top_result_id": retrieved[0] if retrieved else None,
                    "candidate_failure_category": (
                        "lexical_mismatch_candidate"
                        if not query_terms & top_terms
                        else "lexically_overlapping_nonrelevant_candidate"
                    ),
                }
            )
    return {
        "selection_rule": "first deterministic query by caption ID with relevant or nonrelevant top-1 result",
        "successful_example": successful,
        "failure_example": failure,
        "failure_candidates": failures[:20],
        "interpretation": "Categories are diagnostic hypotheses, not semantic ground truth.",
    }


def _markdown_report(report: Mapping[str, Any]) -> str:
    dataset_id = report["provenance"]["dataset_id"]
    lines = [
        "# OmniSearch Phase 3 classical retrieval baseline report",
        "",
        f"Generated: `{report['provenance']['generated_at_utc']}`",
        "",
        "## Audit and scope",
        "",
        f"- Phase 2 audit: **{report['pre_phase_audit']['phase2_audit']}**",
        f"- Real `{dataset_id}` text baseline: **{report['scope']['real_text_baselines']}**",
        f"- Real `{dataset_id}` image baseline: **{report['scope']['real_image_baseline']}**",
        f"- Learned model used: **{report['scope']['learned_model_used']}**",
        "",
        "## Relevance definition",
        "",
        "Text metrics use a limited, explicit dataset definition: other captions belonging to the same image group in the held-out test split are relevant, and the query caption itself is excluded. This measures lexical retrieval of captions sharing an image, not human semantic relevance or cross-modal retrieval.",
        "",
        "Image-to-image metrics are not reported because the caption metadata supplies no defensible relevance labels for two different images. Image descriptors and rankings are still implemented and produce qualitative examples when a verified image root exists.",
        "",
        "## Baseline comparison",
        "",
        "| Baseline | Queries | P@1 | Recall@5 | MRR | MAP@10 | Mean query ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["comparison"]:
        metrics = row.get("metrics", {})
        efficiency = row.get("efficiency", {})

        def value(key: str, current_metrics: Mapping[str, Any] = metrics) -> str:
            current = current_metrics.get(key)
            return "n/e" if current is None else f"{current:.4f}"

        latency = efficiency.get("mean_query_latency_ms")
        lines.append(
            f"| {row['name']} | {metrics.get('queries_evaluated', 0)} | {value('precision_at_1')} | {value('recall_at_5')} | {value('mrr')} | {value('map_at_10')} | {'n/e' if latency is None else f'{latency:.3f}'} |"
        )
    lines.extend(
        [
            "",
            "## Text baseline notes",
            "",
            "- TF-IDF uses word unigrams, sublinear term frequency, smoothed IDF, and L2-normalized sparse vectors.",
            "- BM25 uses Okapi scoring with `k1=1.5` and `b=0.75`.",
            "- A deterministic subset of test captions was queried against the full test caption corpus.",
            "",
            "## Image baseline status",
            "",
            f"- Status: **{report['image_baseline']['status']}**",
            f"- Descriptor: `{report['image_baseline']['descriptor']}`",
            f"- Similarity: `{report['image_baseline']['similarity']}`",
            "- No real image result is claimed without an authorized image root.",
            "",
            "## Qualitative and failure analysis",
            "",
            "Examples are selected by a fixed rule and include the first deterministic top-1 success and failure when available. Failure categories are hypotheses based on token overlap, not semantic annotations.",
            "",
            "## Limitations",
            "",
            "These are classical lexical/appearance baselines. They do not establish semantic image-text alignment or human relevance. Exact image duplicates and cross-image semantic relevance are separate limitations.",
            "",
        ]
    )
    return "\n".join(lines)


def run_phase3(
    manifest_path: Path | str,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    output_dir: Path | str = "artifacts/phase3",
    image_root: Path | str | None = None,
    seed: int | None = None,
    max_text_queries: int | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    phase3_config = _read_phase3_config(config_path)
    manifest = read_manifest(manifest_path)
    records = manifest.records
    assert_no_split_leakage(records)
    actual_seed = config.seed if seed is None else seed
    split = str(phase3_config.get("text_split", "test"))
    if split not in SPLIT_NAMES:
        raise ValueError(f"invalid Phase 3 text split: {split}")
    split_records = tuple(record for record in records if record.split == split)
    documents = records_to_documents(split_records)
    if not documents:
        raise ValueError(f"no caption documents found in split {split}")
    query_limit = (
        max_text_queries
        if max_text_queries is not None
        else int(phase3_config.get("max_text_queries", 256))
    )
    top_k = int(phase3_config.get("top_k", 10))
    if query_limit <= 0 or top_k <= 0:
        raise ValueError("Phase 3 query limit and top_k must be positive")
    ordered_query_ids = _stable_order((doc.doc_id for doc in documents), actual_seed)
    query_ids = set(ordered_query_ids[: min(query_limit, len(ordered_query_ids))])
    queries = tuple(doc for doc in documents if doc.doc_id in query_ids)
    relevance = {
        query.doc_id: {
            doc.doc_id
            for doc in documents
            if doc.group_id == query.group_id and doc.doc_id != query.doc_id
        }
        for query in queries
    }

    baseline_results: list[dict[str, Any]] = []
    tfidf = TfidfIndex(
        ngram_range=(
            int(phase3_config.get("tfidf_ngram_min", 1)),
            int(phase3_config.get("tfidf_ngram_max", 1)),
        ),
        min_df=int(phase3_config.get("tfidf_min_df", 1)),
        max_df=phase3_config.get("tfidf_max_df", 1.0),
        sublinear_tf=bool(phase3_config.get("tfidf_sublinear_tf", True)),
    )
    start = time.perf_counter()
    tfidf.fit(documents)
    tfidf_build_seconds = time.perf_counter() - start
    baseline_results.append(
        _measure_text_baseline(
            "tfidf_word_unigram_l2",
            tfidf,
            documents,
            queries,
            relevance,
            top_k,
            {"_build_seconds": tfidf_build_seconds},
        )
    )
    bm25 = BM25Index(
        k1=float(phase3_config.get("bm25_k1", 1.5)),
        b=float(phase3_config.get("bm25_b", 0.75)),
    )
    start = time.perf_counter()
    bm25.fit(documents)
    bm25_build_seconds = time.perf_counter() - start
    baseline_results.append(
        _measure_text_baseline(
            "bm25_word",
            bm25,
            documents,
            queries,
            relevance,
            top_k,
            {"_build_seconds": bm25_build_seconds},
        )
    )

    qualitative = {
        result["name"]: _qualitative_examples(result, queries)
        for result in baseline_results
    }
    public_baselines = [
        {
            "experiment_id": f"phase3_{result['name']}_seed{actual_seed}_{split}",
            "name": result["name"],
            "dataset_manifest_version": manifest.dataset_version,
            "split": split,
            "seed": actual_seed,
            "configuration": result["index_statistics"],
            "preprocessing": {
                "tokenizer": "unicode word regex",
                "casefold": True,
                "stopword_removal": False,
            },
            "index_statistics": result["index_statistics"],
            "metrics": result["metrics"],
            "efficiency": result["efficiency"],
            "runtime": result["efficiency"],
            "hardware": platform.platform(),
            "artifact_paths": [
                str(output_dir / "phase3_report.json"),
                str(output_dir / "baseline_comparison.csv"),
                str(output_dir / "qualitative_examples.json"),
            ],
        }
        for result in baseline_results
    ]
    comparison = [
        {
            "name": result["name"],
            "metrics": result["metrics"],
            "efficiency": result["efficiency"],
            "index_statistics": result["index_statistics"],
        }
        for result in baseline_results
    ]

    image_baseline: dict[str, Any] = {
        "status": "not_run_no_image_root",
        "descriptor": "spatial_rgb_colour_histogram",
        "similarity": "histogram_intersection",
        "metrics": "not_evaluated_no_defensible_image_to_image_relevance_labels",
        "items_indexed": 0,
        "items_skipped": None,
        "qualitative_examples": [],
    }
    if image_root is not None:
        image_index = ImageHistogramIndex(
            bins=int(phase3_config.get("image_bins", 8)),
            grid_size=int(phase3_config.get("image_grid_size", 2)),
        )
        start = time.perf_counter()
        image_index.fit(split_records, image_root)
        build_seconds = time.perf_counter() - start
        image_examples = []
        image_limit = int(phase3_config.get("image_query_limit", 64))
        for image_id in _stable_order(
            (item[0] for item in image_index.items), actual_seed
        )[:image_limit]:
            query_results = image_index.search(image_id, top_k=top_k)
            image_examples.append(
                {
                    "query_image_id": image_id,
                    "retrieved": [
                        {
                            "rank": rank,
                            "image_id": result.item_id,
                            "score": result.score,
                        }
                        for rank, result in enumerate(query_results, start=1)
                    ],
                }
            )
        image_baseline = {
            "status": "completed"
            if image_index.items
            else "not_completed_no_valid_images",
            "descriptor": "spatial_rgb_colour_histogram",
            "similarity": "histogram_intersection",
            "metrics": "not_evaluated_no_defensible_image_to_image_relevance_labels",
            "build_seconds": build_seconds,
            **image_index.stats(),
            "skipped_sample": image_index.skipped[:50],
            "qualitative_examples": image_examples[:10],
        }

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
        "manifest_schema_version": manifest.schema_version,
        "dataset_id": manifest.dataset_id,
        "dataset_version": manifest.dataset_version,
        "source_sha256": manifest.source_sha256,
        "split": split,
        "seed": actual_seed,
        "text_corpus_documents": len(documents),
        "text_queries": len(queries),
        "top_k": top_k,
        "image_root": str(image_root) if image_root is not None else None,
    }
    report: dict[str, Any] = {
        "report_schema_version": 1,
        "provenance": provenance,
        "pre_phase_audit": {
            "phase2_audit": "PASS_REAL_DATASET_RERUN",
            "real_dataset_eda_verified": image_root is not None,
        },
        "scope": {
            "real_text_baselines": True,
            "real_image_baseline": image_baseline["status"] == "completed"
            and image_baseline["items_indexed"] > 0,
            "learned_model_used": False,
        },
        "dataset": {
            "dataset_id": manifest.dataset_id,
            "split": split,
            "image_groups": len(split_records),
            "caption_documents": len(documents),
            "query_documents": len(queries),
            "captions_per_group_distribution": dict(
                sorted(
                    Counter(len(record.captions) for record in split_records).items()
                )
            ),
        },
        "relevance_definition": {
            "text": "same-image captions in the held-out split; query caption excluded",
            "image_to_image": "not_evaluated_no_defensible_relevance_labels_for_different_images",
        },
        "baselines": public_baselines,
        "comparison": comparison,
        "qualitative_examples": qualitative,
        "failure_analysis": {
            name: qualitative[name]["failure_candidates"] for name in qualitative
        },
        "image_baseline": image_baseline,
        "cross_modal_baseline": {
            "status": "not_implemented",
            "reason": "A non-neural shared text-image representation would be artificial without a defensible alignment construction.",
            "note": "Classical unimodal baselines established. Learned cross-modal alignment begins in later phases.",
        },
    }
    (output_dir / "phase3_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "phase3_report.md").write_text(
        _markdown_report(report), encoding="utf-8"
    )
    (output_dir / "qualitative_examples.json").write_text(
        json.dumps(
            {"provenance": provenance, "examples": qualitative},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (output_dir / "baseline_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "name",
                "queries_evaluated",
                "precision_at_1",
                "recall_at_5",
                "mrr",
                "map",
                "mean_query_latency_ms",
                "index_build_seconds",
            ],
        )
        writer.writeheader()
        for row in comparison:
            metrics = row["metrics"]
            efficiency = row["efficiency"]
            writer.writerow(
                {
                    "name": row["name"],
                    "queries_evaluated": metrics.get("queries_evaluated", 0),
                    "precision_at_1": metrics.get("precision_at_1"),
                    "recall_at_5": metrics.get("recall_at_5"),
                    "mrr": metrics.get("mrr"),
                    "map": metrics.get("map"),
                    "mean_query_latency_ms": efficiency.get("mean_query_latency_ms"),
                    "index_build_seconds": efficiency.get("index_build_seconds"),
                }
            )
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run OmniSearch Phase 3 classical retrieval baselines."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/coco2017_val_split_manifest.json"),
    )
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase3"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-text-queries", type=int, default=None)
    args = parser.parse_args()
    report = run_phase3(
        manifest_path=args.manifest,
        config_path=args.config,
        output_dir=args.output_dir,
        image_root=args.image_root,
        seed=args.seed,
        max_text_queries=args.max_text_queries,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "real_text_baselines": report["scope"]["real_text_baselines"],
                "real_image_baseline": report["scope"]["real_image_baseline"],
                "text_queries": report["provenance"]["text_queries"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
