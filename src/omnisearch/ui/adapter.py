"""Framework-neutral helpers for the Phase 22 interactive demo.

The adapter deliberately delegates all model, preprocessing, and retrieval work
to :class:`omnisearch.api.retrieval.RetrievalService`.  It only validates UI
controls, measures the service call wall time, and formats safe display data.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from typing import Any

from ..api.errors import (
    MalformedImageError,
    ResourceUnavailableError,
    RetrievalExecutionError,
    ServiceError,
    StartupValidationError,
)

UI_MIN_TOP_K = 1
UI_MAX_TOP_K = 20
UI_DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def validate_text_query(query: str, *, max_chars: int = 1_000) -> str:
    """Validate and normalize the text field before calling the service."""

    normalized = str(query).strip()
    if not normalized:
        raise ValueError("Enter a non-empty text query.")
    if len(normalized) > max_chars:
        raise ValueError(f"Text query must be at most {max_chars} characters.")
    return normalized


def validate_top_k(top_k: int) -> int:
    """Keep the demo control smaller than the service's broader API bound."""

    if isinstance(top_k, bool) or not UI_MIN_TOP_K <= int(top_k) <= UI_MAX_TOP_K:
        raise ValueError(f"Top-k must be between {UI_MIN_TOP_K} and {UI_MAX_TOP_K}.")
    return int(top_k)


def make_request_id(prefix: str = "ui") -> str:
    """Create a short non-content request identifier for local diagnostics."""

    return f"{prefix}-{uuid.uuid4().hex[:16]}"


def safe_error_message(error: BaseException) -> str:
    """Return a concise user-facing message without exposing a traceback."""

    if isinstance(error, MalformedImageError):
        return "The uploaded file is not a supported, readable JPEG, PNG, or WEBP image."
    if isinstance(error, ResourceUnavailableError | StartupValidationError):
        return "The retrieval service is not ready. Check that the validated model and indexes are available."
    if isinstance(error, RetrievalExecutionError):
        return "The retrieval request could not be completed. Try again with another input."
    if isinstance(error, ValueError):
        return str(error)
    if isinstance(error, ServiceError):
        return "The retrieval service could not complete this request."
    return "The interactive demo encountered an unexpected service error."


def _timed_service_call(call: Any) -> dict[str, Any]:
    started = time.perf_counter()
    response = call()
    wall_ms = (time.perf_counter() - started) * 1000.0
    output = dict(response)
    latency = dict(output.get("latency_ms", {}))
    server_ms = float(latency.get("total_server_ms", wall_ms))
    latency["ui_service_call_wall_ms"] = max(0.0, float(wall_ms))
    latency["ui_adapter_overhead_ms"] = max(0.0, float(wall_ms) - server_ms)
    output["latency_ms"] = latency
    return output


def run_text_search(service: Any, query: str, top_k: int, *, request_id: str | None = None) -> dict[str, Any]:
    """Validate the UI fields and delegate text-to-image retrieval."""

    normalized = validate_text_query(query, max_chars=int(service.config.max_query_chars))
    bounded_top_k = validate_top_k(top_k)
    return _timed_service_call(
        lambda: service.search_text_to_image(normalized, bounded_top_k, request_id or make_request_id())
    )


def run_image_search(service: Any, payload: bytes, top_k: int, *, request_id: str | None = None) -> dict[str, Any]:
    """Validate the UI fields and delegate image-to-text retrieval in memory."""

    max_upload_bytes = int(getattr(service.config, "max_upload_bytes", UI_DEFAULT_MAX_UPLOAD_BYTES))
    if not payload:
        raise MalformedImageError("image payload is empty")
    if len(payload) > max_upload_bytes:
        raise MalformedImageError(f"image upload must be at most {max_upload_bytes // (1024 * 1024)} MiB")
    bounded_top_k = validate_top_k(top_k)
    return _timed_service_call(
        lambda: service.search_image_to_text(payload, bounded_top_k, request_id or make_request_id())
    )


def format_result(result: Mapping[str, Any], mode: str) -> dict[str, Any]:
    """Select stable, display-safe fields from a service result."""

    metadata = result.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    formatted = {
        "id": str(result.get("id", "")),
        "rank": int(result.get("rank", 0)),
        "score": float(result.get("score", 0.0)),
        "metadata": dict(metadata),
    }
    if mode == "text-to-image":
        formatted["image_id"] = str(metadata.get("image_id", result.get("id", "")))
        formatted["filename"] = str(metadata.get("filename", ""))
    elif mode == "image-to-text":
        formatted["caption_id"] = str(metadata.get("caption_id", result.get("id", "")))
        formatted["image_id"] = str(metadata.get("image_id", ""))
        formatted["text"] = str(metadata.get("text", ""))
    else:
        raise ValueError(f"unsupported UI result mode: {mode}")
    return formatted


def format_results(response: Mapping[str, Any], mode: str) -> list[dict[str, Any]]:
    """Format all returned result rows without changing ranking or scores."""

    return [format_result(row, mode) for row in response.get("results", [])]
