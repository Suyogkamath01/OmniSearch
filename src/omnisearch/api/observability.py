"""Small in-process observability primitives for the research service."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import UTC, datetime
from typing import Any

_ABSOLUTE_PATH = re.compile(r"/(?:Users|Volumes|home|private/tmp)/[^\s:'\"]+")
_SENSITIVE_ASSIGNMENT = re.compile(r"(?i)\b(query|prompt|payload|token|secret|authorization)=\S+")


class JsonLogFormatter(logging.Formatter):
    """Emit safe, stable JSON fields without request payloads or filesystem paths."""

    def format(self, record: logging.LogRecord) -> str:
        message = _ABSOLUTE_PATH.sub("<local-path>", record.getMessage())
        message = _SENSITIVE_ASSIGNMENT.sub(r"\1=<redacted>", message)
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": message,
        }
        for name in (
            "request_id",
            "endpoint",
            "method",
            "status",
            "latency_ms",
            "preprocessing_ms",
            "query_encoding_ms",
            "search_ms",
            "total_server_ms",
            "error_category",
            "result_count",
            "device",
            "ready",
        ):
            value = getattr(record, name, None)
            if value is not None:
                payload[name] = value
        if record.exc_info:
            payload["exception_type"] = record.exc_info[0].__name__ if record.exc_info[0] else "unknown"
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging() -> logging.Logger:
    """Configure one stdout JSON handler for OmniSearch-owned logs."""

    logger = logging.getLogger("omnisearch")
    logger.setLevel(os.environ.get("OMNISEARCH_LOG_LEVEL", "INFO").upper())
    logger.propagate = False
    if not any(isinstance(handler, logging.StreamHandler) and isinstance(handler.formatter, JsonLogFormatter) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonLogFormatter())
        logger.addHandler(handler)
    return logger


class RuntimeMetrics:
    """Thread-safe, process-local counters; not a production telemetry store."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._text_requests = 0
        self._image_requests = 0
        self._errors: dict[str, int] = {}
        self._startup_state = "not_started"
        self._startup_error_category: str | None = None
        self._shutdown_state = "running"

    def set_startup(self, state: str, error_category: str | None = None) -> None:
        with self._lock:
            self._startup_state = state
            self._startup_error_category = error_category

    def set_shutdown(self, state: str) -> None:
        with self._lock:
            self._shutdown_state = state

    def record_request(self, endpoint: str, status: int, latency_ms: float, error_category: str | None = None) -> None:
        with self._lock:
            self._total_requests += 1
            if status < 400:
                self._successful_requests += 1
            else:
                self._failed_requests += 1
            if endpoint.endswith("/text-to-image"):
                self._text_requests += 1
            elif endpoint.endswith("/image-to-text"):
                self._image_requests += 1
            if error_category:
                self._errors[error_category] = self._errors.get(error_category, 0) + 1

    def snapshot(self, service: Any) -> dict[str, Any]:
        with self._lock:
            return {
                "service_status": "ok" if getattr(service, "ready", False) else "degraded",
                "ready": bool(getattr(service, "ready", False)),
                "startup_state": self._startup_state,
                "shutdown_state": self._shutdown_state,
                "device": getattr(service, "device", None),
                "model_identity": getattr(getattr(service, "config", None), "model_id", "unknown"),
                "index_identity": "phase10_cached_faiss_flat",
                "uptime_seconds": max(0.0, time.monotonic() - self._started),
                "request_counts": {
                    "total": self._total_requests,
                    "successful": self._successful_requests,
                    "failed": self._failed_requests,
                    "text_search": self._text_requests,
                    "image_search": self._image_requests,
                },
                "error_counts": dict(sorted(self._errors.items())),
                "startup_error_category": self._startup_error_category,
                "persistence": "process-local only",
            }
