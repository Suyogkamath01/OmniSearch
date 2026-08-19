"""FastAPI application factory for the Phase 21 retrieval service."""

import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import ServiceConfig
from .errors import (
    IndexUnavailableError,
    MalformedImageError,
    ResourceUnavailableError,
    RetrievalExecutionError,
    ServiceError,
    StartupValidationError,
)
from .observability import RuntimeMetrics, configure_logging
from .retrieval import RetrievalService
from .schemas import (
    HealthResponse,
    InfoResponse,
    ReadinessResponse,
    RetrievalResponse,
    TextSearchRequest,
)

_ALLOWED_UPLOAD_MEDIA_TYPES = {
    "",
    "application/octet-stream",
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}


def _set_security_headers(response: Any) -> Any:
    """Apply low-risk browser/API response headers without claiming auth."""

    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault("Cache-Control", "no-store")
    return response


def _validate_upload_media_type(image: UploadFile) -> None:
    """Reject obviously mismatched MIME declarations before decoding bytes.

    Generic ``application/octet-stream`` is accepted because local clients may
    omit image MIME metadata. Pillow remains the authoritative byte and format
    validator.
    """

    media_type = (image.content_type or "").split(";", 1)[0].strip().casefold()
    if media_type not in _ALLOWED_UPLOAD_MEDIA_TYPES:
        raise MalformedImageError("unsupported image content type")


def _request_id(request: Request) -> str:
    existing = getattr(request.state, "request_id", None)
    if existing:
        return str(existing)
    supplied = request.headers.get("X-Request-ID", "")
    identifier = supplied if supplied and len(supplied) <= 64 and supplied.isprintable() else str(uuid.uuid4())
    request.state.request_id = identifier
    return identifier


def _error_payload(code: str, message: str, request_id: str | None = None, error_category: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"error": code, "message": message}
    if request_id:
        payload["request_id"] = request_id
    if error_category:
        payload["error_category"] = error_category
    return payload


def create_app(
    config: ServiceConfig | None = None,
    service: Any | None = None,
    *,
    load_on_startup: bool = True,
) -> FastAPI:
    """Create the application without importing or loading model resources."""

    service_config = config or ServiceConfig.from_env()
    retrieval_service = service or RetrievalService(service_config)
    logger = configure_logging()
    metrics = RuntimeMetrics()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.startup_error = None
        metrics.set_shutdown("running")
        if load_on_startup:
            metrics.set_startup("loading")
            try:
                retrieval_service.load()
            except ServiceError as exc:
                application.state.startup_error = str(exc)
                metrics.set_startup("failed", "STARTUP_ERROR")
                logger.error("startup_failed", extra={"error_category": "STARTUP_ERROR"})
                raise RuntimeError("OmniSearch retrieval service startup validation failed") from exc
            metrics.set_startup("ready")
            logger.info(
                "startup_ready",
                extra={
                    "device": getattr(retrieval_service, "device", None),
                    "ready": bool(getattr(retrieval_service, "ready", False)),
                },
            )
        else:
            metrics.set_startup("bypassed")
        application.state.retrieval_service = retrieval_service
        try:
            yield
        finally:
            metrics.set_shutdown("closing")
            close = getattr(retrieval_service, "close", None)
            if callable(close):
                close()
            metrics.set_shutdown("stopped")
            logger.info("shutdown_complete")

    app = FastAPI(
        title="OmniSearch Retrieval API",
        description="Research-only HTTP interface for validated OmniSearch image-text retrieval.",
        version=service_config.api_version,
        lifespan=lifespan,
    )
    app.state.retrieval_service = retrieval_service
    app.state.startup_error = None
    app.state.metrics = metrics

    @app.middleware("http")
    async def observability_middleware(request: Request, call_next: Any) -> Any:
        request_identifier = _request_id(request)
        started = time.perf_counter()
        response: Any
        try:
            response = await call_next(request)
        except Exception:
            latency_ms = (time.perf_counter() - started) * 1000.0
            request.state.error_category = "INTERNAL_ERROR"
            metrics.record_request(request.url.path, 500, latency_ms, "INTERNAL_ERROR")
            logger.exception(
                "http_request_failed",
                extra={
                    "request_id": request_identifier,
                    "endpoint": request.url.path,
                    "method": request.method,
                    "status": 500,
                    "latency_ms": round(latency_ms, 3),
                    "error_category": "INTERNAL_ERROR",
                },
            )
            response = JSONResponse(
                status_code=500,
                content=_error_payload("internal_error", "the request could not be completed", request_identifier, "INTERNAL_ERROR"),
            )
            response.headers["X-Request-ID"] = request_identifier
            return _set_security_headers(response)
        latency_ms = (time.perf_counter() - started) * 1000.0
        error_category = getattr(request.state, "error_category", None)
        metrics.record_request(request.url.path, response.status_code, latency_ms, error_category)
        log_extra: dict[str, Any] = {
            "request_id": request_identifier,
            "endpoint": request.url.path,
            "method": request.method,
            "status": response.status_code,
            "latency_ms": round(latency_ms, 3),
            "error_category": error_category,
        }
        breakdown = getattr(request.state, "latency_breakdown", None)
        if isinstance(breakdown, dict):
            log_extra.update(
                {
                    "preprocessing_ms": breakdown.get("preprocessing_ms"),
                    "query_encoding_ms": breakdown.get("query_encoding_ms"),
                    "search_ms": breakdown.get("search_ms"),
                    "total_server_ms": breakdown.get("total_server_ms"),
                }
            )
        logger.info("http_request", extra=log_extra)
        response.headers["X-Request-ID"] = request_identifier
        return _set_security_headers(response)

    async def service_dependency(request: Request) -> RetrievalService:
        current = getattr(request.app.state, "retrieval_service", retrieval_service)
        if not getattr(current, "ready", False):
            raise ResourceUnavailableError("retrieval resources are not ready")
        return current

    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
        request_identifier = _request_id(request)
        if isinstance(exc, MalformedImageError):
            status_code, code, message, category = 400, "malformed_image", str(exc), "IMAGE_DECODE_ERROR"
        elif isinstance(exc, IndexUnavailableError):
            status_code, code, message, category = 503, "index_unavailable", "retrieval index is unavailable", "INDEX_ERROR"
        elif isinstance(exc, ResourceUnavailableError):
            status_code, code, message, category = 503, "service_not_ready", "retrieval resources are not ready", "RESOURCE_NOT_READY"
        elif isinstance(exc, StartupValidationError):
            status_code, code, message, category = 503, "startup_validation_failed", "required retrieval resources are incompatible or unavailable", "STARTUP_ERROR"
        elif isinstance(exc, RetrievalExecutionError):
            status_code, code, message, category = 500, "retrieval_failed", "retrieval could not be completed", "MODEL_ERROR"
        else:
            status_code, code, message, category = 500, "service_error", "the request could not be completed", "INTERNAL_ERROR"
        request.state.error_category = category
        logger.error(
            "api_error",
            extra={"endpoint": request.url.path, "request_id": request_identifier, "error_category": category},
        )
        return JSONResponse(status_code=status_code, content=_error_payload(code, message, request_identifier, category))

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        del exc
        request_identifier = _request_id(request)
        request.state.error_category = "VALIDATION_ERROR"
        return JSONResponse(
            status_code=422,
            content=_error_payload("request_validation_failed", "request schema validation failed", request_identifier, "VALIDATION_ERROR"),
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        del exc
        request_identifier = _request_id(request)
        request.state.error_category = "VALIDATION_ERROR"
        return JSONResponse(
            status_code=422,
            content=_error_payload("invalid_request_value", "one or more request values are invalid", request_identifier, "VALIDATION_ERROR"),
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        request_identifier = _request_id(request)
        request.state.error_category = "INTERNAL_ERROR"
        logger.exception(
            "unexpected_api_error",
            exc_info=exc,
            extra={"endpoint": request.url.path, "request_id": request_identifier, "error_category": "INTERNAL_ERROR"},
        )
        return JSONResponse(
            status_code=500,
            content=_error_payload("internal_error", "the request could not be completed", request_identifier, "INTERNAL_ERROR"),
        )

    @app.get("/health", response_model=HealthResponse, tags=["service"], summary="Lightweight service health")
    async def health() -> dict[str, Any]:
        return retrieval_service.health()

    @app.get("/ready", response_model=ReadinessResponse, tags=["service"], summary="Retrieval readiness")
    async def ready() -> dict[str, Any]:
        is_ready, reason = retrieval_service.readiness()
        if not is_ready:
            raise ResourceUnavailableError(reason or "retrieval resources are not ready")
        return {"ready": True, "reason": None, "api_version": service_config.api_version}

    @app.get("/metrics", tags=["service"], summary="Process-local runtime counters")
    async def metrics_endpoint() -> dict[str, Any]:
        return metrics.snapshot(retrieval_service)

    @app.get("/info", response_model=InfoResponse, tags=["service"], summary="Safe service metadata")
    async def info() -> dict[str, Any]:
        return retrieval_service.info()

    @app.post(
        "/search/text-to-image",
        response_model=RetrievalResponse,
        tags=["retrieval"],
        summary="Retrieve images for a text query",
    )
    async def text_to_image(
        payload: TextSearchRequest,
        request: Request,
        current: Annotated[RetrievalService, Depends(service_dependency)],
    ) -> dict[str, Any]:
        if len(payload.query) > service_config.max_query_chars:
            raise ValueError(f"query exceeds the {service_config.max_query_chars}-character limit")
        request_identifier = _request_id(request)
        started = time.perf_counter()
        result = current.search_text_to_image(payload.query, payload.top_k, request_identifier)
        result.setdefault("latency_ms", {})
        result["latency_ms"]["api_route_overhead_ms"] = max(0.0, (time.perf_counter() - started) * 1000.0 - result["latency_ms"]["total_server_ms"])
        request.state.latency_breakdown = result["latency_ms"]
        return result

    @app.post(
        "/search/image-to-text",
        response_model=RetrievalResponse,
        tags=["retrieval"],
        summary="Retrieve captions for an uploaded image",
    )
    async def image_to_text(
        request: Request,
        image: Annotated[UploadFile, File(description="JPEG, PNG, or WEBP image; not persisted")],
        current: Annotated[RetrievalService, Depends(service_dependency)],
        top_k: Annotated[int, Form(ge=1, le=50)] = 5,
    ) -> dict[str, Any]:
        request_identifier = _request_id(request)
        _validate_upload_media_type(image)
        payload = await image.read(service_config.max_upload_bytes + 1)
        started = time.perf_counter()
        result = current.search_image_to_text(payload, top_k, request_identifier)
        result["latency_ms"]["api_route_overhead_ms"] = max(0.0, (time.perf_counter() - started) * 1000.0 - result["latency_ms"]["total_server_ms"])
        request.state.latency_breakdown = result["latency_ms"]
        return result

    return app
