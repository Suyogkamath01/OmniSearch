"""Typed service errors mapped to safe HTTP responses."""

from __future__ import annotations


class ServiceError(Exception):
    """Base class for expected API failures."""


class StartupValidationError(ServiceError):
    """Required model/index artifacts are missing or incompatible."""


class ResourceUnavailableError(ServiceError):
    """A request arrived before the service was ready."""


class IndexUnavailableError(ServiceError):
    """A retrieval index cannot serve the request."""


class MalformedImageError(ServiceError):
    """An uploaded payload is not a supported decodable image."""


class RetrievalExecutionError(ServiceError):
    """A model or index operation failed after startup validation."""
