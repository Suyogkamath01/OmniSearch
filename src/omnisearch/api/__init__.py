"""FastAPI adapter for the validated OmniSearch retrieval service."""

from __future__ import annotations

from .app import create_app
from .config import ServiceConfig

__all__ = ["ServiceConfig", "create_app"]
