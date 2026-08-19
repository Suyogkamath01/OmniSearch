"""Stable Pydantic request and response schemas for Phase 21."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TextSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=1_000, description="Text query")
    top_k: int = Field(default=5, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty or whitespace-only")
        return value


class ResultItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    rank: int = Field(ge=1)
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class LatencyBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preprocessing_ms: float = Field(ge=0)
    query_encoding_ms: float = Field(ge=0)
    search_ms: float = Field(ge=0)
    total_server_ms: float = Field(ge=0)
    api_route_overhead_ms: float = Field(default=0.0, ge=0)


class RetrievalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_type: str
    query: str | None = None
    results: list[ResultItem]
    model_system: str
    retrieval_backend: str
    latency_ms: LatencyBreakdown
    request_id: str


class HealthResponse(BaseModel):
    status: str
    project: str
    model_loaded: bool
    indexes_loaded: bool
    device: str | None
    api_version: str


class ReadinessResponse(BaseModel):
    ready: bool
    reason: str | None = None
    api_version: str


class InfoResponse(BaseModel):
    project: str
    model_family: str
    model_id: str
    embedding_dimension: int | None
    retrieval_backend: str
    supported_query_modes: list[str]
    default_top_k: int
    max_top_k: int
    api_version: str
    protocol_version: str
    device: str | None
    reranker_enabled: bool
    content_safety_filtering: str
    research_system: bool
