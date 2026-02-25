"""Domain models used across every layer of the system.

All models are **immutable** (`frozen=True`) to prevent accidental mutation
once they leave the layer that created them.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum, auto, Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

# class QueryIntent(StrEnum):
#     """High-level intent extracted from the user's natural-language query."""
#
#     SEARCH_DOCTORS = auto()
#     DOCTOR_DETAIL = auto()
#     GENERAL_HEALTH = auto()
#     UNKNOWN = auto()


# class AgentType(StrEnum):
#     """Identifies which agent produced a result."""
#
#     API = auto()
#     WEB_SEARCH = auto()
#     SUMMARIZER = auto()


# class ResultStatus(StrEnum):
#     """Outcome status attached to every agent result."""
#
#     SUCCESS = auto()
#     PARTIAL = auto()
#     ERROR = auto()


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

# class QueryContext(BaseModel):
#     """Parsed representation of a user query that flows through the pipeline."""
#
#     model_config = {"frozen": True}
#
#     query_id: UUID = Field(default_factory=uuid4)
#     raw_query: str
#     intent: QueryIntent = QueryIntent.UNKNOWN
#     entities: dict[str, Any] = Field(default_factory=dict)
#     metadata: dict[str, Any] = Field(default_factory=dict)
#     created_at: datetime = Field(
#         default_factory=lambda: datetime.now(timezone.utc),
#     )
#

# ---------------------------------------------------------------------------
# Data records returned by individual agents
# ---------------------------------------------------------------------------

class ApiRecord(BaseModel):
    """A single structured record returned by the backend API."""

    model_config = {"frozen": True}

    id: str
    name: str
    specialty: str | None = None
    city: str | None = None
    rating: float | None = None
    address: str | None = None
    phone: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """A single result from a web search."""

    model_config = {"frozen": True}

    title: str
    url: str
    snippet: str
    source: str | None = None


# ---------------------------------------------------------------------------
# Agent results
# ---------------------------------------------------------------------------

class AgentResult(BaseModel):
    """Wrapper returned by every agent, regardless of type."""

    model_config = {"frozen": True}

    agent_type: AgentType
    status: ResultStatus
    data: list[dict[str, Any]] = Field(default_factory=list)
    error_message: str | None = None
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class FusedResult(BaseModel):
    """Final output produced by the summarizer / fusion layer."""

    model_config = {"frozen": True}

    query_id: UUID
    summary: str
    api_results: AgentResult
    web_results: AgentResult
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

class QueryIntent(str, Enum):
    SEARCH_DOCTORS = "search_doctors"
    DOCTOR_DETAIL = "doctor_detail"
    LIST_LOCATIONS = "list_locations"
    LIST_SPECIALIZATIONS = "list_specializations"
    GENERAL_HEALTH = "general_health"
    UNKNOWN = "unknown"


class AgentType(str, Enum):
    API = "api"
    WEB_SEARCH = "web_search"
    SUMMARIZER = "summarizer"


class ResultStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    ERROR = "error"
    NO_RESULT = "no_result"


# ── Core data objects ────────────────────────────────────────────────────

class QueryContext(BaseModel):
    """Rich representation of a parsed user query."""

    query_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    raw_query: str
    intent: QueryIntent = QueryIntent.UNKNOWN
    entities: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


# class AgentResult(BaseModel):
#     """Uniform result envelope returned by every agent."""
#
#     agent_type: AgentType
#     status: ResultStatus = ResultStatus.NO_RESULT
#     data: list[dict[str, Any]] = Field(default_factory=list)
#     error_message: str | None = None
#
#
# class FusedResult(BaseModel):
#     """Final output of the pipeline after merging all agent results."""
#
#     query_id: uuid.UUID
#     summary: str = ""
#     api_results: AgentResult | None = None
#     web_results: AgentResult | None = None
#     confidence: float = 0.0


class ToolCall(BaseModel):
    """Represents a single parsed tool/function call."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)