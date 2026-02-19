"""Unit tests for domain models."""

import pytest
from pydantic import ValidationError

from agent_system.core.models import (
    AgentResult,
    AgentType,
    ApiRecord,
    FusedResult,
    QueryContext,
    QueryIntent,
    ResultStatus,
    SearchResult,
)


class TestQueryContext:
    """Tests for QueryContext model."""

    def test_creates_with_defaults(self) -> None:
        ctx = QueryContext(raw_query="hello")
        assert ctx.raw_query == "hello"
        assert ctx.intent == QueryIntent.UNKNOWN
        assert ctx.entities == {}
        assert ctx.query_id is not None

    def test_frozen_model_rejects_mutation(self) -> None:
        ctx = QueryContext(raw_query="hello")
        with pytest.raises(ValidationError):
            ctx.raw_query = "changed"  # type: ignore[misc]


class TestApiRecord:
    def test_minimal_record(self) -> None:
        record = ApiRecord(id="1", name="Dr. Test")
        assert record.specialty is None
        assert record.extra == {}


class TestSearchResult:
    def test_creation(self) -> None:
        sr = SearchResult(title="Title", url="https://x.com", snippet="Snip")
        assert sr.source is None


class TestAgentResult:
    def test_success_result(self) -> None:
        result = AgentResult(
            agent_type=AgentType.API,
            status=ResultStatus.SUCCESS,
            data=[{"id": "1"}],
        )
        assert len(result.data) == 1

    def test_error_result(self) -> None:
        result = AgentResult(
            agent_type=AgentType.WEB_SEARCH,
            status=ResultStatus.ERROR,
            error_message="timeout",
        )
        assert result.data == []


class TestFusedResult:
    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            FusedResult(
                query_id="bad",  # type: ignore[arg-type]
                summary="test",
                api_results=AgentResult(agent_type=AgentType.API, status=ResultStatus.SUCCESS),
                web_results=AgentResult(agent_type=AgentType.WEB_SEARCH, status=ResultStatus.SUCCESS),
                confidence=1.5,  # out of range
            )