"""Shared pytest fixtures used by both unit and integration tests."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
import pytest_asyncio

from agent_system.config.settings import Settings
from agent_system.core.interfaces import DataFetcher, SearchProvider
from agent_system.core.models import QueryContext, QueryIntent


# ---------------------------------------------------------------------------
# Stub / Fake implementations
# ---------------------------------------------------------------------------

class FakeDataFetcher(DataFetcher):
    """In-memory data fetcher for testing."""

    def __init__(self, data: list[dict[str, Any]] | None = None) -> None:
        self._data = data or []
        self.call_log: list[tuple[str, dict | None]] = []

    async def fetch(
        self,
        endpoint: str,
        params: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        self.call_log.append((endpoint, params))
        return self._data


class FakeSearchProvider(SearchProvider):
    """In-memory search provider for testing."""

    def __init__(self, results: list[dict[str, Any]] | None = None) -> None:
        self._results = results or []
        self.call_log: list[tuple[str, int]] = []

    async def search(self, query: str, num_results: int = 5) -> list[dict[str, Any]]:
        self.call_log.append((query, num_results))
        return self._results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_api_data() -> list[dict[str, Any]]:
    return [
        {
            "id": "doc-1",
            "name": "Dr. Alice Smith",
            "specialty": "cardiologist",
            "city": "Austin",
            "rating": 4.8,
            "address": "123 Heart Lane, Austin, TX",
            "phone": "+1-555-0101",
        },
        {
            "id": "doc-2",
            "name": "Dr. Bob Jones",
            "specialty": "cardiologist",
            "city": "Austin",
            "rating": 4.5,
            "address": "456 Pulse Ave, Austin, TX",
            "phone": "+1-555-0102",
        },
    ]


@pytest.fixture()
def sample_web_results() -> list[dict[str, Any]]:
    return [
        {
            "title": "Top Cardiologists in Austin",
            "url": "https://example.com/top-cardiologists",
            "snippet": "Austin is home to many top-rated cardiologists…",
            "source": "example.com",
        },
        {
            "title": "Heart Health Tips",
            "url": "https://health.example.com/heart",
            "snippet": "Keep your heart healthy with these tips…",
            "source": "health.example.com",
        },
    ]


@pytest.fixture()
def fake_data_fetcher(sample_api_data: list[dict]) -> FakeDataFetcher:
    return FakeDataFetcher(data=sample_api_data)


@pytest.fixture()
def fake_search_provider(sample_web_results: list[dict]) -> FakeSearchProvider:
    return FakeSearchProvider(results=sample_web_results)


@pytest.fixture()
def sample_context() -> QueryContext:
    return QueryContext(
        raw_query="Find cardiologists in Austin",
        intent=QueryIntent.SEARCH_DOCTORS,
        entities={"city": "Austin", "specialty": "cardiologist"},
    )


@pytest.fixture()
def test_settings() -> Settings:
    """Minimal settings for unit tests (no real keys needed)."""
    return Settings(
        openai_api_key="sk-test-key-not-real",
        search_api_key="test-search-key",
        backend_api_base_url="https://test-api.example.com",
        log_level="DEBUG",
        environment="testing",
    )