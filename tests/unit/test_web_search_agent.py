"""Unit tests for the web search agent."""

import pytest

from agent_system.agents.web_search_agent import WebSearchAgent
from agent_system.core.models import QueryContext, QueryIntent, ResultStatus


@pytest.fixture()
def agent(fake_search_provider) -> WebSearchAgent:
    return WebSearchAgent(search_provider=fake_search_provider, max_results=3)


@pytest.mark.asyncio
class TestWebSearchAgent:
    async def test_returns_success_with_results(
        self, agent: WebSearchAgent, sample_context: QueryContext
    ) -> None:
        result = await agent.execute(sample_context)
        assert result.status == ResultStatus.SUCCESS
        assert len(result.data) == 2

    async def test_search_query_includes_entities(
        self, agent: WebSearchAgent, fake_search_provider, sample_context: QueryContext
    ) -> None:
        await agent.execute(sample_context)
        query, _ = fake_search_provider.call_log[0]
        assert "Austin" in query
        assert "cardiologist" in query

    async def test_returns_partial_when_empty(self, fake_search_provider) -> None:
        fake_search_provider._results = []
        agent = WebSearchAgent(search_provider=fake_search_provider)
        ctx = QueryContext(raw_query="test", intent=QueryIntent.UNKNOWN)
        result = await agent.execute(ctx)
        assert result.status == ResultStatus.PARTIAL