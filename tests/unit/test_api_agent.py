"""Unit tests for the API agent."""

import pytest

from agent_system.agents.api_agent import ApiAgent
from agent_system.core.models import QueryContext, QueryIntent, ResultStatus


@pytest.fixture()
def agent(fake_data_fetcher) -> ApiAgent:
    return ApiAgent(data_fetcher=fake_data_fetcher)


@pytest.mark.asyncio
class TestApiAgent:
    async def test_returns_success_with_data(
        self, agent: ApiAgent, sample_context: QueryContext
    ) -> None:
        result = await agent.execute(sample_context)
        assert result.status == ResultStatus.SUCCESS
        assert len(result.data) == 2

    async def test_passes_entities_as_params(
        self, agent: ApiAgent, fake_data_fetcher, sample_context: QueryContext
    ) -> None:
        await agent.execute(sample_context)
        _, params = fake_data_fetcher.call_log[0]
        assert params["city"] == "Austin"
        assert params["specialty"] == "cardiologist"

    async def test_returns_partial_when_no_data(self, fake_data_fetcher) -> None:
        fake_data_fetcher._data = []
        agent = ApiAgent(data_fetcher=fake_data_fetcher)
        ctx = QueryContext(raw_query="test", intent=QueryIntent.SEARCH_DOCTORS)
        result = await agent.execute(ctx)
        assert result.status == ResultStatus.PARTIAL

    async def test_name_property(self, agent: ApiAgent) -> None:
        assert agent.name == "api_agent"