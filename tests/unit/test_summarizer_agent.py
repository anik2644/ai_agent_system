"""Unit tests for the summarizer agent (LLM call is mocked)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_system.agents.summarizer_agent import SummarizerAgent
from agent_system.core.models import AgentType, QueryContext, QueryIntent, ResultStatus


@pytest.fixture()
def mock_settings(test_settings):
    return test_settings


@pytest.fixture()
def agent(mock_settings, monkeypatch) -> SummarizerAgent:
    summarizer = SummarizerAgent(settings=mock_settings)

    # Mock the OpenAI client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Here is a summary of the doctors found."
    mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)

    mock_create = AsyncMock(return_value=mock_response)
    summarizer._client = MagicMock()
    summarizer._client.chat.completions.create = mock_create

    return summarizer


@pytest.mark.asyncio
class TestSummarizerAgent:
    async def test_returns_summary(self, agent: SummarizerAgent) -> None:
        ctx = QueryContext(
            raw_query="Find cardiologists in Austin",
            intent=QueryIntent.SEARCH_DOCTORS,
            metadata={
                "api_data": [{"name": "Dr. Smith"}],
                "web_data": [{"title": "Top Doctors"}],
            },
        )
        result = await agent.execute(ctx)
        assert result.status == ResultStatus.SUCCESS
        assert result.agent_type == AgentType.SUMMARIZER
        assert "summary" in result.data[0]
        assert len(result.data[0]["summary"]) > 0

    async def test_handles_empty_data(self, agent: SummarizerAgent) -> None:
        ctx = QueryContext(
            raw_query="test",
            intent=QueryIntent.UNKNOWN,
            metadata={"api_data": [], "web_data": []},
        )
        result = await agent.execute(ctx)
        assert result.status == ResultStatus.SUCCESS