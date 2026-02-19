"""Agent that enriches results with web-search data."""

from __future__ import annotations

import structlog

from agent_system.agents.base import BaseAgent
from agent_system.core.interfaces import SearchProvider
from agent_system.core.models import (
    AgentResult,
    AgentType,
    QueryContext,
    ResultStatus,
)

logger = structlog.get_logger(__name__)


class WebSearchAgent(BaseAgent):
    """Performs a web search to complement structured API data."""

    def __init__(
        self,
        search_provider: SearchProvider,
        max_results: int = 5,
    ) -> None:
        super().__init__(AgentType.WEB_SEARCH)
        self._search = search_provider
        self._max_results = max_results

    @property
    def name(self) -> str:  # noqa: D401
        return "web_search_agent"

    async def _run(self, context: QueryContext) -> AgentResult:
        # Build a search-engine-friendly query string
        parts = [context.raw_query]
        if "city" in context.entities:
            parts.append(context.entities["city"])
        if "specialty" in context.entities:
            parts.append(context.entities["specialty"])
        search_query = " ".join(parts)

        logger.info("web_search_agent_query", query=search_query)

        results = await self._search.search(
            search_query,
            num_results=self._max_results,
        )

        return AgentResult(
            agent_type=AgentType.WEB_SEARCH,
            status=ResultStatus.SUCCESS if results else ResultStatus.PARTIAL,
            data=results,
            metadata={"search_query": search_query},
        )