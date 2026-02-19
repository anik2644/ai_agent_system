"""Agent that fetches structured data from the backend API."""

from __future__ import annotations

import structlog

from agent_system.agents.base import BaseAgent
from agent_system.core.interfaces import DataFetcher
from agent_system.core.models import (
    AgentResult,
    AgentType,
    QueryContext,
    QueryIntent,
    ResultStatus,
)

logger = structlog.get_logger(__name__)

# Map intents to API endpoints
_INTENT_ENDPOINT_MAP: dict[QueryIntent, str] = {
    QueryIntent.SEARCH_DOCTORS: "/doctors/search",
    QueryIntent.DOCTOR_DETAIL: "/doctors/detail",
    QueryIntent.GENERAL_HEALTH: "/health/info",
}


class ApiAgent(BaseAgent):
    """Retrieves structured records from the backend data API."""

    def __init__(self, data_fetcher: DataFetcher) -> None:
        super().__init__(AgentType.API)
        self._fetcher = data_fetcher

    @property
    def name(self) -> str:  # noqa: D401
        return "api_agent"

    async def _run(self, context: QueryContext) -> AgentResult:
        endpoint = _INTENT_ENDPOINT_MAP.get(
            context.intent,
            "/doctors/search",
        )

        # Build query params from extracted entities
        params: dict[str, str] = {}
        if "city" in context.entities:
            params["city"] = context.entities["city"]
        if "specialty" in context.entities:
            params["specialty"] = context.entities["specialty"]
        if not params:
            params["q"] = context.raw_query

        logger.info(
            "api_agent_fetch",
            endpoint=endpoint,
            params=params,
        )

        records = await self._fetcher.fetch(endpoint, params)

        return AgentResult(
            agent_type=AgentType.API,
            status=ResultStatus.SUCCESS if records else ResultStatus.PARTIAL,
            data=records,
            metadata={"endpoint": endpoint, "params": params},
        )