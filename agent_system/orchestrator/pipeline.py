"""Central pipeline that coordinates agent execution.

Execution flow:
1. Parse the user query → `QueryContext`
2. Run the **API agent** and **web-search agent** concurrently.
3. Feed both results into the **summarizer agent**.
4. Return a `FusedResult`.
"""

from __future__ import annotations

import asyncio

import structlog

from agent_system.core.interfaces import Agent
from agent_system.core.models import (
    AgentResult,
    AgentType,
    FusedResult,
    QueryContext,
    ResultStatus,
)
from agent_system.services.query_parser import QueryParser

logger = structlog.get_logger(__name__)


class AgentPipeline:
    """Orchestrates multi-agent execution with concurrent I/O."""

    def __init__(
        self,
        query_parser: QueryParser,
        api_agent: Agent,
        web_search_agent: Agent | None = None,
        summarizer_agent: Agent | None = None,
    ) -> None:
        self._parser = query_parser
        self._api_agent = api_agent
        self._web_agent = web_search_agent
        self._summarizer = summarizer_agent

    async def run(self, raw_query: str) -> FusedResult:
        """Run the full pipeline for *raw_query* and return a `FusedResult`."""

        # --- 1. Parse --------------------------------------------------
        context = self._parser.parse(raw_query)
        logger.info("pipeline_start", query_id=str(context.query_id))

        # --- 2. Gather data concurrently --------------------------------
        tasks = [self._safe_execute(self._api_agent, context)]

        if self._web_agent:
            tasks.append(self._safe_execute(self._web_agent, context))

        results = await asyncio.gather(*tasks)

        api_result = results[0]
        web_result = results[1] if len(results) > 1 else AgentResult(
            agent_type=AgentType.WEB_SEARCH,
            status=ResultStatus.NO_RESULT,
        )

        # --- 3. Fuse / summarise ----------------------------------------
        summary_result = AgentResult(
            agent_type=AgentType.SUMMARIZER,
            status=ResultStatus.NO_RESULT,
        )

        if self._summarizer:
            summary_context = context.model_copy(
                update={
                    "metadata": {
                        **context.metadata,
                        "api_data": api_result.data,
                        "web_data": web_result.data,
                    },
                }
            )
            summary_result = await self._safe_execute(
                self._summarizer,
                summary_context,
            )

        summary_text = ""
        if summary_result.data:
            summary_text = summary_result.data[0].get("summary", "")

        # If no summarizer, build summary from API results
        if not summary_text and api_result.data:
            summary_text = self._build_simple_summary(api_result)

        confidence = self._compute_confidence(api_result, web_result, summary_result)

        fused = FusedResult(
            query_id=context.query_id,
            summary=summary_text,
            api_results=api_result,
            web_results=web_result,
            confidence=confidence,
        )

        logger.info(
            "pipeline_complete",
            query_id=str(context.query_id),
            confidence=confidence,
        )
        return fused

    # -- helpers ----------------------------------------------------------

    @staticmethod
    async def _safe_execute(agent: Agent, context: QueryContext) -> AgentResult:
        """Execute an agent, catching unexpected exceptions."""
        try:
            return await agent.execute(context)
        except Exception as exc:
            logger.error("pipeline_agent_error", agent=agent.name, error=str(exc))
            return AgentResult(
                agent_type=AgentType.API,
                status=ResultStatus.ERROR,
                error_message=str(exc),
            )

    @staticmethod
    def _compute_confidence(
        api_result: AgentResult,
        web_result: AgentResult,
        summary_result: AgentResult,
    ) -> float:
        """Heuristic confidence score between 0 and 1."""
        score = 0.0
        if api_result.status == ResultStatus.SUCCESS and api_result.data:
            score += 0.5
        elif api_result.status == ResultStatus.PARTIAL:
            score += 0.25

        if web_result.status == ResultStatus.SUCCESS and web_result.data:
            score += 0.2
        elif web_result.status == ResultStatus.PARTIAL:
            score += 0.1

        if summary_result.status == ResultStatus.SUCCESS:
            score += 0.3

        return min(round(score, 2), 1.0)

    @staticmethod
    def _build_simple_summary(api_result: AgentResult) -> str:
        """Fallback summary when no summarizer agent is configured."""
        import json
        if not api_result.data:
            return "No results found."
        return json.dumps(api_result.data, indent=2, default=str)