"""Summarizer Agent – optional agent that generates a human-friendly summary.

Uses the LLM to synthesise API and web-search results into a single
coherent answer.
"""

from __future__ import annotations

import json

import structlog

from agent_system.core.interfaces import Agent
from agent_system.core.models import (
    AgentResult,
    AgentType,
    QueryContext,
    ResultStatus,
)
from agent_system.llm.response_generator import ResponseGenerator

logger = structlog.get_logger(__name__)

SUMMARIZER_SYSTEM_PROMPT = """You are a helpful medical assistant. 
You are given raw data about doctors from a database and possibly from web searches.
Summarise the information in a clear, friendly, well-formatted response for the user.
If the data is empty or contains errors, tell the user politely that no results were found."""


class SummarizerAgent(Agent):
    """Synthesises raw agent outputs into a user-friendly summary."""

    def __init__(self, response_generator: ResponseGenerator) -> None:
        self._generator = response_generator

    @property
    def name(self) -> str:
        return "summarizer_agent"

    async def execute(self, context: QueryContext) -> AgentResult:
        """Generate a summary from the metadata attached to *context*."""
        api_data = context.metadata.get("api_data", [])
        web_data = context.metadata.get("web_data", [])

        user_content = (
            f"Original user query: {context.raw_query}\n\n"
            f"API results:\n{json.dumps(api_data, indent=2, default=str)}\n\n"
            f"Web search results:\n{json.dumps(web_data, indent=2, default=str)}\n\n"
            "Please provide a clear, concise summary for the user."
        )

        messages = [
            {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            summary = self._generator.generate(messages)
            return AgentResult(
                agent_type=AgentType.SUMMARIZER,
                status=ResultStatus.SUCCESS,
                data=[{"summary": summary}],
            )
        except Exception as exc:
            logger.error("summarizer_error", error=str(exc))
            return AgentResult(
                agent_type=AgentType.SUMMARIZER,
                status=ResultStatus.ERROR,
                error_message=str(exc),
            )