"""Fusion / summarizer agent – combines API and web results via an LLM."""

from __future__ import annotations

import json
from typing import Any

import structlog
from openai import AsyncOpenAI

from agent_system.agents.base import BaseAgent
from agent_system.config.settings import Settings
from agent_system.core.models import (
    AgentResult,
    AgentType,
    QueryContext,
    ResultStatus,
)
from agent_system.utils.helpers import truncate

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a helpful medical-information assistant. You will receive:

1. **Structured API data** – authoritative records from a healthcare database.
2. **Web search snippets** – supplementary information from the open web.

Your task:
- Merge and de-duplicate the information.
- Prioritise the structured API data for factual accuracy.
- Enrich with useful context from the web results.
- Present a clear, well-organised answer to the user's original question.
- If information is limited, say so honestly.
"""


class SummarizerAgent(BaseAgent):
    """Fuses API and web-search results into a coherent summary using an LLM."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(AgentType.SUMMARIZER)
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.summarizer_model
        self._temperature = settings.summarizer_temperature
        self._max_tokens = settings.summarizer_max_tokens

    @property
    def name(self) -> str:  # noqa: D401
        return "summarizer_agent"

    async def _run(self, context: QueryContext) -> AgentResult:
        # The orchestrator injects upstream results via context.metadata
        api_data: list[dict[str, Any]] = context.metadata.get("api_data", [])
        web_data: list[dict[str, Any]] = context.metadata.get("web_data", [])

        user_prompt = self._build_prompt(context.raw_query, api_data, web_data)

        logger.info(
            "summarizer_request",
            model=self._model,
            prompt_length=len(user_prompt),
        )

        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        summary = (response.choices[0].message.content or "").strip()
        usage = response.usage

        return AgentResult(
            agent_type=AgentType.SUMMARIZER,
            status=ResultStatus.SUCCESS,
            data=[{"summary": summary}],
            metadata={
                "model": self._model,
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
            },
        )

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _build_prompt(
        query: str,
        api_data: list[dict[str, Any]],
        web_data: list[dict[str, Any]],
    ) -> str:
        sections = [f"## User Question\n{query}\n"]

        if api_data:
            pretty = json.dumps(api_data[:20], indent=2, default=str)
            sections.append(f"## Structured API Data\n```json\n{truncate(pretty, 3000)}\n```\n")
        else:
            sections.append("## Structured API Data\nNo results available.\n")

        if web_data:
            web_lines = "\n".join(
                f"- **{r.get('title', 'N/A')}**: {truncate(r.get('snippet', ''), 200)} "
                f"[link]({r.get('url', '')})"
                for r in web_data[:10]
            )
            sections.append(f"## Web Search Results\n{web_lines}\n")
        else:
            sections.append("## Web Search Results\nNo results available.\n")

        return "\n".join(sections)