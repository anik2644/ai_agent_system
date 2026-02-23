"""Application entry point.

Wires up all dependencies and runs the interactive query loop.
"""

from __future__ import annotations

import asyncio
import sys

import structlog

from agent_system.agents import ApiAgent, SummarizerAgent, WebSearchAgent
from agent_system.config.settings import get_settings
from agent_system.core.registry import AgentRegistry
from agent_system.orchestrator.pipeline import AgentPipeline
from agent_system.services.api_client import BackendApiClient
from agent_system.services.query_parser import QueryParser
from agent_system.services.web_scraper import WebSearchClient
from agent_system.utils.logging_config import setup_logging
from agent_system.config.settings import Settings

logger = structlog.get_logger(__name__)


async def build_pipeline(settings: Settings) -> tuple[
    AgentPipeline,
    BackendApiClient,
    WebSearchClient,
]:
    """Construct and wire up the full agent pipeline (Composition Root)."""

    # --- services -------------------------------------------------------
    api_client = BackendApiClient(settings)
    web_client = WebSearchClient(settings)

    # --- agents ---------------------------------------------------------
    api_agent = ApiAgent(data_fetcher=api_client)
    web_agent = WebSearchAgent(search_provider=web_client)
    summarizer = SummarizerAgent(settings=settings)

    # --- registry (optional – useful for plugin architectures) ----------
    registry = AgentRegistry()
    registry.register(api_agent)
    registry.register(web_agent)
    registry.register(summarizer)
    logger.info("registered_agents", agents=registry.list_agents())

    # --- pipeline -------------------------------------------------------
    pipeline = AgentPipeline(
        query_parser=QueryParser(),
        api_agent=api_agent,
        web_search_agent=web_agent,
        summarizer_agent=summarizer,
    )

    return pipeline, api_client, web_client


async def interactive_loop() -> None:
    """Run a simple REPL that takes user queries and prints fused results."""

    settings = get_settings()
    setup_logging(settings.log_level)

    pipeline, api_client, web_client = await build_pipeline(settings)

    async with api_client, web_client:
        print("\n🩺  AI Doctor Finder Agent System")
        print("=" * 50)
        print("Type your query below (or 'quit' to exit).\n")

        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not query:
                continue
            if query.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break

            try:
                result = await pipeline.run(query)
                print(f"\n{'─' * 50}")
                print(f"📋 Summary (confidence={result.confidence}):\n")
                print(result.summary or "[No summary generated]")
                print(f"\n📊 API records returned: {len(result.api_results.data)}")
                print(f"🌐 Web results returned: {len(result.web_results.data)}")
                print(f"{'─' * 50}\n")
            except Exception as exc:
                logger.exception("pipeline_error")
                print(f"\n⚠️  Error: {exc}\n")


def main() -> None:
    """Synchronous wrapper for the async entry point."""
    try:
        asyncio.run(interactive_loop())
    except KeyboardInterrupt:
        print("\nShutting down…")
        sys.exit(0)


if __name__ == "__main__":
    main()