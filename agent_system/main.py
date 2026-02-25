"""Entry point for the Doctor Finder Agent System.

Wires together all components and provides both single-query
execution and interactive chat modes.
"""

from __future__ import annotations

import asyncio
import sys

import structlog

from agent_system.config.settings import Settings
from agent_system.services.query_parser import QueryParser
from agent_system.services.api_client import BackendApiClient
from agent_system.orchestrator.pipeline import AgentPipeline
from agent_system.agents.api_agent import ApiAgent
from agent_system.agents.summarizer_agent import SummarizerAgent
from agent_system.agents.tool_registry import build_function_registry
from agent_system.llm.model_loader import ModelLoader
from agent_system.llm.response_generator import ResponseGenerator
from agent_system.llm.tool_call_parser import ToolCallParser

logger = structlog.get_logger(__name__)


async def run_query(pipeline: AgentPipeline, query: str) -> str:
    """Run a single query through the pipeline and print the result."""
    print(f"\nUser: {query}")
    print("-" * 60)

    result = await pipeline.run(query)

    # Extract the answer
    answer = result.summary
    if not answer and result.api_results and result.api_results.data:
        data = result.api_results.data[0]
        answer = data.get("answer", str(data))

    print(f"\nAgent: {answer}")
    print(f"[Confidence: {result.confidence}]")
    print("=" * 60)
    return answer


async def interactive_mode(pipeline: AgentPipeline) -> None:
    """Run the agent in an interactive chat loop."""
    print("=" * 60)
    print("Doctor Finder Agent (Qwen2.5-1.5B-Instruct)")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        await run_query(pipeline, user_input)


async def run_test_suite(pipeline: AgentPipeline) -> None:
    """Run the comprehensive test suite from the original code."""

    test_queries = [

        "Find me all cardiologists available in Dhaka.",
        "I need a dentist. Can you search for one?",

        "Show me all doctors near Chittagong.",

        "Find me doctors who charge less than 500 taka.",
        "Show me doctors with consultation fees between 300 and 2000 taka in Dhaka.",


        "Show me affordable doctors in Chittagong, max budget 600.",


        "I've been having chest pain lately. What kind of doctor should I see and can you find one for me?",

        "Find doctors at Square Hospital in Dhaka.",

        "Tell me the qualifications of doctor number 3.",
    ]

    for query in test_queries:
        try:
            await run_query(pipeline, query)
        except Exception as exc:
            logger.error("test_query_failed", query=query, error=str(exc))
            print(f"[ERROR] {query}: {exc}")


async def main() -> None:
    """Wire everything together and run."""
    # --- Configuration ---
    settings = Settings()

    # --- Load LLM ---
    model_loader = ModelLoader(settings)
    model_loader.load()

    # --- Build components ---
    response_generator = ResponseGenerator(model_loader, settings)
    tool_call_parser = ToolCallParser()
    query_parser = QueryParser()

    # --- API Client & Function Registry ---
    async with BackendApiClient(settings) as api_client:
        function_registry = build_function_registry(api_client)

        # --- Build Agents ---
        api_agent = ApiAgent(
            response_generator=response_generator,
            tool_call_parser=tool_call_parser,
            function_registry=function_registry,
            max_iterations=settings.max_agent_iterations,
        )

        summarizer_agent = SummarizerAgent(
            response_generator=response_generator,
        )

        # --- Build Pipeline ---
        pipeline = AgentPipeline(
            query_parser=query_parser,
            api_agent=api_agent,
            web_search_agent=None,  # No web search agent for now
            summarizer_agent=summarizer_agent,
        )

        # --- Run ---
        if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
            await interactive_mode(pipeline)
        elif len(sys.argv) > 1 and sys.argv[1] == "--query":
            query = " ".join(sys.argv[2:])
            await run_query(pipeline, query)
        else:
            # Default: run test suite
            await run_test_suite(pipeline)


if __name__ == "__main__":
    # Configure structured logging
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    )

    asyncio.run(main())