"""API Agent – uses the LLM with tool-calling to answer doctor queries.

This agent replaces the original monolithic `run_agent` function.
It handles the iterative tool-calling loop:
  1. Send user query + tool schema to LLM
  2. Parse any tool calls from the response
  3. Execute tool calls against the backend API
  4. Feed results back and repeat until a final answer is produced
"""

from __future__ import annotations

import json
import re

import structlog

from agent_system.core.interfaces import Agent
from agent_system.core.models import (
    AgentResult,
    AgentType,
    QueryContext,
    ResultStatus,
    ToolCall,
)
from agent_system.agents.tool_registry import TOOLS_SCHEMA, ToolFunction
from agent_system.llm.response_generator import ResponseGenerator
from agent_system.llm.tool_call_parser import ToolCallParser

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are a helpful medical assistant that helps users find doctors. 
You have access to the following tools to look up doctor information from a database.
Always use the tools when users ask about doctors, locations, or specializations.
After receiving tool results, provide a clear, friendly, and well-formatted response to the user."""


class ApiAgent(Agent):
    """Agent that queries the doctor API via LLM-driven tool calls."""

    def __init__(
        self,
        response_generator: ResponseGenerator,
        tool_call_parser: ToolCallParser,
        function_registry: dict[str, ToolFunction],
        max_iterations: int = 5,
    ) -> None:
        self._generator = response_generator
        self._parser = tool_call_parser
        self._functions = function_registry
        self._max_iterations = max_iterations

    @property
    def name(self) -> str:
        return "api_agent"

    async def execute(self, context: QueryContext) -> AgentResult:
        """Run the tool-calling loop and return structured results."""
        raw_query = context.raw_query
        logger.info("api_agent_start", query=raw_query)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": raw_query},
        ]

        all_tool_results: list[dict] = []

        for iteration in range(self._max_iterations):
            logger.info("api_agent_iteration", iteration=iteration + 1)

            # Generate LLM response
            response_text = self._generator.generate(messages, tools=TOOLS_SCHEMA)
            logger.debug("api_agent_raw_response", text=response_text[:500])

            # Parse tool calls
            tool_calls = self._parser.parse(response_text)

            if not tool_calls:
                # Final answer – no more tool calls
                final_answer = re.sub(
                    r'<tool_call>.*?</tool_call>', '', response_text, flags=re.DOTALL
                ).strip()

                logger.info("api_agent_final_answer", length=len(final_answer))
                return AgentResult(
                    agent_type=AgentType.API,
                    status=ResultStatus.SUCCESS if all_tool_results else ResultStatus.PARTIAL,
                    data=[{"answer": final_answer, "tool_results": all_tool_results}],
                )

            # Execute tool calls
            logger.info("api_agent_tool_calls", count=len(tool_calls))
            messages.append({"role": "assistant", "content": response_text})

            for tc in tool_calls:
                logger.info("api_agent_tool_exec", name=tc.name, args=tc.arguments)
                result = await self._execute_tool(tc)
                result_str = json.dumps(result, indent=2, default=str)
                all_tool_results.append({"tool": tc.name, "args": tc.arguments, "result": result})

                messages.append({
                    "role": "tool",
                    "name": tc.name,
                    "content": result_str,
                })

        # Exhausted iterations – generate final response
        logger.warning("api_agent_max_iterations")
        response_text = self._generator.generate(messages, tools=TOOLS_SCHEMA)
        final_answer = re.sub(
            r'<tool_call>.*?</tool_call>', '', response_text, flags=re.DOTALL
        ).strip()

        return AgentResult(
            agent_type=AgentType.API,
            status=ResultStatus.PARTIAL,
            data=[{"answer": final_answer, "tool_results": all_tool_results}],
        )

    async def _execute_tool(self, tool_call: ToolCall) -> dict | list:
        """Execute a single tool call and return the result."""
        func_name = tool_call.name
        arguments = dict(tool_call.arguments)

        if func_name not in self._functions:
            return {"error": f"Unknown function: {func_name}"}

        func = self._functions[func_name]

        try:
            # Type coercions
            if "doctor_id" in arguments:
                arguments["doctor_id"] = int(arguments["doctor_id"])
            if "min_fee" in arguments and arguments["min_fee"] is not None:
                arguments["min_fee"] = float(arguments["min_fee"])
            if "max_fee" in arguments and arguments["max_fee"] is not None:
                arguments["max_fee"] = float(arguments["max_fee"])

            # Remove None values
            arguments = {k: v for k, v in arguments.items() if v is not None}

            result = await func(**arguments)
            return result
        except Exception as e:
            logger.error("tool_execution_error", func=func_name, error=str(e))
            return {"error": f"Error executing {func_name}: {str(e)}"}