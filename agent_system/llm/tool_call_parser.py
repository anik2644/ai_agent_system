"""Parses tool/function calls from LLM-generated text.

Handles multiple output formats that Qwen2.5-Instruct may produce.
"""

from __future__ import annotations

import json
import re

import structlog

from agent_system.core.models import ToolCall

logger = structlog.get_logger(__name__)

# Known function names for validation
_KNOWN_FUNCTIONS = {
    "fetch_doctor_details",
    "find_doctors",
    "list_available_locations",
    "list_available_specializations",
}


class ToolCallParser:
    """Stateless parser that extracts `ToolCall` objects from raw LLM text."""

    def parse(self, response_text: str) -> list[ToolCall]:
        """Extract all tool calls from *response_text*."""
        tool_calls = (
            self._parse_xml_format(response_text)
            or self._parse_json_format(response_text)
            or self._parse_function_call_format(response_text)
        )

        if tool_calls:
            logger.info("tool_calls_parsed", count=len(tool_calls),
                        names=[tc.name for tc in tool_calls])
        else:
            logger.debug("no_tool_calls_found")

        return tool_calls

    # -- Pattern 1: <tool_call>{"name": ..., "arguments": ...}</tool_call>

    @staticmethod
    def _parse_xml_format(text: str) -> list[ToolCall]:
        pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(pattern, text, re.DOTALL)

        results: list[ToolCall] = []
        for match in matches:
            try:
                call = json.loads(match)
                name = call.get("name") or call.get("function", {}).get("name")
                arguments = call.get("arguments") or call.get("function", {}).get("arguments", {})
                if name and name in _KNOWN_FUNCTIONS:
                    results.append(ToolCall(name=name, arguments=arguments))
            except json.JSONDecodeError:
                continue
        return results

    # -- Pattern 2: {"name": "func", "arguments": {...}}

    @staticmethod
    def _parse_json_format(text: str) -> list[ToolCall]:
        pattern = r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}'
        matches = re.findall(pattern, text, re.DOTALL)

        results: list[ToolCall] = []
        for name, args_str in matches:
            try:
                arguments = json.loads(args_str)
                if name in _KNOWN_FUNCTIONS:
                    results.append(ToolCall(name=name, arguments=arguments))
            except json.JSONDecodeError:
                continue
        return results

    # -- Pattern 3: func_name({...})

    @staticmethod
    def _parse_function_call_format(text: str) -> list[ToolCall]:
        results: list[ToolCall] = []
        for func_name in _KNOWN_FUNCTIONS:
            pattern = rf'{func_name}\s*\(\s*(\{{.*?\}})\s*\)'
            matches = re.findall(pattern, text, re.DOTALL)
            for args_str in matches:
                try:
                    arguments = json.loads(args_str)
                    results.append(ToolCall(name=func_name, arguments=arguments))
                except json.JSONDecodeError:
                    continue
        return results