"""Tool definitions and registry for function-calling.

Defines the TOOLS_SCHEMA (OpenAI-compatible) and FUNCTION_REGISTRY
that maps tool names to async callables backed by the API client.
"""

from __future__ import annotations

from typing import Any, Callable, Awaitable, Optional

from agent_system.services.api_client import BackendApiClient

# Type alias for async tool functions
ToolFunction = Callable[..., Awaitable[Any]]


# ── Tool Schema (OpenAI / Qwen compatible) ──────────────────────────────

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "fetch_doctor_details",
            "description": "Get detailed information about a specific doctor by their ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_id": {
                        "type": "integer",
                        "description": "The unique ID of the doctor."
                    }
                },
                "required": ["doctor_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_doctors",
            "description": "Search for doctors based on specialization, city location, or price range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "specialization": {
                        "type": "string",
                        "description": "The medical specialization to filter by (e.g., 'Cardiology')."
                    },
                    "location": {
                        "type": "string",
                        "description": "The city to filter by (e.g., 'New York')."
                    },
                    "min_fee": {
                        "type": "number",
                        "description": "Minimum consultation fee."
                    },
                    "max_fee": {
                        "type": "number",
                        "description": "Maximum consultation fee."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_locations",
            "description": "Returns a list of all cities where doctors are available.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_specializations",
            "description": "Returns a list of all available medical specializations.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


def build_function_registry(client: BackendApiClient) -> dict[str, ToolFunction]:
    """Build a name→callable mapping using the given API client."""

    async def fetch_doctor_details(doctor_id: int) -> list[dict[str, Any]]:
        return await client.get_doctor_by_id(int(doctor_id))

    async def find_doctors(
        specialization: Optional[str] = None,
        location: Optional[str] = None,
        min_fee: Optional[float] = None,
        max_fee: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        return await client.search_doctors(
            specialization=specialization,
            location=location,
            min_fee=float(min_fee) if min_fee is not None else None,
            max_fee=float(max_fee) if max_fee is not None else None,
        )

    async def list_available_locations() -> list[dict[str, Any]]:
        return await client.get_metadata_locations()

    async def list_available_specializations() -> list[dict[str, Any]]:
        return await client.get_metadata_specializations()

    return {
        "fetch_doctor_details": fetch_doctor_details,
        "find_doctors": find_doctors,
        "list_available_locations": list_available_locations,
        "list_available_specializations": list_available_specializations,
    }