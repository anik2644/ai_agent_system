"""Core package – domain models, interfaces, and shared exceptions."""

from agent_system.core.interfaces import Agent, DataFetcher, SearchProvider
from agent_system.core.models import (
    AgentResult,
    ApiRecord,
    FusedResult,
    QueryContext,
    SearchResult,
)
from agent_system.core.exceptions import (
    AgentError,
    ApiClientError,
    ConfigurationError,
    OrchestratorError,
    WebSearchError,
)

__all__ = [
    "Agent",
    "AgentError",
    "AgentResult",
    "ApiClientError",
    "ApiRecord",
    "ConfigurationError",
    "DataFetcher",
    "FusedResult",
    "OrchestratorError",
    "QueryContext",
    "SearchProvider",
    "SearchResult",
    "WebSearchError",
]