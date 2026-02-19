"""Abstract interfaces (ports) that concrete agents and services implement.

Using ABCs keeps the orchestration layer decoupled from any concrete
implementation, making it trivial to swap providers or add new agents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from agent_system.core.models import AgentResult, QueryContext


class Agent(ABC):
    """Base contract every agent must satisfy."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable agent name for logging / tracing."""

    @abstractmethod
    async def execute(self, context: QueryContext) -> AgentResult:
        """Run the agent's logic and return a structured result."""


class DataFetcher(ABC):
    """Port for any service that retrieves structured records from an API."""

    @abstractmethod
    async def fetch(
        self,
        endpoint: str,
        params: dict[str, str] | None = None,
    ) -> list[dict]:
        """Fetch data from a remote API endpoint."""


class SearchProvider(ABC):
    """Port for any web-search backend (Google, Bing, SerpAPI, …)."""

    @abstractmethod
    async def search(self, query: str, num_results: int = 5) -> list[dict]:
        """Return a list of search-result dicts."""