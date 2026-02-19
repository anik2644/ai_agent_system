"""Web-search provider implementation.

Uses a generic search API (easily swappable for Google Custom Search,
SerpAPI, Brave Search, etc.).  Implements the `SearchProvider` port.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agent_system.config.settings import Settings
from agent_system.core.exceptions import WebSearchError
from agent_system.core.interfaces import SearchProvider

logger = structlog.get_logger(__name__)

# Default to Google Custom Search JSON API format; swap as needed.
_DEFAULT_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"


class WebSearchClient(SearchProvider):
    """Async web-search client."""

    def __init__(
        self,
        settings: Settings,
        search_url: str = _DEFAULT_SEARCH_URL,
    ) -> None:
        self._api_key = settings.search_api_key
        self._timeout = settings.request_timeout
        self._search_url = search_url
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "WebSearchClient":
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self._timeout))
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client:
            await self._client.aclose()

    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def search(
        self,
        query: str,
        num_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Execute a web search and return normalised results."""
        if self._client is None:
            raise WebSearchError("Client not initialised – use `async with`.")

        params = {
            "q": query,
            "key": self._api_key,
            "num": str(num_results),
        }
        logger.info("web_search_request", query=query, num_results=num_results)

        try:
            response = await self._client.get(self._search_url, params=params)
            response.raise_for_status()
            data = response.json()

            results = [
                {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "source": item.get("displayLink", ""),
                }
                for item in data.get("items", [])
            ]
            logger.info("web_search_response", count=len(results))
            return results

        except httpx.HTTPStatusError as exc:
            raise WebSearchError(
                f"Search API returned {exc.response.status_code}",
            ) from exc
        except httpx.TransportError as exc:
            raise WebSearchError(
                f"Transport error during search: {exc}",
            ) from exc