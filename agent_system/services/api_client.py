"""HTTP client for the backend data API.

Implements the `DataFetcher` port so that the API agent is
decoupled from HTTP details.
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
from agent_system.core.exceptions import ApiClientError
from agent_system.core.interfaces import DataFetcher

logger = structlog.get_logger(__name__)


class BackendApiClient(DataFetcher):
    """Async HTTP client wrapping the backend REST API."""

    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.backend_api_base_url.rstrip("/")
        self._timeout = settings.request_timeout
        self._max_retries = settings.max_retries
        self._client: httpx.AsyncClient | None = None

    # -- lifecycle --------------------------------------------------------

    async def __aenter__(self) -> "BackendApiClient":
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout),
            headers={"Accept": "application/json"},
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client:
            await self._client.aclose()

    # -- DataFetcher port -------------------------------------------------

    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def fetch(
        self,
        endpoint: str,
        params: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """GET *endpoint* and return the JSON payload as a list of dicts."""
        if self._client is None:
            raise ApiClientError("Client not initialised – use `async with`.")

        url = f"/{endpoint.lstrip('/')}"
        logger.info("api_request", url=url, params=params)

        try:
            response = await self._client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()

            # Normalise: always return a list
            if isinstance(payload, dict):
                payload = payload.get("results", [payload])

            logger.info("api_response", url=url, count=len(payload))
            return payload  # type: ignore[return-value]

        except httpx.HTTPStatusError as exc:
            raise ApiClientError(
                f"API returned {exc.response.status_code} for {url}",
                details={"status_code": exc.response.status_code},
            ) from exc
        except httpx.TransportError as exc:
            raise ApiClientError(
                f"Transport error for {url}: {exc}",
            ) from exc