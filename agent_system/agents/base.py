"""Shared base class for all concrete agents.

Provides timing, error handling, and logging out of the box so that
concrete sub-classes only need to implement `_run`.
"""

from __future__ import annotations

import time
from abc import abstractmethod

import structlog

from agent_system.core.exceptions import AgentError
from agent_system.core.interfaces import Agent
from agent_system.core.models import AgentResult, AgentType, QueryContext, ResultStatus

logger = structlog.get_logger(__name__)


class BaseAgent(Agent):
    """Template-method base class that wraps `_run` with observability."""

    def __init__(self, agent_type: AgentType) -> None:
        self._agent_type = agent_type

    # -- public entry point -----------------------------------------------

    async def execute(self, context: QueryContext) -> AgentResult:
        """Execute the agent with automatic timing and error capture."""
        start = time.perf_counter()
        try:
            result = await self._run(context)
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(
                "agent_success",
                agent=self.name,
                elapsed_ms=round(elapsed, 2),
                records=len(result.data),
            )
            return result.model_copy(update={"execution_time_ms": elapsed})

        except AgentError:
            raise  # already structured – let it propagate

        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(
                "agent_failure",
                agent=self.name,
                error=str(exc),
                elapsed_ms=round(elapsed, 2),
            )
            return AgentResult(
                agent_type=self._agent_type,
                status=ResultStatus.ERROR,
                error_message=str(exc),
                execution_time_ms=elapsed,
            )

    # -- subclass hook ----------------------------------------------------

    @abstractmethod
    async def _run(self, context: QueryContext) -> AgentResult:
        """Implement the agent-specific logic here."""