"""Lightweight agent registry for dynamic look-up and plug-in support.

New agents can be registered at application start-up.  The orchestrator
resolves agents by name instead of importing concrete classes, which
keeps coupling low.
"""

from __future__ import annotations

import structlog

from agent_system.core.exceptions import ConfigurationError
from agent_system.core.interfaces import Agent

logger = structlog.get_logger(__name__)


class AgentRegistry:
    """Thread-safe, singleton-style registry of agent instances."""

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    # -- mutators ---------------------------------------------------------

    def register(self, agent: Agent) -> None:
        """Register an agent; raises on duplicate names."""
        if agent.name in self._agents:
            raise ConfigurationError(
                f"Agent '{agent.name}' is already registered."
            )
        self._agents[agent.name] = agent
        logger.info("agent_registered", agent_name=agent.name)

    def unregister(self, name: str) -> None:
        """Remove an agent by name (idempotent)."""
        self._agents.pop(name, None)

    # -- queries ----------------------------------------------------------

    def get(self, name: str) -> Agent:
        """Retrieve a registered agent by name."""
        try:
            return self._agents[name]
        except KeyError as exc:
            raise ConfigurationError(
                f"Agent '{name}' is not registered."
            ) from exc

    def list_agents(self) -> list[str]:
        """Return the names of all registered agents."""
        return list(self._agents.keys())

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, name: str) -> bool:
        return name in self._agents