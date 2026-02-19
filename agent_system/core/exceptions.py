"""Hierarchy of domain-specific exceptions.

A narrow exception tree makes it easy to handle errors precisely
at every layer without catching overly broad built-ins.
"""


class AgentSystemError(Exception):
    """Root exception for the entire agent system."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class ConfigurationError(AgentSystemError):
    """Raised when a required configuration value is missing or invalid."""


class AgentError(AgentSystemError):
    """Raised when an agent encounters a non-recoverable problem."""


class ApiClientError(AgentSystemError):
    """Raised on failures communicating with the backend API."""


class WebSearchError(AgentSystemError):
    """Raised on failures communicating with a web-search provider."""


class OrchestratorError(AgentSystemError):
    """Raised when the pipeline / orchestrator cannot fulfil a request."""