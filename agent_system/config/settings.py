"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Immutable application-wide configuration.

    Values are loaded from environment variables and/or a `.env` file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        frozen=True,
    )

    # --- API Keys ---
    openai_api_key: str = Field(
        ...,
        description="OpenAI API key for the summarizer agent",
    )
    search_api_key: str = Field(
        default="",
        description="API key for the web-search provider",
    )
    backend_api_base_url: str = Field(
        default="https://api.example.com/v1",
        description="Base URL for the backend data API",
    )

    # --- General ---
    log_level: str = Field(default="INFO")
    max_retries: int = Field(default=3, ge=1, le=10)
    request_timeout: int = Field(default=30, ge=5, le=120)
    environment: str = Field(default="development")

    # --- Summarizer Agent ---
    summarizer_model: str = Field(default="gpt-4o")
    summarizer_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    summarizer_max_tokens: int = Field(default=2048, ge=100, le=8192)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of application settings."""
    return Settings()