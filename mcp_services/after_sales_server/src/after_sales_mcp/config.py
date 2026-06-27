from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration owned by the independently deployable MCP service."""

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    database_url: str
    host: str = "0.0.0.0"
    port: int = 8100
    approval_signing_secret: str
    approval_issuer: str = "customer-service-agent"


@lru_cache
def get_settings() -> Settings:
    return Settings()
