"""Centralized application configuration.

All runtime configuration is sourced from environment variables (or a local
.env file in development) via pydantic-settings, so the same image can be
promoted across environments without code changes.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:5173"

    database_url: str = (
        "postgresql+psycopg2://bms_user:changeme@localhost:5432/bms"
    )

    model_registry_path: str = "../ml/models/registry"

    llm_provider: str = "anthropic"
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-5"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def model_registry_dir(self) -> Path:
        return Path(self.model_registry_path).resolve()

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor - environment is read once per process."""
    return Settings()
