"""Process configuration.

One settings object, loaded from the environment, validated at import of the service
entrypoint (not lazily) so a misconfigured pod dies at startup rather than at 3am on
the first request that touches the bad field.

Secrets arrive as env vars locally (from a gitignored .env) and from Key Vault via
managed identity in Azure. Either way they are `SecretStr`, so a stray f-string in a
log line prints `**********` instead of the key.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "ci", "dev", "prod"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ARGUS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    env: Environment = "local"
    log_level: str = "INFO"
    log_json: bool = False

    postgres_dsn: str = "postgresql+asyncpg://argus:argus@localhost:5432/argus"
    kafka_bootstrap: str = "localhost:9092"

    openfga_url: str = "http://localhost:8080"
    openfga_store_id: str = ""
    opa_url: str = "http://localhost:8181"

    otel_endpoint: str = "http://localhost:4317"
    prometheus_pushgateway: str = "http://localhost:9091"
    mlflow_tracking_uri: str = "http://localhost:5000"

    azure_openai_endpoint: str = ""
    azure_openai_api_key: SecretStr = SecretStr("")
    llm_model: str = "gpt-4o-mini"

    sim_seed: int = Field(default=42, description="Global RNG seed for reproducibility")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor. Import this, never a module-level `settings = Settings()` —
    a module-level instance reads the environment at import time, which makes tests
    that monkeypatch env vars silently ineffective."""
    return Settings()
