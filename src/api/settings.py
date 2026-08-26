"""Application settings — pydantic-settings (context7-validated patterns).

Sources, in priority order (highest first): init args → env vars (``FINALZE_*``) →
``.env`` → ``secrets_dir`` files (Docker/K8s) → defaults. Typed DSN fields validate
the Postgres/Redis/RabbitMQ URLs at startup.

Generate a real secret: ``openssl rand -hex 32`` → ``FINALZE_SECRET_KEY``.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import AmqpDsn, Field, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

# Docker/K8s file-secrets mount. Only enable when present so local dev/tests don't warn;
# in containers /run/secrets exists and provides e.g. /run/secrets/finalze_secret_key.
_SECRETS_DIR = "/run/secrets" if os.path.isdir("/run/secrets") else None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FINALZE_",
        env_file=".env",
        env_file_encoding="utf-8",
        secrets_dir=_SECRETS_DIR,  # Docker/K8s secrets when mounted; None on dev (see above)
        extra="ignore",
    )

    # --- Security ---------------------------------------------------------
    secret_key: str = Field(
        default="change-me--openssl-rand-hex-32", min_length=16,
        description="Signing/token secret. Override in every environment.",
    )
    access_token_lifetime_seconds: int = 1800  # short-lived (RedisStrategy revocable)

    # --- Datastores (typed DSNs) -----------------------------------------
    database_url: PostgresDsn = Field(
        default="postgresql+asyncpg://finalze:finalze@localhost:5432/finalze",
        description="Async SQLAlchemy DSN (asyncpg driver).",
    )
    redis_url: RedisDsn = Field(default="redis://localhost:6379/0")          # tokens, rate-limit, cache
    broker_url: AmqpDsn = Field(default="amqp://guest:guest@localhost:5672//")  # Celery broker (RabbitMQ)
    result_backend_url: RedisDsn = Field(default="redis://localhost:6379/1")    # Celery results

    # --- API ---------------------------------------------------------------
    allowed_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8501"])
    max_upload_mb: int = 25
    docs_enabled: bool = True          # gate /docs & /redoc in production
    root_path: str = ""                # set when behind a proxy subpath (Traefik)

    # --- Object storage (MinIO / S3) --------------------------------------
    s3_endpoint_url: str | None = None
    s3_bucket: str = "finalze-documents"
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_secure: bool = False

    # --- Pipeline defaults ------------------------------------------------
    default_accounting_standard: str = "NCT"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Cached singleton; import this rather than instantiating ``Settings`` directly."""
    return Settings()
