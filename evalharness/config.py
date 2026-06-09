"""Application settings powered by pydantic-settings.

All knobs are configurable via environment variables prefixed with ``EVAL_HARNESS_``.
For example, ``EVAL_HARNESS_DB_PATH`` overrides the default SQLite path.

Docker Compose passes these env vars directly; the Settings class normalises them
into the fields the application uses.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for the Eval Harness."""

    # --- Database ---
    # Either supply a full async SQLAlchemy URL, or set DB_PATH for SQLite shorthand.
    DATABASE_URL: str = ""
    DB_PATH: str = ""  # e.g. /app/data/evalharness.db (Docker) or ./evalharness.db

    # --- Authentication ---
    API_KEY: str = ""  # If set, all /api/* requests must supply Bearer <API_KEY>

    # --- Provider Keys ---
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""

    # --- CORS ---
    CORS_ORIGINS: str = "*"  # Comma-separated origins, or "*" for all

    # --- Evaluation Engine ---
    DEFAULT_CONCURRENCY: int = 5
    DEFAULT_TIMEOUT: int = 60
    DEFAULT_RETRIES: int = 3
    EVAL_RUNS_PER_TASK: int = 3
    MAX_WORKERS: int = 4

    # --- Rate Limiting (requests / minute) ---
    RATE_LIMIT: int = 100

    # --- Logging ---
    LOG_LEVEL: str = "INFO"

    # --- Runtime ---
    ENV: str = "production"  # development | production

    model_config = {
        "env_prefix": "EVAL_HARNESS_",
        "case_sensitive": False,
    }

    def get_database_url(self) -> str:
        """Resolve the effective async database URL.

        Priority:
        1. ``EVAL_HARNESS_DATABASE_URL`` — explicit full SQLAlchemy URL
        2. ``EVAL_HARNESS_DB_PATH``       — short SQLite path → converted to URL
        3. Default local SQLite file
        """
        if self.DATABASE_URL:
            return self.DATABASE_URL
        if self.DB_PATH:
            return f"sqlite+aiosqlite:///{self.DB_PATH}"
        return "sqlite+aiosqlite:///./evalharness.db"

    def get_cors_origins(self) -> list[str]:
        """Parse CORS_ORIGINS into a list. ``*`` returns the wildcard list."""
        if self.CORS_ORIGINS.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    return Settings()
