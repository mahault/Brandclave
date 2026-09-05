"""Typed application settings, validated at boot.

Central place for every environment variable the app reads. Values come from
the process environment and the project ``.env`` file (see ``.env.example``).

Design notes:
- Malformed values (e.g. a non-integer interval) fail fast at boot with a
  clear pydantic validation error instead of deep inside a request.
- API keys are ``Optional`` so the app can still boot for the dashboard
  without any secrets configured. Features that need a key must call
  :meth:`Settings.get_required` at the point of use, which raises a
  ``RuntimeError`` naming the missing variable.
- Use :func:`get_settings` (cached singleton) rather than instantiating
  ``Settings`` directly.
"""

from functools import lru_cache
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_VALID_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}


class Settings(BaseSettings):
    """Application configuration loaded from environment variables / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # .env may hold vars for tools/features not modelled here
    )

    # --- Secrets & integrations (optional so the app boots without keys) ---
    mistral_api_key: Optional[str] = None
    genius_api_url: Optional[str] = None
    genius_api_key: Optional[str] = None
    genius_agent_id: Optional[str] = None
    genius_license_key: Optional[str] = None
    sentry_dsn: Optional[str] = None

    # Bluesky (AT Protocol). Unauthenticated searchPosts is now WAF-blocked on
    # every public AppView (403, verified 2026-09-04); bsky.social answers 401,
    # i.e. it serves the same lexicon to an authenticated session. Use a handle
    # plus an app password from Settings -> Privacy and security -> App passwords
    # (never the account password). Absent these, the scraper no-ops with a warning.
    bluesky_handle: Optional[str] = None
    bluesky_app_password: Optional[str] = None

    # OpenAI, used only for image generation (brand concept renders).
    openai_api_key: Optional[str] = None
    openai_image_model: str = "gpt-image-1.5"

    # --- Auth ---
    # Secret used to sign JWT access tokens (env JWT_SECRET). Optional: when
    # unset, services.auth generates an ephemeral secret at boot and logs a
    # warning that sessions won't survive restarts.
    jwt_secret: Optional[str] = None

    # --- Storage ---
    database_url: str = "sqlite:///./data/brandclave.db"
    chroma_persist_dir: str = "./data/chroma"
    redis_url: str = "redis://localhost:6379/0"

    # --- Embeddings ---
    embedding_provider: str = "mistral"  # "mistral" or "local"

    # --- Scraping ---
    scraper_user_agent: str = "BrandClave-Aggregator/1.0"
    adaptive_scrape_interval_minutes: int = 30

    # --- Scheduler / startup behaviour ---
    scheduler_enabled: bool = True
    prewarm_services: bool = False

    # --- Logging / observability ---
    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "text"

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        """Normalize LOG_LEVEL and reject unknown levels at boot."""
        level = value.strip().upper()
        if level not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"LOG_LEVEL must be one of {sorted(_VALID_LOG_LEVELS)}, got {value!r}"
            )
        return level

    @field_validator("log_format")
    @classmethod
    def _validate_log_format(cls, value: str) -> str:
        """Normalize LOG_FORMAT and reject unknown formats at boot."""
        fmt = value.strip().lower()
        if fmt not in {"json", "text"}:
            raise ValueError(f"LOG_FORMAT must be 'json' or 'text', got {value!r}")
        return fmt

    def get_required(self, name: str) -> str:
        """Return a setting value, failing loudly if it is missing.

        Use this in code paths that genuinely need a secret (e.g. calling
        Mistral), so a missing key produces one clear error instead of an
        opaque failure deep inside a request.

        Args:
            name: Setting name, case-insensitive (e.g. "MISTRAL_API_KEY").

        Raises:
            RuntimeError: If the setting is unknown, unset, or blank.
        """
        field = name.lower()
        if field not in type(self).model_fields:
            raise RuntimeError(f"Unknown setting: {name}")
        value = getattr(self, field)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise RuntimeError(
                f"Missing required configuration: {name.upper()} is not set. "
                "Add it to your environment or .env file (see .env.example)."
            )
        return str(value)

    def integrations_summary(self) -> str:
        """One-line summary of configured integrations. Never includes secret values."""
        db_backend = self.database_url.split(":", 1)[0]
        flags = {
            "mistral": bool(self.mistral_api_key),
            "genius": bool(self.genius_api_key),
            "sentry": bool(self.sentry_dsn),
        }
        integrations = ", ".join(
            f"{name}={'configured' if on else 'off'}" for name, on in flags.items()
        )
        return (
            f"db={db_backend}, embeddings={self.embedding_provider}, "
            f"scheduler={'on' if self.scheduler_enabled else 'off'}, {integrations}"
        )


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings singleton, validating on first access."""
    return Settings()
