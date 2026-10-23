"""Server configuration, read from ``LEGIBLE_*`` environment variables (or ``.env``)."""

from __future__ import annotations

import os
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LEGIBLE_", env_file=".env", extra="ignore")

    environment: Literal["development", "test", "production"] = "development"
    data_dir: Path = Path("data")
    database_url: str | None = None          # default: sqlite:///<data_dir>/legible.db
    jwt_secret: SecretStr | None = None      # required in production
    jwt_ttl_hours: int = 24 * 14
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    default_timezone: str = "UTC"
    serve_frontend: bool = True
    frontend_dist: Path = Path("web/dist")
    max_upload_bytes: int = 15 * 1024 * 1024

    # Phase 1: handwriting model
    htr_checkpoint: Path | None = None       # a checkpoint written by legible-train
    htr_device: str = "auto"                 # auto | cpu | cuda | cuda:1 | mps
    htr_beam_size: int = 6
    personalize_min_samples: int = 8         # corrected lines before a personal adapter is trained
    personalize_steps: int = 300

    # Claude, for "what was probably meant", conversation, and comprehension questions.
    # "auto" uses Claude when an API key is available and falls back to an offline demo otherwise.
    llm_mode: Literal["auto", "anthropic", "off"] = "auto"
    anthropic_api_key: SecretStr | None = None
    llm_model: str = "claude-opus-5-5"

    # Phase 4: optional realistic avatar (Tavus CVI in echo mode). Unset = built-in avatar.
    # replica_id is a Tavus *face* id; persona_id must be a PAL created with
    # "pipeline_mode": "echo", so Legible supplies the words and Tavus renders face and voice.
    tavus_api_key: SecretStr | None = None
    tavus_replica_id: str | None = None
    tavus_persona_id: str | None = None
    tavus_api_base: str = "https://tavusapi.com/v2"
    tavus_max_call_seconds: int = 900

    @property
    def resolved_database_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.data_dir / 'legible.db').as_posix()}"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def adapters_dir(self) -> Path:
        return self.data_dir / "adapters"

    def secret_key(self) -> str:
        """JWT signing key. Development generates one and keeps it in the data directory."""
        if self.jwt_secret is not None:
            return self.jwt_secret.get_secret_value()
        if self.environment == "production":
            raise RuntimeError("LEGIBLE_JWT_SECRET must be set in production")
        path = self.data_dir / ".jwt_secret"
        if path.exists():
            return path.read_text().strip()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        key = secrets.token_urlsafe(48)
        path.write_text(key)
        os.chmod(path, 0o600)
        return key

    def anthropic_key(self) -> str | None:
        if self.anthropic_api_key is not None:
            return self.anthropic_api_key.get_secret_value()
        return os.environ.get("ANTHROPIC_API_KEY") or None

    @property
    def llm_configured(self) -> bool:
        if self.llm_mode == "off":
            return False
        if self.llm_mode == "anthropic":
            return True  # the SDK resolves credentials itself (env, `ant auth login` profile, ...)
        return bool(self.anthropic_key() or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

    @property
    def tavus_configured(self) -> bool:
        return bool(self.tavus_api_key and self.tavus_replica_id and self.tavus_persona_id)


@lru_cache
def get_settings() -> Settings:
    return Settings()
