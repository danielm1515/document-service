import os
from dataclasses import dataclass, field


def _as_bool(value: str | None, default: bool) -> bool:
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    """Read when the app is built, never at import time, so tests control the environment."""

    database_url: str = field(default_factory=lambda: _env("DATABASE_URL", "sqlite:///./documents.db"))
    api_auth_enabled: bool = field(default_factory=lambda: _as_bool(os.getenv("API_AUTH_ENABLED"), True))
    api_key: str = field(default_factory=lambda: _env("DOCUMENT_API_KEY"))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-5.6-luna"))
    s3_bucket: str = field(default_factory=lambda: _env("S3_BUCKET"))
    aws_region: str = field(default_factory=lambda: _env("AWS_REGION", "eu-north-1"))
    max_upload_bytes: int = 10 * 1024 * 1024
