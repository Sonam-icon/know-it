"""Validated application configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration loaded from environment variables and an optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Know-It"
    app_version: str = "0.1.0"
    environment: str = "development"

    google_api_key: SecretStr | None = None
    chat_model: str = "gemini-3.5-flash-lite"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimension: int = Field(default=3072, ge=128, le=3072)

    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    max_query_rewrites: int = Field(default=2, ge=0, le=5)
    provider_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    provider_max_attempts: int = Field(default=3, ge=1, le=5)
    grading_concurrency: int = Field(default=3, ge=1, le=10)
    max_question_length: int = Field(default=2000, ge=100, le=10_000)
    max_context_characters: int = Field(default=30_000, ge=1000, le=200_000)

    chunk_size_tokens: int = Field(default=700, ge=100, le=1800)
    chunk_overlap_tokens: int = Field(default=80, ge=0, le=500)
    max_upload_bytes: int = Field(default=5 * 1024 * 1024, ge=1024)
    max_fetch_bytes: int = Field(default=5 * 1024 * 1024, ge=1024)
    fetch_timeout_seconds: float = Field(default=15.0, gt=0, le=120)
    max_fetch_redirects: int = Field(default=3, ge=0, le=10)

    chroma_path: Path = Path("data/chroma")
    chroma_collection: str = "technical_docs"
    application_db_path: Path = Path("data/application.sqlite3")
    checkpoint_db_path: Path = Path("data/checkpoints.sqlite3")

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS")
        return self

    @property
    def provider_configured(self) -> bool:
        """Report whether live Gemini operations can be used."""

        return self.google_api_key is not None and bool(self.google_api_key.get_secret_value())


@lru_cache
def get_settings() -> Settings:
    """Return one immutable-by-convention settings instance per process."""

    return Settings()
