"""Environment-based application settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Research Paper Assistant"
    app_version: str = "0.1.0"
    app_env: Literal["development", "test", "production"] = "development"
    app_host: str = "127.0.0.1"
    app_port: int = Field(default=8000, ge=1, le=65535)
    api_prefix: str = "/api/v1"

    database_url: str = "sqlite+aiosqlite:///./data/research_paper_assistant.db"
    paper_storage_path: Path = Path("./data/papers")
    max_pdf_size_bytes: int = Field(default=50 * 1024 * 1024, ge=1)
    qdrant_url: str = "http://127.0.0.1:6333"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_generation_model: str = "gemma4:12b"
    ollama_embedding_model: str = "qwen3-embedding:0.6b"
    embedding_dimensions: int = Field(default=1024, ge=1)
    embedding_batch_size: int = Field(default=16, ge=1, le=128)
    indexing_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    qdrant_collection: str = Field(
        default="paper_chunks_qwen3_0_6b_v1", pattern=r"^[a-zA-Z0-9_-]+$"
    )
    chunk_tokens: int = Field(default=700, ge=1, le=8000)
    chunk_overlap_tokens: int = Field(default=100, ge=0)
    ingestion_poll_seconds: float = Field(default=1.0, gt=0, le=60)
    retrieval_debug_enabled: bool = False

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> Self:
        if self.chunk_overlap_tokens >= self.chunk_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_TOKENS")
        return self

    health_timeout_seconds: float = Field(default=3.0, gt=0, le=30)
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""

    return Settings()
