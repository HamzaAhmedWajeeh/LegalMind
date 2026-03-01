"""
core/config.py
==============
Centralised application configuration using Pydantic BaseSettings.

Design Pattern: Singleton via @lru_cache — the config object is instantiated
once and reused everywhere. This avoids repeated disk I/O and makes mocking
in tests trivial (just override get_settings()).

Usage:
    from core.config import get_settings
    settings = get_settings()
    print(settings.anthropic_model)
"""

from functools import lru_cache
from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All application settings sourced from environment variables / .env file.
    Pydantic validates types and raises clear errors on misconfiguration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,   # ANTHROPIC_API_KEY == anthropic_api_key
        extra="ignore",         # Ignore unknown env vars gracefully
    )

    # ------------------------------------------------------------------
    # App
    # ------------------------------------------------------------------
    environment: Literal["development", "staging", "production"] = "development"
    log_level: Literal["debug", "info", "warning", "error"] = "info"

    # ------------------------------------------------------------------
    # Anthropic — Claude
    # ------------------------------------------------------------------
    anthropic_api_key: str = Field(..., description="Anthropic API key")
    anthropic_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Claude model string",
    )

    # ------------------------------------------------------------------
    # Cohere — Reranker
    # ------------------------------------------------------------------
    cohere_api_key: str = Field(..., description="Cohere API key")
    cohere_rerank_model: str = Field(
        default="rerank-english-v3.0",
        description="Cohere rerank model",
    )

    # ------------------------------------------------------------------
    # Qdrant — Vector DB
    # ------------------------------------------------------------------
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "legalmind_docs"

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    # ------------------------------------------------------------------
    # PostgreSQL
    # ------------------------------------------------------------------
    postgres_url: str = Field(
        ...,
        description="Async SQLAlchemy connection string (asyncpg)",
    )

    # ------------------------------------------------------------------
    # Redis
    # ------------------------------------------------------------------
    redis_url: str = "redis://redis:6379/0"
    semantic_cache_threshold: float = Field(
        default=0.92,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for cache hits",
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        gt=0,
        description="Cache entry TTL in seconds",
    )

    # ------------------------------------------------------------------
    # Ingestion Pipeline
    # ------------------------------------------------------------------
    chunk_size: int = Field(
        default=512,
        gt=0,
        description="Chunk size in tokens",
    )
    chunk_overlap: int = Field(
        default=51,
        ge=0,
        description="Overlap between consecutive chunks in tokens (~10%)",
    )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    retrieval_top_k: int = Field(
        default=20,
        gt=0,
        description="Number of chunks to retrieve before reranking",
    )
    rerank_top_n: int = Field(
        default=5,
        gt=0,
        description="Number of chunks to pass to LLM after reranking",
    )

    # ------------------------------------------------------------------
    # Evaluation — DeepEval
    # ------------------------------------------------------------------
    # DeepEval uses ANTHROPIC_API_KEY to run Claude as the judge.
    # No separate DeepEval API key is required.
    min_faithfulness_score: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Minimum faithfulness score to pass CI/CD gate",
    )
    golden_dataset_size: int = Field(
        default=50,
        gt=0,
        description="Number of synthetic QA pairs to generate",
    )

    # ------------------------------------------------------------------
    # API pagination
    # ------------------------------------------------------------------
    max_results_per_page: int = Field(default=50, gt=0)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_chunk(cls, v: int, info) -> int:
        """Overlap must be less than chunk size."""
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})"
            )
        return v

    @field_validator("rerank_top_n")
    @classmethod
    def rerank_less_than_topk(cls, v: int, info) -> int:
        """Can't rerank to more chunks than we retrieved."""
        top_k = info.data.get("retrieval_top_k", 20)
        if v > top_k:
            raise ValueError(
                f"rerank_top_n ({v}) must be <= retrieval_top_k ({top_k})"
            )
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns the cached Settings singleton.
    Use this everywhere instead of instantiating Settings() directly.
    In tests, call get_settings.cache_clear() to reset between test cases.
    """
    return Settings()
