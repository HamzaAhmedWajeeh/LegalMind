"""
core/config.py
==============
Centralised application configuration using Pydantic BaseSettings.
"""

from functools import lru_cache
from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # App
    # ------------------------------------------------------------------
    environment: Literal["development", "staging", "production"] = "development"
    log_level: Literal["debug", "info", "warning", "error"] = "info"

    # ------------------------------------------------------------------
    # Google Gemini
    # ------------------------------------------------------------------
    gemini_api_key: str = Field(..., description="Google Gemini API key")
    gemini_model: str = Field(
        default="gemini-3-flash-preview",
        description="Gemini model string",
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
    postgres_url: str = Field(..., description="Async SQLAlchemy connection string")

    # ------------------------------------------------------------------
    # Redis
    # ------------------------------------------------------------------
    redis_url: str = "redis://redis:6379/0"
    semantic_cache_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    cache_ttl_seconds: int = Field(default=3600, gt=0)

    # ------------------------------------------------------------------
    # Ingestion Pipeline
    # ------------------------------------------------------------------
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=51, ge=0)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    retrieval_top_k: int = Field(default=20, gt=0)
    rerank_top_n: int = Field(default=5, gt=0)

    # ------------------------------------------------------------------
    # Evaluation — DeepEval (uses Gemini as judge)
    # ------------------------------------------------------------------
    min_faithfulness_score: float = Field(default=0.9, ge=0.0, le=1.0)
    golden_dataset_size: int = Field(default=50, gt=0)
    adversarial_batch_size: int = Field(default=6, gt=0)
    adversarial_max_chunks_per_batch: int = Field(default=5, gt=0)
    adversarial_max_output_tokens: int = Field(default=1536, gt=0)

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
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError(f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})")
        return v

    @field_validator("rerank_top_n")
    @classmethod
    def rerank_less_than_topk(cls, v: int, info) -> int:
        top_k = info.data.get("retrieval_top_k", 20)
        if v > top_k:
            raise ValueError(f"rerank_top_n ({v}) must be <= retrieval_top_k ({top_k})")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
