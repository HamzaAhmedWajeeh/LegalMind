"""
core/retrieval/reranker.py
==========================
Cohere Cross-Encoder reranker — refines the top-K hybrid results
down to the top-N most relevant chunks for the LLM context window.

Why rerank?
  Vector search + BM25 retrieve broadly relevant chunks, but their
  ranking can be noisy. A Cross-Encoder model reads the query AND
  each candidate chunk together (not separately like bi-encoders),
  producing a much more accurate relevance score.

  Spec requirement:
    - Input  : top 20 retrieved chunks (from hybrid.py)
    - Output : top 5 most relevant chunks (passed to Claude)

  Cohere Rerank v3 is used as specified. It's accessible via API,
  requires no local GPU, and is state-of-the-art for retrieval tasks.

Design Pattern: Facade Pattern (consistent with hybrid.py)
  Reranker wraps the Cohere API behind a clean interface so the
  generation layer only calls `rerank(query, chunks)` and receives
  an ordered list back — it has no knowledge of Cohere internals.
"""

from dataclasses import dataclass, field
from typing import Optional

import cohere
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.config import get_settings
from core.retrieval.vector_store import RetrievedChunk

logger = structlog.get_logger(__name__)
settings = get_settings()


# ──────────────────────────────────────────────────────────────────
# Reranked result
# ──────────────────────────────────────────────────────────────────
@dataclass
class RankedChunk:
    """
    A chunk after cross-encoder reranking.
    Extends RetrievedChunk with the Cohere relevance score
    and the original rank before reranking (for debugging/logging).
    """
    qdrant_id: str
    document_id: str
    filename: str
    text: str
    chunk_index: int
    relevance_score: float            # Cohere's score (0.0 – 1.0)
    original_rank: int                # Rank in hybrid results before reranking
    payload: dict = field(default_factory=dict)

    @property
    def doc_type(self) -> Optional[str]:
        return self.payload.get("doc_type")

    @property
    def client_id(self) -> Optional[str]:
        return self.payload.get("client_id")

    @property
    def date_filed(self) -> Optional[str]:
        return self.payload.get("date_filed")

    @property
    def page_number(self) -> Optional[int]:
        return self.payload.get("page_number")


# ──────────────────────────────────────────────────────────────────
# CohereReranker
# ──────────────────────────────────────────────────────────────────
class CohereReranker:
    """
    Wraps the Cohere Rerank v3 API.

    Lazy-initialises the Cohere client on first use.
    Uses tenacity for automatic retries on transient API errors.
    """

    def __init__(self):
        self._client: Optional[cohere.Client] = None

    def _get_client(self) -> cohere.Client:
        if self._client is None:
            self._client = cohere.Client(api_key=settings.cohere_api_key)
        return self._client

    # ── Rerank ─────────────────────────────────────────────────────
    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: Optional[int] = None,
    ) -> list[RankedChunk]:
        """
        Rerank retrieved chunks using Cohere's cross-encoder model.

        Args:
            query  : The original user query
            chunks : Hybrid search results (typically top-20)
            top_n  : How many to return after reranking (default from config)

        Returns:
            List of RankedChunk ordered by Cohere relevance score descending,
            length = top_n
        """
        top_n = top_n or settings.rerank_top_n

        if not chunks:
            logger.warning("Reranker called with empty chunks list")
            return []

        # Truncate input if more than top_n chunks provided
        # (Cohere supports up to 1000 documents but we cap at retrieval_top_k)
        candidates = chunks[: settings.retrieval_top_k]

        logger.info(
            "Reranking",
            query_preview=query[:60],
            input_count=len(candidates),
            top_n=top_n,
            model=settings.cohere_rerank_model,
        )

        client = self._get_client()

        # Cohere expects a list of strings — we pass the chunk text
        response = client.rerank(
            model=settings.cohere_rerank_model,
            query=query,
            documents=[chunk.text for chunk in candidates],
            top_n=top_n,
            return_documents=False,  # We already have the documents; just want scores
        )

        # Map Cohere results back to RankedChunk objects
        ranked: list[RankedChunk] = []
        for result in response.results:
            original_chunk = candidates[result.index]
            ranked.append(
                RankedChunk(
                    qdrant_id=original_chunk.qdrant_id,
                    document_id=original_chunk.document_id,
                    filename=original_chunk.filename,
                    text=original_chunk.text,
                    chunk_index=original_chunk.chunk_index,
                    relevance_score=result.relevance_score,
                    original_rank=result.index + 1,
                    payload=original_chunk.payload,
                )
            )

        logger.info(
            "Reranking complete",
            returned=len(ranked),
            top_score=ranked[0].relevance_score if ranked else None,
            bottom_score=ranked[-1].relevance_score if ranked else None,
        )

        return ranked


# ──────────────────────────────────────────────────────────────────
# Module-level singleton
# ──────────────────────────────────────────────────────────────────
reranker = CohereReranker()
