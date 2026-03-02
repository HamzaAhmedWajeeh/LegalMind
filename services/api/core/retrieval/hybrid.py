"""
core/retrieval/hybrid.py
========================
Hybrid retrieval — merges vector search and BM25 results using
Reciprocal Rank Fusion (RRF).

Why RRF?
  Simply averaging scores from two systems with different scales
  (cosine similarity 0-1 vs BM25 0-∞) is unreliable.
  RRF uses rank position instead of raw scores, making it
  robust to score scale differences.

  RRF formula for a document d across rankers:
    RRF_score(d) = Σ  1 / (k + rank_i(d))
                   i
  where k=60 is a constant that dampens the impact of very high ranks.
  Documents appearing near the top of BOTH result lists get the
  highest fused scores.

Design Pattern: Facade Pattern
  HybridRetriever hides the complexity of running two searches,
  deduplicating results, and fusing scores behind a single
  `search()` method. Callers don't need to know how many
  underlying retrievers exist.
"""

from typing import Optional

import structlog

from core.retrieval.vector_store import RetrievedChunk, vector_store
from core.retrieval.bm25 import bm25_retriever

logger = structlog.get_logger(__name__)

# RRF constant — standard value from the original paper (Cormack et al., 2009)
_RRF_K = 60

# How many results to fetch from each retriever before fusion.
# We fetch more than top_k from each so fusion has enough candidates.
_FETCH_MULTIPLIER = 2


# ──────────────────────────────────────────────────────────────────
# HybridRetriever — Facade
# ──────────────────────────────────────────────────────────────────
class HybridRetriever:
    """
    Runs vector search and BM25 in parallel, then fuses results
    using Reciprocal Rank Fusion.

    The fused list is what gets passed to the Cohere reranker.
    """

    async def search(
        self,
        query: str,
        top_k: int,
        filter_client_id: Optional[str] = None,
        filter_doc_type: Optional[str] = None,
        filter_date_from: Optional[str] = None,
        filter_date_to: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Hybrid search: vector + BM25 → RRF fusion.

        Args:
            query           : User's legal question
            top_k           : Final number of candidates after fusion
                              (before reranking — typically 20)
            filter_client_id: Metadata pre-filter
            filter_doc_type : Metadata pre-filter
            filter_date_from: Metadata pre-filter
            filter_date_to  : Metadata pre-filter

        Returns:
            List of RetrievedChunk ordered by RRF score descending,
            length = top_k (or fewer if not enough results exist)
        """
        fetch_k = top_k * _FETCH_MULTIPLIER

        # ── Run both searches (concurrently) ──────────────────────
        import asyncio
        vector_results, bm25_results = await asyncio.gather(
            vector_store.search(
                query=query,
                top_k=fetch_k,
                filter_client_id=filter_client_id,
                filter_doc_type=filter_doc_type,
                filter_date_from=filter_date_from,
                filter_date_to=filter_date_to,
            ),
            bm25_retriever.search(
                query=query,
                top_k=fetch_k,
                filter_client_id=filter_client_id,
                filter_doc_type=filter_doc_type,
            ),
        )

        logger.info(
            "Hybrid search: individual results",
            vector_count=len(vector_results),
            bm25_count=len(bm25_results),
        )

        # ── Fuse via RRF ──────────────────────────────────────────
        fused = _reciprocal_rank_fusion(
            result_lists=[vector_results, bm25_results],
            k=_RRF_K,
        )

        final = fused[:top_k]

        logger.info(
            "Hybrid fusion complete",
            fused_total=len(fused),
            returned=len(final),
            top_score=final[0].score if final else None,
        )

        return final


# ──────────────────────────────────────────────────────────────────
# RRF Implementation
# ──────────────────────────────────────────────────────────────────
def _reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]],
    k: int = 60,
) -> list[RetrievedChunk]:
    """
    Reciprocal Rank Fusion across multiple ranked result lists.

    For each unique chunk (identified by qdrant_id), accumulates:
        score += 1 / (k + rank)
    across all lists it appears in.

    Chunks appearing in both vector AND BM25 results receive
    contributions from both, naturally boosting high-confidence results.

    Args:
        result_lists : List of ranked result lists (one per retriever)
        k            : RRF smoothing constant (default 60)

    Returns:
        Deduplicated list of RetrievedChunk sorted by RRF score descending.
        The `.score` field is replaced with the RRF score.
    """
    # Map from qdrant_id → (rrf_score, chunk_object)
    fused_scores: dict[str, float] = {}
    chunk_registry: dict[str, RetrievedChunk] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list, start=1):
            uid = chunk.qdrant_id
            rrf_contribution = 1.0 / (k + rank)

            if uid in fused_scores:
                fused_scores[uid] += rrf_contribution
            else:
                fused_scores[uid] = rrf_contribution
                chunk_registry[uid] = chunk

    # Sort by RRF score descending
    sorted_ids = sorted(fused_scores, key=lambda x: fused_scores[x], reverse=True)

    # Return RetrievedChunk objects with the RRF score as the `.score`
    fused_chunks: list[RetrievedChunk] = []
    for uid in sorted_ids:
        chunk = chunk_registry[uid]
        # Replace the raw retriever score with the fused RRF score
        fused_chunk = RetrievedChunk(
            qdrant_id=chunk.qdrant_id,
            document_id=chunk.document_id,
            filename=chunk.filename,
            text=chunk.text,
            chunk_index=chunk.chunk_index,
            score=fused_scores[uid],
            payload=chunk.payload,
        )
        fused_chunks.append(fused_chunk)

    return fused_chunks


# ──────────────────────────────────────────────────────────────────
# Module-level singleton
# ──────────────────────────────────────────────────────────────────
hybrid_retriever = HybridRetriever()
