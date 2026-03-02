"""
core/generation/rag_service.py
==============================
RAGService — the central Facade for the entire query pipeline.
"""

import time
from typing import Optional, Callable, Awaitable

import structlog

from core.config import get_settings
from core.retrieval.hybrid import hybrid_retriever
from core.retrieval.reranker import reranker, RankedChunk
from core.generation.llm import llm, GenerationResult
from api.models.schemas import (
    QueryRequest,
    QueryResponse,
    SourceChunk,
)

logger = structlog.get_logger(__name__)
settings = get_settings()

# Updated: hook now receives ranked_chunks so the compliance auditor
# can extract context_chunks without needing a separate argument.
ObserverHook = Callable[[QueryRequest, QueryResponse, list[RankedChunk]], Awaitable[None]]


class RAGService:
    """
    Facade over the full RAG pipeline.

    Design Patterns:
      - Facade   : Single interface over retrieve → rerank → generate
      - Observer : Post-generation hooks for evaluation agents
    """

    def __init__(self):
        self._post_generation_hooks: list[ObserverHook] = []

    def register_hook(self, hook: ObserverHook) -> None:
        self._post_generation_hooks.append(hook)
        logger.info("Observer hook registered", hook=hook.__name__)

    async def query(
        self,
        request: QueryRequest,
        cache_enabled: bool = True,
    ) -> QueryResponse:
        start = time.monotonic()
        log = logger.bind(
            query_preview=request.query[:60],
            session_id=request.session_id,
        )
        log.info("RAG pipeline started")

        # ── Stage 1: Semantic cache lookup ─────────────────────────
        if cache_enabled:
            cached = await self._check_cache(request.query)
            if cached:
                log.info("Cache hit — returning cached response")
                return cached

        # ── Stage 2: Hybrid retrieval ──────────────────────────────
        top_k = request.top_k or settings.retrieval_top_k

        log.info("Hybrid retrieval", top_k=top_k)
        hybrid_results = await hybrid_retriever.search(
            query=request.query,
            top_k=top_k,
            filter_client_id=request.filter_client_id,
            filter_doc_type=request.filter_doc_type,
            filter_date_from=(
                request.filter_date_from.isoformat()
                if request.filter_date_from else None
            ),
            filter_date_to=(
                request.filter_date_to.isoformat()
                if request.filter_date_to else None
            ),
        )

        if not hybrid_results:
            log.warning("No results from hybrid retrieval")
            return self._empty_response(request, start)

        # ── Stage 3: Cohere reranking ──────────────────────────────
        log.info("Reranking", candidates=len(hybrid_results))
        ranked_chunks = await reranker.rerank(
            query=request.query,
            chunks=hybrid_results,
            top_n=settings.rerank_top_n,
        )

        if not ranked_chunks:
            log.warning("Reranker returned no results")
            return self._empty_response(request, start)

        # ── Stage 4: Generation ────────────────────────────────────
        log.info("Generating answer", ranked_chunks=len(ranked_chunks))
        generation_result: GenerationResult = await llm.generate(
            query=request.query,
            ranked_chunks=ranked_chunks,
            session_id=request.session_id,
        )

        # ── Stage 5: Build response ────────────────────────────────
        total_latency_ms = int((time.monotonic() - start) * 1000)

        response = QueryResponse(
            query=request.query,
            answer=generation_result.answer,
            sources=_to_source_chunks(generation_result.cited_sources),
            cache_hit=False,
            latency_ms=total_latency_ms,
            session_id=request.session_id,
        )

        # ── Stage 6: Populate cache ────────────────────────────────
        if cache_enabled:
            await self._populate_cache(request.query, response)

        # ── Stage 7: Fire observer hooks (pass ranked_chunks) ──────
        for hook in self._post_generation_hooks:
            try:
                await hook(request, response, ranked_chunks)
            except Exception as exc:
                logger.error(
                    "Observer hook failed",
                    hook=hook.__name__,
                    error=str(exc),
                )

        log.info(
            "RAG pipeline complete",
            total_latency_ms=total_latency_ms,
            sources=len(response.sources),
            cache_hit=False,
        )

        return response

    async def _check_cache(self, query: str) -> Optional[QueryResponse]:
        try:
            from core.cache.semantic_cache import semantic_cache
            return await semantic_cache.get(query)
        except Exception as exc:
            logger.warning("Cache lookup failed — proceeding without cache", error=str(exc))
            return None

    async def _populate_cache(self, query: str, response: QueryResponse) -> None:
        try:
            from core.cache.semantic_cache import semantic_cache
            await semantic_cache.set(query, response)
        except Exception as exc:
            logger.warning("Cache store failed", error=str(exc))

    def _empty_response(self, request: QueryRequest, start: float) -> QueryResponse:
        return QueryResponse(
            query=request.query,
            answer=(
                "I don't have sufficient information in the provided documents "
                "to answer this question accurately. No relevant documents were "
                "found matching your query and the applied filters."
            ),
            sources=[],
            cache_hit=False,
            latency_ms=int((time.monotonic() - start) * 1000),
            session_id=request.session_id,
        )


def _to_source_chunks(cited_sources) -> list[SourceChunk]:
    import uuid as _uuid
    result = []
    for src in cited_sources:
        try:
            doc_id = _uuid.UUID(src.document_id)
        except (ValueError, AttributeError):
            continue
        result.append(SourceChunk(
            document_id=doc_id,
            filename=src.filename,
            chunk_index=src.chunk_index,
            text=src.text,
            relevance_score=src.relevance_score,
            doc_type=src.doc_type,
            client_id=src.client_id,
            date_filed=src.date_filed,
        ))
    return result


rag_service = RAGService()
