"""
api/routes/query.py
====================
Legal query endpoint — the main RAG pipeline entry point.

POST /query  — Run a legal question through the full pipeline:
               hybrid retrieval → rerank → Claude → Shepardize
               Returns the answer, sources, and a citation health report.

GET  /query/history — Retrieve past queries from the audit log.
"""

import uuid
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from core.db import get_db_session
from core.generation.rag_service import rag_service
from core.agents.shepardizer import shepardizer
from core.models.db_models import QueryLog
from api.models.schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["Query"])
logger = structlog.get_logger(__name__)


# ──────────────────────────────────────────────────────────────────
# POST /query
# ──────────────────────────────────────────────────────────────────
@router.post("", response_model=QueryResponse)
async def legal_query(request: QueryRequest):
    """
    Submit a legal question to the LegalMind RAG pipeline.

    Pipeline:
      1. Semantic cache lookup
      2. Hybrid retrieval (vector + BM25 → RRF)
      3. Cohere cross-encoder reranking (top-20 → top-5)
      4. Claude generation with mandatory citations
      5. Shepardizer citation validation
      6. Compliance Auditor faithfulness check (async, non-blocking)

    Returns the answer with fully cited sources and citation health.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    logger.info("Query received", query_preview=request.query[:80])

    # ── Run full RAG pipeline ──────────────────────────────────────
    response = await rag_service.query(request)

    # ── Run Shepardizer (citation validator) ───────────────────────
    # We need the ranked_chunks to shepardize — retrieve them from the
    # response sources (they carry the same payload info).
    if not response.cache_hit and response.sources:
        try:
            from core.retrieval.reranker import RankedChunk

            # Reconstruct RankedChunk objects from the response sources
            ranked_for_shep = [
                RankedChunk(
                    qdrant_id=str(src.document_id),
                    document_id=str(src.document_id),
                    filename=src.filename,
                    text=src.text,
                    chunk_index=src.chunk_index,
                    relevance_score=src.relevance_score,
                    original_rank=i + 1,
                    payload={
                        "doc_type": src.doc_type,
                        "client_id": src.client_id,
                        "date_filed": (
                            src.date_filed.isoformat() if src.date_filed else None
                        ),
                    },
                )
                for i, src in enumerate(response.sources)
            ]

            shep_report = await shepardizer.shepardize(
                response_text=response.answer,
                ranked_chunks=ranked_for_shep,
            )

            # Attach the shepardization summary to the response metadata
            # by appending it to the answer (visible but clearly separated)
            if not shep_report.passed:
                response.answer += (
                    f"\n\n---\n⚠️ **Citation Audit**: {shep_report.summary}"
                )
            else:
                response.answer += (
                    f"\n\n---\n{shep_report.summary}"
                )

        except Exception as exc:
            logger.warning("Shepardizer failed — continuing", error=str(exc))

    logger.info(
        "Query complete",
        cache_hit=response.cache_hit,
        source_count=len(response.sources),
        latency_ms=response.latency_ms,
    )

    return response


# ──────────────────────────────────────────────────────────────────
# GET /query/history
# ──────────────────────────────────────────────────────────────────
@router.get("/history")
async def query_history(
    session_id: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Retrieve past query logs for a session or all sessions.
    Useful for the audit trail required by legal compliance.
    """
    stmt = select(QueryLog).order_by(desc(QueryLog.created_at)).limit(limit)

    if session_id:
        stmt = stmt.where(QueryLog.session_id == session_id)

    result = await db.execute(stmt)
    logs = result.scalars().all()

    return [
        {
            "id": str(log.id),
            "session_id": log.session_id,
            "query": log.query_text,
            "response_preview": (
                log.response_text[:200] + "..." if log.response_text else None
            ),
            "cache_hit": log.cache_hit,
            "latency_ms": log.latency_ms,
            "source_count": (
                len(log.source_doc_ids) if log.source_doc_ids else 0
            ),
            "created_at": log.created_at.isoformat(),
        }
        for log in logs
    ]
