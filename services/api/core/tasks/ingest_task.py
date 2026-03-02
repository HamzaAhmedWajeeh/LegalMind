"""
core/tasks/ingest_task.py
=========================
Celery task that orchestrates the full document ingestion pipeline.

Pipeline stages (in order):
  1. parse     — Extract raw text from PDF/DOCX/TXT
  2. chunk     — Split into overlapping token windows
  3. enrich    — Attach metadata to each chunk
  4. embed     — Generate vector embeddings (via sentence-transformers)
  5. store     — Write vectors to Qdrant + rows to Postgres
  6. update    — Mark document status as 'indexed' in Postgres

Design Pattern: Pipeline / Chain of Responsibility
  Each stage receives the output of the previous stage.
  Failures at any stage roll back the Postgres row to 'failed'
  and the Qdrant points are not written (or are cleaned up).

This task is triggered by the POST /ingest route (Step 8).
"""

import asyncio
import uuid
from typing import Optional

import structlog
from celery import Task

from core.tasks.celery_app import celery_app
from core.ingestion.parser import parse_document
from core.ingestion.chunker import get_chunker
from core.ingestion.enricher import enricher

logger = structlog.get_logger(__name__)


# ──────────────────────────────────────────────────────────────────
# Helper: run async code from a sync Celery task
# ──────────────────────────────────────────────────────────────────
def _run_async(coro):
    """Execute an async coroutine from synchronous Celery context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ──────────────────────────────────────────────────────────────────
# Celery Task
# ──────────────────────────────────────────────────────────────────
@celery_app.task(
    bind=True,
    name="legalmind.ingest_document",
    max_retries=3,
    default_retry_delay=10,
    acks_late=True,
)
def ingest_document_task(
    self: Task,
    document_id: str,
    filename: str,
    file_bytes_hex: str,          # bytes serialised as hex for JSON transport
    doc_type: Optional[str] = None,
    client_id: Optional[str] = None,
    matter_id: Optional[str] = None,
    date_filed: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
    chunking_strategy: str = "recursive",
) -> dict:
    """
    Full document ingestion pipeline.

    Args:
        document_id       : UUID string of the pre-created Postgres row
        filename          : Original filename
        file_bytes_hex    : Raw file bytes encoded as hex string
        doc_type          : Document type for metadata filtering
        client_id         : Client identifier
        matter_id         : Matter/case identifier
        date_filed        : ISO date string
        extra_metadata    : Additional metadata key-value pairs
        chunking_strategy : 'recursive' | 'semantic'

    Returns:
        dict with document_id, chunk_count, and status
    """
    doc_uuid = uuid.UUID(document_id)
    log = logger.bind(document_id=document_id, filename=filename, task_id=self.request.id)
    log.info("Ingestion task started")

    try:
        # ── Stage 1: Mark document as processing ──────────────────
        _run_async(_update_document_status(doc_uuid, "processing"))

        # ── Stage 2: Parse ────────────────────────────────────────
        log.info("Stage 1/5: Parsing document")
        file_bytes = bytes.fromhex(file_bytes_hex)
        parsed_doc = parse_document(file_bytes, filename)

        # ── Stage 3: Chunk ────────────────────────────────────────
        log.info("Stage 2/5: Chunking", strategy=chunking_strategy)
        chunker = get_chunker(strategy_name=chunking_strategy)
        chunks = chunker.chunk(parsed_doc)

        # ── Stage 4: Enrich ───────────────────────────────────────
        log.info("Stage 3/5: Enriching metadata", chunk_count=len(chunks))
        enriched_chunks = enricher.enrich(
            chunks=chunks,
            parsed_doc=parsed_doc,
            document_id=doc_uuid,
            doc_type=doc_type,
            client_id=client_id,
            matter_id=matter_id,
            date_filed=date_filed,
            extra_metadata=extra_metadata,
        )

        # ── Stage 5: Embed + Store ────────────────────────────────
        log.info("Stage 4/5: Embedding and storing vectors")
        _run_async(_embed_and_store(enriched_chunks, doc_uuid))

        # ── Stage 6: Mark document as indexed ─────────────────────
        log.info("Stage 5/5: Finalising")
        _run_async(_update_document_status(
            doc_uuid, "indexed", chunk_count=len(enriched_chunks)
        ))

        log.info(
            "Ingestion complete",
            chunk_count=len(enriched_chunks),
            ocr_used=parsed_doc.ocr_used,
        )

        return {
            "document_id": document_id,
            "chunk_count": len(enriched_chunks),
            "status": "indexed",
            "ocr_used": parsed_doc.ocr_used,
        }

    except Exception as exc:
        log.exception("Ingestion failed", error=str(exc))

        # Mark document as failed so the UI can surface the error
        try:
            _run_async(_update_document_status(doc_uuid, "failed"))
        except Exception:
            pass

        # Retry with exponential back-off
        raise self.retry(exc=exc, countdown=10 * (self.request.retries + 1))


# ──────────────────────────────────────────────────────────────────
# Async helpers called via _run_async
# ──────────────────────────────────────────────────────────────────
async def _update_document_status(
    document_id: uuid.UUID,
    status: str,
    chunk_count: Optional[int] = None,
) -> None:
    """Update the document status and optional chunk_count in Postgres."""
    from sqlalchemy import update
    from core.db import get_db_context
    from core.models.db_models import Document

    async with get_db_context() as db:
        values: dict = {"status": status}
        if chunk_count is not None:
            values["chunk_count"] = chunk_count

        await db.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(**values)
        )


async def _embed_and_store(enriched_chunks, document_id: uuid.UUID) -> None:
    """
    Generate embeddings for all chunks and write to Qdrant + Postgres.
    """
    from core.retrieval.vector_store import vector_store
    from core.db import get_db_context
    from core.models.db_models import Chunk

    # 1) Embed and upsert vectors so semantic retrieval has fresh points.
    await vector_store.upsert_chunks(enriched_chunks)

    # 2) Persist chunk rows/metadata in Postgres.
    async with get_db_context() as db:
        for ec in enriched_chunks:
            chunk_row = Chunk(
                id=uuid.UUID(ec.qdrant_id),
                document_id=document_id,
                chunk_index=ec.chunk_index,
                text=ec.text,
                token_count=ec.token_count,
                qdrant_id=ec.qdrant_id,
                metadata_=ec.payload,
            )
            db.add(chunk_row)

    logger.info(
        "Chunks written to Qdrant and Postgres",
        document_id=str(document_id),
        count=len(enriched_chunks),
    )
