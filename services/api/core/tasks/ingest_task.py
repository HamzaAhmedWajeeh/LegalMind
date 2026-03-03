"""
core/tasks/ingest_task.py
=========================
Celery task that orchestrates the full document ingestion pipeline.

Pipeline stages (in order):
  1. parse     - Extract raw text from PDF/DOCX/TXT
  2. chunk     - Split into overlapping token windows
  3. enrich    - Attach metadata to each chunk
  4. embed     - Generate vector embeddings
  5. store     - Write vectors to Qdrant + rows to Postgres
  6. update    - Mark document status as 'indexed' in Postgres

Design Pattern: Pipeline / Chain of Responsibility
"""
import asyncio
import time
import uuid
from typing import Optional

import structlog
from celery import Task

from core.tasks.celery_app import celery_app
from core.ingestion.parser import parse_document
from core.ingestion.chunker import get_chunker
from core.ingestion.enricher import enricher

logger = structlog.get_logger(__name__)


def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


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
    file_bytes_hex: str,
    doc_type: Optional[str] = None,
    client_id: Optional[str] = None,
    matter_id: Optional[str] = None,
    date_filed: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
    chunking_strategy: str = "recursive",
) -> dict:
    doc_uuid = uuid.UUID(document_id)
    log = logger.bind(document_id=document_id, filename=filename, task_id=self.request.id)
    log.info("Ingestion task started")

    try:
        # Wait for the document row to be committed.
        # The API inserts the document row then immediately dispatches this task.
        # Because the worker runs in a separate process, it can arrive before the
        # API transaction is committed -- causing FK violations when chunks are
        # inserted. We poll for up to 10 seconds to resolve the race condition.
        _run_async(_wait_for_document(doc_uuid, timeout_seconds=10))

        _run_async(_update_document_status(doc_uuid, "processing"))

        log.info("Stage 1/5: Parsing document")
        file_bytes = bytes.fromhex(file_bytes_hex)
        parsed_doc = parse_document(file_bytes, filename)

        log.info("Stage 2/5: Chunking", strategy=chunking_strategy)
        chunker = get_chunker(strategy_name=chunking_strategy)
        chunks = chunker.chunk(parsed_doc)

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

        log.info("Stage 4/5: Embedding and storing vectors")
        _run_async(_embed_and_store(enriched_chunks, doc_uuid))

        log.info("Stage 5/5: Finalising")
        _run_async(_update_document_status(
            doc_uuid, "indexed", chunk_count=len(enriched_chunks)
        ))

        # Rebuild BM25 index so new document is immediately searchable
        try:
            from core.retrieval.bm25 import bm25_retriever
            _run_async(bm25_retriever.build_index())
            log.info("BM25 index rebuilt after ingestion")
        except Exception as bm25_exc:
            log.warning("BM25 rebuild failed (non-fatal)", error=str(bm25_exc))

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
        try:
            _run_async(_update_document_status(doc_uuid, "failed"))
        except Exception:
            pass
        raise self.retry(exc=exc, countdown=10 * (self.request.retries + 1))


async def _wait_for_document(
    document_id: uuid.UUID,
    timeout_seconds: int = 10,
) -> None:
    """
    Poll until the document row is visible in Postgres.
    Resolves the race condition where the Celery worker starts before
    the API has committed the document INSERT transaction.
    """
    from sqlalchemy import select
    from core.db import get_db_context
    from core.models.db_models import Document

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        async with get_db_context() as db:
            result = await db.execute(
                select(Document.id).where(Document.id == document_id)
            )
            if result.scalar_one_or_none() is not None:
                return  # Row is visible -- safe to proceed
        await asyncio.sleep(0.5)

    raise RuntimeError(
        f"Document {document_id} not found in Postgres after {timeout_seconds}s. "
        "The API did not commit the INSERT before dispatching the task."
    )


async def _update_document_status(
    document_id: uuid.UUID,
    status: str,
    chunk_count: Optional[int] = None,
) -> None:
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
    Write embeddings to Qdrant then chunk metadata to Postgres.
    Document row is guaranteed to exist at this point.
    """
    from core.retrieval.vector_store import vector_store
    from core.db import get_db_context
    from core.models.db_models import Chunk

    # Qdrant upsert is idempotent -- safe to retry
    await vector_store.upsert_chunks(enriched_chunks)

    # Postgres chunk rows -- FK to documents is safe now
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
