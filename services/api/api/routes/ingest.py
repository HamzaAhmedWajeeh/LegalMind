"""
api/routes/ingest.py
====================
Document ingestion endpoints.

POST /ingest/upload  — Accept a file upload, create a Document row,
                       dispatch a Celery task, return task_id immediately.
GET  /ingest/status/{task_id} — Poll Celery task status.
GET  /ingest/documents        — List all ingested documents with filters.
DELETE /ingest/documents/{id} — Remove a document and its vectors.
"""

import hashlib
import uuid
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.db import get_db_session
from core.models.db_models import Document, Chunk
from core.tasks.ingest_task import ingest_document_task
from api.models.schemas import (
    DocumentOut,
    DocumentFilter,
    IngestionTaskOut,
    IngestionStatusOut,
    PaginatedResponse,
)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])
logger = structlog.get_logger(__name__)
settings = get_settings()

# Max file size: 50 MB
_MAX_FILE_SIZE = 50 * 1024 * 1024
_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


# ──────────────────────────────────────────────────────────────────
# POST /ingest/upload
# ──────────────────────────────────────────────────────────────────
@router.post("/upload", response_model=IngestionTaskOut, status_code=202)
async def upload_document(
    file: UploadFile = File(..., description="PDF, DOCX, or TXT file to ingest"),
    doc_type: Optional[str] = Form(
        default=None,
        description="contract | case_file | brief | memo | other",
    ),
    client_id: Optional[str] = Form(default=None),
    matter_id: Optional[str] = Form(default=None),
    date_filed: Optional[str] = Form(
        default=None,
        description="ISO date string e.g. 2024-03-15",
    ),
    chunking_strategy: str = Form(
        default="recursive",
        description="recursive | semantic",
    ),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Upload a legal document for ingestion into LegalMind.

    Returns a task_id immediately (HTTP 202 Accepted).
    Poll GET /ingest/status/{task_id} for progress.
    """
    # ── Validate file ──────────────────────────────────────────────
    filename = file.filename or "unnamed"
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {_ALLOWED_EXTENSIONS}",
        )

    file_bytes = await file.read()

    if len(file_bytes) > _MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(file_bytes)} bytes). Max: {_MAX_FILE_SIZE} bytes",
        )

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # ── Deduplication check ────────────────────────────────────────
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    existing = await db.execute(
        select(Document).where(Document.file_hash == file_hash)
    )
    existing_doc = existing.scalar_one_or_none()

    if existing_doc:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Document already ingested (id={existing_doc.id}, "
                f"status={existing_doc.status}). "
                "Use DELETE first if you want to re-ingest."
            ),
        )

    # ── Create Document row (status=pending) ───────────────────────
    doc_id = uuid.uuid4()
    document = Document(
        id=doc_id,
        filename=filename,
        file_hash=file_hash,
        doc_type=doc_type,
        client_id=client_id,
        matter_id=matter_id,
        date_filed=date_filed,
        status="pending",
    )
    db.add(document)
    await db.flush()   # Write to DB before dispatching Celery task

    # ── Dispatch Celery task ───────────────────────────────────────
    task = ingest_document_task.delay(
        document_id=str(doc_id),
        filename=filename,
        file_bytes_hex=file_bytes.hex(),
        doc_type=doc_type,
        client_id=client_id,
        matter_id=matter_id,
        date_filed=date_filed,
        chunking_strategy=chunking_strategy,
    )

    logger.info(
        "Ingestion task dispatched",
        document_id=str(doc_id),
        task_id=task.id,
        filename=filename,
        size_bytes=len(file_bytes),
    )

    return IngestionTaskOut(
        task_id=task.id,
        document_id=doc_id,
        filename=filename,
        status="pending",
        message="Document accepted. Use /ingest/status/{task_id} to track progress.",
    )


# ──────────────────────────────────────────────────────────────────
# GET /ingest/status/{task_id}
# ──────────────────────────────────────────────────────────────────
@router.get("/status/{task_id}", response_model=IngestionStatusOut)
async def get_ingestion_status(
    task_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """Poll the status of an ingestion task by Celery task ID."""
    from celery.result import AsyncResult
    from core.tasks.celery_app import celery_app

    result = AsyncResult(task_id, app=celery_app)

    # Map Celery states to our status vocabulary
    celery_to_status = {
        "PENDING":  "pending",
        "STARTED":  "processing",
        "SUCCESS":  "indexed",
        "FAILURE":  "failed",
        "RETRY":    "processing",
        "REVOKED":  "failed",
    }
    status = celery_to_status.get(result.state, "pending")

    # If the task succeeded, get the document_id from the result
    task_result = result.result if result.state == "SUCCESS" else {}
    document_id_str = (
        task_result.get("document_id") if isinstance(task_result, dict) else None
    )

    # Fetch chunk count from DB if we have the document_id
    chunk_count = 0
    error_msg = None
    doc_uuid = None

    if document_id_str:
        try:
            doc_uuid = uuid.UUID(document_id_str)
            doc_result = await db.execute(
                select(Document).where(Document.id == doc_uuid)
            )
            doc = doc_result.scalar_one_or_none()
            if doc:
                chunk_count = doc.chunk_count
        except Exception:
            pass

    if result.state == "FAILURE":
        error_msg = str(result.result) if result.result else "Unknown error"

    return IngestionStatusOut(
        task_id=task_id,
        document_id=doc_uuid or uuid.uuid4(),
        status=status,
        chunk_count=chunk_count,
        error=error_msg,
    )


# ──────────────────────────────────────────────────────────────────
# GET /ingest/documents
# ──────────────────────────────────────────────────────────────────
@router.get("/documents", response_model=PaginatedResponse)
async def list_documents(
    doc_type: Optional[str] = None,
    client_id: Optional[str] = None,
    status: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db_session),
):
    """List all ingested documents with optional filters and pagination."""
    from sqlalchemy import func

    stmt = select(Document)

    if doc_type:
        stmt = stmt.where(Document.doc_type == doc_type)
    if client_id:
        stmt = stmt.where(Document.client_id == client_id)
    if status:
        stmt = stmt.where(Document.status == status)

    # Count total
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_result = await db.execute(count_stmt)
    total = total_result.scalar() or 0

    # Paginate
    offset = (page - 1) * page_size
    stmt = stmt.order_by(Document.ingested_at.desc()).offset(offset).limit(page_size)
    result = await db.execute(stmt)
    documents = result.scalars().all()

    import math
    return PaginatedResponse(
        items=[DocumentOut.model_validate(d) for d in documents],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=math.ceil(total / page_size) if total > 0 else 0,
    )


# ──────────────────────────────────────────────────────────────────
# DELETE /ingest/documents/{document_id}
# ──────────────────────────────────────────────────────────────────
@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Delete a document, its chunks from Postgres, and its vectors from Qdrant.
    Also invalidates the semantic cache since responses may reference this doc.
    """
    # Verify document exists
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete vectors from Qdrant
    try:
        from core.retrieval.vector_store import vector_store
        await vector_store.delete_document_chunks(str(document_id))
    except Exception as exc:
        logger.warning("Qdrant deletion failed", error=str(exc))

    # Delete chunks and document from Postgres
    await db.execute(delete(Chunk).where(Chunk.document_id == document_id))
    await db.execute(delete(Document).where(Document.id == document_id))

    # Invalidate semantic cache
    try:
        from core.cache.semantic_cache import semantic_cache
        await semantic_cache.invalidate_all()
    except Exception as exc:
        logger.warning("Cache invalidation failed", error=str(exc))

    # Rebuild BM25 index
    try:
        from core.retrieval.bm25 import bm25_retriever
        await bm25_retriever.invalidate()
    except Exception as exc:
        logger.warning("BM25 rebuild failed", error=str(exc))

    logger.info("Document deleted", document_id=str(document_id))
