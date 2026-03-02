"""
api/models/schemas.py
=====================
Pydantic v2 schemas for all API request and response contracts.

Design Pattern: Separation of Concerns.
- DB models (db_models.py) handle persistence.
- These schemas handle HTTP serialization/validation.
- Routes never expose ORM models directly to the client.

Naming convention:
  <Entity>Create  — request body for POST (creation)
  <Entity>Update  — request body for PATCH (partial update)
  <Entity>Out     — response body (what the client receives)
  <Entity>Filter  — query parameters for filtering/searching
"""

import uuid
from datetime import date, datetime
from typing import Any, Optional
from pydantic import BaseModel, Field, ConfigDict


# ──────────────────────────────────────────────────────────────────
# Shared base — enables ORM mode for all response models
# ──────────────────────────────────────────────────────────────────
class ORMBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ══════════════════════════════════════════════════════════════════
# DOCUMENT schemas
# ══════════════════════════════════════════════════════════════════

class DocumentCreate(BaseModel):
    """
    Metadata provided when uploading a new document.
    The file itself is sent as multipart/form-data separately.
    """
    doc_type: Optional[str] = Field(
        default=None,
        description="contract | case_file | brief | memo | other",
        examples=["contract"],
    )
    client_id: Optional[str] = Field(
        default=None,
        description="Client identifier for metadata filtering",
        examples=["CLIENT-001"],
    )
    matter_id: Optional[str] = Field(
        default=None,
        description="Matter/case identifier",
        examples=["MATTER-2024-042"],
    )
    date_filed: Optional[date] = Field(
        default=None,
        description="Date the document was filed or signed",
        examples=["2024-03-15"],
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional key-value metadata",
    )


class DocumentOut(ORMBase):
    """Full document record returned to the client."""
    id: uuid.UUID
    filename: str
    doc_type: Optional[str]
    client_id: Optional[str]
    matter_id: Optional[str]
    date_filed: Optional[date]
    ingested_at: datetime
    status: str
    chunk_count: int

    model_config = ConfigDict(from_attributes=True)


class DocumentFilter(BaseModel):
    """Query parameters for filtering document listings."""
    doc_type: Optional[str] = None
    client_id: Optional[str] = None
    matter_id: Optional[str] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    status: Optional[str] = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)


# ══════════════════════════════════════════════════════════════════
# QUERY (RAG) schemas
# ══════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    """
    A legal query sent to the RAG pipeline.
    Supports optional metadata pre-filtering before vector search.
    """
    query: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="The legal question to answer",
        examples=["What are the indemnification obligations in the ABC Corp contract?"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for grouping queries in audit logs",
    )

    # --- Metadata pre-filters (applied before vector search) ---
    filter_client_id: Optional[str] = Field(
        default=None,
        description="Restrict search to a specific client's documents",
    )
    filter_doc_type: Optional[str] = Field(
        default=None,
        description="Restrict search to a specific document type",
    )
    filter_date_from: Optional[date] = Field(
        default=None,
        description="Only search documents filed on or after this date",
    )
    filter_date_to: Optional[date] = Field(
        default=None,
        description="Only search documents filed on or before this date",
    )

    # --- Retrieval overrides (optional, uses .env defaults otherwise) ---
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Override default retrieval_top_k for this query",
    )


class SourceChunk(BaseModel):
    """A single retrieved and cited source chunk."""
    document_id: uuid.UUID
    filename: str
    chunk_index: int
    text: str
    relevance_score: float = Field(
        description="Reranker score (0.0 - 1.0)",
    )
    doc_type: Optional[str] = None
    client_id: Optional[str] = None
    date_filed: Optional[date] = None


class QueryResponse(BaseModel):
    """
    Complete RAG response including the answer and all cited sources.
    Every response MUST include source citations (enforced by system prompt).
    """
    query: str
    answer: str = Field(description="Claude's answer, grounded in retrieved context")
    sources: list[SourceChunk] = Field(
        description="All source chunks cited in the answer, ranked by relevance",
    )
    cache_hit: bool = Field(
        default=False,
        description="True if this response was served from the semantic cache",
    )
    latency_ms: int = Field(description="Total end-to-end latency in milliseconds")
    session_id: Optional[str] = None


# ══════════════════════════════════════════════════════════════════
# INGESTION TASK schemas
# ══════════════════════════════════════════════════════════════════

class IngestionTaskOut(BaseModel):
    """Response returned immediately when a document upload is accepted."""
    task_id: str = Field(description="Celery task ID — use to poll status")
    document_id: uuid.UUID
    filename: str
    status: str = Field(default="pending")
    message: str = Field(default="Document accepted for ingestion")


class IngestionStatusOut(BaseModel):
    """Polling response for ingestion task status."""
    task_id: str
    document_id: uuid.UUID
    status: str           # pending | processing | indexed | failed
    chunk_count: int
    error: Optional[str] = None


# ══════════════════════════════════════════════════════════════════
# EVALUATION schemas
# ══════════════════════════════════════════════════════════════════

class GoldenDatasetEntryOut(ORMBase):
    """A single synthetic QA pair from the golden dataset."""
    id: uuid.UUID
    question: str
    reference_context: str
    expected_answer: str
    source_doc_ids: Optional[list[uuid.UUID]] = None
    generated_by: str
    created_at: datetime


class EvalRunOut(ORMBase):
    """Results of a completed evaluation run."""
    id: uuid.UUID
    run_id: str
    faithfulness: Optional[float]
    answer_relevance: Optional[float]
    context_precision: Optional[float]
    total_cases: Optional[int]
    passed_cases: Optional[int]
    failed_cases: Optional[int]
    passed: Optional[bool]
    ran_at: datetime


class EvalTriggerRequest(BaseModel):
    """Request to trigger a new evaluation run."""
    run_id: str = Field(
        description="Unique identifier for this run, e.g. GitHub Actions run ID",
        examples=["gh-actions-12345"],
    )
    dataset_size: Optional[int] = Field(
        default=None,
        description="Override golden dataset size for this run",
    )


class EvalTriggerResponse(BaseModel):
    """Immediate response when an eval run is kicked off."""
    run_id: str
    task_id: str
    message: str = "Evaluation run started"


# ══════════════════════════════════════════════════════════════════
# GENERIC / SHARED schemas
# ══════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    status: str
    service: str
    environment: str
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None


class PaginatedResponse(BaseModel):
    """Generic paginated list wrapper."""
    items: list[Any]
    total: int
    page: int
    page_size: int
    total_pages: int
