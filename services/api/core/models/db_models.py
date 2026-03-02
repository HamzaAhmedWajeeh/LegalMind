"""
core/models/db_models.py
========================
SQLAlchemy ORM models — Python representations of the database tables.

These mirror the schema in db/init.sql exactly.
Alembic uses these models to generate future migration scripts.

Design Pattern: Repository Pattern support — these models are only used
inside Repository classes, never directly in route handlers. Routes use
Pydantic schemas (api/models/schemas.py) for input/output.
"""

import uuid
from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    ARRAY,
    UUID,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from core.db import Base


def gen_uuid() -> uuid.UUID:
    return uuid.uuid4()


# ------------------------------------------------------------------
# Document — Master registry of all ingested files
# ------------------------------------------------------------------
class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=gen_uuid
    )
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_hash: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False,
        comment="SHA-256 hash — prevents duplicate ingestion",
    )
    doc_type: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True,
        comment="contract | case_file | brief | memo | other",
    )
    client_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    matter_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    date_filed: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    status: Mapped[str] = mapped_column(
        String(20), default="pending",
        comment="pending | processing | indexed | failed",
    )
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    metadata_: Mapped[dict] = mapped_column(
        "metadata", JSONB, default=dict,
        comment="Flexible additional metadata",
    )

    # Composite indexes (match init.sql)
    __table_args__ = (
        Index("idx_documents_client_id", "client_id"),
        Index("idx_documents_doc_type", "doc_type"),
        Index("idx_documents_date_filed", "date_filed"),
        Index("idx_documents_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename={self.filename} status={self.status}>"


# ------------------------------------------------------------------
# Chunk — Individual text chunks linked to a document
# ------------------------------------------------------------------
class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=gen_uuid
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False,
        comment="FK to documents.id",
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    qdrant_id: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True,
        comment="Corresponding point ID in Qdrant",
    )
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    __table_args__ = (
        Index("idx_chunks_document_id", "document_id"),
    )

    def __repr__(self) -> str:
        return f"<Chunk id={self.id} doc={self.document_id} index={self.chunk_index}>"


# ------------------------------------------------------------------
# QueryLog — Full audit trail of every query + response
# ------------------------------------------------------------------
class QueryLog(Base):
    __tablename__ = "query_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=gen_uuid
    )
    session_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    response_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # PostgreSQL ARRAY of UUIDs — all source doc IDs cited in the response
    source_doc_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=True)
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    __table_args__ = (
        Index("idx_query_logs_session", "session_id"),
        Index("idx_query_logs_created", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<QueryLog id={self.id} cache_hit={self.cache_hit}>"


# ------------------------------------------------------------------
# GoldenDataset — Synthetic QA pairs for RAG evaluation
# ------------------------------------------------------------------
class GoldenDatasetEntry(Base):
    __tablename__ = "golden_dataset"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=gen_uuid
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    reference_context: Mapped[str] = mapped_column(Text, nullable=False)
    expected_answer: Mapped[str] = mapped_column(Text, nullable=False)
    source_doc_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=True)
    generated_by: Mapped[str] = mapped_column(
        String(50), default="adversarial_lawyer_agent"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    def __repr__(self) -> str:
        return f"<GoldenDatasetEntry id={self.id} question={self.question[:50]}...>"


# ------------------------------------------------------------------
# EvalRun — Results of each CI/CD evaluation run
# ------------------------------------------------------------------
class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=gen_uuid
    )
    run_id: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False,
        comment="e.g. GitHub Actions run ID",
    )
    faithfulness: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    answer_relevance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    context_precision: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_cases: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    passed_cases: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    failed_cases: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    passed: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    ran_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    def __repr__(self) -> str:
        return f"<EvalRun run_id={self.run_id} passed={self.passed} faithfulness={self.faithfulness}>"
