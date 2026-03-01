"""
core/ingestion/enricher.py
==========================
Metadata enricher — augments each TextChunk with structured metadata
before it is written to Qdrant (as payload) and PostgreSQL.

Why metadata matters:
  The spec requires Metadata Filtering — the ability to pre-filter
  by date, client_id, or doc_type BEFORE semantic search. Qdrant
  supports payload filtering natively, so we attach all filterable
  fields to each point's payload here.

What the enricher adds to each chunk:
  - Document-level: filename, doc_type, client_id, matter_id, date_filed
  - Chunk-level   : chunk_index, token_count, page_number (estimated)
  - System        : ingested_at timestamp, document_id (UUID)

Returns:
  List of EnrichedChunk objects — ready for the vector store writer.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import structlog

from core.ingestion.chunker import TextChunk
from core.ingestion.parser import ParsedDocument

logger = structlog.get_logger(__name__)


# ──────────────────────────────────────────────────────────────────
# Output dataclass
# ──────────────────────────────────────────────────────────────────
@dataclass
class EnrichedChunk:
    """
    A TextChunk with all metadata attached.
    This is the final form before writing to Qdrant + Postgres.

    Attributes:
        qdrant_id      : UUID used as the point ID in Qdrant
        document_id    : UUID of the parent Document row in Postgres
        text           : Chunk content (for embedding)
        chunk_index    : Position in document
        token_count    : Approximate token count
        payload        : Full metadata dict stored as Qdrant point payload
                         and also mirrors the Postgres chunks.metadata column
    """
    qdrant_id: str              # str(uuid) — Qdrant expects strings
    document_id: uuid.UUID
    text: str
    chunk_index: int
    token_count: int
    payload: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────
# Enricher
# ──────────────────────────────────────────────────────────────────
class MetadataEnricher:
    """
    Enriches a list of TextChunks with document and system metadata.

    Keeps enrichment logic in one place so adding a new metadata
    field only requires changing this class — not the chunker or
    the vector store writer.
    """

    def enrich(
        self,
        chunks: list[TextChunk],
        parsed_doc: ParsedDocument,
        document_id: uuid.UUID,
        doc_type: Optional[str] = None,
        client_id: Optional[str] = None,
        matter_id: Optional[str] = None,
        date_filed: Optional[str] = None,   # ISO date string e.g. "2024-03-15"
        extra_metadata: Optional[dict] = None,
    ) -> list[EnrichedChunk]:
        """
        Attach metadata to each chunk.

        Args:
            chunks        : Raw TextChunks from chunker.py
            parsed_doc    : ParsedDocument from parser.py (for doc-level info)
            document_id   : Postgres UUID for the parent document row
            doc_type      : 'contract' | 'case_file' | 'brief' | 'memo' | 'other'
            client_id     : Client identifier for metadata filtering
            matter_id     : Matter/case identifier
            date_filed    : ISO date string of when the document was filed
            extra_metadata: Any additional caller-supplied key-value pairs

        Returns:
            List of EnrichedChunk objects
        """
        ingested_at = datetime.now(timezone.utc).isoformat()
        enriched: list[EnrichedChunk] = []

        for chunk in chunks:
            qdrant_id = str(uuid.uuid4())

            # Estimate which page this chunk falls on
            page_number = _estimate_page(
                chunk_index=chunk.chunk_index,
                total_chunks=len(chunks),
                total_pages=parsed_doc.page_count,
            )

            # Build the full payload stored in Qdrant point + Postgres metadata
            payload = {
                # ── Document-level (filterable in Qdrant) ──────────
                "document_id": str(document_id),
                "filename": parsed_doc.filename,
                "file_type": parsed_doc.file_type,
                "doc_type": doc_type,
                "client_id": client_id,
                "matter_id": matter_id,
                "date_filed": date_filed,

                # ── Chunk-level ────────────────────────────────────
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "page_number": page_number,

                # ── System ─────────────────────────────────────────
                "ingested_at": ingested_at,
                "ocr_used": parsed_doc.ocr_used,
                "chunking_strategy": chunk.metadata.get("strategy", "unknown"),

                # ── Merge any caller-supplied extras ───────────────
                **(extra_metadata or {}),
            }

            # Remove None values — Qdrant handles missing keys better than None
            payload = {k: v for k, v in payload.items() if v is not None}

            enriched.append(
                EnrichedChunk(
                    qdrant_id=qdrant_id,
                    document_id=document_id,
                    text=chunk.text,
                    chunk_index=chunk.chunk_index,
                    token_count=chunk.token_count,
                    payload=payload,
                )
            )

        logger.info(
            "Metadata enrichment complete",
            filename=parsed_doc.filename,
            chunk_count=len(enriched),
            client_id=client_id,
            doc_type=doc_type,
        )

        return enriched


# ──────────────────────────────────────────────────────────────────
# Module-level singleton (reused across requests)
# ──────────────────────────────────────────────────────────────────
enricher = MetadataEnricher()


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────
def _estimate_page(
    chunk_index: int,
    total_chunks: int,
    total_pages: int,
) -> int:
    """
    Estimate which page a chunk falls on based on its relative position.
    Not precise for PDFs with very unequal page lengths, but good enough
    for metadata display purposes.

    Returns 1-based page number.
    """
    if total_chunks == 0 or total_pages == 0:
        return 1
    ratio = chunk_index / max(total_chunks - 1, 1)
    return max(1, round(ratio * (total_pages - 1)) + 1)
