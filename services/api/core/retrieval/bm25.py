"""
core/retrieval/bm25.py
======================
BM25 keyword search — catches exact legal terminology that semantic
search can miss (e.g., specific clause numbers, defined terms,
case citations like "Smith v. Jones [2019]").

Why BM25 for legal text?
  Semantic search excels at conceptual similarity but can rank a
  paragraph about "general liability" higher than one containing the
  exact term "indemnification clause" when the user searched for
  "indemnification clause". BM25 rewards exact term overlap.

Architecture:
  - The BM25 index is built in-memory from the Postgres `chunks` table.
  - It is rebuilt on application startup and whenever new documents
    are ingested (cache-invalidated by the ingest task).
  - For 10,000+ documents this is fast: rank-bm25 handles 500K+
    documents in memory with sub-second query latency.

Design Pattern: Repository Pattern
  BM25Retriever exposes the same search interface as QdrantVectorStore
  so the hybrid fusion layer can treat them uniformly.
"""

import string
from dataclasses import dataclass, field
from typing import Optional

import structlog
from rank_bm25 import BM25Okapi

from core.retrieval.vector_store import RetrievedChunk

logger = structlog.get_logger(__name__)

# Legal-domain stopwords — extend the standard English list with legal
# function words that add noise without aiding retrieval
_LEGAL_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "is", "was", "are",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall",
    "must", "that", "this", "these", "those", "it", "its", "which",
    "who", "whom", "what", "when", "where", "how", "not", "no", "nor",
    # Legal function words
    "herein", "hereof", "hereto", "hereby", "hereunder", "hereinafter",
    "thereof", "thereto", "therein", "thereunder", "therewith",
    "pursuant", "per", "such", "said", "party", "parties",
}


# ──────────────────────────────────────────────────────────────────
# Internal index entry
# ──────────────────────────────────────────────────────────────────
@dataclass
class _IndexedChunk:
    """Maps a BM25 corpus index position back to chunk metadata."""
    qdrant_id: str
    document_id: str
    filename: str
    text: str
    chunk_index: int
    payload: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────
# BM25 Retriever
# ──────────────────────────────────────────────────────────────────
class BM25Retriever:
    """
    In-memory BM25 index over all ingested document chunks.

    The index must be built (or rebuilt) before searching.
    Call `await build_index()` on startup and after each ingestion.
    """

    def __init__(self):
        self._index: Optional[BM25Okapi] = None
        self._corpus: list[_IndexedChunk] = []
        self._is_built: bool = False

    # ── Index construction ─────────────────────────────────────────
    async def build_index(self) -> None:
        """
        Load all chunks from Postgres and build the BM25 index.

        This is an async method because it reads from the database.
        The tokenisation and index construction are CPU-bound but fast
        enough to run in the event loop for typical corpus sizes.
        """
        from sqlalchemy import select, join
        from core.db import get_db_context
        from core.models.db_models import Chunk, Document

        logger.info("Building BM25 index from Postgres...")

        async with get_db_context() as db:
            # Join chunks with documents to get filename for each chunk
            stmt = (
                select(
                    Chunk.id,
                    Chunk.text,
                    Chunk.chunk_index,
                    Chunk.qdrant_id,
                    Chunk.metadata_,
                    Document.id.label("document_id"),
                    Document.filename,
                )
                .join(Document, Chunk.document_id == Document.id)
                .where(Document.status == "indexed")
                .order_by(Document.id, Chunk.chunk_index)
            )
            result = await db.execute(stmt)
            rows = result.fetchall()

        if not rows:
            logger.warning("BM25 index is empty — no indexed documents found")
            self._index = None
            self._corpus = []
            self._is_built = False
            return

        # Build tokenised corpus
        self._corpus = []
        tokenised_corpus: list[list[str]] = []

        for row in rows:
            chunk = _IndexedChunk(
                qdrant_id=row.qdrant_id or str(row.id),
                document_id=str(row.document_id),
                filename=row.filename,
                text=row.text,
                chunk_index=row.chunk_index,
                payload=row.metadata_ or {},
            )
            self._corpus.append(chunk)
            tokenised_corpus.append(tokenise(row.text))

        self._index = BM25Okapi(tokenised_corpus)
        self._is_built = True

        logger.info(
            "BM25 index built",
            total_chunks=len(self._corpus),
        )

    # ── Search ─────────────────────────────────────────────────────
    async def search(
        self,
        query: str,
        top_k: int,
        filter_client_id: Optional[str] = None,
        filter_doc_type: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        BM25 keyword search.

        Applies optional metadata post-filtering after scoring
        (BM25 doesn't support payload indexes like Qdrant, so
        filtering is done after retrieval).

        Args:
            query           : The user's question
            top_k           : Number of results to return
            filter_client_id: Restrict to a specific client's documents
            filter_doc_type : Restrict to a specific document type

        Returns:
            List of RetrievedChunk ordered by descending BM25 score
        """
        if not self._is_built or self._index is None:
            logger.warning("BM25 index not built — skipping keyword search")
            return []

        query_tokens = tokenise(query)
        if not query_tokens:
            return []

        # Score all documents
        scores = self._index.get_scores(query_tokens)

        # Pair scores with corpus entries and sort descending
        scored = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # Apply metadata filters and collect top_k
        results: list[RetrievedChunk] = []
        for idx, score in scored:
            if score <= 0:
                break   # BM25 scores are non-negative; 0 means no term overlap

            chunk = self._corpus[idx]

            # Post-filter by client_id
            if filter_client_id and chunk.payload.get("client_id") != filter_client_id:
                continue

            # Post-filter by doc_type
            if filter_doc_type and chunk.payload.get("doc_type") != filter_doc_type:
                continue

            results.append(
                RetrievedChunk(
                    qdrant_id=chunk.qdrant_id,
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    text=chunk.text,
                    chunk_index=chunk.chunk_index,
                    score=float(score),
                    payload=chunk.payload,
                )
            )

            if len(results) >= top_k:
                break

        logger.info(
            "BM25 search complete",
            query_preview=query[:60],
            results=len(results),
            top_score=results[0].score if results else None,
        )

        return results

    # ── Incremental update ─────────────────────────────────────────
    async def invalidate(self) -> None:
        """
        Rebuild the index. Called by the ingest task after a new
        document is successfully indexed.
        """
        self._is_built = False
        self._index = None
        self._corpus = []
        await self.build_index()


# ──────────────────────────────────────────────────────────────────
# Module-level singleton
# ──────────────────────────────────────────────────────────────────
bm25_retriever = BM25Retriever()


# ──────────────────────────────────────────────────────────────────
# Tokeniser
# ──────────────────────────────────────────────────────────────────
def tokenise(text: str) -> list[str]:
    """
    Legal-aware tokeniser for BM25.

    Steps:
      1. Lowercase
      2. Preserve hyphenated legal terms (e.g., "force-majeure")
      3. Remove punctuation (except hyphens within words)
      4. Split on whitespace
      5. Remove stopwords
      6. Keep tokens with 2+ characters

    This preserves terms like:
      - "indemnification", "force-majeure", "sub-clause"
      - Case citations: "smith", "jones", "2019"
      - Defined terms: "licensor", "licensee"
    """
    text = text.lower()

    # Remove punctuation but keep hyphens inside legal identifiers/terms.
    punctuation_without_hyphen = string.punctuation.replace("-", "")
    translation = str.maketrans({ch: " " for ch in punctuation_without_hyphen})
    text = text.translate(translation)

    tokens = text.split()
    cleaned_tokens: list[str] = []
    for token in tokens:
        # Drop leading/trailing hyphens but preserve internal ones (e.g., 2024-001).
        token = token.strip("-")
        if not token:
            continue
        if token in _LEGAL_STOPWORDS:
            continue
        if len(token) < 2:
            continue
        cleaned_tokens.append(token)

    return cleaned_tokens
