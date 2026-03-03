"""
core/retrieval/vector_store.py
==============================
Qdrant vector store — handles:
  1. Collection initialisation (creates it if it doesn't exist)
  2. Upserting enriched chunks as Qdrant points (called from ingest_task)
  3. Semantic similarity search with optional metadata pre-filtering

Design Pattern: Repository Pattern
  All Qdrant interactions are encapsulated here behind a clean interface.
  The rest of the system never imports qdrant_client directly — it only
  calls methods on QdrantVectorStore. Swapping to a different vector DB
  (e.g., Pinecone, Weaviate) only requires replacing this file.

Embedding model: sentence-transformers/all-mpnet-base-v2
  - 768-dimensional embeddings
  - Strong performance on semantic similarity tasks
  - Runs locally — no external API cost per embedding
"""

import uuid
from typing import Optional

import structlog
from qdrant_client import AsyncQdrantClient, models
from sentence_transformers import SentenceTransformer

from core.config import get_settings
from core.ingestion.enricher import EnrichedChunk

logger = structlog.get_logger(__name__)
settings = get_settings()

# ── Embedding model config ─────────────────────────────────────────
_EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
_EMBEDDING_DIM = 768          # Must match the model output dimension
_BATCH_SIZE = 32              # Chunks to embed per batch


# ──────────────────────────────────────────────────────────────────
# Retrieval result dataclass
# ──────────────────────────────────────────────────────────────────
from dataclasses import dataclass, field

@dataclass
class RetrievedChunk:
    """
    A single chunk returned from vector or keyword search.
    Shared by both VectorStore and BM25 so the hybrid fusion
    layer can work with a uniform type.
    """
    qdrant_id: str
    document_id: str
    filename: str
    text: str
    chunk_index: int
    score: float                          # Raw search score (not normalised)
    payload: dict = field(default_factory=dict)

    # Convenience properties pulled from payload
    @property
    def doc_type(self) -> Optional[str]:
        return self.payload.get("doc_type")

    @property
    def client_id(self) -> Optional[str]:
        return self.payload.get("client_id")

    @property
    def date_filed(self) -> Optional[str]:
        return self.payload.get("date_filed")


# ──────────────────────────────────────────────────────────────────
# QdrantVectorStore
# ──────────────────────────────────────────────────────────────────
class QdrantVectorStore:
    """
    Repository Pattern wrapper around Qdrant.

    Initialise once at application startup via the module-level
    `vector_store` singleton below.
    """

    def __init__(self):
        self._client: Optional[AsyncQdrantClient] = None
        self._embedding_model: Optional[SentenceTransformer] = None

    # ── Lazy initialisation ────────────────────────────────────────
    async def _get_client(self) -> AsyncQdrantClient:
        if self._client is None:
            self._client = AsyncQdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
            )
            await self._ensure_collection()
        return self._client

    def _get_embedding_model(self) -> SentenceTransformer:
        """Lazy-load the embedding model once."""
        if self._embedding_model is None:
            logger.info("Loading embedding model", model=_EMBEDDING_MODEL_NAME)
            self._embedding_model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
            logger.info("Embedding model loaded")
        return self._embedding_model

    # ── Collection setup ───────────────────────────────────────────
    async def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it doesn't already exist."""
        client = self._client
        collections = await client.get_collections()
        names = [c.name for c in collections.collections]

        if settings.qdrant_collection_name not in names:
            await client.create_collection(
                collection_name=settings.qdrant_collection_name,
                vectors_config=models.VectorParams(
                    size=_EMBEDDING_DIM,
                    distance=models.Distance.COSINE,
                ),
                # HNSW index for fast approximate nearest-neighbour search
                hnsw_config=models.HnswConfigDiff(
                    m=16,               # Number of edges per node
                    ef_construct=100,   # Build-time accuracy vs speed
                ),
                # Payload indexes for metadata pre-filtering
                # These allow Qdrant to filter BEFORE the vector search
                # (much faster than post-filtering)
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=20_000,
                ),
            )

            # Create payload indexes for the filterable fields
            for field_name in ["client_id", "doc_type", "date_filed", "document_id"]:
                await client.create_payload_index(
                    collection_name=settings.qdrant_collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

            logger.info(
                "Qdrant collection created",
                collection=settings.qdrant_collection_name,
                dim=_EMBEDDING_DIM,
            )
        else:
            logger.info(
                "Qdrant collection already exists",
                collection=settings.qdrant_collection_name,
            )

    # ── Embedding ──────────────────────────────────────────────────
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.
        Runs in batches to avoid OOM on large documents.
        """
        model = self._get_embedding_model()
        embeddings = model.encode(
            texts,
            batch_size=_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,   # Unit vectors → cosine similarity = dot product
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_texts([query])[0]

    # ── Upsert ─────────────────────────────────────────────────────
    async def upsert_chunks(self, enriched_chunks: list[EnrichedChunk]) -> None:
        """
        Embed and upsert a batch of EnrichedChunks into Qdrant.
        Called from ingest_task._embed_and_store().

        Uses batch upsert for efficiency — single network round-trip
        per batch rather than one per chunk.
        """
        client = await self._get_client()
        # Re-check collection exists on every upsert.
        # The worker process caches the client across requests, so if Qdrant
        # data was wiped (docker compose down -v) while the worker was running,
        # the collection disappears but _get_client() won't re-create it because
        # self._client is already set. Calling _ensure_collection() here is
        # idempotent (it checks first) and costs one GET /collections per upsert.
        await self._ensure_collection()

        if not enriched_chunks:
            return

        texts = [ec.text for ec in enriched_chunks]
        embeddings = self.embed_texts(texts)

        points = [
            models.PointStruct(
                id=ec.qdrant_id,
                vector=embedding,
                payload={**ec.payload, "text": ec.text},  # text MUST be in payload for retrieval
            )
            for ec, embedding in zip(enriched_chunks, embeddings)
        ]

        # Upsert in batches of 100 to stay within Qdrant request size limits
        for i in range(0, len(points), 100):
            batch = points[i : i + 100]
            await client.upsert(
                collection_name=settings.qdrant_collection_name,
                points=batch,
                wait=True,   # Wait for indexing before returning
            )

        logger.info(
            "Vectors upserted to Qdrant",
            collection=settings.qdrant_collection_name,
            count=len(points),
        )

    # ── Search ─────────────────────────────────────────────────────
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
        Semantic similarity search with optional metadata pre-filtering.

        Metadata filters are applied BEFORE the vector search in Qdrant
        (index-level filtering), not as a post-processing step.
        This is dramatically faster for large collections.

        Args:
            query           : The user's question
            top_k           : Number of results to return
            filter_client_id: Restrict to a specific client's documents
            filter_doc_type : Restrict to a specific document type
            filter_date_from: ISO date string lower bound on date_filed
            filter_date_to  : ISO date string upper bound on date_filed

        Returns:
            List of RetrievedChunk ordered by descending similarity score
        """
        client = await self._get_client()
        query_vector = self.embed_query(query)

        # Build Qdrant filter from the provided criteria
        qdrant_filter = _build_filter(
            filter_client_id=filter_client_id,
            filter_doc_type=filter_doc_type,
            filter_date_from=filter_date_from,
            filter_date_to=filter_date_to,
        )

        results = await client.search(
            collection_name=settings.qdrant_collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=False,   # No need to return vectors to the caller
        )

        retrieved = [
            RetrievedChunk(
                qdrant_id=str(hit.id),
                document_id=hit.payload.get("document_id", ""),
                filename=hit.payload.get("filename", ""),
                text=hit.payload.get("text", "") or _get_text_from_payload(hit),
                chunk_index=hit.payload.get("chunk_index", 0),
                score=hit.score,
                payload=hit.payload,
            )
            for hit in results
        ]

        logger.info(
            "Vector search complete",
            query_preview=query[:60],
            results=len(retrieved),
            top_score=retrieved[0].score if retrieved else None,
        )

        return retrieved

    # ── Delete ─────────────────────────────────────────────────────
    async def delete_document_chunks(self, document_id: str) -> None:
        """Remove all Qdrant points belonging to a document."""
        client = await self._get_client()
        await client.delete(
            collection_name=settings.qdrant_collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )
        logger.info("Deleted Qdrant points for document", document_id=document_id)


# ──────────────────────────────────────────────────────────────────
# Module-level singleton
# ──────────────────────────────────────────────────────────────────
vector_store = QdrantVectorStore()


# ──────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────
def _build_filter(
    filter_client_id: Optional[str],
    filter_doc_type: Optional[str],
    filter_date_from: Optional[str],
    filter_date_to: Optional[str],
) -> Optional[models.Filter]:
    """
    Build a Qdrant Filter from optional metadata criteria.
    Returns None if no filters are active (no filtering applied).
    """
    conditions = []

    if filter_client_id:
        conditions.append(
            models.FieldCondition(
                key="client_id",
                match=models.MatchValue(value=filter_client_id),
            )
        )

    if filter_doc_type:
        conditions.append(
            models.FieldCondition(
                key="doc_type",
                match=models.MatchValue(value=filter_doc_type),
            )
        )

    if filter_date_from or filter_date_to:
        conditions.append(
            models.FieldCondition(
                key="date_filed",
                range=models.DatetimeRange(
                    gte=filter_date_from,
                    lte=filter_date_to,
                ),
            )
        )

    if not conditions:
        return None

    return models.Filter(must=conditions)


def _get_text_from_payload(hit) -> str:
    """
    Qdrant doesn't store the text separately from the payload.
    The text is stored inside the payload under the 'text' key
    by the enricher. This helper handles the case where it might
    be missing (shouldn't happen in practice).
    """
    return hit.payload.get("text", "[text not available]")
