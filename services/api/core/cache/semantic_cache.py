"""
core/cache/semantic_cache.py
============================
Semantic cache — serves instant answers for queries that are
semantically similar to previously answered ones.

Why semantic (not exact-match) caching?
  Legal users often rephrase the same question:
    "What are the indemnification limits in the Apex contract?"
    "What is the indemnification cap for Apex?"
    "How much can Apex be indemnified for?"

  All three questions should hit the same cached answer.
  Exact-match caching misses all of this. Semantic caching
  embeds the query and compares vector similarity.

Implementation:
  - Redis is used as the vector store for the cache index.
  - Each cache entry stores:
      • The query embedding (768-dim float32 vector)
      • The serialised QueryResponse JSON
      • A TTL (default 1 hour, configurable via CACHE_TTL_SECONDS)
  - On lookup: embed the incoming query, run a vector search
    against all cached query embeddings, return the stored response
    if the top similarity score ≥ SEMANTIC_CACHE_THRESHOLD (default 0.92).

Design Pattern: Proxy Pattern
  SemanticCache acts as a transparent proxy in front of the RAG pipeline.
  The RAGService calls cache.get() and cache.set() without knowing
  the implementation — the cache is completely substitutable.

  If Redis is unavailable, all methods fail gracefully and return None/False
  so the RAG pipeline continues unaffected.
"""

import json
import time
from typing import Optional

import numpy as np
import structlog
from redis import asyncio as aioredis

from core.config import get_settings
from api.models.schemas import QueryResponse, SourceChunk

logger = structlog.get_logger(__name__)
settings = get_settings()

# Redis key prefixes
_CACHE_EMBEDDING_PREFIX = "legalmind:cache:emb:"    # stores the query embedding
_CACHE_RESPONSE_PREFIX  = "legalmind:cache:resp:"   # stores the serialised response
_CACHE_INDEX_KEY        = "legalmind:cache:index"   # sorted set of all cache keys

# Embedding vector dimension (must match vector_store.py)
_EMBEDDING_DIM = 768


class SemanticCache:
    """
    Redis-backed semantic query cache.

    Design Pattern: Proxy Pattern — transparent cache layer in front
    of the full RAG pipeline. The RAGService calls get/set without
    knowing whether Redis is available.

    Thread safety: Uses aioredis (async Redis client) — safe for
    concurrent FastAPI requests.
    """

    def __init__(self):
        self._redis: Optional[aioredis.Redis] = None
        self._embedding_model = None       # Lazy-loaded

    # ── Redis connection ───────────────────────────────────────────
    async def _get_redis(self) -> aioredis.Redis:
        """Return a connected Redis client, creating one if needed."""
        if self._redis is None:
            self._redis = await aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=False,   # We handle encoding ourselves for binary vectors
                socket_connect_timeout=3,
                socket_timeout=3,
            )
        return self._redis

    # ── Embedding model ────────────────────────────────────────────
    def _get_model(self):
        """Lazy-load the sentence transformer (same model as vector_store.py)."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(
                "sentence-transformers/all-mpnet-base-v2"
            )
            logger.info("Semantic cache embedding model loaded")
        return self._embedding_model

    def _embed(self, text: str) -> np.ndarray:
        """Embed a single query string into a normalised float32 vector."""
        model = self._get_model()
        vector = model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector[0].astype(np.float32)

    # ── Cache GET ──────────────────────────────────────────────────
    async def get(self, query: str) -> Optional[QueryResponse]:
        """
        Look up a semantically similar past query in the cache.

        Steps:
          1. Embed the incoming query
          2. Retrieve all cached query embeddings from Redis
          3. Compute cosine similarity between the query and each cached embedding
          4. If best match ≥ SEMANTIC_CACHE_THRESHOLD, return the stored response
          5. Otherwise return None (cache miss)

        Args:
            query: The user's incoming legal question

        Returns:
            QueryResponse if cache hit, None otherwise
        """
        try:
            redis = await self._get_redis()
            start = time.monotonic()

            # Get all cached entry keys from the index
            cache_keys = await redis.smembers(_CACHE_INDEX_KEY)

            if not cache_keys:
                return None   # Cache is empty

            # Embed the incoming query
            query_vector = self._embed(query)

            # Retrieve all cached embeddings and compute similarities
            best_key: Optional[str] = None
            best_score: float = -1.0

            for raw_key in cache_keys:
                key = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else raw_key
                emb_key = _CACHE_EMBEDDING_PREFIX + key

                cached_emb_bytes = await redis.get(emb_key)
                if cached_emb_bytes is None:
                    # TTL expired for the embedding — clean up the index
                    await redis.srem(_CACHE_INDEX_KEY, key)
                    continue

                # Deserialise the stored vector
                cached_vector = np.frombuffer(cached_emb_bytes, dtype=np.float32)

                # Cosine similarity (vectors are already normalised → dot product)
                score = float(np.dot(query_vector, cached_vector))

                if score > best_score:
                    best_score = score
                    best_key = key

            # Check if we have a hit above the threshold
            if best_key is None or best_score < settings.semantic_cache_threshold:
                logger.debug(
                    "Cache miss",
                    best_score=round(best_score, 4),
                    threshold=settings.semantic_cache_threshold,
                )
                return None

            # Retrieve the stored response
            resp_key = _CACHE_RESPONSE_PREFIX + best_key
            response_json = await redis.get(resp_key)

            if response_json is None:
                # Response TTL expired but embedding didn't — clean up
                await redis.delete(_CACHE_EMBEDDING_PREFIX + best_key)
                await redis.srem(_CACHE_INDEX_KEY, best_key)
                return None

            # Deserialise and return
            response_data = json.loads(response_json)
            response = _deserialise_response(response_data, cache_hit=True)

            latency_ms = int((time.monotonic() - start) * 1000)
            logger.info(
                "Cache hit",
                similarity=round(best_score, 4),
                lookup_latency_ms=latency_ms,
                matched_key=best_key[:40],
            )

            return response

        except Exception as exc:
            logger.warning("Semantic cache GET failed", error=str(exc))
            return None   # Fail open — continue to full RAG pipeline

    # ── Cache SET ──────────────────────────────────────────────────
    async def set(self, query: str, response: QueryResponse) -> bool:
        """
        Store a query + response pair in the semantic cache.

        Stores:
          - The query embedding as raw bytes (float32) with TTL
          - The serialised QueryResponse as JSON with TTL
          - The cache key in the index set

        Args:
            query    : The original query string
            response : The QueryResponse to cache

        Returns:
            True if stored successfully, False on failure
        """
        try:
            redis = await self._get_redis()

            # Use a hash of the query as the cache key
            import hashlib
            cache_key = hashlib.sha256(query.encode()).hexdigest()[:32]

            # Embed the query
            query_vector = self._embed(query)

            # Serialise the vector to bytes
            emb_bytes = query_vector.tobytes()

            # Serialise the response to JSON
            response_json = json.dumps(_serialise_response(response))

            ttl = settings.cache_ttl_seconds

            # Store embedding
            await redis.set(
                _CACHE_EMBEDDING_PREFIX + cache_key,
                emb_bytes,
                ex=ttl,
            )

            # Store response
            await redis.set(
                _CACHE_RESPONSE_PREFIX + cache_key,
                response_json,
                ex=ttl,
            )

            # Add to index (no TTL on the index — we clean it lazily on GET)
            await redis.sadd(_CACHE_INDEX_KEY, cache_key)

            logger.info(
                "Response cached",
                cache_key=cache_key,
                ttl_seconds=ttl,
                query_preview=query[:60],
            )

            return True

        except Exception as exc:
            logger.warning("Semantic cache SET failed", error=str(exc))
            return False   # Fail open

    # ── Cache invalidation ─────────────────────────────────────────
    async def invalidate_all(self) -> int:
        """
        Clear the entire semantic cache.
        Called when documents are deleted or re-ingested, as cached
        answers may reference outdated information.

        Returns:
            Number of entries deleted
        """
        try:
            redis = await self._get_redis()
            cache_keys = await redis.smembers(_CACHE_INDEX_KEY)
            count = 0

            for raw_key in cache_keys:
                key = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else raw_key
                await redis.delete(_CACHE_EMBEDDING_PREFIX + key)
                await redis.delete(_CACHE_RESPONSE_PREFIX + key)
                count += 1

            await redis.delete(_CACHE_INDEX_KEY)

            logger.info("Semantic cache cleared", entries_deleted=count)
            return count

        except Exception as exc:
            logger.error("Cache invalidation failed", error=str(exc))
            return 0

    # ── Cache stats ────────────────────────────────────────────────
    async def stats(self) -> dict:
        """
        Return basic cache statistics for the evaluation dashboard.

        Returns:
            Dict with entry_count, threshold, ttl_seconds
        """
        try:
            redis = await self._get_redis()
            entry_count = await redis.scard(_CACHE_INDEX_KEY)
            return {
                "entry_count": entry_count,
                "threshold": settings.semantic_cache_threshold,
                "ttl_seconds": settings.cache_ttl_seconds,
                "status": "connected",
            }
        except Exception as exc:
            return {
                "entry_count": 0,
                "threshold": settings.semantic_cache_threshold,
                "ttl_seconds": settings.cache_ttl_seconds,
                "status": f"unavailable: {exc}",
            }


# ──────────────────────────────────────────────────────────────────
# Serialisation helpers
# ──────────────────────────────────────────────────────────────────

def _serialise_response(response: QueryResponse) -> dict:
    """
    Convert a QueryResponse to a JSON-serialisable dict.
    Handles UUID and date fields that aren't natively JSON serialisable.
    """
    return {
        "query": response.query,
        "answer": response.answer,
        "sources": [
            {
                "document_id": str(src.document_id),
                "filename": src.filename,
                "chunk_index": src.chunk_index,
                "text": src.text,
                "relevance_score": src.relevance_score,
                "doc_type": src.doc_type,
                "client_id": src.client_id,
                "date_filed": src.date_filed.isoformat() if src.date_filed else None,
            }
            for src in response.sources
        ],
        "cache_hit": True,      # Always True when served from cache
        "latency_ms": response.latency_ms,
        "session_id": response.session_id,
    }


def _deserialise_response(data: dict, cache_hit: bool = True) -> QueryResponse:
    """Reconstruct a QueryResponse from a cached JSON dict."""
    import uuid
    from datetime import date

    sources = []
    for src in data.get("sources", []):
        date_filed = None
        if src.get("date_filed"):
            try:
                date_filed = date.fromisoformat(src["date_filed"])
            except (ValueError, TypeError):
                pass

        sources.append(SourceChunk(
            document_id=uuid.UUID(src["document_id"]),
            filename=src["filename"],
            chunk_index=src["chunk_index"],
            text=src["text"],
            relevance_score=src["relevance_score"],
            doc_type=src.get("doc_type"),
            client_id=src.get("client_id"),
            date_filed=date_filed,
        ))

    return QueryResponse(
        query=data["query"],
        answer=data["answer"],
        sources=sources,
        cache_hit=cache_hit,
        latency_ms=data.get("latency_ms", 0),
        session_id=data.get("session_id"),
    )


# ──────────────────────────────────────────────────────────────────
# Module-level singleton
# ──────────────────────────────────────────────────────────────────
semantic_cache = SemanticCache()
