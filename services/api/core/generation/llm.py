"""
core/generation/llm.py
======================
Google Gemini client — the generation layer of the RAG pipeline.

Uses Google Gemini 2.5 Pro.
All other behaviour (citation parsing, audit logging, retry) unchanged.

Design Pattern: Facade Pattern
"""

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.config import get_settings
from core.generation.prompts import SYSTEM_PROMPT, build_user_message
from core.retrieval.reranker import RankedChunk

logger = structlog.get_logger(__name__)
settings = get_settings()


# ──────────────────────────────────────────────────────────────────
# Result dataclasses
# ──────────────────────────────────────────────────────────────────
@dataclass
class CitedSource:
    document_id: str
    filename: str
    chunk_index: int
    text: str
    relevance_score: float
    doc_type: Optional[str] = None
    client_id: Optional[str] = None
    date_filed: Optional[str] = None


@dataclass
class GenerationResult:
    answer: str
    cited_sources: list[CitedSource]
    raw_response: str
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0


# ──────────────────────────────────────────────────────────────────
# LLM Client
# ──────────────────────────────────────────────────────────────────
class LegalMindLLM:
    """
    Gemini API wrapper for the LegalMind RAG generation stage.
    Singleton — lazy-initialised on first use.
    """

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=settings.gemini_api_key)
            self._client = genai.GenerativeModel(
                model_name=settings.gemini_model,
                system_instruction=SYSTEM_PROMPT,
            )
        return self._client

    # ── Main generation method ─────────────────────────────────────
    async def generate(
        self,
        query: str,
        ranked_chunks: list[RankedChunk],
        session_id: Optional[str] = None,
    ) -> GenerationResult:
        log = logger.bind(
            query_preview=query[:60],
            chunk_count=len(ranked_chunks),
            session_id=session_id,
        )
        log.info("Generation started")

        user_message = build_user_message(query=query, chunks=ranked_chunks)

        start = time.monotonic()
        raw_response, input_tokens, output_tokens = await self._call_gemini(
            user_message=user_message
        )
        latency_ms = int((time.monotonic() - start) * 1000)

        log.info(
            "Gemini response received",
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        cited_sources = _parse_citations(
            response_text=raw_response,
            ranked_chunks=ranked_chunks,
        )

        clean_answer = re.sub(r'\n{3,}', '\n\n', raw_response).strip()

        await _save_query_log(
            query=query,
            response=clean_answer,
            cited_sources=cited_sources,
            latency_ms=latency_ms,
            session_id=session_id,
            cache_hit=False,
        )

        result = GenerationResult(
            answer=clean_answer,
            cited_sources=cited_sources,
            raw_response=raw_response,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        log.info(
            "Generation complete",
            cited_source_count=len(cited_sources),
            answer_chars=len(clean_answer),
        )

        return result

    # ── Gemini API call with retry ─────────────────────────────────
    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def _call_gemini(
        self,
        user_message: str,
    ) -> tuple[str, int, int]:
        """
        Call the Gemini API with the formatted prompt.
        Runs the synchronous SDK in a thread pool to avoid blocking the event loop.

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        import asyncio
        import google.generativeai as genai

        client = self._get_client()

        generation_config = genai.types.GenerationConfig(
            max_output_tokens=2048,
            temperature=0.0,   # Deterministic for legal accuracy
        )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.generate_content(
                user_message,
                generation_config=generation_config,
            )
        )

        response_text = response.text

        # Extract token counts from usage metadata
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        return response_text, input_tokens, output_tokens


# ──────────────────────────────────────────────────────────────────
# Citation Parser (unchanged from previous version)
# ──────────────────────────────────────────────────────────────────
def _parse_citations(
    response_text: str,
    ranked_chunks: list[RankedChunk],
) -> list[CitedSource]:
    citation_pattern = re.compile(
        r'\[SOURCE:\s*(.+?)\s*\|\s*Chunk\s*(\d+)\s*\]',
        re.IGNORECASE,
    )

    matches = citation_pattern.findall(response_text)

    if not matches:
        logger.warning(
            "No citations found in Gemini response — potential grounding issue",
            response_preview=response_text[:200],
        )
        return []

    chunk_map: dict[tuple[str, int], RankedChunk] = {}
    for chunk in ranked_chunks:
        key = (chunk.filename.strip().lower(), chunk.chunk_index)
        chunk_map[key] = chunk

    seen_keys: set[tuple[str, int]] = set()
    cited_sources: list[CitedSource] = []

    for filename_raw, chunk_idx_str in matches:
        filename = filename_raw.strip()
        chunk_index = int(chunk_idx_str)
        key = (filename.lower(), chunk_index)

        if key in seen_keys:
            continue
        seen_keys.add(key)

        matched_chunk = chunk_map.get(key)
        if matched_chunk is None:
            logger.warning(
                "Citation references unknown chunk — will be flagged by Shepardizer",
                filename=filename,
                chunk_index=chunk_index,
            )
            cited_sources.append(CitedSource(
                document_id="unknown",
                filename=filename,
                chunk_index=chunk_index,
                text="[Source not found in retrieved context]",
                relevance_score=0.0,
            ))
            continue

        cited_sources.append(CitedSource(
            document_id=matched_chunk.document_id,
            filename=matched_chunk.filename,
            chunk_index=matched_chunk.chunk_index,
            text=matched_chunk.text,
            relevance_score=matched_chunk.relevance_score,
            doc_type=matched_chunk.doc_type,
            client_id=matched_chunk.client_id,
            date_filed=matched_chunk.date_filed,
        ))

    logger.info(
        "Citations parsed",
        total_mentions=len(matches),
        unique_sources=len(cited_sources),
    )

    return cited_sources


# ──────────────────────────────────────────────────────────────────
# Audit log writer (unchanged)
# ──────────────────────────────────────────────────────────────────
async def _save_query_log(
    query: str,
    response: str,
    cited_sources: list[CitedSource],
    latency_ms: int,
    session_id: Optional[str],
    cache_hit: bool,
) -> None:
    try:
        from core.db import get_db_context
        from core.models.db_models import QueryLog

        source_doc_ids = []
        for src in cited_sources:
            if src.document_id and src.document_id != "unknown":
                try:
                    source_doc_ids.append(uuid.UUID(src.document_id))
                except ValueError:
                    pass

        async with get_db_context() as db:
            log_entry = QueryLog(
                session_id=session_id,
                query_text=query,
                response_text=response,
                source_doc_ids=source_doc_ids or None,
                cache_hit=cache_hit,
                latency_ms=latency_ms,
            )
            db.add(log_entry)

    except Exception as exc:
        logger.error("Failed to save query log", error=str(exc))


# ──────────────────────────────────────────────────────────────────
# Module-level singleton
# ──────────────────────────────────────────────────────────────────
llm = LegalMindLLM()
