"""
core/generation/llm.py
======================
Anthropic Claude client — the generation layer of the RAG pipeline.

Responsibilities:
  1. Accept the user query + reranked chunks
  2. Build the formatted prompt (via prompts.py)
  3. Call the Anthropic API
  4. Parse the response to extract the answer text and cited sources
  5. Log the query + response to the audit trail (Postgres)
  6. Return a GenerationResult with everything the API route needs

Design Pattern: Facade Pattern
  The RAGService (wired together in Step 8) calls a single method:
    result = await llm.generate(query, ranked_chunks, session_id)
  All Claude API details, prompt formatting, citation parsing,
  and audit logging are hidden behind this interface.

Retry strategy: tenacity with exponential back-off handles
  Anthropic API rate limits and transient 5xx errors gracefully.
"""

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import anthropic
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
# Result dataclass returned to the caller
# ──────────────────────────────────────────────────────────────────
@dataclass
class CitedSource:
    """
    A single source citation extracted from Claude's response.
    Maps directly to the SourceChunk Pydantic schema in schemas.py.
    """
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
    """
    Complete output from the generation layer.

    Attributes:
        answer        : Claude's response text (cleaned of raw citation tags)
        cited_sources : Parsed and deduplicated list of cited sources
        raw_response  : Full unmodified Claude response (for debugging)
        latency_ms    : Time taken for the API call in milliseconds
        input_tokens  : Tokens consumed in the prompt
        output_tokens : Tokens consumed in the completion
    """
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
    Claude API wrapper for the LegalMind RAG generation stage.

    Instantiated once at module level as a singleton.
    The Anthropic client is lazy-initialised on first use.
    """

    def __init__(self):
        self._client: Optional[anthropic.Anthropic] = None

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key,
            )
        return self._client

    # ── Main generation method ─────────────────────────────────────
    async def generate(
        self,
        query: str,
        ranked_chunks: list[RankedChunk],
        session_id: Optional[str] = None,
    ) -> GenerationResult:
        """
        Full RAG generation pipeline:
          1. Format prompt from query + reranked chunks
          2. Call Claude API
          3. Parse citations from response
          4. Persist to audit log
          5. Return GenerationResult

        Args:
            query         : The user's legal question
            ranked_chunks : Top-N chunks from the Cohere reranker
            session_id    : Optional session ID for audit logging

        Returns:
            GenerationResult with answer, sources, and usage stats
        """
        log = logger.bind(
            query_preview=query[:60],
            chunk_count=len(ranked_chunks),
            session_id=session_id,
        )
        log.info("Generation started")

        # ── 1. Build prompt ────────────────────────────────────────
        user_message = build_user_message(query=query, chunks=ranked_chunks)

        # ── 2. Call Claude ─────────────────────────────────────────
        start = time.monotonic()
        raw_response, input_tokens, output_tokens = await self._call_claude(
            user_message=user_message
        )
        latency_ms = int((time.monotonic() - start) * 1000)

        log.info(
            "Claude response received",
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        # ── 3. Parse citations ─────────────────────────────────────
        cited_sources = _parse_citations(
            response_text=raw_response,
            ranked_chunks=ranked_chunks,
        )

        # ── 4. Clean the answer text ───────────────────────────────
        # Keep [SOURCE: ...] tags in the answer — they're valuable for
        # the user to see which claims are backed by which documents.
        # We just strip any double blank lines introduced by formatting.
        clean_answer = re.sub(r'\n{3,}', '\n\n', raw_response).strip()

        # ── 5. Persist to audit log ────────────────────────────────
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

    # ── Claude API call with retry ─────────────────────────────────
    @retry(
        retry=retry_if_exception_type(
            (anthropic.RateLimitError, anthropic.InternalServerError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def _call_claude(
        self,
        user_message: str,
    ) -> tuple[str, int, int]:
        """
        Call the Claude API with the formatted prompt.

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        import asyncio

        client = self._get_client()

        # Run the synchronous Anthropic client in a thread pool
        # to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model=settings.anthropic_model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_message}
                ],
            )
        )

        response_text = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        return response_text, input_tokens, output_tokens


# ──────────────────────────────────────────────────────────────────
# Citation Parser
# ──────────────────────────────────────────────────────────────────
def _parse_citations(
    response_text: str,
    ranked_chunks: list[RankedChunk],
) -> list[CitedSource]:
    """
    Extract and validate cited sources from Claude's response.

    Parses [SOURCE: filename | Chunk N] patterns, cross-references
    them against the actual ranked chunks, and returns only valid
    citations (i.e., ones where the filename + chunk_index match
    a real chunk we provided).

    This prevents Claude from fabricating citation references —
    any citation not found in ranked_chunks is silently dropped
    (the Shepardizer agent in Step 7 will flag this as a broken citation).

    Args:
        response_text : Raw text from Claude
        ranked_chunks : The chunks actually provided in the prompt

    Returns:
        Deduplicated list of CitedSource objects
    """
    # Pattern matches: [SOURCE: some_file.pdf | Chunk 4]
    citation_pattern = re.compile(
        r'\[SOURCE:\s*(.+?)\s*\|\s*Chunk\s*(\d+)\s*\]',
        re.IGNORECASE,
    )

    matches = citation_pattern.findall(response_text)

    if not matches:
        logger.warning(
            "No citations found in Claude response — potential grounding issue",
            response_preview=response_text[:200],
        )
        return []

    # Build a lookup map from (filename, chunk_index) → RankedChunk
    chunk_map: dict[tuple[str, int], RankedChunk] = {}
    for chunk in ranked_chunks:
        key = (chunk.filename.strip().lower(), chunk.chunk_index)
        chunk_map[key] = chunk

    # Resolve citations against actual chunks, deduplicate
    seen_keys: set[tuple[str, int]] = set()
    cited_sources: list[CitedSource] = []

    for filename_raw, chunk_idx_str in matches:
        filename = filename_raw.strip()
        chunk_index = int(chunk_idx_str)
        key = (filename.lower(), chunk_index)

        if key in seen_keys:
            continue    # Deduplicate
        seen_keys.add(key)

        matched_chunk = chunk_map.get(key)
        if matched_chunk is None:
            logger.warning(
                "Citation references unknown chunk — will be flagged by Shepardizer",
                filename=filename,
                chunk_index=chunk_index,
            )
            # Still include it but with minimal info so the Shepardizer can flag it
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
# Audit log writer
# ──────────────────────────────────────────────────────────────────
async def _save_query_log(
    query: str,
    response: str,
    cited_sources: list[CitedSource],
    latency_ms: int,
    session_id: Optional[str],
    cache_hit: bool,
) -> None:
    """
    Persist the query + response to the query_logs Postgres table.
    Non-fatal — if this fails we log the error but don't raise,
    so a DB write failure never breaks the user-facing response.
    """
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
