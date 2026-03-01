"""
core/agents/adversarial_lawyer.py
==================================
Adversarial Lawyer Agent — generates a high-quality "Golden Dataset"
of synthetic question-context-answer triples by analysing ingested
legal documents.

Role (from spec):
  "It analyses internal case files to find complex, multi-hop legal
   questions (e.g., 'How does Clause X in Contract A interact with
   the liability limits in Contract B?'). It creates ground-truth
   pairs (Question, Reference Context, Expected Answer) that you can
   use as a benchmark to measure your RAG system's performance."

How it works:
  1. Loads a random sample of chunks from the Golden Dataset corpus
     (Postgres chunks table, status='indexed')
  2. Groups chunks by document so it can generate single-doc AND
     cross-document (multi-hop) questions
  3. Calls Claude with a specialised "question generator" prompt
     that instructs it to create adversarial, legally specific
     questions that require precise clause-level reasoning
  4. Parses the structured JSON response into GoldenDatasetEntry rows
  5. Persists entries to the golden_dataset Postgres table

Design Pattern: Agent Pattern (autonomous task execution)
  The agent operates independently — it reads from the DB,
  calls an LLM, and writes results back without human intervention.
  It is triggered either manually via the /evaluate/generate-dataset
  API endpoint or automatically before a CI/CD eval run.
"""

import json
import random
import uuid
from typing import Optional

import anthropic
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import get_settings
from core.db import get_db_context

logger = structlog.get_logger(__name__)
settings = get_settings()

# ── Prompt for the Adversarial Lawyer ─────────────────────────────
_ADVERSARIAL_LAWYER_SYSTEM_PROMPT = """You are an adversarial legal examiner creating a benchmark test suite.
Your job is to generate challenging, legally precise question-answer pairs from the provided document chunks.

RULES:
1. Generate questions that require specific clause-level reasoning — not vague summaries.
2. Include multi-hop questions where the answer requires synthesising information from MULTIPLE chunks.
3. Questions should be adversarial — they should trip up a RAG system that retrieves the wrong chunks.
4. Every answer must be grounded ONLY in the provided text. Do not add external legal knowledge.
5. Include edge cases: conflicting clauses, exceptions to rules, defined-term dependencies.

OUTPUT FORMAT — respond ONLY with a valid JSON array, no preamble or markdown:
[
  {
    "question": "...",
    "reference_context": "The exact text from the chunks that answers this question",
    "expected_answer": "A precise, citation-backed answer",
    "question_type": "single_hop | multi_hop | edge_case",
    "source_filenames": ["filename1.pdf", "filename2.pdf"]
  }
]
"""

_ADVERSARIAL_LAWYER_USER_TEMPLATE = """Below are {chunk_count} legal document chunks. 
Generate exactly {n_questions} question-answer pairs following the rules above.

Ensure at least:
- {n_single} single-hop questions (answerable from one chunk)
- {n_multi} multi-hop questions (require synthesising 2+ chunks)  
- {n_edge} edge case questions (exceptions, conflicts, or defined-term traps)

DOCUMENT CHUNKS:
{chunks_text}

Generate {n_questions} QA pairs as a JSON array:"""


class AdversarialLawyerAgent:
    """
    Generates synthetic QA pairs for the evaluation Golden Dataset.

    The "adversarial" framing means questions are designed to be hard —
    they target the exact failure modes of RAG systems:
    - Questions requiring exact clause numbers
    - Cross-document reasoning
    - Defined-term lookups
    - Exception clauses that override general rules
    """

    def __init__(self):
        self._client: Optional[anthropic.Anthropic] = None

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        return self._client

    async def generate_dataset(
        self,
        target_size: Optional[int] = None,
        batch_size: int = 10,
        max_chunks_per_batch: int = 8,
    ) -> int:
        """
        Generate synthetic QA pairs and persist them to the golden_dataset table.

        Args:
            target_size           : Total QA pairs to generate (default from config)
            batch_size            : Questions per LLM call
            max_chunks_per_batch  : Chunks to show the LLM per call

        Returns:
            Number of QA pairs successfully generated and stored
        """
        target = target_size or settings.golden_dataset_size
        log = logger.bind(target_size=target, batch_size=batch_size)
        log.info("Adversarial Lawyer Agent starting")

        # ── Load available chunks from Postgres ───────────────────
        chunks = await self._load_indexed_chunks()
        if not chunks:
            log.warning("No indexed chunks found — cannot generate dataset")
            return 0

        log.info("Loaded chunks for generation", total_chunks=len(chunks))

        # ── Generate in batches ───────────────────────────────────
        total_generated = 0
        batches_needed = (target + batch_size - 1) // batch_size

        for batch_num in range(batches_needed):
            remaining = target - total_generated
            n_questions = min(batch_size, remaining)

            # Sample a diverse set of chunks for this batch
            batch_chunks = self._sample_chunks(
                chunks,
                n=max_chunks_per_batch,
            )

            log.info(
                "Generating batch",
                batch=batch_num + 1,
                total_batches=batches_needed,
                n_questions=n_questions,
            )

            try:
                entries = await self._generate_batch(
                    chunks=batch_chunks,
                    n_questions=n_questions,
                )
                stored = await self._store_entries(entries, batch_chunks)
                total_generated += stored
                log.info("Batch stored", stored=stored, total=total_generated)

            except Exception as exc:
                log.error("Batch generation failed", batch=batch_num + 1, error=str(exc))
                continue   # Keep going with remaining batches

        log.info("Adversarial Lawyer Agent complete", total_generated=total_generated)
        return total_generated

    # ── LLM call ──────────────────────────────────────────────────
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        reraise=True,
    )
    async def _generate_batch(
        self,
        chunks: list[dict],
        n_questions: int,
    ) -> list[dict]:
        """
        Call Claude to generate n_questions QA pairs from the given chunks.

        Returns:
            List of raw QA pair dicts parsed from Claude's JSON response
        """
        import asyncio

        # Format chunks for the prompt
        chunks_text = "\n\n".join([
            f"[CHUNK from {c['filename']} | Index {c['chunk_index']}]\n{c['text']}"
            for c in chunks
        ])

        # Distribute question types
        n_single = max(1, n_questions // 2)
        n_multi  = max(1, n_questions // 3)
        n_edge   = max(1, n_questions - n_single - n_multi)

        user_message = _ADVERSARIAL_LAWYER_USER_TEMPLATE.format(
            chunk_count=len(chunks),
            n_questions=n_questions,
            n_single=n_single,
            n_multi=n_multi,
            n_edge=n_edge,
            chunks_text=chunks_text,
        )

        client = self._get_client()
        loop = asyncio.get_event_loop()

        response = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model=settings.anthropic_model,
                max_tokens=4096,
                system=_ADVERSARIAL_LAWYER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
        )

        raw_text = response.content[0].text.strip()

        # Strip markdown fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        qa_pairs = json.loads(raw_text.strip())

        if not isinstance(qa_pairs, list):
            raise ValueError("Expected JSON array from Adversarial Lawyer")

        return qa_pairs

    # ── Persist to DB ──────────────────────────────────────────────
    async def _store_entries(
        self,
        qa_pairs: list[dict],
        source_chunks: list[dict],
    ) -> int:
        """
        Persist generated QA pairs to the golden_dataset table.

        Returns:
            Number of entries successfully stored
        """
        from core.models.db_models import GoldenDatasetEntry

        # Build a filename → document_id lookup from the source chunks
        filename_to_doc_id: dict[str, uuid.UUID] = {}
        for chunk in source_chunks:
            filename_to_doc_id[chunk["filename"]] = chunk["document_id"]

        stored = 0
        async with get_db_context() as db:
            for pair in qa_pairs:
                try:
                    # Resolve source doc IDs from filenames
                    source_filenames = pair.get("source_filenames", [])
                    source_doc_ids = [
                        filename_to_doc_id[fn]
                        for fn in source_filenames
                        if fn in filename_to_doc_id
                    ]

                    entry = GoldenDatasetEntry(
                        question=pair["question"],
                        reference_context=pair["reference_context"],
                        expected_answer=pair["expected_answer"],
                        source_doc_ids=source_doc_ids or None,
                        generated_by="adversarial_lawyer_agent",
                    )
                    db.add(entry)
                    stored += 1
                except (KeyError, Exception) as exc:
                    logger.warning("Skipping malformed QA pair", error=str(exc))
                    continue

        return stored

    # ── Data loading helpers ───────────────────────────────────────
    async def _load_indexed_chunks(self) -> list[dict]:
        """Load all indexed chunks from Postgres as plain dicts."""
        from sqlalchemy import select
        from core.models.db_models import Chunk, Document

        async with get_db_context() as db:
            stmt = (
                select(
                    Chunk.id,
                    Chunk.text,
                    Chunk.chunk_index,
                    Document.id.label("document_id"),
                    Document.filename,
                    Document.doc_type,
                    Document.client_id,
                )
                .join(Document, Chunk.document_id == Document.id)
                .where(Document.status == "indexed")
                .where(Chunk.token_count > 50)   # Skip trivially short chunks
            )
            result = await db.execute(stmt)
            rows = result.fetchall()

        return [
            {
                "id": str(row.id),
                "text": row.text,
                "chunk_index": row.chunk_index,
                "document_id": row.document_id,
                "filename": row.filename,
                "doc_type": row.doc_type,
                "client_id": row.client_id,
            }
            for row in rows
        ]

    def _sample_chunks(
        self,
        chunks: list[dict],
        n: int,
    ) -> list[dict]:
        """
        Sample n chunks, favouring diversity across different documents.
        For multi-hop question generation we want chunks from different
        files in the same batch.
        """
        if len(chunks) <= n:
            return chunks

        # Group by document
        by_doc: dict[str, list[dict]] = {}
        for chunk in chunks:
            doc_id = str(chunk["document_id"])
            by_doc.setdefault(doc_id, []).append(chunk)

        sampled: list[dict] = []
        doc_ids = list(by_doc.keys())
        random.shuffle(doc_ids)

        # Round-robin across documents
        while len(sampled) < n:
            for doc_id in doc_ids:
                if len(sampled) >= n:
                    break
                doc_chunks = by_doc[doc_id]
                if doc_chunks:
                    sampled.append(
                        doc_chunks.pop(random.randrange(len(doc_chunks)))
                    )

        return sampled


# ── Module-level singleton ─────────────────────────────────────────
adversarial_lawyer = AdversarialLawyerAgent()
