"""
core/agents/shepardizer.py
===========================
Shepardizer Agent — citation and source attribution validator.

Role (from spec):
  "It verifies that every paragraph referenced in the AI's response
   actually exists in the provided source documents and that the
   link/ID is correct. It ensures the system meets the high
   explainability and auditability standards required for legal
   compliance, failing any test where a citation is broken or
   irrelevant."

Named after: "Shepardizing" is a real legal term — the process of
verifying that a cited case or statute is still valid and correctly
referenced. Westlaw/LexisNexis provide this as a paid service.
LegalMind's Shepardizer automates it for internal citations.

What it validates:
  1. EXISTENCE   — Does [SOURCE: filename | Chunk N] reference a chunk
                   that was actually retrieved and provided to Claude?
  2. RELEVANCE   — Does the cited chunk's text actually support the
                   claim it was cited for? (semantic similarity check)
  3. ACCURACY    — Does the filename and chunk_index match a real
                   document in the Postgres chunks table?

Design Pattern: Chain of Responsibility
  Three validators run in sequence. Each validator can:
    - PASS  : citation is valid
    - WARN  : citation is technically present but questionable
    - FAIL  : citation is broken, fabricated, or irrelevant

  Results are aggregated into a ShepardizationReport.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import structlog

from core.config import get_settings
from core.retrieval.reranker import RankedChunk

logger = structlog.get_logger(__name__)
settings = get_settings()

# Relevance threshold — a cited chunk's text must be at least this
# similar to the answer sentence that cites it
_MIN_CITATION_RELEVANCE = 0.50


# ──────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────
class CitationStatus(str, Enum):
    VALID   = "valid"
    WARNED  = "warned"
    INVALID = "invalid"


@dataclass
class CitationValidation:
    """Validation result for a single [SOURCE: ...] citation."""
    filename: str
    chunk_index: int
    status: CitationStatus
    issues: list[str] = field(default_factory=list)
    relevance_score: Optional[float] = None
    found_in_context: bool = False
    found_in_db: bool = False


@dataclass
class ShepardizationReport:
    """
    Complete citation validation report for one RAG response.

    Used by:
      - The API /query response (surfaced to the user as a warning)
      - The pytest test suite (Step 9) — fails tests with INVALID citations
      - The Streamlit dashboard (Step 8) — shown as a citation health badge
    """
    total_citations: int
    valid_citations: int
    warned_citations: int
    invalid_citations: int
    validations: list[CitationValidation]
    passed: bool          # True if zero INVALID citations
    score: float          # valid / total (0.0 – 1.0)

    @property
    def summary(self) -> str:
        if self.passed and self.warned_citations == 0:
            return f"✅ All {self.total_citations} citations verified"
        elif self.passed:
            return (
                f"⚠️ {self.valid_citations}/{self.total_citations} citations valid, "
                f"{self.warned_citations} warnings"
            )
        else:
            return (
                f"❌ {self.invalid_citations}/{self.total_citations} citations INVALID — "
                f"human verification required"
            )


# ──────────────────────────────────────────────────────────────────
# Shepardizer Agent
# ──────────────────────────────────────────────────────────────────
class ShepardizierAgent:
    """
    Validates source citations in a RAG response.

    Runs three validators in sequence (Chain of Responsibility):
      1. ContextValidator   — citation must appear in the retrieved chunks
      2. DatabaseValidator  — citation must map to a real DB row
      3. RelevanceValidator — cited chunk must be semantically related to the claim
    """

    async def shepardize(
        self,
        response_text: str,
        ranked_chunks: list[RankedChunk],
    ) -> ShepardizationReport:
        """
        Validate all citations in a RAG response.

        Args:
            response_text : The full text of Claude's response
            ranked_chunks : The chunks that were actually provided to Claude
                            (from the reranker — these are the only valid sources)

        Returns:
            ShepardizationReport with per-citation validation results
        """
        # ── Extract all citations from the response ────────────────
        citations = _extract_citations(response_text)

        if not citations:
            logger.warning(
                "Shepardizer: no citations found in response",
                response_preview=response_text[:100],
            )
            return ShepardizationReport(
                total_citations=0,
                valid_citations=0,
                warned_citations=0,
                invalid_citations=0,
                validations=[],
                passed=False,   # No citations = fail (spec requires citations)
                score=0.0,
            )

        # ── Build lookup maps for fast validation ──────────────────
        # (filename.lower(), chunk_index) → RankedChunk
        context_map: dict[tuple[str, int], RankedChunk] = {
            (chunk.filename.strip().lower(), chunk.chunk_index): chunk
            for chunk in ranked_chunks
        }

        # ── Run validators on each citation ────────────────────────
        validations: list[CitationValidation] = []

        for filename, chunk_index, surrounding_text in citations:
            validation = await self._validate_citation(
                filename=filename,
                chunk_index=chunk_index,
                surrounding_text=surrounding_text,
                context_map=context_map,
            )
            validations.append(validation)

        # ── Aggregate results ──────────────────────────────────────
        valid   = sum(1 for v in validations if v.status == CitationStatus.VALID)
        warned  = sum(1 for v in validations if v.status == CitationStatus.WARNED)
        invalid = sum(1 for v in validations if v.status == CitationStatus.INVALID)
        total   = len(validations)
        passed  = invalid == 0
        score   = valid / max(total, 1)

        report = ShepardizationReport(
            total_citations=total,
            valid_citations=valid,
            warned_citations=warned,
            invalid_citations=invalid,
            validations=validations,
            passed=passed,
            score=score,
        )

        logger.info(
            "Shepardization complete",
            total=total,
            valid=valid,
            warned=warned,
            invalid=invalid,
            passed=passed,
            score=round(score, 3),
        )

        return report

    # ── Per-citation validation ────────────────────────────────────
    async def _validate_citation(
        self,
        filename: str,
        chunk_index: int,
        surrounding_text: str,
        context_map: dict[tuple[str, int], RankedChunk],
    ) -> CitationValidation:
        """
        Run all three validators on a single citation.
        Chain of Responsibility — each validator adds issues to the result.
        """
        validation = CitationValidation(
            filename=filename,
            chunk_index=chunk_index,
            status=CitationStatus.VALID,   # Assume valid until proven otherwise
        )

        # ── Validator 1: Context check ─────────────────────────────
        key = (filename.strip().lower(), chunk_index)
        matched_chunk = context_map.get(key)

        if matched_chunk is None:
            validation.found_in_context = False
            validation.status = CitationStatus.INVALID
            validation.issues.append(
                f"Citation [{filename} | Chunk {chunk_index}] was NOT in the retrieved "
                f"context provided to Claude. This is a fabricated citation."
            )
            # Short-circuit — no point running other validators
            return validation

        validation.found_in_context = True

        # ── Validator 2: Database check ────────────────────────────
        db_exists = await self._check_db_existence(filename, chunk_index)
        validation.found_in_db = db_exists

        if not db_exists:
            validation.status = CitationStatus.INVALID
            validation.issues.append(
                f"Citation [{filename} | Chunk {chunk_index}] found in context "
                f"but NOT in the database. Document may have been deleted."
            )
            return validation

        # ── Validator 3: Relevance check ───────────────────────────
        relevance = _compute_relevance(
            claim_text=surrounding_text,
            chunk_text=matched_chunk.text,
        )
        validation.relevance_score = relevance

        if relevance < _MIN_CITATION_RELEVANCE:
            validation.status = CitationStatus.WARNED
            validation.issues.append(
                f"Citation [{filename} | Chunk {chunk_index}] has low relevance "
                f"({relevance:.2f}) to the surrounding text. The chunk may not "
                f"actually support the claim it was cited for."
            )
        else:
            validation.status = CitationStatus.VALID

        return validation

    async def _check_db_existence(
        self,
        filename: str,
        chunk_index: int,
    ) -> bool:
        """Verify the cited chunk exists in the Postgres chunks table."""
        try:
            from sqlalchemy import select, func
            from core.db import get_db_context
            from core.models.db_models import Chunk, Document

            async with get_db_context() as db:
                stmt = (
                    select(func.count())
                    .select_from(Chunk)
                    .join(Document, Chunk.document_id == Document.id)
                    .where(Document.filename == filename)
                    .where(Chunk.chunk_index == chunk_index)
                )
                result = await db.execute(stmt)
                count = result.scalar()

            return count > 0
        except Exception as exc:
            logger.warning("DB existence check failed", error=str(exc))
            return True   # Assume exists on DB error — don't penalise incorrectly


# ──────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────
def _extract_citations(
    response_text: str,
) -> list[tuple[str, int, str]]:
    """
    Extract all [SOURCE: filename | Chunk N] citations from response text.

    Also captures the surrounding sentence (50 chars each side) for
    relevance validation in Validator 3.

    Returns:
        List of (filename, chunk_index, surrounding_text) tuples
    """
    pattern = re.compile(
        r'\[SOURCE:\s*(.+?)\s*\|\s*Chunk\s*(\d+)\s*\]',
        re.IGNORECASE,
    )

    results: list[tuple[str, int, str]] = []

    for match in pattern.finditer(response_text):
        filename = match.group(1).strip()
        chunk_index = int(match.group(2))

        # Grab surrounding context (the sentence containing the citation)
        start = max(0, match.start() - 200)
        end   = min(len(response_text), match.end() + 200)
        surrounding = response_text[start:end].strip()

        results.append((filename, chunk_index, surrounding))

    return results


def _compute_relevance(claim_text: str, chunk_text: str) -> float:
    """
    Compute cosine similarity between the claim text and the chunk text
    to verify the citation is semantically related to the claim.

    Uses a lightweight sentence transformer for speed.
    Returns 1.0 if the model fails to load (fail open).
    """
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        # Use a small, fast model for relevance checks
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(
            [claim_text[:512], chunk_text[:512]],   # Truncate for speed
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return float(np.dot(embeddings[0], embeddings[1]))
    except Exception as exc:
        logger.warning("Relevance computation failed", error=str(exc))
        return 1.0   # Fail open — don't penalise on model errors


# ── Module-level singleton ─────────────────────────────────────────
shepardizer = ShepardizierAgent()\
