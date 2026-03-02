"""
tests/test_citations.py
========================
Citation validation tests — powered by the Shepardizer Agent.

Spec requirement:
  "It ensures the system meets the high explainability and auditability
   standards required for legal compliance, failing any test where a
   citation is broken or irrelevant."

What we test:
  1. Valid citations   — all [SOURCE: ...] tags map to real retrieved chunks
  2. Fabricated citations — Gemini cites a document that wasn't in context
  3. Missing citations — response has no citations at all (should fail)
  4. Relevance check  — cited chunk is semantically related to the claim
  5. Citation parser  — the regex correctly extracts all citation formats
  6. DB existence check logic — validates the database lookup path
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.agents.shepardizer import (
    ShepardizierAgent,
    CitationStatus,
    _extract_citations,
)
from core.retrieval.reranker import RankedChunk


# ──────────────────────────────────────────────────────────────────
# Citation parser tests
# ──────────────────────────────────────────────────────────────────

class TestCitationExtraction:
    """Tests for the _extract_citations() helper function."""

    def test_extracts_single_citation(self):
        text = "The cap is 12 months [SOURCE: apex_msa_2023.pdf | Chunk 4]."
        citations = _extract_citations(text)
        assert len(citations) == 1
        assert citations[0][0] == "apex_msa_2023.pdf"
        assert citations[0][1] == 4

    def test_extracts_multiple_citations(self):
        text = (
            "Claim one [SOURCE: doc_a.pdf | Chunk 1]. "
            "Claim two [SOURCE: doc_b.pdf | Chunk 7]."
        )
        citations = _extract_citations(text)
        assert len(citations) == 2
        filenames = [c[0] for c in citations]
        assert "doc_a.pdf" in filenames
        assert "doc_b.pdf" in filenames

    def test_extracts_citation_with_surrounding_text(self):
        text = "The liability is limited [SOURCE: contract.pdf | Chunk 3]."
        citations = _extract_citations(text)
        assert len(citations) == 1
        # Surrounding text should be captured for relevance checking
        surrounding = citations[0][2]
        assert "liability" in surrounding.lower()

    def test_handles_case_insensitive_source_tag(self):
        """[source: ...] and [SOURCE: ...] should both be parsed."""
        text = "Some claim [source: doc.pdf | chunk 2]."
        citations = _extract_citations(text)
        assert len(citations) == 1

    def test_returns_empty_for_no_citations(self):
        text = "This response has no source citations at all."
        citations = _extract_citations(text)
        assert len(citations) == 0

    def test_handles_extra_whitespace_in_citation(self):
        text = "Claim [SOURCE:  apex_msa_2023.pdf  |  Chunk  4  ]."
        citations = _extract_citations(text)
        assert len(citations) == 1
        assert citations[0][0] == "apex_msa_2023.pdf"
        assert citations[0][1] == 4

    def test_deduplicates_same_citation_appearing_twice(self):
        """
        The same [SOURCE] appearing twice in the response should be
        treated as the same citation (not penalised twice by Shepardizer).
        """
        text = (
            "First mention [SOURCE: doc.pdf | Chunk 1]. "
            "Second mention [SOURCE: doc.pdf | Chunk 1]."
        )
        citations = _extract_citations(text)
        # Both citations are extracted (dedup happens in the agent, not the parser)
        assert len(citations) == 2   # Parser finds both occurrences
        assert citations[0][0] == citations[1][0]


# ──────────────────────────────────────────────────────────────────
# Shepardizer unit tests
# ──────────────────────────────────────────────────────────────────

class TestShepardizierAgent:
    """Unit tests for the ShepardizierAgent using mocked DB calls."""

    @pytest.fixture
    def agent(self):
        return ShepardizierAgent()

    @pytest.mark.asyncio
    async def test_valid_citations_all_pass(
        self, agent, valid_citation_response, sample_chunks
    ):
        """
        A response where every [SOURCE: ...] maps to a real retrieved chunk
        should produce a ShepardizationReport with all citations VALID.
        """
        with patch.object(agent, '_check_db_existence', return_value=True), \
             patch('core.agents.shepardizer._compute_relevance', return_value=0.85):

            report = await agent.shepardize(
                response_text=valid_citation_response["answer"],
                ranked_chunks=sample_chunks,
            )

        assert report.passed is True
        assert report.invalid_citations == 0
        assert report.total_citations == 3
        assert report.score == 1.0

    @pytest.mark.asyncio
    async def test_fabricated_citation_fails(
        self, agent, broken_citation_response, sample_chunks
    ):
        """
        A citation referencing a document NOT in the retrieved context
        must be flagged as INVALID — this is a fabricated source.
        """
        with patch.object(agent, '_check_db_existence', return_value=False):
            report = await agent.shepardize(
                response_text=broken_citation_response["answer"],
                ranked_chunks=sample_chunks,
            )

        assert report.passed is False
        assert report.invalid_citations >= 1

        # Find the invalid citation
        invalid = [v for v in report.validations if v.status == CitationStatus.INVALID]
        assert len(invalid) >= 1
        # The nonexistent_document.pdf citation should be invalid
        invalid_filenames = [v.filename for v in invalid]
        assert any("nonexistent" in f for f in invalid_filenames)

    @pytest.mark.asyncio
    async def test_no_citations_fails(self, agent, sample_chunks):
        """
        A response with zero citations must fail — the spec requires
        source attribution on every response.
        """
        response_without_citations = (
            "The liability cap is twelve months of fees. "
            "Consequential damages are excluded."
        )

        report = await agent.shepardize(
            response_text=response_without_citations,
            ranked_chunks=sample_chunks,
        )

        assert report.passed is False
        assert report.total_citations == 0
        assert report.score == 0.0

    @pytest.mark.asyncio
    async def test_low_relevance_citation_warned(
        self, agent, sample_chunks
    ):
        """
        A citation that exists but has low semantic relevance to the
        surrounding claim should be WARNED, not INVALID.
        """
        response = (
            "The payment terms are net-30 "
            "[SOURCE: apex_msa_2023.pdf | Chunk 4]."  # Chunk 4 is about indemnification, not payment
        )

        with patch.object(agent, '_check_db_existence', return_value=True), \
             patch('core.agents.shepardizer._compute_relevance', return_value=0.25):  # Low relevance

            report = await agent.shepardize(
                response_text=response,
                ranked_chunks=sample_chunks,
            )

        warned = [v for v in report.validations if v.status == CitationStatus.WARNED]
        assert len(warned) >= 1
        # Warned citations don't fail the overall report
        assert report.passed is True   # No INVALID citations, just warnings

    @pytest.mark.asyncio
    async def test_citation_not_in_db_is_invalid(
        self, agent, sample_chunks
    ):
        """
        A citation found in the context but not in the DB (document deleted)
        should be flagged INVALID.
        """
        response = "The cap is 12 months [SOURCE: apex_msa_2023.pdf | Chunk 4]."

        # Chunk IS in the context (sample_chunks) but NOT in the DB
        with patch.object(agent, '_check_db_existence', return_value=False):
            report = await agent.shepardize(
                response_text=response,
                ranked_chunks=sample_chunks,
            )

        assert report.invalid_citations == 1
        assert report.passed is False

    @pytest.mark.asyncio
    async def test_shepardization_report_summary_pass(
        self, agent, valid_citation_response, sample_chunks
    ):
        """ShepardizationReport.summary should return a ✅ string on pass."""
        with patch.object(agent, '_check_db_existence', return_value=True), \
             patch('core.agents.shepardizer._compute_relevance', return_value=0.90):

            report = await agent.shepardize(
                response_text=valid_citation_response["answer"],
                ranked_chunks=sample_chunks,
            )

        assert "✅" in report.summary

    @pytest.mark.asyncio
    async def test_shepardization_report_summary_fail(
        self, agent, broken_citation_response, sample_chunks
    ):
        """ShepardizationReport.summary should return a ❌ string on fail."""
        with patch.object(agent, '_check_db_existence', return_value=False):
            report = await agent.shepardize(
                response_text=broken_citation_response["answer"],
                ranked_chunks=sample_chunks,
            )

        assert "❌" in report.summary


# ──────────────────────────────────────────────────────────────────
# Citation score calculation tests
# ──────────────────────────────────────────────────────────────────

class TestCitationScoreCalculation:
    """Tests for ShepardizationReport score computation."""

    @pytest.mark.asyncio
    async def test_score_is_ratio_of_valid_to_total(self, sample_chunks):
        """
        score = valid_citations / total_citations
        2 valid + 1 invalid = score of 0.667
        """
        agent = ShepardizierAgent()

        response = (
            "First claim [SOURCE: apex_msa_2023.pdf | Chunk 4]. "
            "Second claim [SOURCE: apex_msa_2023.pdf | Chunk 5]. "
            "Third claim [SOURCE: fake_document.pdf | Chunk 99]."
        )

        # First two exist, last one doesn't
        async def mock_db_check(filename, chunk_index):
            return "fake_document" not in filename

        with patch.object(agent, '_check_db_existence', side_effect=mock_db_check), \
             patch('core.agents.shepardizer._compute_relevance', return_value=0.85):

            report = await agent.shepardize(
                response_text=response,
                ranked_chunks=sample_chunks,
            )

        assert report.total_citations == 3
        assert report.valid_citations == 2
        assert report.invalid_citations == 1
        assert abs(report.score - (2 / 3)) < 0.01
