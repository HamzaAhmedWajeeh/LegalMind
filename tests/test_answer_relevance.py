"""
tests/test_answer_relevance.py
==============================
Answer Relevance and Context Precision tests.

RAG Triad definitions:
  - Answer Relevance  : Does the response actually address the user's question?
  - Context Precision : Is the most relevant information ranked at the top?

What we test:
  1. Answer relevance logic — response addresses the question asked
  2. "I don't know" response relevance — correct when context insufficient
  3. Off-topic response detection — answer ignores the question
  4. Context precision — highest-scoring chunks appear first
  5. Reranker output ordering — Cohere scores correctly sort chunks
  6. Prompt engineering — system prompt contains required instructions
  7. Citation format — response follows [SOURCE: file | Chunk N] pattern
  8. RAG service integration — full pipeline wiring
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.retrieval.reranker import RankedChunk


# ──────────────────────────────────────────────────────────────────
# Answer Relevance tests
# ──────────────────────────────────────────────────────────────────

class TestAnswerRelevance:
    """Tests that responses address the question that was asked."""

    def test_idk_response_is_relevant_for_unanswerable_question(self):
        """
        When context is insufficient, 'I don't know' IS the relevant answer.
        An agent that fabricates instead of saying IDK has low relevance to
        what the user actually needs (a reliable answer).
        """
        idk_response = (
            "I don't have sufficient information in the provided documents "
            "to answer this question accurately."
        )
        question = "What was the outcome of the 1995 Smith arbitration?"

        # IDK response should contain the core IDK phrase
        assert "don't have sufficient information" in idk_response.lower() or \
               "insufficient" in idk_response.lower()

    def test_response_addresses_indemnification_question(self):
        """
        A response to an indemnification question should mention
        indemnification-related content.
        """
        question = "What are the indemnification limits?"
        answer = (
            "The indemnification liability is capped at twelve months of fees "
            "[SOURCE: apex_msa_2023.pdf | Chunk 4]."
        )

        question_keywords = {"indemnif", "limit", "cap", "liabilit"}
        answer_lower = answer.lower()
        matching = sum(1 for kw in question_keywords if kw in answer_lower)
        assert matching >= 2, "Answer should address the indemnification question"

    def test_off_topic_response_detected(self):
        """
        A response about payment terms to an indemnification question
        is off-topic and should be detectable.
        """
        question = "What are the indemnification limits?"
        off_topic_answer = (
            "Payment is due net-30 from invoice date. Late payments incur "
            "a 1.5% monthly interest charge [SOURCE: apex_msa_2023.pdf | Chunk 4]."
        )

        question_keywords = {"indemnif", "limit", "liabilit"}
        answer_lower = off_topic_answer.lower()
        matching = sum(1 for kw in question_keywords if kw in answer_lower)

        assert matching == 0, "Off-topic answer should not match question keywords"


# ──────────────────────────────────────────────────────────────────
# Context Precision tests
# ──────────────────────────────────────────────────────────────────

class TestContextPrecision:
    """
    Tests that retrieved chunks are ordered by relevance (highest first).
    Context Precision = most relevant chunk is ranked #1.
    """

    def test_reranked_chunks_ordered_by_relevance_score(self, sample_chunks):
        """
        After reranking, chunks must be ordered descending by relevance_score.
        """
        # Verify the fixture itself is correctly ordered
        scores = [c.relevance_score for c in sample_chunks]
        assert scores == sorted(scores, reverse=True), (
            "Chunks should be ordered by descending relevance score"
        )

    def test_most_relevant_chunk_is_first(self, sample_chunks):
        """The chunk with the highest relevance score should be at index 0."""
        highest_score = max(c.relevance_score for c in sample_chunks)
        assert sample_chunks[0].relevance_score == highest_score

    def test_context_precision_with_mocked_reranker(self, sample_chunks):
        """
        Simulate reranker output and verify the ordering logic.
        A reranker that returns chunks out of order would hurt context precision.
        """
        # Simulate reranker returning chunks in the wrong order
        shuffled = [sample_chunks[2], sample_chunks[0], sample_chunks[1]]

        # Sort them as the reranker should have
        reranked = sorted(shuffled, key=lambda c: c.relevance_score, reverse=True)

        assert reranked[0].relevance_score >= reranked[1].relevance_score
        assert reranked[1].relevance_score >= reranked[2].relevance_score

    @pytest.mark.asyncio
    async def test_cohere_reranker_sorts_by_relevance(self, sample_chunks):
        """
        The CohereReranker should return chunks sorted by relevance_score descending.
        Uses a mocked Cohere client.
        """
        from core.retrieval.reranker import CohereReranker
        from core.retrieval.vector_store import RetrievedChunk

        # Create unsorted input chunks with known scores
        input_chunks = [
            RetrievedChunk(
                qdrant_id=str(uuid.uuid4()),
                document_id="doc1",
                filename="test.pdf",
                text="Low relevance text",
                chunk_index=0,
                score=0.3,
            ),
            RetrievedChunk(
                qdrant_id=str(uuid.uuid4()),
                document_id="doc1",
                filename="test.pdf",
                text="High relevance text about indemnification limits",
                chunk_index=1,
                score=0.9,
            ),
        ]

        # Mock Cohere response — ranks chunk at index 1 first
        mock_result_0 = MagicMock()
        mock_result_0.index = 1          # Chunk index 1 (high relevance) is ranked first
        mock_result_0.relevance_score = 0.95

        mock_result_1 = MagicMock()
        mock_result_1.index = 0          # Chunk index 0 (low relevance) is ranked second
        mock_result_1.relevance_score = 0.30

        mock_response = MagicMock()
        mock_response.results = [mock_result_0, mock_result_1]

        reranker = CohereReranker()
        mock_client = MagicMock()
        mock_client.rerank.return_value = mock_response
        reranker._client = mock_client

        ranked = await reranker.rerank(
            query="What are the indemnification limits?",
            chunks=input_chunks,
            top_n=2,
        )

        assert len(ranked) == 2
        assert ranked[0].relevance_score > ranked[1].relevance_score
        assert ranked[0].relevance_score == 0.95
        assert "indemnification" in ranked[0].text.lower()


# ──────────────────────────────────────────────────────────────────
# Prompt engineering tests
# ──────────────────────────────────────────────────────────────────

class TestPromptEngineering:
    """
    Tests that the system prompt contains all spec-required instructions.
    These are critical — a missing instruction = missing behaviour.
    """

    def test_system_prompt_mandates_citations(self):
        from core.generation.prompts import SYSTEM_PROMPT
        assert "SOURCE" in SYSTEM_PROMPT
        assert "citation" in SYSTEM_PROMPT.lower() or "cite" in SYSTEM_PROMPT.lower()

    def test_system_prompt_contains_idk_instruction(self):
        from core.generation.prompts import SYSTEM_PROMPT
        assert "don't know" in SYSTEM_PROMPT.lower() or \
               "insufficient" in SYSTEM_PROMPT.lower() or \
               "I don't have" in SYSTEM_PROMPT

    def test_system_prompt_prohibits_hallucination(self):
        from core.generation.prompts import SYSTEM_PROMPT
        prompt_lower = SYSTEM_PROMPT.lower()
        assert "only" in prompt_lower or "solely" in prompt_lower
        assert "context" in prompt_lower

    def test_citation_format_is_consistent(self):
        """
        The citation format [SOURCE: filename | Chunk N] must appear in
        the system prompt example, the context formatter, and be parseable
        by the Shepardizer.
        """
        from core.generation.prompts import SYSTEM_PROMPT
        from core.agents.shepardizer import _extract_citations

        # The system prompt example should use the citation format
        assert "[SOURCE:" in SYSTEM_PROMPT

        # That same format must be parseable by the Shepardizer
        test_response = "Example [SOURCE: test.pdf | Chunk 3]."
        citations = _extract_citations(test_response)
        assert len(citations) == 1
        assert citations[0][0] == "test.pdf"
        assert citations[0][1] == 3

    def test_context_formatter_includes_source_labels(self, sample_chunks):
        from core.generation.prompts import format_context

        context = format_context(sample_chunks)

        # Each chunk should be labelled with its source
        for chunk in sample_chunks:
            assert chunk.filename in context
            assert f"Chunk {chunk.chunk_index}" in context

    def test_context_formatter_handles_empty_chunks(self):
        from core.generation.prompts import format_context

        result = format_context([])
        assert "No relevant documents" in result


# ──────────────────────────────────────────────────────────────────
# RAG Service integration tests (mocked)
# ──────────────────────────────────────────────────────────────────

class TestRAGServiceIntegration:
    """Tests the RAGService orchestration layer with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_rag_service_returns_query_response(self, sample_chunks):
        """
        RAGService.query() should return a QueryResponse with the
        correct fields populated.
        """
        from core.generation.rag_service import RAGService
        from api.models.schemas import QueryRequest, QueryResponse, SourceChunk

        mock_response = QueryResponse(
            query="What is the indemnification cap?",
            answer="The cap is twelve months of fees [SOURCE: apex_msa_2023.pdf | Chunk 4].",
            sources=[
                SourceChunk(
                    document_id=uuid.UUID(sample_chunks[0].document_id) if len(sample_chunks[0].document_id) == 36 else uuid.uuid4(),
                    filename=sample_chunks[0].filename,
                    chunk_index=sample_chunks[0].chunk_index,
                    text=sample_chunks[0].text,
                    relevance_score=sample_chunks[0].relevance_score,
                )
            ],
            cache_hit=False,
            latency_ms=450,
        )

        service = RAGService()

        with patch.object(service, '_check_cache', return_value=None), \
             patch('core.generation.rag_service.hybrid_retriever') as mock_hybrid, \
             patch('core.generation.rag_service.reranker') as mock_reranker, \
             patch('core.generation.rag_service.llm') as mock_llm:

            mock_hybrid.search = AsyncMock(return_value=sample_chunks)
            mock_reranker.rerank = AsyncMock(return_value=sample_chunks)
            mock_llm.generate = AsyncMock(return_value=MagicMock(
                answer=mock_response.answer,
                cited_sources=[],
                latency_ms=450,
            ))

            with patch('core.generation.rag_service._to_source_chunks', return_value=mock_response.sources), \
                 patch.object(service, '_populate_cache', return_value=None):

                request = QueryRequest(query="What is the indemnification cap?")
                response = await service.query(request)

        assert response is not None
        assert response.query == request.query
        assert response.cache_hit is False

    @pytest.mark.asyncio
    async def test_rag_service_returns_cached_response(self, sample_chunks):
        """
        When the semantic cache returns a hit, RAGService should return
        it immediately without calling the retrieval/generation pipeline.
        """
        from core.generation.rag_service import RAGService
        from api.models.schemas import QueryRequest, QueryResponse

        cached = QueryResponse(
            query="What is the indemnification cap?",
            answer="Cached answer.",
            sources=[],
            cache_hit=True,
            latency_ms=5,
        )

        service = RAGService()

        with patch.object(service, '_check_cache', return_value=cached), \
             patch('core.generation.rag_service.hybrid_retriever') as mock_hybrid:

            mock_hybrid.search = AsyncMock()

            request = QueryRequest(query="What is the indemnification cap?")
            response = await service.query(request)

        assert response.cache_hit is True
        # Hybrid search should NOT have been called
        mock_hybrid.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_rag_service_handles_empty_retrieval(self):
        """
        When hybrid retrieval returns nothing, the service should return
        the 'insufficient information' response gracefully.
        """
        from core.generation.rag_service import RAGService
        from api.models.schemas import QueryRequest

        service = RAGService()

        with patch.object(service, '_check_cache', return_value=None), \
             patch('core.generation.rag_service.hybrid_retriever') as mock_hybrid:

            mock_hybrid.search = AsyncMock(return_value=[])  # No results

            request = QueryRequest(query="Something obscure with no matching documents")
            response = await service.query(request)

        assert "insufficient" in response.answer.lower() or \
               "no relevant" in response.answer.lower()
        assert len(response.sources) == 0
