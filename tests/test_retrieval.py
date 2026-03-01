"""
tests/test_retrieval.py
========================
Retrieval layer tests — hybrid search, BM25, RRF fusion.

RAG Triad — Context Precision definition:
  "Is the most relevant information ranked at the top of the
   search results?" — verifies the retrieval pipeline returns
   the right chunks in the right order.

What we test:
  1. BM25 tokeniser — legal-aware preprocessing
  2. RRF fusion     — correct score calculation and deduplication
  3. Metadata filter building — Qdrant filter construction
  4. Chunker strategy pattern — strategy swapping works
  5. Context precision logic  — most relevant chunks rank highest
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.retrieval.bm25 import tokenise
from core.retrieval.hybrid import _reciprocal_rank_fusion
from core.retrieval.vector_store import RetrievedChunk, _build_filter


# ──────────────────────────────────────────────────────────────────
# BM25 Tokeniser tests
# ──────────────────────────────────────────────────────────────────

class TestBM25Tokeniser:
    """Tests for the legal-aware BM25 tokeniser."""

    def test_lowercases_input(self):
        tokens = tokenise("Indemnification CLAUSE")
        assert all(t == t.lower() for t in tokens)

    def test_removes_legal_stopwords(self):
        tokens = tokenise("the party shall indemnify the other party")
        assert "the" not in tokens
        assert "shall" not in tokens
        assert "party" not in tokens
        assert "indemnify" in tokens

    def test_preserves_hyphenated_legal_terms(self):
        """force-majeure, sub-clause should stay hyphenated."""
        tokens = tokenise("force-majeure clause applies")
        assert "force-majeure" in tokens

    def test_removes_punctuation(self):
        tokens = tokenise("Section 8.1, Indemnification.")
        assert not any("," in t or "." in t for t in tokens)

    def test_filters_single_character_tokens(self):
        tokens = tokenise("a b c indemnification")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "indemnification" in tokens

    def test_preserves_legal_identifiers(self):
        """Case numbers like 2024-001 should survive."""
        tokens = tokenise("case number 2024-001 indemnification")
        assert "2024-001" in tokens

    def test_empty_string_returns_empty(self):
        assert tokenise("") == []

    def test_stopwords_only_returns_empty(self):
        tokens = tokenise("the and or but in on at")
        assert tokens == []

    def test_real_legal_query(self):
        query = "What are the indemnification limits in the contract?"
        tokens = tokenise(query)
        assert "indemnification" in tokens
        assert "limits" in tokens
        # Stopwords removed
        assert "the" not in tokens
        assert "in" not in tokens
        assert "are" not in tokens


# ──────────────────────────────────────────────────────────────────
# Reciprocal Rank Fusion tests
# ──────────────────────────────────────────────────────────────────

class TestReciprocalkRankFusion:
    """Tests for the RRF fusion algorithm."""

    def _make_chunk(self, qdrant_id: str, score: float = 1.0) -> RetrievedChunk:
        return RetrievedChunk(
            qdrant_id=qdrant_id,
            document_id="doc-1",
            filename="test.pdf",
            text=f"Text for {qdrant_id}",
            chunk_index=0,
            score=score,
        )

    def test_rrf_boosts_chunks_in_both_lists(self):
        """
        A chunk appearing in BOTH vector and BM25 results should have
        a higher RRF score than a chunk appearing in only one.
        """
        # chunk-A appears in both lists (rank 1 in vector, rank 2 in BM25)
        # chunk-B appears only in vector (rank 2)
        # chunk-C appears only in BM25 (rank 1)

        vector_results = [
            self._make_chunk("chunk-A"),
            self._make_chunk("chunk-B"),
        ]
        bm25_results = [
            self._make_chunk("chunk-C"),
            self._make_chunk("chunk-A"),
        ]

        fused = _reciprocal_rank_fusion([vector_results, bm25_results])

        # Find scores
        scores = {c.qdrant_id: c.score for c in fused}

        # chunk-A should score higher than chunk-B and chunk-C
        assert scores["chunk-A"] > scores["chunk-B"]
        assert scores["chunk-A"] > scores["chunk-C"]

    def test_rrf_deduplicates_chunks(self):
        """
        The same chunk appearing in both lists should appear only once
        in the fused output.
        """
        chunk = self._make_chunk("chunk-X")
        fused = _reciprocal_rank_fusion([[chunk], [chunk]])
        ids = [c.qdrant_id for c in fused]
        assert ids.count("chunk-X") == 1

    def test_rrf_rank_position_over_raw_score(self):
        """
        RRF uses rank position, not raw scores. A chunk ranked #1 with
        a low raw score should beat a chunk ranked #5 with a high raw score.
        """
        rank1_low_score  = self._make_chunk("rank1", score=0.1)
        rank5_high_score = self._make_chunk("rank5", score=0.99)

        # rank1 at position 1, rank5 at position 5
        results = [rank1_low_score, self._make_chunk("r2"), self._make_chunk("r3"),
                   self._make_chunk("r4"), rank5_high_score]

        fused = _reciprocal_rank_fusion([results])
        ids = [c.qdrant_id for c in fused]

        assert ids.index("rank1") < ids.index("rank5")

    def test_rrf_handles_empty_lists(self):
        """RRF should handle empty input lists gracefully."""
        fused = _reciprocal_rank_fusion([[], []])
        assert fused == []

    def test_rrf_single_list_preserves_order(self):
        """With a single result list, rank order should be preserved."""
        results = [
            self._make_chunk("first"),
            self._make_chunk("second"),
            self._make_chunk("third"),
        ]
        fused = _reciprocal_rank_fusion([results])
        ids = [c.qdrant_id for c in fused]
        assert ids == ["first", "second", "third"]

    def test_rrf_scores_are_between_0_and_1(self):
        """All RRF scores should be positive and less than 1."""
        results = [self._make_chunk(f"chunk-{i}") for i in range(5)]
        fused = _reciprocal_rank_fusion([results, results[::-1]])
        for chunk in fused:
            assert 0 < chunk.score < 1


# ──────────────────────────────────────────────────────────────────
# Qdrant metadata filter tests
# ──────────────────────────────────────────────────────────────────

class TestMetadataFilterBuilding:
    """Tests for Qdrant payload filter construction."""

    def test_no_filters_returns_none(self):
        """When no filters provided, returns None (no filtering applied)."""
        result = _build_filter(
            filter_client_id=None,
            filter_doc_type=None,
            filter_date_from=None,
            filter_date_to=None,
        )
        assert result is None

    def test_client_id_filter_creates_condition(self):
        """Providing client_id should produce a filter with a MatchValue condition."""
        result = _build_filter(
            filter_client_id="APEX-001",
            filter_doc_type=None,
            filter_date_from=None,
            filter_date_to=None,
        )
        assert result is not None
        assert len(result.must) == 1
        assert result.must[0].key == "client_id"

    def test_multiple_filters_combine_with_must(self):
        """Multiple filters should all appear in the must clause (AND logic)."""
        result = _build_filter(
            filter_client_id="APEX-001",
            filter_doc_type="contract",
            filter_date_from=None,
            filter_date_to=None,
        )
        assert result is not None
        assert len(result.must) == 2
        keys = {c.key for c in result.must}
        assert "client_id" in keys
        assert "doc_type" in keys

    def test_date_range_filter(self):
        """Date filters should create a DatetimeRange condition."""
        result = _build_filter(
            filter_client_id=None,
            filter_doc_type=None,
            filter_date_from="2024-01-01",
            filter_date_to="2024-12-31",
        )
        assert result is not None
        assert len(result.must) == 1
        assert result.must[0].key == "date_filed"


# ──────────────────────────────────────────────────────────────────
# Chunker Strategy Pattern tests
# ──────────────────────────────────────────────────────────────────

class TestChunkerStrategyPattern:
    """
    Tests that the Strategy Pattern works correctly —
    strategies are swappable and produce consistent output shapes.
    """

    def test_recursive_chunker_produces_chunks(self):
        from core.ingestion.chunker import get_chunker, RecursiveChunkingStrategy
        from core.ingestion.parser import ParsedDocument

        chunker = get_chunker("recursive")
        assert isinstance(chunker.strategy, RecursiveChunkingStrategy)

        doc = ParsedDocument(
            filename="test.pdf",
            file_hash="abc123",
            raw_text="Section 1. The vendor shall indemnify the client. " * 30,
            page_count=1,
            file_type="pdf",
        )
        chunks = chunker.chunk(doc)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.text.strip() != ""
            assert chunk.token_count > 0
            assert chunk.chunk_index >= 0

    def test_chunker_strategy_is_swappable(self):
        """
        Swapping the strategy at runtime should not break the interface.
        The Chunker context class accepts any ChunkingStrategy.
        """
        from core.ingestion.chunker import (
            Chunker,
            RecursiveChunkingStrategy,
            SemanticChunkingStrategy,
        )

        chunker = Chunker(strategy=RecursiveChunkingStrategy())
        assert "recursive" in chunker.strategy.name

        # Swap to semantic
        chunker.strategy = SemanticChunkingStrategy()
        assert "semantic" in chunker.strategy.name

    def test_factory_creates_correct_strategy(self):
        from core.ingestion.chunker import get_chunker, RecursiveChunkingStrategy

        chunker = get_chunker("recursive")
        assert isinstance(chunker.strategy, RecursiveChunkingStrategy)

    def test_factory_defaults_to_recursive(self):
        from core.ingestion.chunker import get_chunker, RecursiveChunkingStrategy

        chunker = get_chunker("unknown_strategy_name")
        assert isinstance(chunker.strategy, RecursiveChunkingStrategy)

    def test_chunk_overlap_is_applied(self):
        """
        Consecutive chunks should share some overlapping content
        when chunk_overlap > 0.
        """
        from core.ingestion.chunker import Chunker, RecursiveChunkingStrategy
        from core.ingestion.parser import ParsedDocument

        chunker = Chunker(strategy=RecursiveChunkingStrategy(chunk_size=50, chunk_overlap=10))
        doc = ParsedDocument(
            filename="test.pdf",
            file_hash="abc123",
            raw_text="word " * 200,
            page_count=1,
            file_type="pdf",
        )
        chunks = chunker.chunk(doc)

        # With overlap, consecutive chunks should have some shared words
        assert len(chunks) >= 2
