"""
tests/conftest.py
=================
Shared pytest fixtures for the LegalMind test suite.

Fixtures provided:
  - mock_settings       : Overrides config with test values (no real API keys needed)
  - sample_chunks       : A list of realistic legal RankedChunk objects
  - sample_golden_entry : A single golden dataset QA pair
  - grounded_response   : A well-formed, fully-cited RAG response (should PASS)
  - hallucinated_response: A response that claims facts not in context (should FAIL)
  - broken_citation_response: A response with a citation to a non-existent chunk
  - valid_citation_response : A response with correct citations (should PASS)

Design note:
  All fixtures use real text structures but avoid requiring live API calls.
  Tests that DO require live APIs are marked @pytest.mark.integration
  and skipped in CI unless LEGALMIND_INTEGRATION_TESTS=true is set.
"""

import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.retrieval.reranker import RankedChunk


# ──────────────────────────────────────────────────────────────────
# Environment guard — skip integration tests unless explicitly enabled
# ──────────────────────────────────────────────────────────────────
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require live API keys and running services",
    )


def pytest_collection_modifyitems(config, items):
    run_integration = os.getenv("LEGALMIND_INTEGRATION_TESTS", "false").lower() == "true"
    skip_integration = pytest.mark.skip(reason="Set LEGALMIND_INTEGRATION_TESTS=true to run")
    for item in items:
        if "integration" in item.keywords and not run_integration:
            item.add_marker(skip_integration)


# ──────────────────────────────────────────────────────────────────
# Settings override
# ──────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """
    Override settings for all tests — no real API keys required.
    Tests that need real keys must be marked @pytest.mark.integration.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY",  "test-anthropic-key")
    monkeypatch.setenv("COHERE_API_KEY",     "test-cohere-key")
    monkeypatch.setenv("POSTGRES_URL",       "postgresql+asyncpg://test:test@localhost/test")
    monkeypatch.setenv("REDIS_URL",          "redis://localhost:6379/0")
    monkeypatch.setenv("QDRANT_HOST",        "localhost")
    monkeypatch.setenv("MIN_FAITHFULNESS_SCORE", "0.9")
    monkeypatch.setenv("ENVIRONMENT",        "development")

    # Clear the lru_cache so the new env vars are picked up
    from core.config import get_settings
    get_settings.cache_clear()

    yield

    get_settings.cache_clear()


# ──────────────────────────────────────────────────────────────────
# Sample legal document chunks
# ──────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_chunks() -> list[RankedChunk]:
    """
    Realistic legal RankedChunk objects representing retrieved context.
    Based on a fictional Master Services Agreement between Apex Corp and Vendor Ltd.
    """
    doc_id = str(uuid.uuid4())

    return [
        RankedChunk(
            qdrant_id=str(uuid.uuid4()),
            document_id=doc_id,
            filename="apex_msa_2023.pdf",
            text=(
                "Section 8.1 — Indemnification. Vendor shall indemnify, defend, and hold "
                "harmless Apex Corp and its officers, directors, and employees from and against "
                "any claims, damages, losses, and expenses arising out of or resulting from "
                "Vendor's breach of this Agreement. The aggregate liability of Vendor under "
                "this Section 8.1 shall not exceed the total fees paid by Apex Corp to Vendor "
                "in the twelve (12) months immediately preceding the claim."
            ),
            chunk_index=4,
            relevance_score=0.94,
            original_rank=1,
            payload={
                "doc_type": "contract",
                "client_id": "APEX-001",
                "date_filed": "2023-06-15",
                "page_number": 12,
            },
        ),
        RankedChunk(
            qdrant_id=str(uuid.uuid4()),
            document_id=doc_id,
            filename="apex_msa_2023.pdf",
            text=(
                "Section 8.3 — Exclusion of Consequential Damages. IN NO EVENT SHALL EITHER "
                "PARTY BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR "
                "CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, LOSS OF DATA, OR BUSINESS "
                "INTERRUPTION, HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN "
                "CONTRACT, STRICT LIABILITY, OR TORT, EVEN IF ADVISED OF THE POSSIBILITY "
                "OF SUCH DAMAGES."
            ),
            chunk_index=5,
            relevance_score=0.87,
            original_rank=2,
            payload={
                "doc_type": "contract",
                "client_id": "APEX-001",
                "date_filed": "2023-06-15",
                "page_number": 12,
            },
        ),
        RankedChunk(
            qdrant_id=str(uuid.uuid4()),
            document_id=str(uuid.uuid4()),
            filename="apex_amendment_2024.pdf",
            text=(
                "Amendment No. 2 to the Master Services Agreement dated June 15, 2023. "
                "The parties agree to increase the liability cap set forth in Section 8.1 "
                "from twelve (12) months of fees to twenty-four (24) months of fees, "
                "effective January 1, 2024. All other terms of the Agreement remain unchanged."
            ),
            chunk_index=2,
            relevance_score=0.81,
            original_rank=3,
            payload={
                "doc_type": "contract",
                "client_id": "APEX-001",
                "date_filed": "2024-01-01",
                "page_number": 1,
            },
        ),
    ]


# ──────────────────────────────────────────────────────────────────
# Sample golden dataset entry
# ──────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_golden_entry() -> dict:
    return {
        "question": (
            "What is the liability cap for indemnification under the Apex MSA, "
            "and was it subsequently amended?"
        ),
        "reference_context": (
            "Section 8.1 limits Vendor liability to 12 months of fees. "
            "Amendment No. 2 increased this to 24 months effective January 1, 2024."
        ),
        "expected_answer": (
            "The original Master Services Agreement limits the Vendor's indemnification "
            "liability to the total fees paid in the preceding 12 months. This was "
            "subsequently amended by Amendment No. 2, which increased the cap to 24 "
            "months of fees, effective January 1, 2024."
        ),
    }


# ──────────────────────────────────────────────────────────────────
# Sample RAG responses for evaluation tests
# ──────────────────────────────────────────────────────────────────
@pytest.fixture
def grounded_response(sample_chunks) -> dict:
    """
    A well-formed RAG response that is fully grounded in the context.
    Should PASS faithfulness evaluation.
    """
    return {
        "question": "What is the indemnification liability cap in the Apex contract?",
        "answer": (
            "The Apex Master Services Agreement limits the Vendor's indemnification "
            "liability to the total fees paid by Apex Corp in the twelve (12) months "
            "immediately preceding the claim [SOURCE: apex_msa_2023.pdf | Chunk 4]. "
            "Consequential damages, including lost profits and business interruption, "
            "are explicitly excluded under Section 8.3 [SOURCE: apex_msa_2023.pdf | Chunk 5]. "
            "Additionally, Amendment No. 2 increased this cap to twenty-four (24) months "
            "of fees effective January 1, 2024 [SOURCE: apex_amendment_2024.pdf | Chunk 2].\n\n"
            "**Sources Referenced:**\n"
            "- [SOURCE: apex_msa_2023.pdf | Chunk 4] — Indemnification liability cap (12 months)\n"
            "- [SOURCE: apex_msa_2023.pdf | Chunk 5] — Exclusion of consequential damages\n"
            "- [SOURCE: apex_amendment_2024.pdf | Chunk 2] — Amendment increasing cap to 24 months"
        ),
        "context": [c.text for c in sample_chunks],
        "sources": sample_chunks,
    }


@pytest.fixture
def hallucinated_response(sample_chunks) -> dict:
    """
    A RAG response that fabricates facts not present in the context.
    Should FAIL faithfulness evaluation.
    The context says nothing about a $5M cap or insurance requirements.
    """
    return {
        "question": "What is the indemnification liability cap in the Apex contract?",
        "answer": (
            "The Apex contract sets a hard indemnification cap of $5,000,000 USD, "
            "which is the maximum liability Vendor can incur under any circumstances. "
            "Additionally, Vendor is required to maintain professional liability insurance "
            "of at least $10,000,000 per occurrence as a condition of this indemnification. "
            "[SOURCE: apex_msa_2023.pdf | Chunk 4]"
        ),
        "context": [c.text for c in sample_chunks],
        "sources": sample_chunks,
    }


@pytest.fixture
def broken_citation_response(sample_chunks) -> dict:
    """
    A response with a citation referencing a chunk that doesn't exist.
    Should FAIL Shepardizer citation validation.
    """
    return {
        "answer": (
            "The liability cap is twelve months of fees "
            "[SOURCE: apex_msa_2023.pdf | Chunk 4]. "
            "There is also a force majeure provision that suspends obligations "
            "[SOURCE: nonexistent_document.pdf | Chunk 99]."  # This chunk doesn't exist
        ),
        "sources": sample_chunks,
    }


@pytest.fixture
def valid_citation_response(sample_chunks) -> dict:
    """
    A response with all citations correctly referencing provided chunks.
    Should PASS Shepardizer citation validation.
    """
    return {
        "answer": (
            "The indemnification cap is twelve months of fees "
            "[SOURCE: apex_msa_2023.pdf | Chunk 4]. "
            "Consequential damages are excluded [SOURCE: apex_msa_2023.pdf | Chunk 5]. "
            "This cap was later increased to 24 months [SOURCE: apex_amendment_2024.pdf | Chunk 2]."
        ),
        "sources": sample_chunks,
    }
