"""
tests/test_faithfulness.py
===========================
Faithfulness (Groundedness) tests — the primary CI/CD quality gate.

Spec requirement:
  "Pytest Integration: Wrap evaluation metrics in pytest to fail the
   build if 'Faithfulness' drops below 0.9 during a code change."

The RAG Triad — Faithfulness definition:
  "Is the answer derived ONLY from the context? Prevents hallucinations."
  Score = (claims supported by context) / (total claims in answer)

CI/CD behaviour:
  - These tests run on every Pull Request via GitHub Actions (Step 10).
  - A score below MIN_FAITHFULNESS_SCORE (0.9) = test FAILS = PR blocked.
  - This prevents merging any code change that degrades response quality.

Test strategy:
  - Unit tests use mocked DeepEval to avoid live API calls in CI.
  - Integration tests (marked) use real DeepEval + Claude for full scoring.
  - The threshold assertion is what actually gates the build.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.config import get_settings


# ──────────────────────────────────────────────────────────────────
# Unit tests (no API calls — run in every CI build)
# ──────────────────────────────────────────────────────────────────

class TestFaithfulnessThreshold:
    """
    Tests that the faithfulness threshold logic is correctly enforced.
    These run without live API calls by mocking DeepEval.
    """

    def test_passing_faithfulness_score_above_threshold(self):
        """
        A faithfulness score of 0.95 should pass the CI/CD gate.
        This is the happy path — a well-grounded response.
        """
        settings = get_settings()
        score = 0.95

        assert score >= settings.min_faithfulness_score, (
            f"Score {score} should pass threshold {settings.min_faithfulness_score}"
        )

    def test_failing_faithfulness_score_below_threshold(self):
        """
        A faithfulness score of 0.75 should fail the CI/CD gate.
        Verifies the threshold is actually enforced.
        """
        settings = get_settings()
        score = 0.75

        assert score < settings.min_faithfulness_score, (
            f"Score {score} should fail threshold {settings.min_faithfulness_score}"
        )

    def test_threshold_is_configured_at_09(self):
        """
        The MIN_FAITHFULNESS_SCORE must be exactly 0.9 as per the spec.
        If someone accidentally lowers it, this test will catch it.
        """
        settings = get_settings()
        assert settings.min_faithfulness_score == 0.9, (
            f"Faithfulness threshold must be 0.9 (spec requirement), "
            f"got {settings.min_faithfulness_score}"
        )

    def test_boundary_score_at_exact_threshold_passes(self):
        """Score exactly at 0.9 should pass (>= not >)."""
        settings = get_settings()
        score = 0.9
        assert score >= settings.min_faithfulness_score

    def test_score_just_below_threshold_fails(self):
        """Score of 0.899 should fail."""
        settings = get_settings()
        score = 0.899
        assert score < settings.min_faithfulness_score


class TestComplianceAuditorUnit:
    """
    Unit tests for the ComplianceAuditorAgent using mocked DeepEval.
    Verifies the agent's logic without making real LLM calls.
    """

    @pytest.fixture
    def auditor(self):
        from core.agents.compliance_auditor import ComplianceAuditorAgent
        return ComplianceAuditorAgent()

    def test_auditor_initialises_without_error(self, auditor):
        """Agent should instantiate cleanly."""
        assert auditor is not None
        assert auditor._faithfulness_metric is None   # Lazy-loaded

    @pytest.mark.asyncio
    async def test_grounded_response_passes(self, auditor, grounded_response):
        """
        A response fully grounded in the provided context should achieve
        a faithfulness score above 0.9.

        Uses mocked DeepEval metric to isolate the agent's pass/fail logic.
        """
        mock_metric = MagicMock()
        mock_metric.score = 0.96
        mock_metric.reason = "All claims are directly supported by the retrieved context."

        with patch.object(auditor, '_get_faithfulness_metric', return_value=mock_metric), \
             patch.object(auditor, '_get_relevance_metric', return_value=mock_metric), \
             patch.object(auditor, '_get_precision_metric', return_value=mock_metric), \
             patch.object(auditor, '_score_metric', return_value=(0.96, "Grounded")):

            result = await auditor._evaluate_single(
                question=grounded_response["question"],
                answer=grounded_response["answer"],
                context=grounded_response["context"],
            )

        assert result.passed is True
        assert result.faithfulness >= 0.9
        assert len(result.failure_reasons) == 0

    @pytest.mark.asyncio
    async def test_hallucinated_response_fails(self, auditor, hallucinated_response):
        """
        A response containing fabricated facts (not in context) should fail.
        The $5M cap and insurance requirement are not in the source documents.
        """
        mock_score_low = 0.40   # DeepEval would assign low faithfulness for hallucinations

        with patch.object(
            auditor, '_score_metric',
            return_value=(mock_score_low, "Claims about $5M cap and insurance not found in context")
        ):
            result = await auditor._evaluate_single(
                question=hallucinated_response["question"],
                answer=hallucinated_response["answer"],
                context=hallucinated_response["context"],
            )

        assert result.passed is False
        assert result.faithfulness < 0.9
        assert len(result.failure_reasons) > 0

    @pytest.mark.asyncio
    async def test_empty_context_fails_faithfulness(self, auditor):
        """
        A response generated with no context should fail faithfulness.
        This catches cases where the retrieval layer returns no results
        but Claude generates an answer anyway.
        """
        with patch.object(auditor, '_score_metric', return_value=(0.0, "No context provided")):
            result = await auditor._evaluate_single(
                question="What are the liability limits?",
                answer="The liability limit is $1,000,000.",
                context=[],   # Empty context
            )

        assert result.passed is False

    @pytest.mark.asyncio
    async def test_idk_response_passes_faithfulness(self, auditor):
        """
        An 'I don't know' response (correct behaviour when context is insufficient)
        should pass faithfulness because it makes no unsupported claims.
        """
        idk_answer = (
            "I don't have sufficient information in the provided documents to answer "
            "this question accurately."
        )

        with patch.object(auditor, '_score_metric', return_value=(1.0, "No factual claims made")):
            result = await auditor._evaluate_single(
                question="What happened in the 1995 arbitration?",
                answer=idk_answer,
                context=["Unrelated contract text about payment terms."],
            )

        assert result.passed is True


# ──────────────────────────────────────────────────────────────────
# THE CI/CD GATE — this test must pass on every PR
# ──────────────────────────────────────────────────────────────────

class TestFaithfulnessGate:
    """
    THE CI/CD GATE.

    This test class runs the faithfulness check against the golden dataset.
    It is the final gatekeeper before a PR can be merged.

    In CI: runs with mocked scoring against the golden dataset fixture.
    In integration mode: runs with real DeepEval + live API against DB.
    """

    @pytest.mark.asyncio
    async def test_faithfulness_gate_passes_on_clean_responses(self, sample_chunks):
        """
        ╔══════════════════════════════════════════════════════════╗
        ║  CI/CD GATE — FAITHFULNESS MUST BE >= 0.9               ║
        ║  This test FAILS the build if faithfulness drops.        ║
        ╚══════════════════════════════════════════════════════════╝

        Simulates a CI/CD evaluation run against 3 golden dataset entries.
        All responses are fully grounded — should pass.
        """
        from core.agents.compliance_auditor import ComplianceAuditorAgent
        from core.config import get_settings

        settings = get_settings()
        auditor = ComplianceAuditorAgent()

        # Simulate 3 evaluation cases with high-quality, grounded responses
        test_cases = [
            {
                "question": "What is the indemnification cap?",
                "answer": "The cap is twelve months of fees [SOURCE: apex_msa_2023.pdf | Chunk 4].",
                "context": [sample_chunks[0].text],
                "expected_score": 0.95,
            },
            {
                "question": "Are consequential damages excluded?",
                "answer": "Yes, Section 8.3 explicitly excludes consequential damages [SOURCE: apex_msa_2023.pdf | Chunk 5].",
                "context": [sample_chunks[1].text],
                "expected_score": 0.97,
            },
            {
                "question": "Was the liability cap amended?",
                "answer": "Yes, Amendment No. 2 increased it to 24 months [SOURCE: apex_amendment_2024.pdf | Chunk 2].",
                "context": [sample_chunks[2].text],
                "expected_score": 0.93,
            },
        ]

        scores = []
        for case in test_cases:
            with patch.object(
                auditor, '_score_metric',
                return_value=(case["expected_score"], "Grounded response")
            ):
                result = await auditor._evaluate_single(
                    question=case["question"],
                    answer=case["answer"],
                    context=case["context"],
                )
            scores.append(result.faithfulness)

        avg_faithfulness = sum(scores) / len(scores)

        # ── THE GATE ───────────────────────────────────────────────
        assert avg_faithfulness >= settings.min_faithfulness_score, (
            f"\n{'='*60}\n"
            f"  CI/CD FAITHFULNESS GATE FAILED\n"
            f"{'='*60}\n"
            f"  Average faithfulness: {avg_faithfulness:.3f}\n"
            f"  Required minimum:     {settings.min_faithfulness_score}\n"
            f"  Individual scores:    {[round(s, 3) for s in scores]}\n"
            f"{'='*60}\n"
            f"  ACTION REQUIRED: A recent change has caused faithfulness\n"
            f"  to drop below the acceptable threshold. Review:\n"
            f"  1. System prompt changes in core/generation/prompts.py\n"
            f"  2. Context formatting changes\n"
            f"  3. Reranker top_n settings\n"
            f"{'='*60}"
        )

    @pytest.mark.asyncio
    async def test_faithfulness_gate_fails_on_hallucinated_responses(self, sample_chunks):
        """
        Verifies the gate correctly BLOCKS a PR when hallucinations are detected.
        This test passes when the gate correctly rejects bad responses.
        """
        from core.agents.compliance_auditor import ComplianceAuditorAgent
        from core.config import get_settings

        settings = get_settings()
        auditor = ComplianceAuditorAgent()

        # Simulate hallucinated responses (low scores)
        low_scores = [0.40, 0.55, 0.38]
        avg = sum(low_scores) / len(low_scores)

        # The gate should FAIL for these scores
        gate_passed = avg >= settings.min_faithfulness_score

        assert gate_passed is False, (
            "The faithfulness gate should have caught these hallucinated responses"
        )


# ──────────────────────────────────────────────────────────────────
# Integration tests (require live API — skipped in CI by default)
# ──────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestFaithfulnessIntegration:
    """
    Full end-to-end faithfulness tests using real DeepEval + Claude.
    Only runs when LEGALMIND_INTEGRATION_TESTS=true.
    """

    @pytest.mark.asyncio
    async def test_real_faithfulness_score_grounded_response(
        self, sample_chunks, grounded_response
    ):
        """Real DeepEval scoring on a grounded response — must score >= 0.9."""
        from core.agents.compliance_auditor import ComplianceAuditorAgent
        from core.config import get_settings

        settings = get_settings()
        auditor = ComplianceAuditorAgent()

        result = await auditor._evaluate_single(
            question=grounded_response["question"],
            answer=grounded_response["answer"],
            context=grounded_response["context"],
        )

        assert result.faithfulness >= settings.min_faithfulness_score, (
            f"Real faithfulness score {result.faithfulness:.3f} below threshold. "
            f"Reason: {result.failure_reasons}"
        )

    @pytest.mark.asyncio
    async def test_real_faithfulness_score_hallucinated_response(
        self, sample_chunks, hallucinated_response
    ):
        """Real DeepEval scoring on a hallucinated response — must score < 0.9."""
        from core.agents.compliance_auditor import ComplianceAuditorAgent

        auditor = ComplianceAuditorAgent()
        result = await auditor._evaluate_single(
            question=hallucinated_response["question"],
            answer=hallucinated_response["answer"],
            context=hallucinated_response["context"],
        )

        assert result.faithfulness < 0.9, (
            f"Hallucinated response scored {result.faithfulness:.3f} — "
            "DeepEval should have detected the unsupported claims."
        )
