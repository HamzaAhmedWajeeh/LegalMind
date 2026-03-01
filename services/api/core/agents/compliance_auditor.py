"""
core/agents/compliance_auditor.py
===================================
Compliance Auditor Agent — LLM-as-judge faithfulness evaluator.

Role (from spec):
  "It extracts individual claims from the RAG system's response and
   cross-references each one against the retrieved legal chunks.
   It calculates the Faithfulness score (Groundedness). If the RAG
   system claims a specific indemnity exists but the source text says
   otherwise, the Auditor agent flags a hallucination and fails the
   unit test."

Design Pattern: OBSERVER PATTERN
  This agent registers itself as a post-generation hook on RAGService:

    rag_service.register_hook(compliance_auditor.evaluate_response)

  After every RAG response is generated, the hook fires automatically.
  The RAGService doesn't know about the Auditor — it just calls all
  registered hooks. This decouples evaluation from generation completely.

Two modes of operation:
  1. ONLINE (per-request): Registered as a hook — runs after every
     query automatically. Results logged but don't block the response.

  2. BATCH (CI/CD eval): Called directly with a full golden dataset
     via `run_evaluation()`. Returns an EvalResult used by pytest
     to pass/fail the CI/CD gate.

Evaluation framework: DeepEval
  Uses DeepEval's FaithfulnessMetric which:
  1. Extracts individual factual claims from the answer
  2. For each claim, asks the LLM: "Is this claim supported by the context?"
  3. Faithfulness = (supported claims) / (total claims)
"""

import uuid
from dataclasses import dataclass, field
from typing import Optional

import structlog
from deepeval import evaluate
from deepeval.models import AnthropicModel
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase

from core.config import get_settings
from api.models.schemas import QueryRequest, QueryResponse

logger = structlog.get_logger(__name__)
settings = get_settings()


# ──────────────────────────────────────────────────────────────────
# Result dataclasses
# ──────────────────────────────────────────────────────────────────
@dataclass
class SingleEvalResult:
    """Evaluation result for one query-response pair."""
    question: str
    faithfulness: float
    answer_relevance: float
    context_precision: float
    passed: bool
    failure_reasons: list[str] = field(default_factory=list)


@dataclass
class BatchEvalResult:
    """Aggregated evaluation result for a full eval run."""
    run_id: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    avg_faithfulness: float
    avg_answer_relevance: float
    avg_context_precision: float
    passed: bool                      # True if avg_faithfulness >= threshold
    individual_results: list[SingleEvalResult] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────
# Compliance Auditor Agent
# ──────────────────────────────────────────────────────────────────
class ComplianceAuditorAgent:
    """
    DeepEval-powered faithfulness evaluator.

    Registered as an Observer on RAGService so it fires automatically
    after each generation without coupling to the generation code.
    """

    def __init__(self):
        # DeepEval metrics — initialised lazily to avoid startup delay
        self._faithfulness_metric: Optional[FaithfulnessMetric] = None
        self._relevance_metric: Optional[AnswerRelevancyMetric] = None
        self._precision_metric: Optional[ContextualPrecisionMetric] = None

    # ── Metric initialisation ──────────────────────────────────────
    def _get_judge_model(self) -> AnthropicModel:
        # DeepEval uses ANTHROPIC_API_KEY directly — no separate key needed.
        # Passing model and api_key explicitly makes the dependency clear.
        return AnthropicModel(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=0,
        )

    def _get_faithfulness_metric(self) -> FaithfulnessMetric:
        if self._faithfulness_metric is None:
            self._faithfulness_metric = FaithfulnessMetric(
                threshold=settings.min_faithfulness_score,
                model=self._get_judge_model(),
                include_reason=True,
            )
        return self._faithfulness_metric

    def _get_relevance_metric(self) -> AnswerRelevancyMetric:
        if self._relevance_metric is None:
            self._relevance_metric = AnswerRelevancyMetric(
                threshold=0.7,
                model=self._get_judge_model(),
                include_reason=True,
            )
        return self._relevance_metric

    def _get_precision_metric(self) -> ContextualPrecisionMetric:
        if self._precision_metric is None:
            self._precision_metric = ContextualPrecisionMetric(
                threshold=0.7,
                model=self._get_judge_model(),
                include_reason=True,
            )
        return self._precision_metric

    # ── Observer hook (called by RAGService after each response) ───
    async def evaluate_response(
        self,
        request: QueryRequest,
        response: QueryResponse,
    ) -> None:
        """
        Observer hook — fires automatically after every RAG generation.

        Logs the faithfulness score. Does NOT block or modify the
        response — the user always receives their answer regardless
        of the evaluation outcome.

        Registered via: rag_service.register_hook(auditor.evaluate_response)
        """
        # Skip evaluation if response came from cache or has no sources
        if response.cache_hit or not response.sources:
            return

        try:
            result = await self._evaluate_single(
                question=request.query,
                answer=response.answer,
                context=[src.text for src in response.sources],
                expected_output=None,   # No expected output for live queries
            )

            if result.passed:
                logger.info(
                    "Compliance audit PASSED",
                    faithfulness=round(result.faithfulness, 3),
                    relevance=round(result.answer_relevance, 3),
                    precision=round(result.context_precision, 3),
                )
            else:
                logger.warning(
                    "Compliance audit FAILED — potential hallucination detected",
                    faithfulness=round(result.faithfulness, 3),
                    threshold=settings.min_faithfulness_score,
                    reasons=result.failure_reasons,
                    query_preview=request.query[:80],
                )

        except Exception as exc:
            # Evaluation failures must never surface to the user
            logger.error("Compliance audit error", error=str(exc))

    # ── Single test case evaluation ────────────────────────────────
    async def _evaluate_single(
        self,
        question: str,
        answer: str,
        context: list[str],
        expected_output: Optional[str] = None,
    ) -> SingleEvalResult:
        """
        Run all three RAG Triad metrics on a single query-response pair.

        Returns:
            SingleEvalResult with faithfulness, relevance, precision scores
        """
        import asyncio

        # Build DeepEval test case
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=expected_output or answer,  # Use actual as fallback
            retrieval_context=context,
        )

        # Run metrics (synchronous DeepEval calls wrapped in executor)
        loop = asyncio.get_event_loop()

        faithfulness_score, faithfulness_reason = await loop.run_in_executor(
            None, lambda: self._score_metric(self._get_faithfulness_metric(), test_case)
        )
        relevance_score, relevance_reason = await loop.run_in_executor(
            None, lambda: self._score_metric(self._get_relevance_metric(), test_case)
        )
        precision_score, precision_reason = await loop.run_in_executor(
            None, lambda: self._score_metric(self._get_precision_metric(), test_case)
        )

        # Determine pass/fail — faithfulness is the hard gate
        passed = faithfulness_score >= settings.min_faithfulness_score
        failure_reasons = []
        if not passed:
            failure_reasons.append(
                f"Faithfulness {faithfulness_score:.3f} < {settings.min_faithfulness_score} threshold"
            )
            if faithfulness_reason:
                failure_reasons.append(faithfulness_reason)

        return SingleEvalResult(
            question=question,
            faithfulness=faithfulness_score,
            answer_relevance=relevance_score,
            context_precision=precision_score,
            passed=passed,
            failure_reasons=failure_reasons,
        )

    def _score_metric(self, metric, test_case: LLMTestCase) -> tuple[float, str]:
        """Run a single DeepEval metric and return (score, reason)."""
        try:
            metric.measure(test_case)
            return metric.score, getattr(metric, "reason", "") or ""
        except Exception as exc:
            logger.error("Metric scoring failed", metric=type(metric).__name__, error=str(exc))
            return 0.0, str(exc)

    # ── Batch evaluation (CI/CD eval runs) ────────────────────────
    async def run_evaluation(
        self,
        run_id: str,
        dataset_size: Optional[int] = None,
    ) -> BatchEvalResult:
        """
        Run a full evaluation against the golden dataset.

        Called by the /evaluate/run endpoint and by pytest (Step 9).
        Fetches QA pairs from the golden_dataset table, runs the RAG
        pipeline on each question, then evaluates the response.

        Args:
            run_id       : Unique identifier for this eval run
            dataset_size : How many golden dataset entries to evaluate
                           (defaults to all active entries)

        Returns:
            BatchEvalResult persisted to eval_runs table
        """
        log = logger.bind(run_id=run_id)
        log.info("Batch evaluation started")

        # ── Load golden dataset ────────────────────────────────────
        entries = await self._load_golden_dataset(limit=dataset_size)
        if not entries:
            log.warning("Golden dataset is empty — run generate-dataset first")
            return BatchEvalResult(
                run_id=run_id,
                total_cases=0,
                passed_cases=0,
                failed_cases=0,
                avg_faithfulness=0.0,
                avg_answer_relevance=0.0,
                avg_context_precision=0.0,
                passed=False,
            )

        log.info("Golden dataset loaded", entries=len(entries))

        # ── Run RAG pipeline on each question ──────────────────────
        from core.generation.rag_service import rag_service
        from api.models.schemas import QueryRequest

        individual_results: list[SingleEvalResult] = []

        for i, entry in enumerate(entries):
            log.info("Evaluating entry", index=i + 1, total=len(entries))
            try:
                # Run the RAG pipeline (cache disabled for eval runs)
                rag_request = QueryRequest(query=entry["question"])
                rag_response = await rag_service.query(
                    rag_request,
                    cache_enabled=False,
                )

                # Evaluate with the expected answer as ground truth
                result = await self._evaluate_single(
                    question=entry["question"],
                    answer=rag_response.answer,
                    context=[src.text for src in rag_response.sources],
                    expected_output=entry["expected_answer"],
                )
                individual_results.append(result)

            except Exception as exc:
                log.error("Evaluation case failed", entry_index=i, error=str(exc))
                individual_results.append(SingleEvalResult(
                    question=entry["question"],
                    faithfulness=0.0,
                    answer_relevance=0.0,
                    context_precision=0.0,
                    passed=False,
                    failure_reasons=[f"Pipeline error: {exc}"],
                ))

        # ── Aggregate results ──────────────────────────────────────
        total = len(individual_results)
        passed_count = sum(1 for r in individual_results if r.passed)
        failed_count = total - passed_count

        avg_faithfulness    = sum(r.faithfulness for r in individual_results) / max(total, 1)
        avg_relevance       = sum(r.answer_relevance for r in individual_results) / max(total, 1)
        avg_precision       = sum(r.context_precision for r in individual_results) / max(total, 1)

        overall_passed = avg_faithfulness >= settings.min_faithfulness_score

        batch_result = BatchEvalResult(
            run_id=run_id,
            total_cases=total,
            passed_cases=passed_count,
            failed_cases=failed_count,
            avg_faithfulness=avg_faithfulness,
            avg_answer_relevance=avg_relevance,
            avg_context_precision=avg_precision,
            passed=overall_passed,
            individual_results=individual_results,
        )

        # ── Persist to DB ──────────────────────────────────────────
        await self._save_eval_run(batch_result)

        log.info(
            "Batch evaluation complete",
            passed=overall_passed,
            avg_faithfulness=round(avg_faithfulness, 3),
            passed_cases=passed_count,
            failed_cases=failed_count,
        )

        return batch_result

    # ── Helpers ────────────────────────────────────────────────────
    async def _load_golden_dataset(
        self,
        limit: Optional[int] = None,
    ) -> list[dict]:
        """Load active golden dataset entries from Postgres."""
        from sqlalchemy import select
        from core.models.db_models import GoldenDatasetEntry

        async with get_db_context() as db:
            stmt = (
                select(GoldenDatasetEntry)
                .where(GoldenDatasetEntry.is_active == True)
                .order_by(GoldenDatasetEntry.created_at)
            )
            if limit:
                stmt = stmt.limit(limit)

            result = await db.execute(stmt)
            entries = result.scalars().all()

        return [
            {
                "question": e.question,
                "reference_context": e.reference_context,
                "expected_answer": e.expected_answer,
            }
            for e in entries
        ]

    async def _save_eval_run(self, result: BatchEvalResult) -> None:
        """Persist batch evaluation results to the eval_runs table."""
        from core.models.db_models import EvalRun

        async with get_db_context() as db:
            run = EvalRun(
                run_id=result.run_id,
                faithfulness=result.avg_faithfulness,
                answer_relevance=result.avg_answer_relevance,
                context_precision=result.avg_context_precision,
                total_cases=result.total_cases,
                passed_cases=result.passed_cases,
                failed_cases=result.failed_cases,
                passed=result.passed,
            )
            db.add(run)

        logger.info("Eval run persisted", run_id=result.run_id, passed=result.passed)


# ── Module-level singleton ─────────────────────────────────────────
compliance_auditor = ComplianceAuditorAgent()
