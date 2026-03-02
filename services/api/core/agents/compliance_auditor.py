"""
core/agents/compliance_auditor.py
===================================
Compliance Auditor Agent — RAG Triad evaluation using DeepEval.

Uses Google Gemini 2.5 Pro as the judge LLM.
Uses a custom DeepEvalBaseLLM wrapper since deepeval 0.21.x predates
the official Gemini integration.

Design Pattern: Observer Pattern
  The RAGService notifies this agent after every generation.
  Online mode: non-blocking background evaluation.
  Batch mode:  blocking evaluation for CI/CD gate.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

import structlog
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase

from api.models.schemas import QueryRequest
from core.config import get_settings
from core.generation.rag_service import rag_service

logger = structlog.get_logger(__name__)
settings = get_settings()


# ──────────────────────────────────────────────────────────────────
# Custom Gemini judge for DeepEval
# ──────────────────────────────────────────────────────────────────
class GeminiJudge(DeepEvalBaseLLM):
    """
    Wraps Google Gemini for use as DeepEval's judge LLM.
    DeepEval calls generate() / a_generate() to score responses.
    """

    def __init__(self):
        self._model = None

    def _get_model(self):
        if self._model is None:
            import google.generativeai as genai
            genai.configure(api_key=settings.gemini_api_key)
            self._model = genai.GenerativeModel(
                model_name=settings.gemini_model,
            )
        return self._model

    def get_model_name(self) -> str:
        return settings.gemini_model

    def generate(self, prompt: str) -> str:
        """Synchronous generate — called by DeepEval internals."""
        model = self._get_model()
        response = model.generate_content(prompt)
        return response.text

    async def a_generate(self, prompt: str) -> str:
        """Async generate — runs sync client in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt)


# ──────────────────────────────────────────────────────────────────
# Evaluation result dataclass
# ──────────────────────────────────────────────────────────────────
@dataclass
class EvaluationResult:
    faithfulness: float
    answer_relevance: float
    context_precision: float
    passed: bool
    details: dict


@dataclass
class BatchEvaluationResult:
    run_id: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    avg_faithfulness: float
    avg_answer_relevance: float
    avg_context_precision: float
    passed: bool


# ──────────────────────────────────────────────────────────────────
# Compliance Auditor Agent
# ──────────────────────────────────────────────────────────────────
class ComplianceAuditorAgent:
    """
    Evaluates RAG responses using the RAG Triad metrics via DeepEval.
    Uses Gemini 2.5 Pro as the judge LLM.

    Metrics:
      - Faithfulness       ≥ 0.9  (CI/CD gate)
      - Answer Relevance   ≥ 0.7
      - Context Precision  ≥ 0.7
    """

    FAITHFULNESS_THRESHOLD = 0.9
    RELEVANCE_THRESHOLD = 0.7
    PRECISION_THRESHOLD = 0.7

    def __init__(self):
        self._judge = None

    def _get_judge(self) -> GeminiJudge:
        if self._judge is None:
            self._judge = GeminiJudge()
        return self._judge

    async def evaluate_response(
        self,
        query: str,
        response: str,
        context_chunks: list[str],
        session_id: Optional[str] = None,
        persist_result: bool = True,
    ) -> EvaluationResult:
        """
        Run RAG Triad evaluation on a single query/response pair.

        Args:
            query         : The user's legal question
            response      : LegalMind's generated answer
            context_chunks: Raw text of the retrieved chunks used
            session_id    : For logging

        Returns:
            EvaluationResult with all three metric scores
        """
        log = logger.bind(session_id=session_id, query_preview=query[:60])
        log.info("Compliance audit started")

        judge = self._get_judge()

        test_case = LLMTestCase(
            input=query,
            actual_output=response,
            retrieval_context=context_chunks,
        )

        faithfulness_metric = FaithfulnessMetric(
            threshold=self.FAITHFULNESS_THRESHOLD,
            model=judge,
            include_reason=True,
        )
        relevance_metric = AnswerRelevancyMetric(
            threshold=self.RELEVANCE_THRESHOLD,
            model=judge,
            include_reason=True,
        )
        precision_metric = ContextualPrecisionMetric(
            threshold=self.PRECISION_THRESHOLD,
            model=judge,
            include_reason=True,
        )

        try:
            loop = asyncio.get_event_loop()

            def _run_metrics():
                faithfulness_metric.measure(test_case)
                relevance_metric.measure(test_case)
                precision_metric.measure(test_case)

            await loop.run_in_executor(None, _run_metrics)

            faithfulness_score = faithfulness_metric.score or 0.0
            relevance_score = relevance_metric.score or 0.0
            precision_score = precision_metric.score or 0.0

            passed = faithfulness_score >= self.FAITHFULNESS_THRESHOLD

            result = EvaluationResult(
                faithfulness=faithfulness_score,
                answer_relevance=relevance_score,
                context_precision=precision_score,
                passed=passed,
                details={
                    "faithfulness_reason": faithfulness_metric.reason,
                    "relevance_reason": relevance_metric.reason,
                    "precision_reason": precision_metric.reason,
                },
            )

            if passed:
                log.info(
                    "Compliance audit PASSED",
                    faithfulness=faithfulness_score,
                    relevance=relevance_score,
                    precision=precision_score,
                )
            else:
                log.warning(
                    "Compliance audit FAILED — faithfulness below threshold",
                    faithfulness=faithfulness_score,
                    threshold=self.FAITHFULNESS_THRESHOLD,
                )

            if persist_result:
                await _save_single_eval_result(result, session_id)
            return result

        except Exception as exc:
            log.error("Compliance audit error", error=str(exc))
            return EvaluationResult(
                faithfulness=0.0,
                answer_relevance=0.0,
                context_precision=0.0,
                passed=False,
                details={"error": str(exc)},
            )

    async def run_evaluation(
        self,
        run_id: str,
        dataset_size: Optional[int] = None,
    ) -> BatchEvaluationResult:
        """
        Run a batch evaluation over active golden dataset entries.
        This is used by /evaluate/run and CI/CD guardrails.
        """
        size = dataset_size or settings.golden_dataset_size
        log = logger.bind(run_id=run_id, dataset_size=size)
        log.info("Batch evaluation started")

        entries = await _load_golden_dataset_entries(limit=size)
        if not entries:
            result = BatchEvaluationResult(
                run_id=run_id,
                total_cases=0,
                passed_cases=0,
                failed_cases=0,
                avg_faithfulness=0.0,
                avg_answer_relevance=0.0,
                avg_context_precision=0.0,
                passed=False,
            )
            await _save_batch_eval_result(result)
            log.warning("Batch evaluation skipped: no active golden dataset entries")
            return result

        scores_f: list[float] = []
        scores_r: list[float] = []
        scores_p: list[float] = []
        passed_cases = 0
        failed_cases = 0

        for idx, entry in enumerate(entries, start=1):
            try:
                query_response = await rag_service.query(
                    QueryRequest(query=entry.question, session_id=f"eval:{run_id}"),
                    cache_enabled=False,
                )
                eval_result = await self.evaluate_response(
                    query=entry.question,
                    response=query_response.answer,
                    context_chunks=[src.text for src in query_response.sources],
                    session_id=f"{run_id}:{idx}",
                    persist_result=False,
                )
            except Exception as exc:
                log.error("Case evaluation failed", case_index=idx, error=str(exc))
                eval_result = EvaluationResult(
                    faithfulness=0.0,
                    answer_relevance=0.0,
                    context_precision=0.0,
                    passed=False,
                    details={"error": str(exc)},
                )

            scores_f.append(eval_result.faithfulness)
            scores_r.append(eval_result.answer_relevance)
            scores_p.append(eval_result.context_precision)
            if eval_result.passed:
                passed_cases += 1
            else:
                failed_cases += 1

        total = len(entries)
        avg_f = sum(scores_f) / total
        avg_r = sum(scores_r) / total
        avg_p = sum(scores_p) / total

        batch_result = BatchEvaluationResult(
            run_id=run_id,
            total_cases=total,
            passed_cases=passed_cases,
            failed_cases=failed_cases,
            avg_faithfulness=avg_f,
            avg_answer_relevance=avg_r,
            avg_context_precision=avg_p,
            passed=avg_f >= self.FAITHFULNESS_THRESHOLD,
        )

        await _save_batch_eval_result(batch_result)
        log.info(
            "Batch evaluation complete",
            total_cases=total,
            passed_cases=passed_cases,
            failed_cases=failed_cases,
            avg_faithfulness=avg_f,
            passed=batch_result.passed,
        )
        return batch_result


# ──────────────────────────────────────────────────────────────────
# Persist evaluation result
# ──────────────────────────────────────────────────────────────────
async def _save_single_eval_result(
    result: EvaluationResult,
    session_id: Optional[str],
) -> None:
    """Store evaluation scores in eval_runs table (non-fatal)."""
    try:
        import uuid
        from core.db import get_db_context
        from core.models.db_models import EvalRun

        async with get_db_context() as db:
            run = EvalRun(
                run_id=str(uuid.uuid4()),
                faithfulness=result.faithfulness,
                answer_relevance=result.answer_relevance,
                context_precision=result.context_precision,
                total_cases=1,
                passed_cases=1 if result.passed else 0,
                failed_cases=0 if result.passed else 1,
                passed=result.passed,
                metadata=result.details,
            )
            db.add(run)
    except Exception as exc:
        logger.error("Failed to save eval result", error=str(exc))


async def _load_golden_dataset_entries(limit: int):
    """Load active golden dataset rows for batch evaluation."""
    try:
        from sqlalchemy import select
        from core.db import get_db_context
        from core.models.db_models import GoldenDatasetEntry

        async with get_db_context() as db:
            result = await db.execute(
                select(GoldenDatasetEntry)
                .where(GoldenDatasetEntry.is_active == True)
                .order_by(GoldenDatasetEntry.created_at.desc())
                .limit(limit)
            )
            return list(result.scalars().all())
    except Exception as exc:
        logger.error("Failed to load golden dataset entries", error=str(exc))
        return []


async def _save_batch_eval_result(result: BatchEvaluationResult) -> None:
    """Persist one batch evaluation aggregate row to eval_runs."""
    try:
        from core.db import get_db_context
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
                metadata={
                    "mode": "batch",
                },
            )
            db.add(run)
    except Exception as exc:
        logger.error("Failed to save batch eval result", error=str(exc), run_id=result.run_id)


# ──────────────────────────────────────────────────────────────────
# Module-level singleton
# ──────────────────────────────────────────────────────────────────
compliance_auditor = ComplianceAuditorAgent()
