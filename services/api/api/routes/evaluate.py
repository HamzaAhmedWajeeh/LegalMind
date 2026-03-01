"""
api/routes/evaluate.py
======================
Evaluation endpoints — trigger agent workflows and retrieve results.

POST /evaluate/generate-dataset  — Adversarial Lawyer Agent generates
                                   golden dataset entries.
POST /evaluate/run               — Compliance Auditor runs full batch
                                   evaluation against golden dataset.
GET  /evaluate/results           — Retrieve past evaluation run results.
GET  /evaluate/dataset           — Browse golden dataset entries.
GET  /evaluate/cache/stats       — Semantic cache statistics.
POST /evaluate/cache/clear       — Invalidate semantic cache.
"""

import uuid
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from core.db import get_db_session
from core.models.db_models import EvalRun, GoldenDatasetEntry
from api.models.schemas import (
    EvalTriggerRequest,
    EvalTriggerResponse,
    EvalRunOut,
    GoldenDatasetEntryOut,
    PaginatedResponse,
)

router = APIRouter(prefix="/evaluate", tags=["Evaluation"])
logger = structlog.get_logger(__name__)


# ──────────────────────────────────────────────────────────────────
# POST /evaluate/generate-dataset
# ──────────────────────────────────────────────────────────────────
@router.post("/generate-dataset", status_code=202)
async def generate_golden_dataset(
    target_size: Optional[int] = Query(
        default=None,
        description="Number of QA pairs to generate (default from config)",
    ),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Trigger the Adversarial Lawyer Agent to generate synthetic QA pairs.

    Runs in the background — returns immediately.
    Check GET /evaluate/dataset to see generated entries.
    """
    from core.agents.adversarial_lawyer import adversarial_lawyer

    async def _run_generation():
        count = await adversarial_lawyer.generate_dataset(target_size=target_size)
        logger.info("Golden dataset generation complete", count=count)

    background_tasks.add_task(_run_generation)

    return {
        "message": "Adversarial Lawyer Agent started",
        "target_size": target_size,
        "status": "running in background",
        "check": "GET /evaluate/dataset to see results",
    }


# ──────────────────────────────────────────────────────────────────
# POST /evaluate/run
# ──────────────────────────────────────────────────────────────────
@router.post("/run", response_model=EvalTriggerResponse, status_code=202)
async def trigger_evaluation(
    request: EvalTriggerRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Trigger the Compliance Auditor Agent to run a full batch evaluation.

    Runs the RAG pipeline on every golden dataset entry and scores
    Faithfulness, Answer Relevance, and Context Precision.

    Returns immediately with a run_id. Poll GET /evaluate/results
    to see the outcome once complete.
    """
    from core.agents.compliance_auditor import compliance_auditor

    async def _run_eval():
        result = await compliance_auditor.run_evaluation(
            run_id=request.run_id,
            dataset_size=request.dataset_size,
        )
        logger.info(
            "Evaluation run complete",
            run_id=request.run_id,
            passed=result.passed,
            faithfulness=round(result.avg_faithfulness, 3),
        )

    background_tasks.add_task(_run_eval)

    return EvalTriggerResponse(
        run_id=request.run_id,
        task_id=f"background-{request.run_id}",
        message="Evaluation started in background. Check GET /evaluate/results.",
    )


# ──────────────────────────────────────────────────────────────────
# GET /evaluate/results
# ──────────────────────────────────────────────────────────────────
@router.get("/results", response_model=list[EvalRunOut])
async def list_eval_results(
    limit: int = Query(default=10, ge=1, le=50),
    db: AsyncSession = Depends(get_db_session),
):
    """
    List past evaluation run results, most recent first.

    Useful for the CI/CD dashboard and regression tracking.
    A run with passed=False means faithfulness dropped below the
    threshold — investigate and fix before merging.
    """
    stmt = (
        select(EvalRun)
        .order_by(desc(EvalRun.ran_at))
        .limit(limit)
    )
    result = await db.execute(stmt)
    runs = result.scalars().all()

    return [EvalRunOut.model_validate(r) for r in runs]


# ──────────────────────────────────────────────────────────────────
# GET /evaluate/results/{run_id}
# ──────────────────────────────────────────────────────────────────
@router.get("/results/{run_id}", response_model=EvalRunOut)
async def get_eval_result(
    run_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """Get a specific evaluation run result by run_id."""
    result = await db.execute(
        select(EvalRun).where(EvalRun.run_id == run_id)
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail=f"Eval run '{run_id}' not found")

    return EvalRunOut.model_validate(run)


# ──────────────────────────────────────────────────────────────────
# GET /evaluate/dataset
# ──────────────────────────────────────────────────────────────────
@router.get("/dataset", response_model=PaginatedResponse)
async def get_golden_dataset(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db_session),
):
    """Browse the golden dataset (synthetic QA pairs for evaluation)."""
    from sqlalchemy import func

    count_result = await db.execute(
        select(func.count(GoldenDatasetEntry.id))
        .where(GoldenDatasetEntry.is_active == True)
    )
    total = count_result.scalar() or 0

    offset = (page - 1) * page_size
    result = await db.execute(
        select(GoldenDatasetEntry)
        .where(GoldenDatasetEntry.is_active == True)
        .order_by(GoldenDatasetEntry.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    entries = result.scalars().all()

    import math
    return PaginatedResponse(
        items=[GoldenDatasetEntryOut.model_validate(e) for e in entries],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=math.ceil(total / page_size) if total > 0 else 0,
    )


# ──────────────────────────────────────────────────────────────────
# GET /evaluate/cache/stats
# ──────────────────────────────────────────────────────────────────
@router.get("/cache/stats")
async def cache_stats():
    """Return semantic cache statistics."""
    from core.cache.semantic_cache import semantic_cache
    return await semantic_cache.stats()


# ──────────────────────────────────────────────────────────────────
# POST /evaluate/cache/clear
# ──────────────────────────────────────────────────────────────────
@router.post("/cache/clear")
async def clear_cache():
    """Invalidate the entire semantic cache."""
    from core.cache.semantic_cache import semantic_cache
    count = await semantic_cache.invalidate_all()
    return {"message": f"Semantic cache cleared", "entries_deleted": count}
