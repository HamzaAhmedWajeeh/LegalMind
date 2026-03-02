"""
main.py
=======
LegalMind API — Application Entry Point.

Wires together:
- FastAPI app with lifespan (startup/shutdown hooks)
- CORS middleware
- Structured logging
- Database connection pool lifecycle
- Route registration (added incrementally in later steps)
"""

import time
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import get_settings
from core.db import close_db, init_db

# ------------------------------------------------------------------
# Structured logging setup
# ------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
        if get_settings().environment == "development"
        else structlog.processors.JSONRenderer(),
    ]
)

logger = structlog.get_logger(__name__)
settings = get_settings()


# ------------------------------------------------------------------
# Lifespan — startup and shutdown hooks
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup and once at shutdown.
    Replaces the deprecated @app.on_event("startup") pattern.
    """
    # STARTUP
    logger.info(
        "LegalMind API starting",
        environment=settings.environment,
        model=settings.gemini_model,
    )
    await init_db()
    logger.info("Database ready")

    # Register agents (Compliance Auditor as Observer on RAGService)
    from core.agents.registry import register_all_agents
    register_all_agents()

    # Build BM25 index from any already-indexed documents
    from core.retrieval.bm25 import bm25_retriever
    await bm25_retriever.build_index()
    logger.info("BM25 index built — startup complete")

    yield  # Application runs here

    # SHUTDOWN
    await close_db()
    logger.info("LegalMind API shut down cleanly")


# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------
app = FastAPI(
    title="LegalMind Knowledge Assistant",
    description=(
        "Modular RAG system for legal document retrieval. "
        "Features hybrid search, cross-encoder reranking, semantic caching, "
        "and automated hallucination detection via DeepEval."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ------------------------------------------------------------------
# Middleware
# ------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with method, path, status, and latency."""
    start = time.monotonic()
    response = await call_next(request)
    latency_ms = int((time.monotonic() - start) * 1000)
    logger.info(
        "HTTP request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=latency_ms,
    )
    return response


# ------------------------------------------------------------------
# Global exception handler
# ------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error_code": "INTERNAL_ERROR"},
    )


# ------------------------------------------------------------------
# Register API routers
# ------------------------------------------------------------------
from api.routes.ingest import router as ingest_router
from api.routes.query import router as query_router
from api.routes.evaluate import router as evaluate_router

app.include_router(ingest_router,   prefix="/api/v1")
app.include_router(query_router,    prefix="/api/v1")
app.include_router(evaluate_router, prefix="/api/v1")


# ------------------------------------------------------------------
# Core routes
# ------------------------------------------------------------------
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint used by Docker and load balancers."""
    return {
        "status": "ok",
        "service": "legalmind-api",
        "environment": settings.environment,
        "version": "1.0.0",
    }


@app.get("/", tags=["System"])
async def root():
    return {
        "message": "LegalMind API is running.",
        "docs": "/docs",
        "health": "/health",
    }


# ------------------------------------------------------------------
# Dev entrypoint
# ------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=(settings.environment == "development"),
    )
