"""
core/agents/registry.py
========================
Agent Registry — bootstraps all agents at application startup.

This module is imported once in main.py during the lifespan startup.
It registers the Compliance Auditor as a post-generation Observer
on RAGService, so evaluation hooks fire automatically after every query.

Design Pattern: Registry + Observer wiring
  Centralising agent registration here means:
  - main.py stays clean (one import, one call)
  - Adding a new agent only requires adding it here
  - RAGService and agents remain fully decoupled
"""

import structlog

logger = structlog.get_logger(__name__)


def register_all_agents() -> None:
    """
    Initialise and register all agents.
    Called once during FastAPI application startup.
    """
    from core.generation.rag_service import rag_service
    from core.agents.compliance_auditor import compliance_auditor

    # Register Compliance Auditor as a post-generation observer
    # It will fire automatically after every non-cached RAG response
    rag_service.register_hook(compliance_auditor.evaluate_response)

    logger.info(
        "All agents registered",
        hooks=["compliance_auditor.evaluate_response"],
    )
