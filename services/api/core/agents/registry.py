"""
core/agents/registry.py
========================
Agent registry — wires agents into the RAGService observer hooks.

Called once at startup from main.py lifespan.
"""

import structlog

logger = structlog.get_logger(__name__)


def register_all_agents() -> None:
    """
    Register all post-generation observer hooks on the RAGService singleton.

    Each hook receives (request, response, ranked_chunks) from rag_service.
    We wrap evaluate_response to adapt signatures cleanly.
    """
    from core.generation.rag_service import rag_service
    from core.agents.compliance_auditor import compliance_auditor

    async def compliance_hook(request, response, ranked_chunks) -> None:
        """
        Adapter: translates RAGService hook signature →
        ComplianceAuditorAgent.evaluate_response signature.
        """
        context_chunks = [chunk.text for chunk in ranked_chunks]
        await compliance_auditor.evaluate_response(
            query=request.query,
            response=response.answer,
            context_chunks=context_chunks,
            session_id=request.session_id,
        )

    compliance_hook.__name__ = "compliance_hook"
    rag_service.register_hook(compliance_hook)
    logger.info("Compliance auditor registered as observer")
