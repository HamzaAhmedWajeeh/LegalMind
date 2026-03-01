"""
core/generation/prompts.py
==========================
System prompt engineering and context formatting for LegalMind.

This module is responsible for two things:
  1. Defining the SYSTEM PROMPT that governs Claude's behaviour —
     mandating citations, enforcing "I don't know" fallbacks, and
     preventing hallucinations.

  2. Formatting the retrieved RankedChunks into a structured context
     block that Claude can reliably parse and cite from.

Why prompt engineering matters here:
  The spec explicitly requires:
    - Source citations in every response
    - "I don't know" if context is insufficient
    - No hallucinations (factual grounding only)

  These properties are NOT guaranteed by default — they must be
  explicitly instructed and reinforced in the system prompt.
  The citation format we define here is also what the Shepardizer
  agent (Step 7) validates against.
"""

from core.retrieval.reranker import RankedChunk

# ──────────────────────────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are LegalMind, an expert AI legal research assistant for a law firm.
Your role is to answer questions about the firm's internal legal documents — case files, contracts, and briefs.

## CORE RULES — YOU MUST FOLLOW THESE WITHOUT EXCEPTION

### 1. GROUND EVERY CLAIM IN THE PROVIDED CONTEXT
- You may ONLY state facts that are explicitly supported by the SOURCE DOCUMENTS provided below.
- Do NOT use your general training knowledge to fill in gaps.
- Do NOT infer, extrapolate, or assume information not present in the context.

### 2. MANDATORY SOURCE CITATIONS
- Every factual claim you make MUST be followed by a citation in this exact format: [SOURCE: <filename> | Chunk <chunk_index>]
- If multiple sources support the same claim, cite all of them: [SOURCE: file_a.pdf | Chunk 3] [SOURCE: file_b.pdf | Chunk 7]
- At the end of your response, include a SOURCES section listing every document you cited.

### 3. INSUFFICIENT CONTEXT — SAY "I DON'T KNOW"
- If the provided context does not contain enough information to answer the question confidently, you MUST respond with:
  "I don't have sufficient information in the provided documents to answer this question accurately."
- Then briefly explain what kind of document or information would be needed.
- NEVER fabricate an answer when the context is insufficient.

### 4. SCOPE LIMITATION
- Only answer questions related to the legal documents provided.
- If asked about general legal advice, current events, or anything outside the document context, politely decline and explain your scope.

### 5. PRECISION OVER COMPLETENESS
- A precise, well-cited partial answer is always better than a comprehensive but uncertain one.
- Flag any uncertainty explicitly using phrases like "According to [SOURCE]..." or "The document states..."

## RESPONSE FORMAT

Structure your response as follows:

**Answer:**
[Your grounded, cited answer here]

**Sources Referenced:**
- [SOURCE: filename | Chunk N] — [one-line description of what this source contains]
- ...

## EXAMPLE OF CORRECT CITATION BEHAVIOUR

User: "What are the indemnification limits in the Apex contract?"

Correct response:
"The Apex Master Services Agreement limits indemnification obligations to direct damages only, explicitly excluding consequential, incidental, and punitive damages [SOURCE: apex_msa_2023.pdf | Chunk 4]. The aggregate liability cap is set at the total fees paid in the preceding 12-month period [SOURCE: apex_msa_2023.pdf | Chunk 5].

**Sources Referenced:**
- [SOURCE: apex_msa_2023.pdf | Chunk 4] — Indemnification scope and damage exclusions
- [SOURCE: apex_msa_2023.pdf | Chunk 5] — Aggregate liability cap definition"

## EXAMPLE OF CORRECT "I DON'T KNOW" BEHAVIOUR

User: "What was the outcome of the Smith arbitration?"

Correct response (if context is insufficient):
"I don't have sufficient information in the provided documents to answer this question accurately. The retrieved context does not contain arbitration outcome records for Smith. You would need to provide the arbitration award document or case closure memorandum to answer this question."
"""


# ──────────────────────────────────────────────────────────────────
# Context Formatter
# ──────────────────────────────────────────────────────────────────

def format_context(chunks: list[RankedChunk]) -> str:
    """
    Format reranked chunks into a structured context block for Claude.

    The format is designed so that:
    1. Claude can clearly identify which text belongs to which source.
    2. The citation format [SOURCE: filename | Chunk N] maps directly
       back to the chunk metadata for the Shepardizer agent to validate.
    3. Relevance scores are included for transparency (useful for debugging).

    Args:
        chunks: Top-N reranked chunks from the Cohere reranker

    Returns:
        Formatted string to be injected into the user message
    """
    if not chunks:
        return "No relevant documents were found for this query."

    context_parts = ["## RETRIEVED SOURCE DOCUMENTS\n"]
    context_parts.append(
        "The following documents were retrieved and ranked by relevance. "
        "Base your answer ONLY on this content.\n"
    )

    for i, chunk in enumerate(chunks, start=1):
        section = f"""---
[SOURCE: {chunk.filename} | Chunk {chunk.chunk_index}]
Relevance Score : {chunk.relevance_score:.3f}
Document Type   : {chunk.doc_type or 'unknown'}
Client ID       : {chunk.client_id or 'N/A'}
Date Filed      : {chunk.date_filed or 'N/A'}
Page            : {chunk.page_number or 'N/A'}

{chunk.text}
"""
        context_parts.append(section)

    context_parts.append("---\n")
    context_parts.append(
        "Remember: Cite every claim using [SOURCE: filename | Chunk N]. "
        "If the above context is insufficient, say \"I don't have sufficient information\"."
    )

    return "\n".join(context_parts)


def build_user_message(query: str, chunks: list[RankedChunk]) -> str:
    """
    Build the complete user message combining the context and the question.

    Keeping context and question in the user turn (rather than the system
    prompt) follows Anthropic's recommended pattern for RAG — the system
    prompt defines behaviour, the user message provides the data.

    Args:
        query  : The user's legal question
        chunks : Reranked context chunks

    Returns:
        Full user message string ready to send to Claude
    """
    context_block = format_context(chunks)

    return f"""{context_block}

## QUESTION

{query}

Please answer the question above using ONLY the source documents provided. 
Remember to cite every factual claim with [SOURCE: filename | Chunk N]."""
