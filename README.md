# ⚖️ LegalMind — AI-Powered Legal Research Assistant

> **Capgemini GenAI Developer Assessment — Case Study Submission**

A production-grade Retrieval-Augmented Generation (RAG) system for querying 10,000+ legal documents with high factual accuracy, mandatory source citations, and zero hallucinations.

> **Model:** `gemini-2.0-flash-preview` (Google Gemini) — used for both answer generation and golden dataset synthesis.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                     USER INTERFACES                          │
│      Streamlit UI (port 8501)  │  FastAPI Docs (port 8000)  │
└──────────────────┬───────────────────────────────────────────┘
                   │ HTTP
┌──────────────────▼───────────────────────────────────────────┐
│                  FastAPI Application                         │
│  POST /api/v1/query        →  RAGService Facade             │
│  POST /api/v1/ingest/upload →  Celery Task (async)          │
│  POST /api/v1/evaluate/run  →  ComplianceAuditor Agent      │
└──────┬────────────────────┬─────────────────────────────────-┘
       │                    │
┌──────▼──────┐    ┌─────────▼─────────────────────────────-──┐
│   Celery    │    │           RAG Pipeline                    │
│   Worker    │    │  Query → Cache → Hybrid → Rerank → Gemini │
└──────┬──────┘    └──────┬───────────────────┬───────────-────┘
       │                  │                   │
┌──────▼──────┐    ┌──────▼──────┐    ┌───────▼──────┐
│   Qdrant    │    │    Redis    │    │  PostgreSQL   │
│  (vectors)  │    │   (cache)   │    │  (metadata)   │
└─────────────┘    └─────────────┘    └──────────────-┘
```

### Six Docker Services

| Service | Port | Purpose |
|---------|------|---------|
| `api` | 8000 | RAG pipeline, agents, REST API |
| `worker` | — | Async document ingestion (Celery) |
| `frontend` | 8501 | Chat UI + evaluation dashboard |
| `qdrant` | 6333 | Vector database |
| `redis` | 6379 | Semantic cache + Celery broker |
| `postgres` | 5432 | Document metadata + audit log |

---

## Design Patterns

| Pattern | Location | Description |
|---------|----------|-------------|
| **Strategy** | `core/ingestion/chunker.py` | Recursive vs Semantic chunking — swappable with zero pipeline changes |
| **Factory** | `core/ingestion/chunker.py` | `get_chunker("semantic")` decouples instantiation from usage |
| **Repository** | `core/retrieval/vector_store.py` | All Qdrant interactions behind one interface — swap to Pinecone by replacing one file |
| **Repository** | `core/retrieval/bm25.py` | Same interface as vector store — interchangeable |
| **Facade** | `core/generation/rag_service.py` | Single `rag_service.query()` hides cache + retrieve + rerank + generate |
| **Facade** | `core/retrieval/hybrid.py` | Hides vector + BM25 + RRF behind one call |
| **Observer** | `core/agents/compliance_auditor.py` | Post-generation hook — fires after every response without coupling |
| **Pipeline** | `core/tasks/ingest_task.py` | Parse → Chunk → Enrich → Embed → Store |
| **Chain of Responsibility** | `core/agents/shepardizer.py` | Three citation validators in sequence |
| **Singleton** | `core/config.py` | `@lru_cache` on `get_settings()` |
| **Proxy** | `core/cache/semantic_cache.py` | Transparent cache layer — fail-open |

---

## Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| LLM | Google Gemini (`gemini-2.0-flash-preview`) | Strong reasoning, long context, fast inference |
| Embeddings | sentence-transformers/all-mpnet-base-v2 | 768-dim, runs locally, no API cost per embed |
| Vector DB | Qdrant | HNSW index, payload filtering, async client |
| Keyword Search | rank-bm25 | Exact legal term matching, no API required |
| Reranker | Cohere Rerank v3 | Cross-encoder, top-20 → top-5 precision |
| Cache | Redis | Semantic similarity cache (cosine ≥ 0.92) |
| Task Queue | Celery + Redis | Async ingestion, retry with back-off |
| PDF Extraction | pdfplumber + pytesseract | Structured + OCR fallback |
| Evaluation | DeepEval | Faithfulness, Relevance, Context Precision |
| API | FastAPI + Pydantic v2 | Async, type-safe, auto-docs |
| UI | Streamlit | Demo-ready three-tab frontend |
| CI/CD | GitHub Actions | Faithfulness gate on every PR |

---

## Quick Start

### Prerequisites
- Docker + Docker Compose v2
- API keys: Gemini, Cohere, DeepEval

### 1. Clone and configure

```bash
git clone <your-repo>
cd legalmind
cp .env.example .env
# Add your API keys to .env:
#   GEMINI_API_KEY=...
#   COHERE_API_KEY=...
#   DEEPEVAL_API_KEY=...
```

### 2. Start all services

```bash
docker compose up --build
```

First run takes ~3 minutes to pull images and build. All 6 services start with health checks.
Wait for this log line before uploading documents:

```
Embedding model ready — startup complete
```

### 3. Ingest sample documents

```bash
# Via the Streamlit UI
open http://localhost:8501   # Documents tab → Upload

# Or via curl
curl -X POST http://localhost:8000/api/v1/ingest/upload \
  -F "file=@sample_docs/apex_msa_2023.txt" \
  -F "doc_type=contract" \
  -F "client_id=APEX-001"
```

Wait for all documents to reach **🟢 indexed** status before querying.

### 4. Ask a legal question

```bash
# Via the Streamlit UI
open http://localhost:8501   # Legal Query tab

# Or via API
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the indemnification limits in the Apex contract?"}'
```

### 5. Run evaluation

```bash
open http://localhost:8501   # Evaluation tab
# 1. Generate Dataset (Adversarial Lawyer Agent)
# 2. Run Evaluation  (Compliance Auditor Agent)
# 3. View faithfulness trend chart
```

---

## Project Structure

```
legalmind/
├── docker-compose.yml
├── .env.example
├── pytest.ini
├── sample_docs/                    # Demo documents (pre-loaded)
│   ├── apex_msa_2023.txt
│   ├── apex_amendment_2024.txt
│   └── legal_glossary.txt
├── tests/
│   ├── conftest.py                 # Shared fixtures
│   ├── test_faithfulness.py        # CI/CD gate (Faithfulness ≥ 0.9)
│   ├── test_citations.py           # Shepardizer tests
│   ├── test_retrieval.py           # Hybrid search + RRF + BM25
│   └── test_answer_relevance.py    # Answer relevance + RAG service
├── .github/workflows/
│   └── rag-eval.yml                # GitHub Actions CI/CD
└── services/
    ├── api/
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   ├── main.py
    │   ├── db/init.sql
    │   ├── core/
    │   │   ├── config.py
    │   │   ├── db.py
    │   │   ├── ingestion/          # parser, chunker (Strategy), enricher
    │   │   ├── retrieval/          # vector_store, bm25, hybrid, reranker
    │   │   ├── generation/         # prompts, llm, rag_service (Facade)
    │   │   ├── cache/              # semantic_cache (Proxy)
    │   │   ├── agents/             # adversarial_lawyer, compliance_auditor, shepardizer
    │   │   ├── models/             # SQLAlchemy ORM models
    │   │   └── tasks/              # celery_app, ingest_task (Pipeline)
    │   └── api/
    │       ├── models/schemas.py
    │       └── routes/             # ingest, query, evaluate
    └── frontend/
        ├── Dockerfile
        ├── requirements.txt
        └── app.py                  # 3-tab Streamlit UI
```

---

## RAG Pipeline

```
User Query
    │
    ▼  1. Semantic cache check (Redis, cosine ≥ 0.92) → instant if hit
    │
    ▼  2. Hybrid Retrieval (parallel)
    │     ├── Vector Search (Qdrant HNSW, top-20)
    │     └── BM25 Keyword Search (top-20)
    │
    ▼  3. Reciprocal Rank Fusion → fused top-20
    │
    ▼  4. Cohere Cross-Encoder Reranking → top-5
    │
    ▼  5. Gemini Generation (gemini-2.0-flash-preview)
    │     System prompt mandates [SOURCE: file | Chunk N] citations
    │     "I don't know" fallback when context is insufficient
    │
    ▼  6. Response returned to user immediately
    │
    ▼  7. Background tasks (asyncio.create_task — non-blocking)
          ├── Shepardizer: Context → Database → Relevance citation checks
          └── Compliance Auditor: DeepEval faithfulness scoring
```

> **Note on response latency:** The Shepardizer and Compliance Auditor run as background
> `asyncio` tasks and do **not** block the HTTP response. The user receives the answer
> as soon as Gemini finishes (~10-12 seconds). Audit results appear in the evaluation
> dashboard shortly after.

---

## Ingestion Pipeline

```
Uploaded File
    │
    ▼  1. Document row inserted to Postgres (status: pending)
    │
    ▼  2. Celery task dispatched to worker
    │     Worker polls until document row is visible (race condition fix)
    │
    ▼  3. Parse  — pdfplumber / docx / plain text + OCR fallback
    │
    ▼  4. Chunk  — Recursive or Semantic strategy (configurable)
    │
    ▼  5. Enrich — Attach metadata (doc_type, client_id, filename, etc.)
    │
    ▼  6. Embed  — sentence-transformers/all-mpnet-base-v2 (local, 768-dim)
    │
    ▼  7. Store  — Qdrant upsert (with text in payload) + Postgres chunk rows
    │              Collection auto-created if missing (restart-safe)
    │
    ▼  8. BM25 index rebuilt
    │
    ▼  9. Document status → indexed
```

---

## The 3 Agents

### 🤖 Adversarial Lawyer Agent
Generates synthetic multi-hop QA pairs for benchmarking. Samples chunks cross-document to create single-hop, multi-hop, and edge-case questions. Stores to `golden_dataset` table. Uses truncation-safe JSON parsing to recover partial results if Gemini hits output limits.

### 🔍 Compliance Auditor Agent
DeepEval-powered faithfulness evaluator. Registered as an Observer — scores every live response asynchronously in the background. In batch mode, runs the full golden dataset for CI/CD reporting.

### ⚖️ Shepardizer Agent
Named after the legal practice of "Shepardizing" citations. Uses Chain of Responsibility: Context → Database → Relevance. Flags fabricated citations and attaches a health badge to every response. Runs in the background and does not block the response to the user.

---

## Evaluation & CI/CD

### RAG Triad Thresholds

| Metric | Threshold | Gate |
|--------|-----------|------|
| Faithfulness | ≥ 0.9 | **Hard — blocks PR** |
| Answer Relevance | ≥ 0.7 | Logged |
| Context Precision | ≥ 0.7 | Logged |

### Running Tests

```bash
# All unit tests (no API keys needed)
docker compose exec api pytest ../../tests/ -m "not integration" -v

# Just the faithfulness gate
docker compose exec api pytest ../../tests/test_faithfulness.py::TestFaithfulnessGate -v

# With coverage
docker compose exec api pytest ../../tests/ --cov=core --cov=api -m "not integration"
```

---

## API Reference

Interactive docs: `http://localhost:8000/docs`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/query` | Submit a legal question |
| `POST` | `/api/v1/ingest/upload` | Upload a document |
| `GET` | `/api/v1/ingest/status/{task_id}` | Poll ingestion progress |
| `GET` | `/api/v1/ingest/documents` | List documents |
| `DELETE` | `/api/v1/ingest/documents/{id}` | Delete a document |
| `POST` | `/api/v1/evaluate/generate-dataset` | Adversarial Lawyer Agent |
| `POST` | `/api/v1/evaluate/run` | Compliance Auditor batch eval |
| `GET` | `/api/v1/evaluate/results` | Evaluation run results |

---

## Known Bugs Fixed

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| All documents fail ingestion | FK violation: Celery worker arrived before API committed document INSERT | Added `_wait_for_document()` polling loop in `ingest_task.py` |
| Chunks stored but text not retrievable | `ec.payload` dict never included `text` field; Qdrant stored vectors with empty payloads | Added `{**ec.payload, "text": ec.text}` in `vector_store.upsert_chunks()` |
| Ingestion fails after `docker compose down -v` | Worker's Qdrant client cached collection existence; collection deleted with volumes but worker didn't recreate it | `_ensure_collection()` now called on every `upsert_chunks()` call |
| Query response takes 3+ minutes | Shepardizer and Compliance Auditor `await`ed inline, blocking HTTP response | Moved all observer hooks to `asyncio.create_task()` in `rag_service.py` |
| Golden dataset generation always returns 0 | `max_output_tokens=1536` too low for 5 QA pairs with full reference contexts; Gemini truncated mid-JSON | Raised to 4096; added truncation-safe `_safe_parse_json_array()` that salvages complete pairs before truncation point |

---

## GitHub Secrets Required

Set in **Settings → Secrets and variables → Actions**:

| Secret | Description |
|--------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key |
| `COHERE_API_KEY` | Cohere API key |
| `DEEPEVAL_API_KEY` | DeepEval API key |

---

*Built for the Capgemini GenAI Developer Assessment.*  
*Stack: Gemini 2.0 Flash · Qdrant · Cohere Rerank v3 · Redis · PostgreSQL · DeepEval · FastAPI · Streamlit*
