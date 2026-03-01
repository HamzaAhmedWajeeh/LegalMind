# ⚖️ LegalMind Knowledge Assistant

> A production-grade, modular RAG system for querying 10,000+ legal documents with source citations, hallucination detection, and automated evaluation.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                      │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   FastAPI    │  │   Qdrant     │  │      Redis       │  │
│  │  (Backend)   │  │ (VectorDB)   │  │ (Semantic Cache) │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Streamlit   │  │  PostgreSQL  │  │ Celery Worker    │  │
│  │  (Frontend)  │  │  (Metadata)  │  │ (Async Ingest)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 🔑 Design Patterns Used

| Pattern | Where Applied |
|---|---|
| **Strategy** | Chunking strategies (fixed vs semantic) are swappable |
| **Factory** | LLM and retriever instances created via factory functions |
| **Pipeline / Chain** | Ingestion and retrieval stages are composable |
| **Repository** | Abstract interface over Qdrant + Postgres |
| **Observer** | Evaluation hooks fire after each generation |
| **Facade** | Single `RAGService` hides retrieval + rerank + generate complexity |

---

## 🚀 Quick Start

### Prerequisites
- Docker + Docker Compose v2
- API keys: Anthropic, Cohere, (optional) DeepEval

### 1. Clone & Configure

```bash
git clone <repo-url>
cd legalmind
cp .env.example .env
# Edit .env and add your API keys
```

### 2. Start the Stack

```bash
docker compose up --build
```

### 3. Access Services

| Service | URL |
|---|---|
| Chat UI (Streamlit) | http://localhost:8501 |
| API (FastAPI) | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Qdrant Dashboard | http://localhost:6333/dashboard |

### 4. Run Evaluations

```bash
# Run the full RAG evaluation suite
docker compose exec api pytest tests/ -v

# Run only faithfulness tests (CI/CD gate)
docker compose exec api pytest tests/test_faithfulness.py -v
```

---

## 📦 Services

| Service | Image | Purpose |
|---|---|---|
| `api` | Custom FastAPI | RAG pipeline, agents, REST API |
| `worker` | Same as api | Async Celery ingestion worker |
| `frontend` | Custom Streamlit | Chat UI + evaluation dashboard |
| `qdrant` | qdrant/qdrant:v1.9.2 | Vector database |
| `redis` | redis:7.2-alpine | Semantic cache + task broker |
| `postgres` | postgres:16-alpine | Metadata, audit logs, golden dataset |

---

## 🤖 Agents

1. **Adversarial Lawyer** — Generates synthetic QA pairs for golden dataset
2. **Compliance Auditor** — Faithfulness / hallucination detection via DeepEval
3. **Shepardizer** — Citation and source attribution validator

---

## 🧪 Evaluation (DeepEval)

The system implements the **RAG Triad**:

- **Faithfulness ≥ 0.9** — Answers grounded in retrieved context only
- **Answer Relevance** — Response addresses the actual question
- **Context Precision** — Most relevant chunks ranked highest

CI/CD via GitHub Actions runs these on every Pull Request.