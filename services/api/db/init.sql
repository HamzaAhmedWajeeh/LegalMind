-- ============================================================
--  LegalMind — Database Schema
--  Runs automatically on first postgres container start
-- ============================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ------------------------------------------------------------
-- documents — Master registry of all ingested files
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename        VARCHAR(500)    NOT NULL,
    file_hash       VARCHAR(64)     UNIQUE NOT NULL,  -- SHA-256 to prevent duplicates
    doc_type        VARCHAR(50),                       -- 'contract', 'case_file', 'brief', etc.
    client_id       VARCHAR(100),
    matter_id       VARCHAR(100),
    date_filed      DATE,
    ingested_at     TIMESTAMPTZ     DEFAULT NOW(),
    status          VARCHAR(20)     DEFAULT 'pending', -- pending | processing | indexed | failed
    chunk_count     INTEGER         DEFAULT 0,
    metadata        JSONB           DEFAULT '{}'
);

-- ------------------------------------------------------------
-- chunks — Individual text chunks linked to documents
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id     UUID            REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER         NOT NULL,
    text            TEXT            NOT NULL,
    token_count     INTEGER,
    qdrant_id       VARCHAR(100),   -- ID in Qdrant vector store
    metadata        JSONB           DEFAULT '{}'
);

-- ------------------------------------------------------------
-- query_logs — Every query made + response (for audit trail)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS query_logs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      VARCHAR(100),
    query_text      TEXT            NOT NULL,
    response_text   TEXT,
    source_doc_ids  UUID[],         -- Array of cited document IDs
    cache_hit       BOOLEAN         DEFAULT FALSE,
    latency_ms      INTEGER,
    created_at      TIMESTAMPTZ     DEFAULT NOW(),
    metadata        JSONB           DEFAULT '{}'
);

-- ------------------------------------------------------------
-- golden_dataset — Synthetic QA pairs for evaluation
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS golden_dataset (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    question        TEXT            NOT NULL,
    reference_context TEXT          NOT NULL,
    expected_answer TEXT            NOT NULL,
    source_doc_ids  UUID[],
    generated_by    VARCHAR(50)     DEFAULT 'adversarial_lawyer_agent',
    created_at      TIMESTAMPTZ     DEFAULT NOW(),
    is_active       BOOLEAN         DEFAULT TRUE
);

-- ------------------------------------------------------------
-- eval_runs — Evaluation run results per CI/CD build
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eval_runs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id          VARCHAR(100)    UNIQUE NOT NULL,  -- e.g. GitHub Actions run ID
    faithfulness    FLOAT,
    answer_relevance FLOAT,
    context_precision FLOAT,
    total_cases     INTEGER,
    passed_cases    INTEGER,
    failed_cases    INTEGER,
    passed          BOOLEAN,
    ran_at          TIMESTAMPTZ     DEFAULT NOW(),
    metadata        JSONB           DEFAULT '{}'
);

-- ------------------------------------------------------------
-- Indexes for common query patterns
-- ------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_documents_client_id  ON documents(client_id);
CREATE INDEX IF NOT EXISTS idx_documents_doc_type   ON documents(doc_type);
CREATE INDEX IF NOT EXISTS idx_documents_date_filed ON documents(date_filed);
CREATE INDEX IF NOT EXISTS idx_documents_status     ON documents(status);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id   ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_query_logs_session   ON query_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_query_logs_created   ON query_logs(created_at DESC);

-- Done
