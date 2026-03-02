"""
LegalMind — Streamlit Frontend
Three-tab interface:
  ⚖️  Query      — Chat-style legal question answering
  📄 Documents  — Upload and manage legal documents
  🧪 Evaluation — Evaluation dashboard and CI/CD metrics
"""
import uuid
from datetime import datetime
from typing import Optional
import httpx
import streamlit as st
st.set_page_config(
    page_title="LegalMind",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)
API_BASE = "http://api:8000/api/v1"
# Increased to 300s — Gemini 2.5 Pro + embedding model cold start can take ~2min
TIMEOUT  = httpx.Timeout(3000.0)
def api_get(path: str, params: dict = None) -> Optional[dict]:
    try:
        r = httpx.get(f"{API_BASE}{path}", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None
def api_post(path: str, json: dict = None, files=None, data=None) -> Optional[dict]:
    try:
        r = httpx.post(f"{API_BASE}{path}", json=json, files=files, data=data, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None
def api_delete(path: str) -> bool:
    try:
        r = httpx.delete(f"{API_BASE}{path}", timeout=TIMEOUT)
        return r.status_code == 204
    except Exception as e:
        st.error(f"Delete error: {e}")
        return False
# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ LegalMind")
    st.caption("AI-Powered Legal Research Assistant")
    st.divider()
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session-{str(uuid.uuid4())[:8]}"
    st.write("**Session ID**")
    st.code(st.session_state.session_id, language=None)
    st.divider()
    st.write("**System Status**")
    try:
        health = httpx.get("http://api:8000/health", timeout=5).json()
        st.success(f"✅ API: {health.get('status', 'unknown')}")
    except Exception:
        st.error("❌ API: unreachable")
    cache_data = api_get("/evaluate/cache/stats")
    if cache_data:
        st.metric("Cached Queries", cache_data.get("entry_count", 0))
    st.divider()
    st.caption("Built for Capgemini GenAI Assessment")
    st.caption("Stack: Gemini · Qdrant · Redis · PostgreSQL")
# ── Tabs ───────────────────────────────────────────────────────────
tab_query, tab_docs, tab_eval = st.tabs([
    "⚖️ Legal Query", "📄 Documents", "🧪 Evaluation"
])
# ══════════════════════════════════════════════════════════════════
# TAB 1: Legal Query
# ══════════════════════════════════════════════════════════════════
with tab_query:
    st.header("⚖️ Legal Research Query")
    st.caption("Ask questions grounded in your indexed legal documents. Every answer includes mandatory source citations.")
    with st.expander("🔍 Metadata Filters (optional)", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_client = st.text_input("Client ID", placeholder="e.g. CLIENT-001")
        with col2:
            filter_doc_type = st.selectbox("Document Type", ["(all)", "contract", "case_file", "brief", "memo", "other"])
        with col3:
            filter_date_from = st.date_input("Filed After", value=None)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "⚖️"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📎 {len(msg['sources'])} Source(s)"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(
                            f"**{i}. [{src['filename']} | Chunk {src['chunk_index']}]**  \n"
                            f"*Relevance: {src['relevance_score']:.3f} · Type: {src.get('doc_type','N/A')} · Client: {src.get('client_id','N/A')}*"
                        )
                        st.caption(src["text"][:400] + ("..." if len(src["text"]) > 400 else ""))
                        st.divider()
            if msg.get("meta"):
                m = msg["meta"]
                c1, c2, c3 = st.columns(3)
                c1.metric("Latency", f"{m.get('latency_ms',0)}ms")
                c2.metric("Sources", len(msg.get("sources", [])))
                c3.metric("Cache Hit", "✅" if m.get("cache_hit") else "❌")
    query = st.chat_input("Ask a legal question about your documents...")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="👤"):
            st.markdown(query)
        request_body = {"query": query, "session_id": st.session_state.session_id}
        if filter_client:
            request_body["filter_client_id"] = filter_client
        if filter_doc_type != "(all)":
            request_body["filter_doc_type"] = filter_doc_type
        if filter_date_from:
            request_body["filter_date_from"] = filter_date_from.isoformat()
        with st.chat_message("assistant", avatar="⚖️"):
            with st.spinner("Searching legal documents and generating answer (first query may take ~60s while the AI model warms up)..."):
                response = api_post("/query", json=request_body)
            if response:
                answer  = response.get("answer", "No answer returned.")
                sources = response.get("sources", [])
                latency = response.get("latency_ms", 0)
                cache_hit = response.get("cache_hit", False)
                st.markdown(answer)
                if sources:
                    with st.expander(f"📎 {len(sources)} Source(s) Cited"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(
                                f"**{i}. [{src['filename']} | Chunk {src['chunk_index']}]**  \n"
                                f"*Relevance: {src['relevance_score']:.3f} · Type: {src.get('doc_type','N/A')} · Client: {src.get('client_id','N/A')}*"
                            )
                            st.caption(src["text"][:400] + ("..." if len(src["text"]) > 400 else ""))
                            st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("Latency", f"{latency}ms")
                c2.metric("Sources", len(sources))
                c3.metric("Cache Hit", "✅" if cache_hit else "❌")
                st.session_state.chat_history.append({
                    "role": "assistant", "content": answer,
                    "sources": sources, "meta": {"latency_ms": latency, "cache_hit": cache_hit}
                })
            else:
                st.error("Failed to get a response.")
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
# ══════════════════════════════════════════════════════════════════
# TAB 2: Document Management
# ══════════════════════════════════════════════════════════════════
with tab_docs:
    st.header("📄 Document Management")
    st.subheader("Upload New Document")
    with st.form("upload_form", clear_on_submit=True):
        uploaded_file = st.file_uploader("Choose a legal document", type=["pdf", "docx", "txt"])
        col1, col2, col3 = st.columns(3)
        with col1:
            doc_type  = st.selectbox("Document Type", ["contract", "case_file", "brief", "memo", "other"])
            client_id = st.text_input("Client ID", placeholder="CLIENT-001")
        with col2:
            matter_id  = st.text_input("Matter ID", placeholder="MATTER-2024-042")
            date_filed = st.date_input("Date Filed", value=None)
        with col3:
            chunking = st.radio("Chunking Strategy", ["recursive", "semantic"],
                                help="Recursive: fast. Semantic: clause-aware.")
        submitted = st.form_submit_button("📤 Upload & Ingest", type="primary")
    if submitted and uploaded_file:
        with st.spinner(f"Uploading {uploaded_file.name}..."):
            files     = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            form_data = {"doc_type": doc_type, "chunking_strategy": chunking}
            if client_id:  form_data["client_id"]  = client_id
            if matter_id:  form_data["matter_id"]   = matter_id
            if date_filed: form_data["date_filed"]  = date_filed.isoformat()
            result = api_post("/ingest/upload", files=files, data=form_data)
        if result:
            st.success(f"✅ Accepted! Task ID: `{result['task_id']}`")
            st.session_state.last_task_id = result["task_id"]
    if "last_task_id" in st.session_state:
        s = api_get(f"/ingest/status/{st.session_state.last_task_id}")
        if s:
            icon = {"indexed":"🟢","processing":"🟡","pending":"🔵","failed":"🔴"}.get(s.get("status",""), "⚪")
            st.write(f"**Last task:** {icon} `{s.get('status')}` — {s.get('chunk_count',0)} chunks")
    st.divider()
    st.subheader("Indexed Documents")
    c1, c2, c3 = st.columns([1,2,2])
    with c1:
        if st.button("🔄 Refresh"): st.rerun()
    with c2:
        list_type = st.selectbox("Filter type", ["(all)","contract","case_file","brief","memo"], key="lt")
    with c3:
        list_client = st.text_input("Filter client", key="lc")
    params = {"page_size": 50}
    if list_type != "(all)": params["doc_type"]   = list_type
    if list_client:          params["client_id"]  = list_client
    docs_data = api_get("/ingest/documents", params=params)
    if docs_data and docs_data.get("items"):
        st.caption(f"Showing {len(docs_data['items'])} of {docs_data['total']} documents")
        for doc in docs_data["items"]:
            status = doc.get("status", "unknown")
            icon   = {"indexed":"🟢","processing":"🟡","pending":"🔵","failed":"🔴"}.get(status,"⚪")
            with st.expander(f"{icon} {doc['filename']} — {doc.get('doc_type','N/A')} | {doc.get('chunk_count',0)} chunks"):
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Status", status)
                c2.metric("Chunks", doc.get("chunk_count",0))
                c3.write(f"**Client:** {doc.get('client_id','N/A')}")
                c4.write(f"**Filed:** {doc.get('date_filed','N/A')}")
                st.caption(f"ID: `{doc['id']}` · Ingested: {doc.get('ingested_at','')[:19]}")
                if st.button("🗑️ Delete", key=f"del_{doc['id']}"):
                    if api_delete(f"/ingest/documents/{doc['id']}"):
                        st.success("Deleted"); st.rerun()
    else:
        st.info("No documents yet. Upload one above to get started.")
# ══════════════════════════════════════════════════════════════════
# TAB 3: Evaluation Dashboard
# ══════════════════════════════════════════════════════════════════
with tab_eval:
    st.header("🧪 Evaluation Dashboard")
    st.caption("RAG quality monitoring via DeepEval. Faithfulness ≥ 0.90 required to pass CI/CD gate.")
    col_gen, col_run = st.columns(2)
    with col_gen:
        st.subheader("1. Generate Golden Dataset")
        dataset_size = st.number_input("Target size", 5, 200, 50)
        if st.button("🤖 Generate Dataset", type="primary"):
            r = api_post(f"/evaluate/generate-dataset?target_size={dataset_size}")
            if r: st.success("Adversarial Lawyer Agent started")
    with col_run:
        st.subheader("2. Run Evaluation")
        run_id = st.text_input("Run ID", value=f"manual-{datetime.now().strftime('%Y%m%d-%H%M')}")
        if st.button("🧪 Run Evaluation", type="primary"):
            r = api_post("/evaluate/run", json={"run_id": run_id})
            if r: st.success(f"Started: `{run_id}`")
    st.divider()
    st.subheader("Golden Dataset Preview")
    dataset = api_get("/evaluate/dataset", params={"page_size": 5})
    if dataset and dataset.get("items"):
        st.caption(f"{dataset['total']} total QA pairs")
        for e in dataset["items"]:
            with st.expander(f"Q: {e['question'][:80]}..."):
                st.write("**Question:**", e["question"])
                st.write("**Expected Answer:**", e["expected_answer"])
                st.caption(f"Generated by: {e['generated_by']} · {e['created_at'][:19]}")
    else:
        st.info("No golden dataset entries yet.")
    st.divider()
    st.subheader("Evaluation Results")
    results = api_get("/evaluate/results", params={"limit": 10})
    if results:
        import pandas as pd
        rows = [{
            "Run ID": r["run_id"],
            "Faithfulness": round(r.get("faithfulness") or 0, 3),
            "Relevance": round(r.get("answer_relevance") or 0, 3),
            "Precision": round(r.get("context_precision") or 0, 3),
            "Cases": r.get("total_cases", 0),
            "✅ Passed": r.get("passed_cases", 0),
            "❌ Failed": r.get("failed_cases", 0),
            "Result": "✅ PASS" if r.get("passed") else "❌ FAIL",
            "Date": r["ran_at"][:19] if r.get("ran_at") else "",
        } for r in results]
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, column_config={
            "Faithfulness": st.column_config.ProgressColumn("Faithfulness", min_value=0, max_value=1, format="%.3f"),
            "Relevance":    st.column_config.ProgressColumn("Relevance",    min_value=0, max_value=1, format="%.3f"),
            "Precision":    st.column_config.ProgressColumn("Precision",    min_value=0, max_value=1, format="%.3f"),
        })
        if len(rows) >= 2:
            import plotly.graph_objects as go
            fig = go.Figure()
            dates = [r["Date"] for r in rows[::-1]]
            fig.add_trace(go.Scatter(x=dates, y=[r["Faithfulness"] for r in rows[::-1]], name="Faithfulness", line=dict(color="#00cc66", width=3)))
            fig.add_trace(go.Scatter(x=dates, y=[r["Relevance"] for r in rows[::-1]],   name="Answer Relevance", line=dict(color="#0099ff", width=2, dash="dash")))
            fig.add_trace(go.Scatter(x=dates, y=[r["Precision"] for r in rows[::-1]],   name="Context Precision", line=dict(color="#ff9900", width=2, dash="dot")))
            fig.add_hline(y=0.9, line_dash="dash", line_color="red", annotation_text="CI/CD Gate (0.90)")
            fig.update_layout(yaxis=dict(range=[0,1]), height=350, margin=dict(t=40,b=40),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No evaluation runs yet.")
    st.divider()
    st.subheader("Semantic Cache")
    cs = api_get("/evaluate/cache/stats")
    if cs:
        c1,c2,c3 = st.columns(3)
        c1.metric("Cached Queries", cs.get("entry_count",0))
        c2.metric("Similarity Threshold", cs.get("threshold", 0.92))
        c3.metric("TTL", f"{cs.get('ttl_seconds',3600)}s")
    if st.button("🗑️ Clear Cache"):
        r = api_post("/evaluate/cache/clear")
        if r: st.success(f"Cleared {r.get('entries_deleted',0)} entries"); st.rerun()
