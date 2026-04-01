# app.py — Streamlit multi-turn chat interface for Product Insights Engine
# Design: no emoji, no custom colors. Inherits Streamlit's dark theme.

import streamlit as st
from pathlib import Path
import chromadb

from conversation import ConversationHistoryManager, ConversationTurn
from rag_with_claude import TinderRAGAnalyzer

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Product Insights Engine",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Minimal CSS — no hardcoded colors, theme-safe
# ============================================================

st.markdown("""
<style>
    /* Sidebar section labels */
    .sidebar-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        opacity: 0.45;
        margin: 1.4rem 0 0.4rem 0;
        padding: 0;
    }

    /* Stat row in sidebar */
    .stat-row {
        display: flex;
        gap: 1.5rem;
        margin: 0.3rem 0 0.6rem 0;
    }
    .stat-item { display: flex; flex-direction: column; }
    .stat-value {
        font-size: 1.05rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .stat-label {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.45;
    }

    /* Header */
    .app-header {
        padding: 0.8rem 0 0.5rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1.2rem;
    }
    .app-header h1 {
        font-size: 1.4rem;
        font-weight: 700;
        margin: 0 0 0.15rem 0;
        letter-spacing: -0.01em;
    }
    .app-header .subtitle {
        font-size: 0.85rem;
        opacity: 0.5;
        margin: 0;
    }

    /* Mode tag — neutral, just border */
    .mode-tag {
        display: inline-block;
        padding: 2px 8px;
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 3px;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.02em;
        margin-left: 0.5rem;
        vertical-align: middle;
        opacity: 0.7;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem 1rem;
    }
    .empty-state h2 {
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .empty-state p {
        font-size: 0.85rem;
        opacity: 0.5;
        margin-bottom: 1.2rem;
    }

    /* Retrieval indicator — just text with a dot */
    .retrieval-tag {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.72rem;
        font-weight: 400;
        opacity: 0.5;
        margin-top: 0.3rem;
    }
    .retrieval-tag .dot {
        width: 5px; height: 5px;
        border-radius: 50%;
        display: inline-block;
        background: currentColor;
    }

    /* Evidence panel */
    .evidence-meta {
        font-size: 0.75rem;
        font-family: monospace;
        opacity: 0.4;
        margin-bottom: 0.2rem;
    }
    .evidence-text {
        font-size: 0.84rem;
        opacity: 0.7;
        line-height: 1.5;
        padding-bottom: 0.6rem;
        margin-bottom: 0.6rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }

    /* Footer */
    .app-footer {
        text-align: center;
        font-size: 0.73rem;
        opacity: 0.3;
        padding: 1.5rem 0 0.8rem 0;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin-top: 2rem;
    }

    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Path Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "tinder_rag_db_recent"

# ============================================================
# Resource Initialization
# ============================================================

@st.cache_resource
def load_analyzer(api_key: str) -> TinderRAGAnalyzer:
    return TinderRAGAnalyzer(api_key=api_key, db_path=str(DB_PATH))

@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_collection("tinder_reviews_recent")

@st.cache_data
def get_collection_stats(_collection):
    meta = _collection.metadata
    return {
        "total_reviews": _collection.count(),
        "date_range": (
            f"{meta.get('date_range_start', 'N/A')} to "
            f"{meta.get('date_range_end', 'N/A')}"
        ),
    }

# ============================================================
# Helpers
# ============================================================

def render_retrieval_indicator(retrieval_triggered: bool) -> None:
    label = "New evidence retrieved" if retrieval_triggered else "Answered from session context"
    st.markdown(
        f'<div class="retrieval-tag"><span class="dot"></span>{label}</div>',
        unsafe_allow_html=True,
    )

# ============================================================
# Session State
# ============================================================

def init_session_state() -> None:
    if "history_manager" not in st.session_state:
        st.session_state.history_manager = ConversationHistoryManager()
    if "display_messages" not in st.session_state:
        st.session_state.display_messages = []
    if "all_session_chunks" not in st.session_state:
        st.session_state.all_session_chunks = []
    if "session_analysis_mode" not in st.session_state:
        st.session_state.session_analysis_mode = None

def reset_session() -> None:
    st.session_state.history_manager = ConversationHistoryManager()
    st.session_state.display_messages = []
    st.session_state.all_session_chunks = []
    st.session_state.session_analysis_mode = None

# ============================================================
# Main
# ============================================================

def main():
    init_session_state()

    # ── Sidebar ───────────────────────────────────────────────
    try:
        collection = load_collection()
        stats = get_collection_stats(collection)
        st.sidebar.caption("SYSTEM READY")
        st.sidebar.markdown(
            f'<div class="stat-row">'
            f'  <div class="stat-item">'
            f'    <span class="stat-value">{stats["total_reviews"]:,}</span>'
            f'    <span class="stat-label">Reviews</span>'
            f'  </div>'
            f'  <div class="stat-item">'
            f'    <span class="stat-value" style="font-size:0.82rem;">'
            f'{stats["date_range"]}</span>'
            f'    <span class="stat-label">Date Range</span>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Error loading vector database: {e}")
        st.stop()

    st.sidebar.markdown(
        '<p class="sidebar-label">Configuration</p>', unsafe_allow_html=True
    )
    api_key = st.sidebar.text_input(
        "API Key",
        type="password",
        help="Anthropic API key — console.anthropic.com",
        label_visibility="collapsed",
        placeholder="Enter Claude API key",
    )
    if not api_key:
        st.info("Enter your Claude API key in the sidebar to start.")
        st.stop()

    st.sidebar.markdown(
        '<p class="sidebar-label">Analysis Mode</p>', unsafe_allow_html=True
    )
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["General Analysis", "Root Cause Analysis", "Feature Requests"],
        help="Changing the mode resets the current session.",
        label_visibility="collapsed",
    )
    if (
        st.session_state.session_analysis_mode is not None
        and st.session_state.session_analysis_mode != analysis_type
        and len(st.session_state.display_messages) > 0
    ):
        reset_session()
        st.toast("Mode changed — session reset.")
    st.session_state.session_analysis_mode = analysis_type

    st.sidebar.markdown(
        '<p class="sidebar-label">Retrieval</p>', unsafe_allow_html=True
    )
    n_reviews = st.sidebar.slider(
        "Reviews per query",
        min_value=5, max_value=30, value=15,
        help="More reviews = richer context, slower response",
        label_visibility="collapsed",
    )

    st.sidebar.markdown(
        '<p class="sidebar-label">Filters</p>', unsafe_allow_html=True
    )
    review_type = st.sidebar.radio(
        "Review Type",
        ["All reviews", "Negative only", "Positive only"],
        label_visibility="collapsed",
    )
    min_thumbs = st.sidebar.number_input(
        "Min. thumbs up",
        min_value=0, max_value=100, value=0,
        help="Focus on highly-upvoted reviews",
    )

    chroma_filters: dict = {}
    if review_type == "Negative only":
        chroma_filters["is_negative"] = True
    elif review_type == "Positive only":
        chroma_filters["is_positive"] = True
    if min_thumbs > 0:
        chroma_filters["thumbs_up"] = {"$gte": min_thumbs}
    active_filters = chroma_filters if chroma_filters else None

    st.sidebar.markdown(
        '<p class="sidebar-label">Session</p>', unsafe_allow_html=True
    )
    if st.sidebar.button("New Session", use_container_width=True):
        reset_session()
        st.rerun()
    if st.session_state.display_messages:
        turn_count = len(st.session_state.display_messages) // 2
        st.sidebar.caption(f"Turn {turn_count}")

    # ── Analyzer ──────────────────────────────────────────────
    try:
        analyzer = load_analyzer(api_key)
    except Exception as e:
        st.error(f"Failed to initialize analyzer: {e}")
        st.stop()

    # ── Header ────────────────────────────────────────────────
    st.markdown(
        f'<div class="app-header">'
        f'  <h1>Product Insights Engine'
        f'    <span class="mode-tag">{analysis_type}</span>'
        f'  </h1>'
        f'  <p class="subtitle">RAG-powered analysis of user feedback</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Chat ──────────────────────────────────────────────────
    chat_container = st.container()

    with chat_container:
        if not st.session_state.display_messages:
            st.markdown(
                '<div class="empty-state">'
                '  <h2>Start an analysis session</h2>'
                '  <p>Ask a question about user feedback, or pick one below.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            quick_questions = [
                "What are the biggest complaints in the last 12 months?",
                "What features are users requesting most?",
                "Why has the rating declined recently?",
                "What do users think about pricing?",
                "What bugs are most commonly reported?",
            ]
            cols = st.columns(2)
            for i, q in enumerate(quick_questions):
                if cols[i % 2].button(q, key=f"quick_{i}", use_container_width=True):
                    st.session_state["prefilled_question"] = q
                    st.rerun()

        for msg in st.session_state.display_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    render_retrieval_indicator(msg.get("retrieval_triggered", False))

    # ── Evidence Panel ────────────────────────────────────────
    if st.session_state.all_session_chunks:
        chunk_count = len(st.session_state.all_session_chunks)
        with st.expander(f"Session evidence — {chunk_count} sources", expanded=False):
            for i, chunk in enumerate(st.session_state.all_session_chunks, 1):
                meta = chunk.get("metadata", {})
                doc = chunk.get("document", "")
                score = meta.get("score", "—")
                date = meta.get("date", "—")
                thumbs = meta.get("thumbs_up", 0)

                st.markdown(
                    f'<div class="evidence-meta">'
                    f'#{i} · {score}/5 · {date} · {thumbs} upvotes'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                content_start = doc.find("User Feedback:")
                if content_start != -1:
                    preview = doc[content_start + 15:].strip()
                    st.markdown(
                        f'<div class="evidence-text">'
                        f'{preview[:300]}{"…" if len(preview) > 300 else ""}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # ── Input ─────────────────────────────────────────────────
    prefilled = st.session_state.pop("prefilled_question", None)
    user_input = st.chat_input("Ask a question or follow up…", key="chat_input")
    question = user_input or prefilled

    if not question:
        st.markdown(
            '<div class="app-footer">'
            'RAG + Claude · Tinder Google Play Reviews (Last 12 months)'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # ── Process ───────────────────────────────────────────────
    with st.chat_message("user"):
        st.markdown(question)

    is_first_turn = len(st.session_state.display_messages) == 0

    with st.chat_message("assistant"):
        with st.spinner("Analyzing…"):
            if is_first_turn:
                raw_results = analyzer.retrieve_reviews(
                    question, n_results=n_reviews, filters=active_filters
                )
                if not raw_results["documents"][0]:
                    st.warning("No reviews matched the current filters.")
                    return
                analysis = analyzer.analyze_with_claude(
                    question, raw_results, analysis_type
                )
                retrieval_triggered = True
                chunks_used = [
                    {"id": cid, "document": doc, "metadata": meta}
                    for cid, doc, meta in zip(
                        raw_results["ids"][0],
                        raw_results["documents"][0],
                        raw_results["metadatas"][0],
                    )
                ]
                answer = analysis
            else:
                result = analyzer.follow_up(
                    question=question,
                    conversation_history=st.session_state.history_manager,
                    filters=active_filters,
                    n_reviews=n_reviews,
                    analysis_type=analysis_type,
                )
                answer = result["answer"]
                retrieval_triggered = result["retrieval_triggered"]
                chunks_used = result["chunks_used"]

        st.markdown(answer)
        render_retrieval_indicator(retrieval_triggered)

    # ── Update state ──────────────────────────────────────────
    st.session_state.history_manager.add_turn(ConversationTurn(
        role="user", content=question,
        retrieved_chunks=None, retrieval_triggered=False,
    ))
    st.session_state.history_manager.add_turn(ConversationTurn(
        role="assistant", content=answer,
        retrieved_chunks=chunks_used if chunks_used else None,
        retrieval_triggered=retrieval_triggered,
    ))
    st.session_state.display_messages.append(
        {"role": "user", "content": question, "retrieval_triggered": False}
    )
    st.session_state.display_messages.append(
        {"role": "assistant", "content": answer, "retrieval_triggered": retrieval_triggered}
    )
    if chunks_used:
        existing_ids = {c.get("id") for c in st.session_state.all_session_chunks}
        for chunk in chunks_used:
            if chunk.get("id") not in existing_ids:
                st.session_state.all_session_chunks.append(chunk)

    st.markdown(
        '<div class="app-footer">'
        'RAG + Claude · Tinder Google Play Reviews (Last 12 months)'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
