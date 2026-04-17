# Product Insights RAG

A conversational product analytics tool that turns raw user reviews into structured, evidence-backed insights. Ask a question, get an answer grounded in real data, then follow up — the system remembers the conversation and decides whether to retrieve new evidence or work with what it already has.

Built with ChromaDB, Claude API, and Streamlit.

---

## How it works
```
User question
     ↓
[First turn]  Embed → Retrieve from ChromaDB → Mode-specific Claude analysis
     ↓
[Follow-ups]  LLM router decides:
              ├── Answer from existing context  →  Claude responds using session history
              └── New retrieval needed          →  Re-query ChromaDB → Claude responds
```

<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/7af7d5d6-71f6-47de-b9b6-ec5c4f4ed921" />


**Three analysis modes, each with a different reasoning framework:**

- **General Analysis:** Identifies dominant themes, assesses severity (cosmetic → functional blocker → trust risk), flags trend shifts with dates, and outputs scoped sprint-level recommendations.
- **Root Cause Analysis:** Traces causal chains from symptom → mechanism → root cause. Distinguishes independent vs. shared root causes across complaint clusters. Rates each chain by evidence strength.
- **Feature Requests:** Clusters demand by theme, performs gap analysis (current workaround vs. desired state), and separates high-frequency/low-intensity requests from low-frequency/high-intensity ones.

These aren't just different output templates — each mode changes how the LLM reasons about the data.

**Retrieval decision layer:** A lightweight `claude-haiku` call acts as a router on every follow-up turn. It reads the conversation history and decides whether existing evidence is sufficient or a new ChromaDB query is needed. If new retrieval is triggered, it also rewrites the user's question into a query optimized for semantic search. Fallback behavior: if the router fails, the system defaults to new retrieval (safe over sorry).

**Context window management:** Recent turns are kept in full; older turns are compressed into one-line summaries. Retrieved chunks from earlier turns are capped and truncated. This keeps the prompt lean enough for 5+ turn sessions without degradation.

---

## Design decisions

**Why LLM-as-router instead of keyword heuristics for retrieval decisions?**

"Tell me more" could mean anything. "What about pricing?" might be a new topic or a drill-down on something already covered. Keyword rules can't reliably make this call. The LLM already has the full conversation context — it's the best judge of whether existing evidence is sufficient. Cost is negligible: one short classification call per turn.

**Why mode-specific analytical frameworks instead of one general prompt?**

Early versions used three prompts that were essentially the same structure with different output checklists. The LLM produced similar-looking responses regardless of mode. Rewriting each mode as a distinct reasoning framework (causal chain tracing for root cause, gap analysis for feature requests) changed the actual analytical approach, not just the formatting.

**Why cap retrieval at 20–30 reviews instead of pulling everything?**

Semantic similarity drops off fast. The top 15 results for "billing complaints" are genuinely about billing. By result 30+, you're getting reviews that happen to mention "charge" in passing. More evidence doesn't mean better analysis — it means more noise and a less focused LLM response. If broader coverage is needed, multi-query retrieval (same question rewritten from multiple angles, results deduplicated) is a better approach than increasing n.

---

## Toward production

This is a portfolio project, but the architecture was designed with real deployment in mind. Here's what changes (and what doesn't) in a production context.

**What stays the same:** The core RAG + LLM analysis loop, the retrieval decision layer, the multi-turn session model. These patterns are domain-agnostic — swap the review data for support tickets, NPS responses, or return reasons and the system works the same way.

**Data ingestion.** The current pipeline reads from a static CSV. In production, this becomes an API connector or warehouse query (Shopify, Zendesk, BigQuery, etc.) with incremental embedding — only new reviews get embedded and appended to the existing vector store, no full rebuilds.

**Evaluation.** The current system cites source reviews as evidence, but doesn't score retrieval quality. A production version would add LLM-as-judge scoring on each retrieved chunk (is this actually relevant to the query?) and track retrieval precision over time. Answer faithfulness evaluation (does the response stay within what the evidence supports?) is conceptually important but requires ground truth data to do properly.

**Scheduling and delivery.** Instead of an interactive UI, production use often means scheduled runs: daily or weekly analysis on new reviews, delivered via Slack or email dashboards. The analytical engine is the same; the trigger and output channel change.

**Scale.** ChromaDB works well for datasets up to low millions of documents. Beyond that, a managed vector database (Pinecone, Weaviate, pgvector) provides better indexing, filtering, and concurrent access.

---

## Project structure
```
├── app.py                                  # Streamlit chat UI + session state
├── rag_with_claude.py                      # TinderRAGAnalyzer: ask() + follow_up()
├── conversation.py                         # ConversationTurn dataclass + history manager
├── build_rag_system_recent.py              # One-time script: CSV → ChromaDB
└── FEATURE_SPEC_conversational_followup.md # Feature spec for the multi-turn upgrade
```

---

## Setup

**Install dependencies**
```bash
pip install streamlit chromadb sentence-transformers anthropic
```

**Build the vector database** (one-time)
```bash
python build_rag_system_recent.py
```

Reads the review CSV, generates embeddings with `all-MiniLM-L6-v2`, and writes to `./tinder_rag_db_recent/`.

**Run the app**
```bash
streamlit run app.py
```

Enter your Claude API key in the sidebar. Get one at [console.anthropic.com](https://console.anthropic.com).

---

## Usage

- First question always triggers a fresh retrieval
- Follow-ups go through the retrieval decision layer automatically
- Each response shows an indicator: new evidence retrieved vs. answered from session context
- Changing analysis mode resets the session (different mode = different analytical lens)
- Evidence panel at the bottom shows all source reviews used across the session

---

## Tech stack

| Component | Role |
|---|---|
| ChromaDB | Vector storage + semantic retrieval |
| all-MiniLM-L6-v2 | Embedding model (sentence-transformers) |
| Claude Sonnet | Primary analysis LLM |
| Claude Haiku | Lightweight retrieval router |
| Streamlit | Interactive chat UI |

---

## Data

21,722 Tinder Google Play reviews from the last 12 months (avg. rating: 2.47 stars). Raw data and the built vector database are excluded from this repo — run `build_rag_system_recent.py` with your own copy of the CSV to generate the database locally.
