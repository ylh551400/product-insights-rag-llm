# Product Insights RAG

A conversational product analytics tool that lets you query and drill into user reviews using retrieval-augmented generation. Ask a question, get an evidence-backed answer, then follow up — the system remembers the conversation and decides intelligently whether to retrieve new evidence or answer from what it already has.

Built with ChromaDB, Claude API, and Streamlit.

---

## How it works

```
User question
     ↓
[First turn]  Embed → Retrieve from ChromaDB → Claude analysis
     ↓
[Follow-ups]  LLM router decides:
              ├── Answer from existing context  →  Claude responds using session history
              └── New retrieval needed          →  Re-query ChromaDB → Claude responds
```

**Retrieval decision layer:** Instead of re-querying the database on every follow-up (wasteful) or always answering from context (risky), a lightweight `claude-haiku` call acts as a router. It reads the conversation history and decides whether existing evidence is sufficient or new retrieval is needed. If new retrieval is triggered, it also rewrites the query for better semantic search.

**Context window management:** Conversation history is serialized with the most recent 3 turns in full and older turns compressed into one-line summaries, keeping the prompt lean across long sessions.

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

### 1. Install dependencies

```bash
pip install streamlit chromadb sentence-transformers anthropic
```

### 2. Prepare your data

The build script expects a CSV file with Tinder Google Play reviews. Place it in the project folder and update the filename in `build_rag_system_recent.py` (line 15) if needed.

### 3. Build the vector database

This only needs to be run once (or when you want to refresh the data).

```bash
python build_rag_system_recent.py
```

This reads the CSV, generates embeddings with `all-MiniLM-L6-v2`, and stores everything in a local ChromaDB database at `./tinder_rag_db_recent/`.

### 4. Run the app

```bash
streamlit run app.py
```

Enter your Claude API key in the sidebar when prompted. Get one at [console.anthropic.com](https://console.anthropic.com).

---

## Usage

- **First question** always triggers a fresh retrieval from the database
- **Follow-up questions** go through the retrieval decision layer automatically
- Each assistant response shows whether it used existing evidence or retrieved new chunks
- **New Session** button in the sidebar clears the conversation and starts fresh
- Changing the analysis mode mid-session also resets the conversation
- The evidence panel at the bottom shows all source reviews retrieved during the session

---

## Tech stack

| Component | Library |
|---|---|
| Vector database | ChromaDB |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| LLM | Claude (Anthropic API) |
| Routing | `claude-haiku` (lightweight classification call) |
| UI | Streamlit |

---

## Data

The knowledge base covers **21,722 Tinder Google Play reviews** from the last 12 months (2025-03-31 to 2025-11-27), with an average rating of 2.47 stars. The raw CSV and the built vector database are excluded from this repo — run `build_rag_system_recent.py` locally to generate the database from your own copy of the data.
