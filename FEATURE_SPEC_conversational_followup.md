# Feature Spec: Conversational Follow-Up for RAG Insights

## Problem

The current system is single-turn: user asks a question → retrieves evidence → generates insight → done. If the user has a follow-up (e.g., "Can you break that down by sentiment?" or "What about pricing specifically?"), they must start a completely new query with no memory of the previous analysis.

This doesn't match how product teams actually use insight tools. In practice, analysis is iterative — you get an initial answer, then drill down, challenge assumptions, or pivot to adjacent questions. The current architecture forces users to re-retrieve and re-analyze from scratch every time.

## Goal

Turn the system from a **single-query tool** into a **multi-turn analysis session** where users can ask follow-ups that build on previous context — and the system intelligently decides whether to answer from existing context or trigger a new retrieval.

## Scope

### In Scope

- Multi-turn conversation UI in Streamlit
- Conversation history management (store and pass prior Q&A pairs)
- Retrieval re-trigger logic: decide per turn whether new evidence is needed
- Context window management to prevent token overflow
- Session reset capability

### Out of Scope

- Cross-session persistence (conversation dies when user closes the tab — that's fine)
- User authentication or multi-user session management
- Streaming responses (nice-to-have, not required for v1)

## Architecture

### Current Flow (Single-Turn)

```
User Question → Embed → Retrieve (Chroma) → LLM Analysis → Response
```

### Target Flow (Multi-Turn)

```
User Question
     ↓
┌─────────────────────────────────┐
│  Conversation History Manager   │
│  (maintains Q&A + retrieved     │
│   evidence from prior turns)    │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│  Retrieval Decision Layer       │
│  "Can I answer from existing    │
│   context, or do I need to      │
│   search again?"                │
└─────┬──────────────┬────────────┘
      ↓              ↓
  [Use existing   [New retrieval
   context]        from Chroma]
      ↓              ↓
      └──────┬───────┘
             ↓
┌─────────────────────────────────┐
│  LLM Analysis (with full       │
│  conversation history +         │
│  available evidence)            │
└────────────┬────────────────────┘
             ↓
         Response
```

## Component Details

### 1. Conversation History Manager

**What it does:** Maintains an ordered list of turns, where each turn contains the user question, retrieved chunks (if any), and the LLM response.

**Data structure:**

```python
# Each turn in the session
@dataclass
class ConversationTurn:
    role: str                          # "user" or "assistant"
    content: str                       # the message text
    retrieved_chunks: list[dict] | None  # chunks used (if retrieval happened)
    retrieval_triggered: bool          # whether this turn triggered new retrieval

# Session state
conversation_history: list[ConversationTurn] = []
```

**Key behaviors:**

- Append each Q&A pair after the LLM responds
- Track which chunks have already been retrieved (avoid duplicate retrieval)
- Expose a method to serialize history into a prompt-ready format

### 2. Retrieval Decision Layer

**What it does:** Before every follow-up, determine whether the existing context (prior retrieved chunks + conversation history) is sufficient to answer, or whether a new Chroma query is needed.

**Implementation approach — LLM-as-router:**

Send a lightweight classification prompt to the LLM:

```
Given the conversation so far and the available evidence,
can the following question be answered from existing context?

Question: "{user_follow_up}"

Respond with exactly one of:
- ANSWER_FROM_CONTEXT — if the existing evidence is sufficient
- NEW_RETRIEVAL_NEEDED — if new evidence must be retrieved

If NEW_RETRIEVAL_NEEDED, also provide a rewritten search query
optimized for semantic retrieval (not the raw user question).
```

**Why LLM-as-router instead of heuristics:**

- Keyword matching is brittle ("tell me more" could mean anything)
- The LLM already has the context — it's the best judge of sufficiency
- Cost is minimal: this is a short classification call, not a full analysis

**Fallback:** If the router call fails or times out, default to triggering a new retrieval. False retrieval is better than a hallucinated answer.

### 3. Context Window Management

**Problem:** After 4–5 turns, conversation history + retrieved chunks will approach or exceed the context window limit.

**Strategy (simple, sufficient for v1):**

- Set a **token budget** for conversation history (e.g., 60% of available context)
- When history exceeds the budget, **truncate oldest turns first** — keep the most recent 2–3 turns in full, summarize or drop earlier ones
- Always preserve the **current turn's retrieved chunks** at full fidelity
- Previously retrieved chunks from older turns: keep only a brief summary (e.g., first 50 tokens of each chunk) rather than full text

**Implementation:**

```python
def build_prompt_context(history: list[ConversationTurn], 
                         token_budget: int) -> str:
    # 1. Always include current turn's evidence in full
    # 2. Include recent turns (last 2-3) in full
    # 3. Summarize older turns to: "User asked about X → Key finding was Y"
    # 4. Drop oldest if still over budget
    ...
```

### 4. Streamlit UI Changes

**Current UI:** Sidebar mode selector + single input box + single output panel.

**Target UI:** Add a chat interface below (or replacing) the current output panel.

**Streamlit components to use:**

- `st.chat_message("user")` / `st.chat_message("assistant")` for conversation bubbles
- `st.chat_input()` for the follow-up input box
- `st.session_state` to persist conversation history within a session
- Keep the sidebar mode selector (General / Root Cause / Feature Requests) — mode applies to the entire session, not per-message

**UI behaviors:**

- First message in a session always triggers retrieval
- Follow-up messages go through the retrieval decision layer
- Each assistant message shows a small indicator: `📄 Used existing evidence` or `🔍 Retrieved new evidence`
- "New Session" button in sidebar clears history and starts fresh
- Evidence panel (source citations) updates to show **all chunks used across the session**, not just the current turn

### 5. Changes to `rag_with_claude.py`

The core `TinderRAGAnalyzer` class needs a new method:

```python
class TinderRAGAnalyzer:
    # Existing
    def ask(self, question, filters=None) -> dict:
        ...

    # New
    def follow_up(self, 
                  question: str, 
                  conversation_history: list[ConversationTurn],
                  filters: dict | None = None) -> dict:
        """
        Handle a follow-up question within an existing analysis session.
        
        1. Call retrieval decision layer
        2. If NEW_RETRIEVAL_NEEDED: run Chroma query with rewritten query
        3. Build prompt with conversation history + evidence
        4. Call LLM for analysis
        5. Return response + metadata (retrieval_triggered, chunks_used)
        """
        ...
```

The original `ask()` method stays unchanged — `follow_up()` is additive, not a refactor.

## Prompt Design

### System Prompt Addition (for follow-up turns)

```
You are a senior product analyst conducting an iterative analysis session.

You have access to the conversation history and previously retrieved user 
review evidence. When answering follow-up questions:

1. Build on prior analysis — don't repeat what you've already said
2. If the user asks to drill down, narrow your focus to the specific area
3. If the user challenges a conclusion, re-examine the evidence honestly
4. If you cannot answer from existing evidence, say so explicitly — 
   do NOT fabricate or speculate beyond what the data shows
5. When citing evidence, clarify whether it comes from previously 
   retrieved reviews or newly retrieved ones
```

## Edge Cases

| Scenario | Handling |
|---|---|
| User asks something completely unrelated to prior context | Router classifies as NEW_RETRIEVAL_NEEDED; fresh query runs normally |
| User says "tell me more" (vague follow-up) | Router interprets based on last turn's topic; retrieves more evidence on the same theme |
| User switches analysis mode mid-session | Clear conversation history; treat as new session (mode change = context change) |
| Retrieved chunks are identical to previous turn | Deduplicate by chunk ID before sending to LLM; note to user that no new evidence was found |
| LLM router fails or returns ambiguous signal | Default to NEW_RETRIEVAL_NEEDED (safe fallback) |

## File Changes Summary

| File | Change Type | What |
|---|---|---|
| `src/rag_with_claude.py` | Modify | Add `follow_up()` method, retrieval decision logic, context builder |
| `src/app.py` | Modify | Replace single-output UI with chat interface, add session state management |
| `src/conversation.py` | **New** | `ConversationTurn` dataclass, history manager, prompt serializer |
| `README.md` | Modify | Update feature description, add conversation demo screenshot |

## Success Criteria

1. User can ask a follow-up question and get an answer that builds on the prior turn (not a cold restart)
2. The system correctly distinguishes "answerable from context" vs. "needs new retrieval" at least ~80% of the time
3. A 5-turn conversation doesn't crash, hallucinate, or lose coherence
4. Evidence attribution remains clear — user can always see which source reviews support which claims
5. Session reset works cleanly with no state leakage

## Non-Goals (Explicitly)

- This is not a general chatbot. The system should stay focused on product analysis — if the user asks "what's the weather?", it should redirect, not answer.
- No need for "memory" across browser sessions. Session state is ephemeral and that's fine.
- No need for streaming in v1. Full response after generation is acceptable.
