# rag_with_claude.py — TinderRAGAnalyzer with single-turn ask() and multi-turn follow_up()

import os
from sentence_transformers import SentenceTransformer
import chromadb
import anthropic

from conversation import ConversationHistoryManager, ConversationTurn

# ------------------------------------------------------------------ #
# Analysis mode prompts                                               #
#                                                                      #
# Each mode defines a different analytical FRAMEWORK — not just a     #
# different output checklist. The LLM should reason differently       #
# depending on the mode.                                              #
# ------------------------------------------------------------------ #

SYSTEM_PROMPTS = {
    "General Analysis": """\
You are a senior product analyst. Your job is to turn raw user reviews into a \
concise briefing that a product manager can act on in their next sprint planning.

Analytical framework:
- WHAT is happening: identify the 3-5 dominant themes, ranked by how many \
reviews support each theme. Cite specific review numbers as evidence.
- HOW SEVERE is it: for each theme, assess severity on a scale \
(cosmetic annoyance → functional blocker → trust/safety risk). \
Use the language and emotion in the reviews to calibrate, not just frequency.
- WHAT CHANGED: flag anything that appears to be getting worse or better over \
time. Reference specific months/dates from review metadata when available.
- SO WHAT: for each theme, give one concrete, scoped recommendation \
(not vague advice like "improve UX"). Specify what to fix, why it matters, \
and relative priority.

Constraints:
- Every claim must reference at least one specific review.
- If the evidence is thin or conflicting, say so. Do not over-generalize.
- Keep the total response under 600 words unless the evidence demands more.""",

    "Root Cause Analysis": """\
You are a senior product analyst specializing in diagnostic root cause analysis. \
Your job is NOT to list complaints — it is to explain WHY those complaints exist \
by tracing causal chains from symptoms back to underlying system failures.

Analytical framework:
- SYMPTOMS → MECHANISMS → ROOT CAUSES: for each major complaint cluster, \
work backwards. Example: "users complain about no matches (symptom) → \
the algorithm deprioritizes inactive profiles (mechanism) → \
re-engagement after a break is penalized by design (root cause)."
- Distinguish between INDEPENDENT root causes and SHARED root causes. \
If billing issues and ban complaints both trace back to the same account \
management system, say that — it changes the fix strategy.
- TIMELINE: when did each issue first appear or spike? Use review dates to \
identify whether this is chronic (steady complaints over months) or acute \
(sudden spike suggesting a specific release or policy change).
- EVIDENCE STRENGTH: rate each causal chain as strong (multiple reviews \
corroborate the full chain), moderate (symptoms are clear but mechanism is \
inferred), or weak (single review or speculation).

Constraints:
- Do not just list problems. Every finding must include at least one causal \
link (A causes B).
- If you cannot identify a plausible root cause from the evidence, say \
"insufficient evidence to determine root cause" — do not guess.
- Prioritize depth on 2-3 issues over shallow coverage of many.""",

    "Feature Requests": """\
You are a senior product analyst evaluating feature requests and unmet needs \
from user feedback. Your job is to separate signal from noise — identify which \
requests represent real, recurring needs vs. one-off wishes.

Analytical framework:
- DEMAND CLUSTERING: group related requests into themes (users may describe \
the same need in different words). Report the cluster, not individual requests.
- GAP ANALYSIS: for each cluster, describe what users currently have to do \
(the workaround or pain) vs. what they want. This "current state → desired \
state" framing reveals the actual problem better than the request itself.
- FREQUENCY vs. INTENSITY: a feature requested by 20 users casually is \
different from one requested by 5 users desperately. Distinguish between \
"nice to have" and "blocking users from getting value."
- FEASIBILITY SIGNAL: if the reviews contain hints about implementation \
complexity (e.g., users comparing to competitor features, or describing \
technical limitations they've hit), note these as context. Do NOT make your \
own feasibility estimates — you are analyzing demand, not engineering effort.

Constraints:
- Rank clusters by a combination of frequency and intensity, not frequency alone.
- Quote the most vivid or specific user language to illustrate each cluster — \
vague summaries lose the urgency that makes PMs act.
- If a "feature request" is actually a bug report in disguise (e.g., "add the \
ability to cancel" when cancellation exists but is broken), flag it as such.""",
}

FOLLOW_UP_ADDENDUM = """\

You are in a multi-turn analysis session. Additional rules for follow-up responses:
- Build on prior analysis — do not repeat what you already said.
- If the user drills down, narrow your focus to that specific area.
- If the user challenges a conclusion, re-examine the evidence honestly.
- If the available evidence cannot answer the question, say so explicitly — \
do NOT fabricate or speculate beyond what the data shows.
- When citing evidence, distinguish between previously retrieved reviews and \
newly retrieved ones."""


class TinderRAGAnalyzer:
    """RAG-powered Product Analyst assistant — last 12 months of Tinder reviews."""

    def __init__(self, api_key: str | None = None, db_path: str = "./tinder_rag_db_recent"):
        print("Loading embedding model...")
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        print("Connecting to vector database (recent data)...")
        chroma_client = chromadb.PersistentClient(path=str(db_path))
        self.collection = chroma_client.get_collection("tinder_reviews_recent")

        print("Initializing Claude API...")
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

        print(f"\n✅ System ready! Knowledge base: {self.collection.count():,} reviews\n")

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def retrieve_reviews(self, query: str, n_results: int = 10, filters: dict | None = None):
        """Embed query and retrieve the top-N nearest reviews from Chroma."""
        query_embedding = self.embed_model.encode([query])
        return self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            where=filters,
        )

    def analyze_with_claude(
        self, query: str, context_reviews: dict, analysis_type: str = "General Analysis"
    ) -> str:
        """Run the LLM over a set of retrieved reviews (single-turn path)."""
        context_text = "\n\n".join(
            f"Review {i + 1}:\n{doc}"
            for i, doc in enumerate(context_reviews["documents"][0])
        )

        system_prompt = SYSTEM_PROMPTS.get(analysis_type, SYSTEM_PROMPTS["General Analysis"])

        user_message = f"""User Reviews (last 12 months):
{context_text}

Question: {query}"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text

    # ------------------------------------------------------------------ #
    # Retrieval Decision Layer (LLM-as-router)                            #
    # ------------------------------------------------------------------ #

    def _decide_retrieval(
        self,
        question: str,
        history_manager: ConversationHistoryManager,
    ) -> tuple[bool, str]:
        """
        Ask a lightweight LLM call whether we can answer from existing context
        or need a fresh Chroma query.

        Returns:
            (needs_retrieval: bool, search_query: str)
            - needs_retrieval=False  → answer from current session context
            - needs_retrieval=True   → run a new Chroma query with search_query
        """
        history_text = history_manager.serialize_for_prompt()

        router_prompt = f"""Given the conversation so far and the available evidence, \
can the following question be answered from existing context?

Conversation history:
{history_text}

Question: "{question}"

Respond with EXACTLY one of:
- ANSWER_FROM_CONTEXT — if the existing evidence is sufficient
- NEW_RETRIEVAL_NEEDED — if new evidence must be retrieved

If NEW_RETRIEVAL_NEEDED, add a second line formatted as:
QUERY: <rewritten search query optimised for semantic retrieval>"""

        try:
            response = self.client.messages.create(
                model="claude-haiku-4-20250514",
                max_tokens=150,
                messages=[{"role": "user", "content": router_prompt}],
            )
            text = response.content[0].text.strip()

            if "ANSWER_FROM_CONTEXT" in text:
                return False, question

            if "NEW_RETRIEVAL_NEEDED" in text:
                rewritten = question  # default fallback
                for line in text.splitlines():
                    if line.startswith("QUERY:"):
                        rewritten = line[len("QUERY:"):].strip()
                        break
                return True, rewritten

        except Exception:
            pass  # network / API error — fall through to safe default

        # Safe fallback: trigger retrieval rather than risk a hallucinated answer
        return True, question

    # ------------------------------------------------------------------ #
    # Public API — single-turn (unchanged)                                #
    # ------------------------------------------------------------------ #

    def ask(self, query: str, n_reviews: int = 10, filters: dict | None = None) -> str:
        """Ask a question and get AI-powered analysis (single-turn, original method)."""
        print(f"🔍 Question: {query}")
        print(f"   Retrieving {n_reviews} relevant reviews...")

        reviews = self.retrieve_reviews(query, n_results=n_reviews, filters=filters)

        if not reviews["documents"][0]:
            return "No relevant reviews found."

        print(f"   ✅ Retrieved {len(reviews['documents'][0])} reviews")
        print(f"   🤖 Analyzing with Claude...\n")

        return self.analyze_with_claude(query, reviews)

    # ------------------------------------------------------------------ #
    # Public API — multi-turn (new)                                       #
    # ------------------------------------------------------------------ #

    def follow_up(
        self,
        question: str,
        conversation_history: ConversationHistoryManager,
        filters: dict | None = None,
        n_reviews: int = 10,
        analysis_type: str = "General Analysis",
    ) -> dict:
        """
        Handle a follow-up question within an ongoing analysis session.

        Steps:
        1. Call retrieval decision layer (LLM-as-router).
        2. If NEW_RETRIEVAL_NEEDED: query Chroma with the (possibly rewritten) query,
           deduplicating against chunks already seen this session.
        3. Build a prompt that includes conversation history + all available evidence.
        4. Call LLM for analysis.
        5. Return response + metadata.

        Returns:
            {
                "answer": str,
                "retrieval_triggered": bool,
                "chunks_used": list[dict],   # new chunks added this turn
                "search_query_used": str | None,
            }
        """
        # Step 1 — routing decision
        needs_retrieval, search_query = self._decide_retrieval(question, conversation_history)

        new_chunks_for_history: list[dict] = []
        new_evidence_text = ""

        # Step 2 — optional new retrieval
        if needs_retrieval:
            raw = self.retrieve_reviews(search_query, n_results=n_reviews, filters=filters)
            seen_ids = conversation_history.get_seen_chunk_ids()

            # Deduplicate: only keep chunks not yet seen in this session
            fresh_docs, fresh_metas, fresh_ids = [], [], []
            if raw["documents"][0]:
                for doc, meta, cid in zip(
                    raw["documents"][0],
                    raw["metadatas"][0],
                    raw["ids"][0],
                ):
                    if cid not in seen_ids:
                        fresh_docs.append(doc)
                        fresh_metas.append(meta)
                        fresh_ids.append(cid)

            if fresh_docs:
                # Build chunk dicts for history tracking
                new_chunks_for_history = [
                    {"id": cid, "document": doc, "metadata": meta}
                    for cid, doc, meta in zip(fresh_ids, fresh_docs, fresh_metas)
                ]
                new_evidence_text = "\n\n".join(
                    f"New Review {i + 1}:\n{doc}"
                    for i, doc in enumerate(fresh_docs)
                )
            else:
                # All retrieved chunks are duplicates — treat as context-only answer
                needs_retrieval = False

        # Step 3 — build prompt context
        history_text = conversation_history.serialize_for_prompt()

        # Prior evidence: brief summaries of older retrieved chunks
        prior_chunks = conversation_history.get_all_chunks()
        prior_evidence_text = ""
        if prior_chunks:
            prior_summaries = [
                f"Prior Review {i + 1}: {chunk.get('document', '')[:200]}…"
                for i, chunk in enumerate(prior_chunks[:6])  # cap at 6 to save tokens
            ]
            prior_evidence_text = "\n".join(prior_summaries)

        # Assemble evidence block
        evidence_blocks: list[str] = []
        if prior_evidence_text:
            evidence_blocks.append(
                f"=== Previously retrieved evidence (summaries) ===\n{prior_evidence_text}"
            )
        if new_evidence_text:
            evidence_blocks.append(
                f"=== Newly retrieved evidence (full) ===\n{new_evidence_text}"
            )
        if not evidence_blocks:
            evidence_blocks.append("No additional evidence retrieved for this turn.")

        evidence_section = "\n\n".join(evidence_blocks)

        base_prompt = SYSTEM_PROMPTS.get(analysis_type, SYSTEM_PROMPTS["General Analysis"])
        system_prompt = base_prompt + FOLLOW_UP_ADDENDUM

        user_message = f"""Conversation so far:
{history_text}

Available evidence:
{evidence_section}

Follow-up question: {question}

Please answer the follow-up question, building on the conversation above."""

        # Step 4 — LLM analysis call
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        answer = response.content[0].text

        # Step 5 — return result + metadata
        return {
            "answer": answer,
            "retrieval_triggered": needs_retrieval,
            "chunks_used": new_chunks_for_history,
            "search_query_used": search_query if needs_retrieval else None,
        }
