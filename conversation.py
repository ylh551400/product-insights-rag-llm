# conversation.py — ConversationTurn dataclass + history manager + prompt serializer

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConversationTurn:
    role: str                              # "user" or "assistant"
    content: str                           # message text
    retrieved_chunks: Optional[list] = None  # chunk dicts used (if retrieval happened)
    retrieval_triggered: bool = False      # whether this turn triggered new retrieval


class ConversationHistoryManager:
    """Manages ordered conversation turns and provides serialization utilities."""

    def __init__(self):
        self.turns: list[ConversationTurn] = []
        self._seen_chunk_ids: set[str] = set()

    # ---------- mutation ----------

    def add_turn(self, turn: ConversationTurn) -> None:
        """Append a turn and register any chunk IDs it contains."""
        self.turns.append(turn)
        if turn.retrieved_chunks:
            for chunk in turn.retrieved_chunks:
                cid = chunk.get("id", "")
                if cid:
                    self._seen_chunk_ids.add(cid)

    def clear(self) -> None:
        """Reset the session (called on 'New Session')."""
        self.turns = []
        self._seen_chunk_ids = set()

    # ---------- queries ----------

    def get_seen_chunk_ids(self) -> set[str]:
        """Return the set of chunk IDs already retrieved in this session."""
        return self._seen_chunk_ids.copy()

    def get_all_chunks(self) -> list[dict]:
        """Return all unique chunks retrieved across the entire session."""
        seen: set[str] = set()
        result: list[dict] = []
        for turn in self.turns:
            if not turn.retrieved_chunks:
                continue
            for chunk in turn.retrieved_chunks:
                cid = chunk.get("id", chunk.get("document", "")[:40])
                if cid not in seen:
                    seen.add(cid)
                    result.append(chunk)
        return result

    # ---------- serialization ----------

    def serialize_for_prompt(self, max_recent_pairs: int = 3) -> str:
        """
        Convert history to a prompt-ready string.

        Strategy (per spec):
        - Keep the most recent `max_recent_pairs` Q&A pairs in full.
        - Summarize older turns as: "User asked about X → Key finding was Y".
        - Drop nothing — just compress older turns.
        """
        if not self.turns:
            return ""

        # Identify the boundary between 'recent' and 'older' turns.
        # Each Q&A pair = 2 turns, so keep the last (max_recent_pairs * 2) turns in full.
        keep_full = max_recent_pairs * 2
        recent = self.turns[-keep_full:] if len(self.turns) > keep_full else self.turns
        older = self.turns[: max(0, len(self.turns) - keep_full)]

        sections: list[str] = []

        # Summarize older turns
        if older:
            summaries: list[str] = []
            i = 0
            while i < len(older):
                user_text = ""
                assistant_text = ""
                if i < len(older) and older[i].role == "user":
                    user_text = older[i].content[:120].replace("\n", " ")
                    i += 1
                if i < len(older) and older[i].role == "assistant":
                    assistant_text = older[i].content[:180].replace("\n", " ")
                    i += 1
                if user_text or assistant_text:
                    summaries.append(
                        f"User asked about: {user_text}  →  Key finding: {assistant_text}"
                    )
            if summaries:
                sections.append("=== Earlier in this session (summarized) ===")
                sections.extend(summaries)

        # Full recent turns
        if recent:
            sections.append("=== Recent conversation ===")
            for turn in recent:
                prefix = "User" if turn.role == "user" else "Assistant"
                sections.append(f"{prefix}: {turn.content}")

        return "\n".join(sections)

    def __len__(self) -> int:
        return len(self.turns)
