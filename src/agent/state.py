"""Agent state definition — this is what flows through the LangGraph pipeline."""

from typing import TypedDict, Literal, Optional


class TicketState(TypedDict):
    """State that gets passed between every node in the agent graph."""

    # --- Input ---
    ticket_text: str                     # Raw customer message
    ticket_id: str                       # Unique ID for tracking

    # --- Tool 1: Intent classifier output ---
    intent: str                          # e.g. "billing", "technical_issue"
    intent_confidence: float             # 0.0 to 1.0

    # --- Tool 2: Priority scorer output ---
    priority: str                        # "low", "medium", "high", "critical"
    sentiment_score: float               # -1.0 (angry) to 1.0 (happy)

    # --- Routing decision ---
    should_escalate: bool                # True → skip to human handoff

    # --- Tool 3: KB retrieval output ---
    retrieved_docs: list[str]            # Relevant FAQ/policy chunks
    retrieval_scores: list[float]        # Similarity scores

    # --- Tool 4: Response drafter output ---
    drafted_response: str                # Generated reply
    response_confidence: float           # How confident the agent is

    # --- Final output ---
    final_action: Literal["send", "escalate", "flag_for_review"]
    reasoning_trace: list[str]           # Step-by-step log for transparency
