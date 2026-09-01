"""
Tool 2: Priority Scorer
Assesses ticket urgency using sentiment analysis and keyword signals.

Owner: Strong technical member
"""

from src.agent.state import TicketState


# Keywords that signal high/critical urgency
ESCALATION_KEYWORDS = [
    "urgent", "immediately", "lawsuit", "legal", "cancel",
    "broken", "down", "outage", "security", "breach", "hacked",
    "unacceptable", "terrible", "worst", "scam", "fraud",
]


def score_priority(state: TicketState) -> dict:
    """
    Score ticket priority based on sentiment and urgency keywords.

    TODO: Implement this tool
    - Use LLM to get sentiment score (-1.0 to 1.0)
    - Check for escalation keywords
    - Combine into priority level
    - Set should_escalate=True for critical priority

    Logic:
        critical → sentiment < -0.7 AND escalation keywords present
        high     → sentiment < -0.4 OR escalation keywords present
        medium   → sentiment < 0.0
        low      → sentiment >= 0.0
    """
    ticket_text = state["ticket_text"]

    # ---- PLACEHOLDER — replace with LLM call ----
    sentiment_score = 0.0
    priority = "medium"
    should_escalate = False
    # ----------------------------------------------

    return {
        "priority": priority,
        "sentiment_score": sentiment_score,
        "should_escalate": should_escalate,
        "reasoning_trace": state.get("reasoning_trace", [])
        + [f"Priority: {priority} (sentiment: {sentiment_score:.2f})"],
    }
