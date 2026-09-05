"""
Tool 2: Priority Scorer
Assesses ticket urgency using sentiment analysis and keyword signals.

Owner: Strong technical member
"""

import re

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from config.settings import LLM_MODEL
from src.agent.state import TicketState

llm = ChatGroq(model=LLM_MODEL, temperature=0)


class SentimentResult(BaseModel):
    sentiment_score: float = Field(ge=-1.0, le=1.0)


structured_llm = llm.with_structured_output(SentimentResult, method="json_mode")


# Keywords that signal high/critical urgency
ESCALATION_KEYWORDS = [
    "urgent", "immediately", "lawsuit", "legal", "cancel",
    "broken", "down", "outage", "security", "breach", "hacked",
    "unacceptable", "terrible", "worst", "scam", "fraud",
]


def has_escalation_keyword(ticket_text: str) -> bool:
    words = set(re.findall(r"\w+", ticket_text.lower()))
    return any(keyword in words for keyword in ESCALATION_KEYWORDS)


def score_priority(state: TicketState) -> dict:
    """Score ticket priority based on sentiment and urgency keywords."""
    ticket_text = state["ticket_text"]

    prompt = (
        "Rate the sentiment of this support ticket from -1.0 (very angry) "
        "to 1.0 (very happy), with 0.0 being neutral.\n"
        'Respond with ONLY a JSON object of the form {"sentiment_score": <float>}.\n\n'
        f'Ticket: "{ticket_text}"'
    )

    try:
        result = structured_llm.invoke(prompt)
        sentiment_score = result.sentiment_score
    except Exception:
        sentiment_score = 0.0

    keyword_hit = has_escalation_keyword(ticket_text)

    if sentiment_score < -0.7 and keyword_hit:
        priority = "critical"
    elif sentiment_score < -0.4 or keyword_hit:
        priority = "high"
    elif sentiment_score < 0.0:
        priority = "medium"
    else:
        priority = "low"

    should_escalate = priority == "critical"

    return {
        "priority": priority,
        "sentiment_score": sentiment_score,
        "should_escalate": should_escalate,
        "reasoning_trace": state.get("reasoning_trace", [])
        + [f"Priority: {priority} (sentiment: {sentiment_score:.2f})"],
    }
