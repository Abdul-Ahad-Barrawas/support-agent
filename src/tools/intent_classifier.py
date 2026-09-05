"""
Tool 1: Intent Classifier
Categorizes incoming ticket into one of the predefined intent labels.

Owner: Strong technical member
"""

from typing import Literal

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from config.settings import INTENT_LABELS, LLM_MODEL
from src.agent.state import TicketState

llm = ChatGroq(model=LLM_MODEL, temperature=0)


class IntentResult(BaseModel):
    intent: Literal[tuple(INTENT_LABELS)]
    confidence: float = Field(ge=0.0, le=1.0)


structured_llm = llm.with_structured_output(IntentResult)


def classify_intent(state: TicketState) -> dict:
    """Classify the ticket's intent using the LLM."""
    ticket_text = state["ticket_text"]

    prompt = (
        f"Classify the following support ticket into one of these categories: "
        f"{', '.join(INTENT_LABELS)}.\n\n"
        f'Ticket: "{ticket_text}"\n\n'
        "Return the category and your confidence (0.0-1.0) that it's correct."
    )

    try:
        result = structured_llm.invoke(prompt)
        intent = result.intent
        confidence = result.confidence
    except Exception:
        intent = "general_inquiry"
        confidence = 0.0

    return {
        "intent": intent,
        "intent_confidence": confidence,
        "reasoning_trace": state.get("reasoning_trace", [])
        + [f"Classified intent: {intent} ({confidence:.2f})"],
    }
