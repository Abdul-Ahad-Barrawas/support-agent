"""
Tool 1: Intent Classifier
Categorizes incoming ticket into one of the predefined intent labels.

Owner: Strong technical member
"""

from src.agent.state import TicketState
from config.settings import INTENT_LABELS


def classify_intent(state: TicketState) -> dict:
    """
    Classify the ticket's intent using the LLM.

    TODO: Implement this tool
    - Call the LLM with a structured prompt
    - Parse the response into (intent, confidence)
    - Prompt should list INTENT_LABELS and ask for JSON output

    Example prompt structure:
        "Classify the following support ticket into one of these
         categories: {INTENT_LABELS}. Return JSON with keys
         'intent' and 'confidence'."
    """
    ticket_text = state["ticket_text"]

    # ---- PLACEHOLDER — replace with LLM call ----
    intent = "general_inquiry"
    confidence = 0.0
    # ----------------------------------------------

    return {
        "intent": intent,
        "intent_confidence": confidence,
        "reasoning_trace": state.get("reasoning_trace", [])
        + [f"Classified intent: {intent} ({confidence:.2f})"],
    }
