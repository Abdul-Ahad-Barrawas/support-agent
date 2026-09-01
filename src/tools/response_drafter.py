"""
Tool 4: Response Drafter
Generates a grounded customer reply using retrieved context.

Owner: Strong technical member (generation logic)
        + Non-technical member (prompt engineering & ablation)
"""

from src.agent.state import TicketState


def draft_response(state: TicketState) -> dict:
    """
    Draft a response grounded in retrieved KB documents.

    TODO: Implement this tool
    - Build a prompt with: ticket_text, intent, priority, retrieved_docs
    - Ask LLM to generate a helpful response
    - Ask LLM to self-rate its confidence (0.0 to 1.0)
    - Parse response and confidence

    Prompt template should include:
        - Role: "You are a customer support agent for [Company]"
        - Context: The retrieved FAQ/policy docs
        - Ticket: The original customer message
        - Instructions: Be helpful, empathetic, reference policy
        - Self-rating: "Rate your confidence 0-1 that this fully
          resolves the customer's issue"
    """
    ticket_text = state["ticket_text"]
    retrieved_docs = state["retrieved_docs"]

    # ---- PLACEHOLDER — replace with LLM call ----
    drafted_response = "Thank you for contacting us. We're looking into your issue."
    response_confidence = 0.0
    # ----------------------------------------------

    return {
        "drafted_response": drafted_response,
        "response_confidence": response_confidence,
        "reasoning_trace": state.get("reasoning_trace", [])
        + [f"Drafted response (confidence: {response_confidence:.2f})"],
    }
