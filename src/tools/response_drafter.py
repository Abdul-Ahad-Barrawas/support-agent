"""
Tool 4: Response Drafter
Generates a grounded customer reply using retrieved context.

Owner: Strong technical member (generation logic)
        + Non-technical member (prompt engineering & ablation)
"""

from pydantic import BaseModel, Field

from config.settings import COMPANY_NAME
from src.agent.state import TicketState
from src.llm import get_llm

llm = get_llm(temperature=0.3)


class DraftResult(BaseModel):
    response: str
    confidence: float = Field(ge=0.0, le=1.0)


# json_schema constrains decoding; with tool calling the model sometimes answered in
# plain prose instead (Groq 400 tool_use_failed on ~15% of golden tickets)
structured_llm = llm.with_structured_output(DraftResult, method="json_schema")

FALLBACK_RESPONSE = "Thank you for contacting us. We're looking into your issue."


def build_prompt(state: TicketState) -> str:
    retrieved_docs = state.get("retrieved_docs", [])
    if retrieved_docs:
        context = "\n\n".join(f"[{i}] {doc}" for i, doc in enumerate(retrieved_docs, 1))
    else:
        context = "(no relevant policy documents found)"

    return (
        f"You are a customer support agent for {COMPANY_NAME}.\n\n"
        f"Company policy documents:\n{context}\n\n"
        f"Ticket intent: {state.get('intent') or 'unknown'}\n"
        f"Ticket priority: {state.get('priority') or 'unknown'}\n"
        f'Customer message: "{state["ticket_text"]}"\n\n'
        "Write a reply to the customer. Be helpful and empathetic, and base any "
        "policy details (timelines, eligibility, steps) ONLY on the documents above. "
        "Never invent policy. If the documents don't cover the issue, say a "
        "specialist will follow up.\n\n"
        "Then rate your confidence from 0.0 to 1.0 that this reply fully resolves "
        "the customer's issue using only the given documents."
    )


def draft_response(state: TicketState) -> dict:
    """Draft a response grounded in retrieved KB documents."""
    try:
        result = structured_llm.invoke(build_prompt(state))
        drafted_response = result.response
        response_confidence = result.confidence
        trace = f"Drafted response (confidence: {response_confidence:.2f})"
    except Exception as e:
        # Confidence 0 guarantees the fallback text is flagged, never sent
        drafted_response = FALLBACK_RESPONSE
        response_confidence = 0.0
        trace = f"Drafting failed ({type(e).__name__}) — using fallback (confidence: 0.00)"

    return {
        "drafted_response": drafted_response,
        "response_confidence": response_confidence,
        "reasoning_trace": state.get("reasoning_trace", []) + [trace],
    }
