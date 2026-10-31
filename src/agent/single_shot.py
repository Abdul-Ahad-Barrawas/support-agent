"""
Single-shot baseline for the ablation study: one LLM call, no tools, no graph.

Exposes the same .invoke(state) -> state interface as the compiled graph,
so the evaluator can run it like any other agent variant.
"""

from typing import Literal

from pydantic import BaseModel, Field

from config.settings import (
    COMPANY_NAME,
    CONFIDENCE_THRESHOLD,
    INTENT_LABELS,
    PRIORITY_LEVELS,
)
from src.agent.state import TicketState
from src.llm import get_llm

llm = get_llm(temperature=0)


class SingleShotResult(BaseModel):
    intent: Literal[tuple(INTENT_LABELS)]
    priority: Literal[tuple(PRIORITY_LEVELS)]
    response: str
    confidence: float = Field(ge=0.0, le=1.0)


# json_schema, not tool calling — see the note in src/tools/response_drafter.py
structured_llm = llm.with_structured_output(SingleShotResult, method="json_schema")


class SingleShotAgent:
    def __init__(self, confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold

    def invoke(self, state: TicketState) -> TicketState:
        prompt = (
            f"You are a customer support agent for {COMPANY_NAME}.\n"
            f'Customer message: "{state["ticket_text"]}"\n\n'
            f"1. Classify the intent as one of: {', '.join(INTENT_LABELS)}.\n"
            f"2. Rate the priority as one of: {', '.join(PRIORITY_LEVELS)}.\n"
            "3. Write a helpful, empathetic reply.\n"
            "4. Rate your confidence 0.0-1.0 that the reply fully resolves the issue."
        )

        try:
            result = structured_llm.invoke(prompt)
        except Exception as e:
            return {**state, "final_action": "flag_for_review",
                    "reasoning_trace": [f"Single-shot call failed ({type(e).__name__})"]}

        if result.priority == "critical":
            final_action = "escalate"
        elif result.confidence >= self.confidence_threshold:
            final_action = "send"
        else:
            final_action = "flag_for_review"

        return {
            **state,
            "intent": result.intent,
            "priority": result.priority,
            "should_escalate": final_action == "escalate",
            "drafted_response": "" if final_action == "escalate" else result.response,
            "response_confidence": result.confidence,
            "final_action": final_action,
            "reasoning_trace": [
                f"Single-shot: {result.intent} / {result.priority} "
                f"(confidence {result.confidence:.2f}) → {final_action}"
            ],
        }
