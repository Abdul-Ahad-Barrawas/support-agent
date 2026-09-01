"""
Main agent graph — defines the multi-step pipeline using LangGraph.

Flow:
  incoming ticket
    → classify_intent
    → score_priority
    → [DECISION] critical? → escalate
    → retrieve_from_kb
    → draft_response
    → [DECISION] confident? → send | flag_for_review
"""

from langgraph.graph import StateGraph, END

from src.agent.state import TicketState
from src.tools.intent_classifier import classify_intent
from src.tools.priority_scorer import score_priority
from src.tools.kb_retriever import retrieve_from_kb
from src.tools.response_drafter import draft_response
from config.settings import CONFIDENCE_THRESHOLD


def route_after_priority(state: TicketState) -> str:
    """Decision gate 1: escalate critical tickets immediately."""
    if state["should_escalate"]:
        return "escalate"
    return "retrieve"


def route_after_draft(state: TicketState) -> str:
    """Decision gate 2: check confidence before sending."""
    if state["response_confidence"] >= CONFIDENCE_THRESHOLD:
        return "send"
    return "flag_for_review"


def escalate_node(state: TicketState) -> dict:
    """Terminal node — mark ticket for human handoff."""
    return {
        "final_action": "escalate",
        "reasoning_trace": state["reasoning_trace"]
        + ["ESCALATED: critical priority detected"],
    }


def send_node(state: TicketState) -> dict:
    """Terminal node — response is good to send."""
    return {
        "final_action": "send",
        "reasoning_trace": state["reasoning_trace"]
        + [f"SENT: confidence {state['response_confidence']:.2f} above threshold"],
    }


def flag_node(state: TicketState) -> dict:
    """Terminal node — low confidence, needs human review."""
    return {
        "final_action": "flag_for_review",
        "reasoning_trace": state["reasoning_trace"]
        + [f"FLAGGED: confidence {state['response_confidence']:.2f} below threshold"],
    }


def build_agent_graph() -> StateGraph:
    """Construct and compile the agent graph."""

    graph = StateGraph(TicketState)

    # Add tool nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("score_priority", score_priority)
    graph.add_node("retrieve_from_kb", retrieve_from_kb)
    graph.add_node("draft_response", draft_response)

    # Add terminal nodes
    graph.add_node("escalate", escalate_node)
    graph.add_node("send", send_node)
    graph.add_node("flag_for_review", flag_node)

    # Wire the edges
    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "score_priority")

    # Decision gate 1: priority check
    graph.add_conditional_edges(
        "score_priority",
        route_after_priority,
        {"escalate": "escalate", "retrieve": "retrieve_from_kb"},
    )

    graph.add_edge("retrieve_from_kb", "draft_response")

    # Decision gate 2: confidence check
    graph.add_conditional_edges(
        "draft_response",
        route_after_draft,
        {"send": "send", "flag_for_review": "flag_for_review"},
    )

    # Terminal edges
    graph.add_edge("escalate", END)
    graph.add_edge("send", END)
    graph.add_edge("flag_for_review", END)

    return graph.compile()
