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

from functools import partial

from langgraph.graph import StateGraph, END

from src.agent.state import TicketState, new_ticket_state
from src.tools.intent_classifier import classify_intent
from src.tools.priority_scorer import score_priority
from src.tools.kb_retriever import retrieve_from_kb
from src.tools.response_drafter import draft_response
from config.settings import CONFIDENCE_THRESHOLD, MIN_RETRIEVAL_SCORE


def route_after_priority(state: TicketState) -> str:
    """Decision gate 1: escalate critical tickets immediately."""
    if state["should_escalate"]:
        return "escalate"
    return "retrieve"


def best_retrieval_score(state: TicketState) -> float:
    return max(state.get("retrieval_scores") or [0.0])


def route_after_draft(state: TicketState, threshold: float = CONFIDENCE_THRESHOLD,
                      min_retrieval_score: float | None = None) -> str:
    """
    Decision gate 2: send only a confident draft. When min_retrieval_score is
    set, the KB must also have held a relevant policy — the model's self-rated
    confidence alone is not trusted to auto-send.
    """
    if state["response_confidence"] < threshold:
        return "flag_for_review"
    if min_retrieval_score is not None and best_retrieval_score(state) < min_retrieval_score:
        return "flag_for_review"
    return "send"


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
        + [f"SENT: confidence {state['response_confidence']:.2f} at or above threshold"],
    }


def flag_node(state: TicketState, threshold: float = CONFIDENCE_THRESHOLD,
              min_retrieval_score: float | None = None) -> dict:
    """Terminal node — needs human review; the trace says which check failed."""
    confidence = state["response_confidence"]
    if confidence < threshold:
        reason = f"confidence {confidence:.2f} below threshold {threshold:.2f}"
    else:
        reason = (f"confidence {confidence:.2f} OK, but best KB match "
                  f"{best_retrieval_score(state):.2f} below {min_retrieval_score:.2f}")
    return {
        "final_action": "flag_for_review",
        "reasoning_trace": state["reasoning_trace"] + [f"FLAGGED: {reason}"],
    }


def build_agent_graph(
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    use_kb: bool = True,
    use_priority_gate: bool = True,
    min_retrieval_score: float = MIN_RETRIEVAL_SCORE,
):
    """
    Construct and compile the agent graph.

    The flags exist for the ablation study — the defaults are the full agent:
      use_kb=False            → skip retrieve_from_kb, draft with no context
                                (the retrieval floor is off: there are no scores)
      use_priority_gate=False → never escalate, even critical tickets
    """
    floor = min_retrieval_score if use_kb else None

    graph = StateGraph(TicketState)

    # Add tool nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("score_priority", score_priority)
    if use_kb:
        graph.add_node("retrieve_from_kb", retrieve_from_kb)
    graph.add_node("draft_response", draft_response)

    # Add terminal nodes
    if use_priority_gate:
        graph.add_node("escalate", escalate_node)
    graph.add_node("send", send_node)
    graph.add_node("flag_for_review", partial(flag_node, threshold=confidence_threshold,
                                              min_retrieval_score=floor))

    # Wire the edges
    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "score_priority")

    after_priority = "retrieve_from_kb" if use_kb else "draft_response"

    # Decision gate 1: priority check
    if use_priority_gate:
        graph.add_conditional_edges(
            "score_priority",
            route_after_priority,
            {"escalate": "escalate", "retrieve": after_priority},
        )
        graph.add_edge("escalate", END)
    else:
        graph.add_edge("score_priority", after_priority)

    if use_kb:
        graph.add_edge("retrieve_from_kb", "draft_response")

    # Decision gate 2: confidence check
    graph.add_conditional_edges(
        "draft_response",
        partial(route_after_draft, threshold=confidence_threshold, min_retrieval_score=floor),
        {"send": "send", "flag_for_review": "flag_for_review"},
    )

    # Terminal edges
    graph.add_edge("send", END)
    graph.add_edge("flag_for_review", END)

    return graph.compile()


def run_agent(ticket_text: str, ticket_id: str = "", agent=None) -> TicketState:
    """Run one ticket through the agent (the full graph unless one is given)."""
    agent = agent or build_agent_graph()
    return agent.invoke(new_ticket_state(ticket_text, ticket_id))
