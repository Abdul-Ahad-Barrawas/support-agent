import pytest

from src.agent import graph as graph_module
from src.agent.graph import build_agent_graph, route_after_draft, run_agent


@pytest.fixture
def stub_tools(monkeypatch):
    """Replace the 4 LLM/vector tools with stubs; `calls` records which ran."""
    calls = []
    config = {"priority": "low", "confidence": 0.9}

    def classify(state):
        calls.append("classify_intent")
        return {"intent": "billing", "intent_confidence": 0.9,
                "reasoning_trace": state["reasoning_trace"] + ["intent"]}

    def score(state):
        calls.append("score_priority")
        return {"priority": config["priority"], "sentiment_score": -0.9,
                "should_escalate": config["priority"] == "critical",
                "reasoning_trace": state["reasoning_trace"] + ["priority"]}

    def retrieve(state):
        calls.append("retrieve_from_kb")
        return {"retrieved_docs": ["Refunds take 5 days."], "retrieval_scores": [0.8],
                "reasoning_trace": state["reasoning_trace"] + ["retrieve"]}

    def draft(state):
        calls.append("draft_response")
        return {"drafted_response": f"Reply using {len(state['retrieved_docs'])} docs",
                "response_confidence": config["confidence"],
                "reasoning_trace": state["reasoning_trace"] + ["draft"]}

    monkeypatch.setattr(graph_module, "classify_intent", classify)
    monkeypatch.setattr(graph_module, "score_priority", score)
    monkeypatch.setattr(graph_module, "retrieve_from_kb", retrieve)
    monkeypatch.setattr(graph_module, "draft_response", draft)
    return calls, config


def test_route_after_draft_threshold_is_inclusive():
    assert route_after_draft({"response_confidence": 0.7}, threshold=0.7) == "send"
    assert route_after_draft({"response_confidence": 0.69}, threshold=0.7) == "flag_for_review"


def test_confident_ticket_runs_full_pipeline_and_sends(stub_tools):
    calls, _ = stub_tools

    result = run_agent("I was charged twice", ticket_id="T1", agent=build_agent_graph())

    assert calls == ["classify_intent", "score_priority", "retrieve_from_kb", "draft_response"]
    assert result["final_action"] == "send"
    assert result["ticket_id"] == "T1"
    assert result["drafted_response"] == "Reply using 1 docs"
    assert result["reasoning_trace"][:4] == ["intent", "priority", "retrieve", "draft"]
    assert result["reasoning_trace"][-1].startswith("SENT")


def test_low_confidence_is_flagged(stub_tools):
    _, config = stub_tools
    config["confidence"] = 0.3

    result = run_agent("vague question", agent=build_agent_graph())

    assert result["final_action"] == "flag_for_review"
    assert result["reasoning_trace"][-1].startswith("FLAGGED")


def test_threshold_is_configurable(stub_tools):
    _, config = stub_tools
    config["confidence"] = 0.6

    assert run_agent("x", agent=build_agent_graph(confidence_threshold=0.5))["final_action"] == "send"
    assert run_agent("x", agent=build_agent_graph(confidence_threshold=0.8))["final_action"] == "flag_for_review"


def test_critical_ticket_escalates_before_retrieval(stub_tools):
    calls, config = stub_tools
    config["priority"] = "critical"

    result = run_agent("I'm calling my lawyer", agent=build_agent_graph())

    assert calls == ["classify_intent", "score_priority"]
    assert result["final_action"] == "escalate"
    assert result["drafted_response"] == ""


def test_no_priority_gate_variant_never_escalates(stub_tools):
    calls, config = stub_tools
    config["priority"] = "critical"

    result = run_agent("I'm calling my lawyer", agent=build_agent_graph(use_priority_gate=False))

    assert result["final_action"] == "send"
    assert "draft_response" in calls


def test_no_kb_variant_skips_retrieval(stub_tools):
    calls, _ = stub_tools

    result = run_agent("refund?", agent=build_agent_graph(use_kb=False))

    assert "retrieve_from_kb" not in calls
    assert result["retrieved_docs"] == []
    assert result["drafted_response"] == "Reply using 0 docs"
