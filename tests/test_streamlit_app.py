import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.agent import graph as graph_module
from src.evaluation import evaluator

APP = "../app/streamlit_app.py"


class FakeAgent:
    def __init__(self, action="send"):
        self.action = action

    def invoke(self, state):
        return {**state, "intent": "billing", "intent_confidence": 0.9, "priority": "high",
                "sentiment_score": -0.5, "retrieved_docs": ["Refunds take 5 days."],
                "retrieval_scores": [0.81], "drafted_response": "Your refund is on its way.",
                "response_confidence": 0.88, "final_action": self.action,
                "reasoning_trace": ["Classified intent: billing (0.90)", "SENT"]}


@pytest.fixture
def fake_agent(monkeypatch):
    agent = FakeAgent()
    st.cache_resource.clear()
    monkeypatch.setattr(graph_module, "build_agent_graph", lambda *args, **kwargs: agent)
    monkeypatch.setattr(evaluator, "default_variants", lambda t: {"Full agent": agent, "Other": agent})
    return agent


def run_ticket(text="I was charged twice"):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_area[0].input(text)
    next(b for b in at.button if b.label == "Run Agent").click()
    return at.run()


def test_app_loads_with_golden_set(fake_agent):
    at = AppTest.from_file(APP, default_timeout=30).run()

    assert not at.exception
    assert any("golden tickets loaded" in c.value for c in at.sidebar.caption)


def test_sent_ticket_shows_result_and_trace(fake_agent):
    at = run_ticket()

    assert not at.exception
    assert "SEND" in at.success[0].value
    assert at.metric[0].value == "billing"
    assert any("Your refund is on its way." in m.value for m in at.markdown)
    assert any("Classified intent" in m.value for m in at.markdown)


def test_escalated_ticket_shows_error_banner(fake_agent):
    fake_agent.action = "escalate"

    at = run_ticket("I'm calling my lawyer")

    assert "ESCALATE" in at.error[0].value


def test_evaluation_and_ablation_buttons(fake_agent):
    at = AppTest.from_file(APP, default_timeout=30).run()

    next(b for b in at.sidebar.button if b.label == "Run evaluation").click()
    at.run()
    next(b for b in at.sidebar.button if b.label == "Run ablation").click()
    at.run()

    assert not at.exception
    assert "eval_metrics" in at.session_state
    assert list(at.session_state["ablation"].index) == ["Full agent", "Other"]


def test_empty_ticket_warns_instead_of_running(fake_agent):
    at = run_ticket("   ")

    assert "Paste a ticket first." in at.warning[0].value
    assert "result" not in at.session_state
