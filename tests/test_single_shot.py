import pytest

from src.agent import single_shot
from src.agent.single_shot import SingleShotAgent, SingleShotResult
from src.agent.state import new_ticket_state


class FakeLLM:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    def invoke(self, prompt):
        if self.error:
            raise self.error
        return self.result


def run(monkeypatch, priority="low", confidence=0.9, threshold=0.7, error=None):
    result = SingleShotResult(intent="billing", priority=priority, response="Reply", confidence=confidence)
    monkeypatch.setattr(single_shot, "structured_llm", FakeLLM(result, error))
    return SingleShotAgent(confidence_threshold=threshold).invoke(new_ticket_state("ticket", "T9"))


def test_confident_reply_is_sent(monkeypatch):
    out = run(monkeypatch)

    assert out["final_action"] == "send"
    assert out["intent"] == "billing" and out["priority"] == "low"
    assert out["drafted_response"] == "Reply"
    assert out["ticket_id"] == "T9"


def test_low_confidence_is_flagged(monkeypatch):
    assert run(monkeypatch, confidence=0.5)["final_action"] == "flag_for_review"


def test_critical_priority_escalates_without_reply(monkeypatch):
    out = run(monkeypatch, priority="critical", confidence=0.99)

    assert out["final_action"] == "escalate"
    assert out["should_escalate"] is True
    assert out["drafted_response"] == ""


def test_llm_failure_flags_for_review(monkeypatch):
    out = run(monkeypatch, error=RuntimeError("timeout"))

    assert out["final_action"] == "flag_for_review"
    assert "failed" in out["reasoning_trace"][0]


def test_invalid_labels_rejected():
    with pytest.raises(ValueError):
        SingleShotResult(intent="spam", priority="low", response="x", confidence=0.5)
