import pytest

from src.agent.state import new_ticket_state
from src.tools import response_drafter
from src.tools.response_drafter import DraftResult, draft_response


class FakeLLM:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.prompts = result, error, []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM(DraftResult(response="Your refund is on its way.", confidence=0.85))
    monkeypatch.setattr(response_drafter, "structured_llm", fake)
    return fake


def ticket(docs=(), trace=None):
    state = new_ticket_state("I was charged twice, please refund me.")
    state.update(intent="billing", priority="high", retrieved_docs=list(docs),
                 reasoning_trace=trace or [])
    return state


def test_returns_llm_response_and_confidence(fake_llm):
    out = draft_response(ticket(["Refunds take 3-5 business days."]))

    assert out["drafted_response"] == "Your refund is on its way."
    assert out["response_confidence"] == 0.85


def test_prompt_is_grounded_in_ticket_and_docs(fake_llm):
    draft_response(ticket(["Refunds take 3-5 business days.", "Duplicate charges are refunded immediately."]))

    prompt = fake_llm.prompts[0]
    assert "I was charged twice" in prompt
    assert "[1] Refunds take 3-5 business days." in prompt
    assert "[2] Duplicate charges are refunded immediately." in prompt
    assert "billing" in prompt and "high" in prompt
    assert response_drafter.COMPANY_NAME in prompt


def test_prompt_says_when_no_docs_found(fake_llm):
    draft_response(ticket([]))

    assert "no relevant policy documents found" in fake_llm.prompts[0]


def test_llm_failure_falls_back_with_zero_confidence(monkeypatch):
    monkeypatch.setattr(response_drafter, "structured_llm", FakeLLM(error=RuntimeError("429 rate limit")))

    out = draft_response(ticket(["doc"]))

    assert out["drafted_response"] == response_drafter.FALLBACK_RESPONSE
    assert out["response_confidence"] == 0.0
    assert "Drafting failed (RuntimeError)" in out["reasoning_trace"][-1]


def test_appends_to_reasoning_trace(fake_llm):
    out = draft_response(ticket(trace=["Retrieved 1 docs from KB"]))

    assert out["reasoning_trace"] == ["Retrieved 1 docs from KB", "Drafted response (confidence: 0.85)"]


def test_confidence_outside_range_rejected():
    with pytest.raises(ValueError):
        DraftResult(response="x", confidence=1.5)
