"""Fallback behaviour of the two LLM tools from the earlier commit (LLM stubbed)."""

import pytest

from src.agent.state import new_ticket_state
from src.tools import intent_classifier, priority_scorer
from src.tools.intent_classifier import IntentResult, classify_intent
from src.tools.priority_scorer import SentimentResult, has_escalation_keyword, score_priority


class FakeLLM:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    def invoke(self, prompt):
        if self.error:
            raise self.error
        return self.result


def test_classifier_success(monkeypatch):
    monkeypatch.setattr(intent_classifier, "structured_llm",
                        FakeLLM(IntentResult(intent="billing", confidence=0.9)))

    out = classify_intent(new_ticket_state("charged twice"))

    assert (out["intent"], out["intent_confidence"]) == ("billing", 0.9)
    assert out["reasoning_trace"] == ["Classified intent: billing (0.90)"]


def test_classifier_failure_is_visible_in_trace(monkeypatch):
    monkeypatch.setattr(intent_classifier, "structured_llm", FakeLLM(error=RuntimeError("429")))

    out = classify_intent(new_ticket_state("charged twice"))

    assert (out["intent"], out["intent_confidence"]) == ("general_inquiry", 0.0)
    assert "failed" in out["reasoning_trace"][-1]


@pytest.mark.parametrize("sentiment, text, priority", [
    (-0.9, "this is a scam, cancel now", "critical"),
    (-0.9, "I am very sad", "high"),
    (0.5, "my account was hacked", "high"),
    (-0.2, "slightly annoyed", "medium"),
    (0.3, "thanks!", "low"),
])
def test_priority_rules(monkeypatch, sentiment, text, priority):
    monkeypatch.setattr(priority_scorer, "structured_llm",
                        FakeLLM(SentimentResult(sentiment_score=sentiment)))

    out = score_priority(new_ticket_state(text))

    assert out["priority"] == priority
    assert out["should_escalate"] == (priority == "critical")


def test_scorer_failure_is_visible_in_trace(monkeypatch):
    monkeypatch.setattr(priority_scorer, "structured_llm", FakeLLM(error=RuntimeError("429")))

    out = score_priority(new_ticket_state("thanks"))

    assert out["sentiment_score"] == 0.0
    assert "failed" in out["reasoning_trace"][0]
    assert out["reasoning_trace"][-1] == "Priority: low (sentiment: 0.00)"


def test_keyword_match_is_whole_word():
    assert has_escalation_keyword("The site is DOWN!")
    assert not has_escalation_keyword("I downloaded the invoice")
