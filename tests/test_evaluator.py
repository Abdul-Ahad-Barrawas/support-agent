import pandas as pd
import pytest

from src.evaluation.evaluator import (
    REQUIRED_COLUMNS,
    load_golden_set,
    predict,
    rouge_l_f1,
    run_ablation,
    run_evaluation,
    score_predictions,
)
from config.settings import GOLDEN_SET_PATH


GOLDEN = pd.DataFrame([
    {"ticket_text": "charged twice", "expected_intent": "billing", "expected_priority": "high",
     "expected_action": "send", "reference_response": "We refunded the duplicate charge"},
    {"ticket_text": "calling my lawyer", "expected_intent": "complaint", "expected_priority": "critical",
     "expected_action": "escalate", "reference_response": "N/A - escalated to human agent"},
    {"ticket_text": "add dark mode", "expected_intent": "feature_request", "expected_priority": "low",
     "expected_action": "send", "reference_response": "Thanks we logged the feature request"},
])


class LookupAgent:
    """Fake agent: answers from a dict keyed on ticket text."""

    def __init__(self, answers):
        self.answers = answers

    def invoke(self, state):
        answer = self.answers[state["ticket_text"]]
        if isinstance(answer, Exception):
            raise answer
        return {**state, **answer}


PERFECT = LookupAgent({
    "charged twice": {"intent": "billing", "priority": "high", "final_action": "send",
                      "drafted_response": "We refunded the duplicate charge"},
    "calling my lawyer": {"intent": "complaint", "priority": "critical", "final_action": "escalate",
                          "drafted_response": ""},
    "add dark mode": {"intent": "feature_request", "priority": "low", "final_action": "send",
                      "drafted_response": "Thanks we logged the feature request"},
})


# --- load_golden_set ---

def test_loads_real_golden_set():
    golden = load_golden_set(GOLDEN_SET_PATH)

    assert set(REQUIRED_COLUMNS) <= set(golden.columns)
    assert len(golden) > 0
    assert golden["expected_action"].isin(["send", "escalate", "flag_for_review"]).all()


def test_missing_column_raises(tmp_path):
    path = tmp_path / "bad.csv"
    GOLDEN.drop(columns=["expected_priority"]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="expected_priority"):
        load_golden_set(str(path))


def test_empty_golden_set_raises(tmp_path):
    path = tmp_path / "empty.csv"
    GOLDEN.head(0).to_csv(path, index=False)

    with pytest.raises(ValueError, match="no rows"):
        load_golden_set(str(path))


# --- rouge_l_f1 ---

def test_rouge_identical_is_one():
    assert rouge_l_f1("Refund issued today", "refund issued today!") == 1.0


def test_rouge_no_overlap_or_empty_is_zero():
    assert rouge_l_f1("hello there", "refund issued") == 0.0
    assert rouge_l_f1("", "refund issued") == 0.0


def test_rouge_partial_overlap():
    # LCS = "refund issued" (2 words); P = 2/3, R = 2/2 → F1 = 0.8
    assert rouge_l_f1("refund was issued", "refund issued") == pytest.approx(0.8)


def test_rouge_respects_word_order():
    assert rouge_l_f1("issued refund", "refund issued") == pytest.approx(0.5)


# --- run_evaluation ---

def test_perfect_agent_scores_one():
    metrics = run_evaluation(PERFECT, GOLDEN)

    assert metrics["intent_accuracy"] == 1.0
    assert metrics["intent_macro_f1"] == 1.0
    assert metrics["priority_accuracy"] == 1.0
    assert metrics["action_accuracy"] == 1.0
    assert metrics["response_quality"] == 1.0
    assert metrics["n_tickets"] == 3 and metrics["n_errors"] == 0


def test_partial_mistakes_are_counted():
    agent = LookupAgent({
        **PERFECT.answers,
        "calling my lawyer": {"intent": "complaint", "priority": "high", "final_action": "send",
                              "drafted_response": "Sorry about that"},
    })

    metrics = run_evaluation(agent, GOLDEN)

    assert metrics["priority_accuracy"] == pytest.approx(2 / 3)
    assert metrics["action_accuracy"] == pytest.approx(2 / 3)
    # Escalation rows are excluded from response quality
    assert metrics["response_quality"] == 1.0


def test_wrongly_escalated_ticket_scores_zero_quality():
    agent = LookupAgent({
        **PERFECT.answers,
        "add dark mode": {"intent": "feature_request", "priority": "critical",
                          "final_action": "escalate", "drafted_response": ""},
    })

    assert run_evaluation(agent, GOLDEN)["response_quality"] == pytest.approx(0.5)


def test_agent_errors_score_as_wrong_not_dropped():
    agent = LookupAgent({**PERFECT.answers, "charged twice": RuntimeError("rate limited")})

    predictions = predict(agent, GOLDEN)
    metrics = score_predictions(predictions)

    assert metrics["n_errors"] == 1
    assert metrics["n_tickets"] == 3
    assert metrics["intent_accuracy"] == pytest.approx(2 / 3)
    assert "rate limited" in predictions.loc[0, "error"]


def test_predictions_keep_golden_columns():
    predictions = predict(PERFECT, GOLDEN)

    assert set(REQUIRED_COLUMNS) <= set(predictions.columns)
    assert predictions.loc[2, "predicted_intent"] == "feature_request"


# --- run_ablation ---

def test_ablation_has_one_row_per_variant():
    lazy = LookupAgent({text: {"intent": "billing", "priority": "low", "final_action": "send",
                               "drafted_response": ""} for text in GOLDEN["ticket_text"]})

    table = run_ablation({"Full agent": PERFECT, "Lazy": lazy}, GOLDEN)

    assert list(table.index) == ["Full agent", "Lazy"]
    assert table.loc["Full agent", "action_accuracy"] == 1.0
    assert table.loc["Lazy", "intent_accuracy"] == pytest.approx(1 / 3)


def test_tool_fallbacks_are_counted_as_degraded():
    agent = LookupAgent({
        **PERFECT.answers,
        "add dark mode": {**PERFECT.answers["add dark mode"],
                          "reasoning_trace": ["Drafting failed (BadRequestError) — using fallback"]},
    })

    metrics = run_evaluation(agent, GOLDEN)

    assert metrics["n_degraded"] == 1
    assert metrics["n_errors"] == 0


def test_crashed_rows_are_also_degraded():
    agent = LookupAgent({**PERFECT.answers, "charged twice": RuntimeError("boom")})

    assert run_evaluation(agent, GOLDEN)["n_degraded"] == 1
