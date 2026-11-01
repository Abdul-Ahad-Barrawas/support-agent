import pandas as pd
import pytest

from config.settings import INTENT_LABELS
from src.evaluation.build_golden_set import (
    label_intent,
    label_priority,
    load_raw_tickets,
    merge_with_hand_rows,
    sample_dev_set,
    sample_golden,
    to_golden_rows,
)
from src.evaluation.evaluator import REQUIRED_COLUMNS


def raw_row(subject="Help", body="Something happened.", queue="Technical Support",
            ticket_type="Incident", priority="medium", tag_list=(), answer="We can help.",
            language="en"):
    row = {"subject": subject, "body": body, "answer": answer, "type": ticket_type,
           "queue": queue, "priority": priority, "language": language}
    for i in range(1, 9):
        row[f"tag_{i}"] = tag_list[i - 1] if i <= len(tag_list) else None
    return pd.Series(row)


# --- label_intent ---

@pytest.mark.parametrize("kwargs, expected", [
    ({"queue": "Billing and Payments", "body": "I can't log in to pay my invoice"}, "billing"),
    ({"queue": "Returns and Exchanges", "body": "When will my refund arrive?"}, "billing"),
    ({"queue": "Billing and Payments", "ticket_type": "Request",
      "body": "Please revise our digital marketing plan."}, None),
    ({"body": "I am locked out after a password reset."}, "account_access"),
    ({"body": "I forgot my password and cannot log in."}, "account_access"),
    ({"body": "This is unacceptable, the third time this month."}, "complaint"),
    ({"ticket_type": "Request", "tag_list": ("Feature", "IT"),
      "body": "Please add an export feature."}, "feature_request"),
    ({"queue": "IT Support", "ticket_type": "Problem"}, "technical_issue"),
    ({"queue": "Customer Service", "ticket_type": "Request"}, "general_inquiry"),
    ({"queue": "Human Resources", "ticket_type": "Request"}, None),
    ({"queue": "Technical Support", "ticket_type": "Change"}, None),
])
def test_label_intent(kwargs, expected):
    assert label_intent(raw_row(**kwargs)) == expected


def test_unauthorized_access_incident_is_not_account_access():
    row = raw_row(body="Unauthorized access to patient data detected; login attempts blocked.")

    assert label_intent(row) == "technical_issue"


def test_reported_complaints_are_not_a_complaint():
    row = raw_row(body="We have received several complaints about delayed payroll processing.")

    assert label_intent(row) == "technical_issue"


def test_feature_tag_without_feature_wording_is_dropped():
    row = raw_row(queue="Human Resources", ticket_type="Request", tag_list=("Feature",),
                  body="Can you help improve our marketing strategy?")

    assert label_intent(row) is None


def test_feature_tag_on_incident_is_not_feature_request():
    row = raw_row(ticket_type="Incident", tag_list=("Feature",))

    assert label_intent(row) == "technical_issue"


def test_keyword_must_be_whole_word():
    # "blogin" / "catalogins" must not trip the account_access rule
    assert label_intent(raw_row(body="Our blogin platform crashed.")) == "technical_issue"


def test_missing_subject_is_handled():
    assert label_intent(raw_row(subject=float("nan"), body="Forgot my password")) == "account_access"


def test_html_line_breaks_become_newlines():
    golden = to_golden_rows(pd.DataFrame([raw_row(subject="Hi", body="Invoice one<br><br>Line two",
                                                  queue="Billing and Payments")]))

    assert golden.iloc[0].ticket_text == "Hi\n\nInvoice one\n\nLine two"


# --- label_priority ---

def test_high_priority_with_breach_becomes_critical():
    assert label_priority(raw_row(priority="high", body="We suspect a data breach.")) == "critical"


def test_breach_mention_alone_does_not_promote_medium():
    assert label_priority(raw_row(priority="medium", body="We suspect a data breach.")) == "medium"


def test_breach_in_a_request_does_not_promote():
    row = raw_row(priority="high", ticket_type="Request", body="Please improve breach protection.")

    assert label_priority(row) == "high"


def test_high_priority_without_signal_stays_high():
    assert label_priority(raw_row(priority="high", body="Printer is slow.")) == "high"


# --- to_golden_rows ---

def test_golden_rows_schema_and_action():
    raw = pd.DataFrame([
        raw_row(priority="high", body="Full outage across all regions.", answer="Investigating."),
        raw_row(queue="Billing and Payments", priority="low", body="Refund my charge.",
                answer="Refund sent."),
        raw_row(queue="Human Resources", ticket_type="Request"),
    ])

    golden = to_golden_rows(raw)

    assert list(golden.columns) == REQUIRED_COLUMNS + ["source"]
    assert len(golden) == 2  # HR ticket dropped
    critical, billing = golden.iloc[0], golden.iloc[1]
    assert (critical.expected_priority, critical.expected_action) == ("critical", "escalate")
    assert critical.reference_response.startswith("N/A")
    assert (billing.expected_intent, billing.expected_action, billing.reference_response) == \
        ("billing", "send", "Refund sent.")
    assert set(golden["source"]) == {"kaggle"}


def test_ticket_text_joins_subject_and_body():
    golden = to_golden_rows(pd.DataFrame([raw_row(subject="Invoice", body="Wrong amount.",
                                                  queue="Billing and Payments")]))

    assert golden.iloc[0].ticket_text == "Invoice\n\nWrong amount."


def test_load_raw_tickets_keeps_english_with_answers(tmp_path):
    path = tmp_path / "raw.csv"
    pd.DataFrame([
        raw_row(body="keep"),
        raw_row(body="german", language="de"),
        raw_row(body="no answer", answer=None),
    ]).to_csv(path, index=False)

    assert list(load_raw_tickets(str(path))["body"]) == ["keep"]


# --- sample_golden ---

def labeled_pool():
    rows = []
    for intent in INTENT_LABELS:
        for i in range(20):
            rows.append({"ticket_text": f"{intent} {i}", "expected_intent": intent,
                         "expected_priority": "critical" if i < 3 else "low"})
    return pd.DataFrame(rows)


def test_sample_is_stratified():
    sample = sample_golden(labeled_pool(), n_per_intent=5, n_critical=4)

    assert (sample["expected_priority"] == "critical").sum() == 4
    non_critical = sample[sample["expected_priority"] != "critical"]
    assert non_critical["expected_intent"].value_counts().to_dict() == {i: 5 for i in INTENT_LABELS}


def test_sample_is_reproducible_and_seed_dependent():
    pool = labeled_pool()

    a = sample_golden(pool, seed=1)
    assert a.equals(sample_golden(pool, seed=1))
    assert not a.equals(sample_golden(pool, seed=2))


def test_sample_handles_rare_intent():
    pool = labeled_pool()
    pool = pool[~((pool.expected_intent == "complaint") & (pool.ticket_text != "complaint 10"))]

    sample = sample_golden(pool, n_per_intent=5)

    assert (sample["expected_intent"] == "complaint").sum() == 1


# --- merge_with_hand_rows ---

def golden_row(text, source=None):
    row = {"ticket_text": text, "expected_intent": "billing", "expected_priority": "low",
           "expected_action": "send", "reference_response": "ok"}
    if source:
        row["source"] = source
    return row


def test_merge_keeps_hand_rows_from_legacy_file():
    existing = pd.DataFrame([golden_row("hand 1"), golden_row("hand 2")])  # no source column
    generated = pd.DataFrame([golden_row("kaggle 1", "kaggle")])

    merged = merge_with_hand_rows(existing, generated)

    assert list(merged["ticket_text"]) == ["hand 1", "hand 2", "kaggle 1"]
    assert list(merged["source"]) == ["hand", "hand", "kaggle"]


def test_merge_replaces_previous_generated_rows():
    existing = pd.DataFrame([golden_row("hand 1", "hand"), golden_row("old kaggle", "kaggle")])
    generated = pd.DataFrame([golden_row("new kaggle", "kaggle")])

    merged = merge_with_hand_rows(existing, generated)

    assert list(merged["ticket_text"]) == ["hand 1", "new kaggle"]


def test_merge_keeps_hand_corrected_kaggle_row():
    corrected = {**golden_row("kaggle ticket", "hand"), "expected_intent": "complaint"}
    existing = pd.DataFrame([corrected])
    generated = pd.DataFrame([golden_row("kaggle ticket", "kaggle"), golden_row("other", "kaggle")])

    merged = merge_with_hand_rows(existing, generated)

    assert list(merged["ticket_text"]) == ["kaggle ticket", "other"]
    assert merged.iloc[0]["expected_intent"] == "complaint"


# --- sample_dev_set ---

def test_dev_set_never_overlaps_golden():
    pool = labeled_pool()
    golden = sample_golden(pool, n_per_intent=5, n_critical=4, seed=42)

    dev = sample_dev_set(pool, golden, n_per_intent=5, n_critical=4)

    assert not set(dev["ticket_text"]) & set(golden["ticket_text"])
    assert len(dev) == len(golden)  # same stratified shape


def test_dev_set_is_reproducible():
    pool = labeled_pool()
    golden = sample_golden(pool, seed=42)

    assert sample_dev_set(pool, golden).equals(sample_dev_set(pool, golden))
