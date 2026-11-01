"""
Builds golden-set rows from the raw Kaggle ticket dataset.

Source: "Customer IT Support - Ticket Dataset" (tobiasbueck, CC BY-NC 4.0),
downloaded into data/raw/customer-support-tickets/ — see README "Data Sources".

The dataset has no labels matching our schema, so they are derived:
  - intent   ← queue + ticket type + tags + keyword rules (see label_intent)
               — the dataset's queues are noisy, so keywords must agree with them
  - priority ← dataset priority, promoted to "critical" when a high-priority
               incident mentions a breach / outage / legal threat
  - action   ← "escalate" for critical tickets, otherwise "send"

These are heuristic labels: every generated row is tagged source="kaggle"
so a human can spot-check them. Hand-written rows (source="hand") are kept.

Run from the repo root:
    python -m src.evaluation.build_golden_set
"""

import re

import pandas as pd

from config.settings import GOLDEN_SET_PATH, INTENT_LABELS
from src.evaluation.evaluator import REQUIRED_COLUMNS

RAW_PATH = "./data/raw/customer-support-tickets/aa_dataset-tickets-multi-lang-5-2-50-version.csv"
DEV_SET_PATH = "./data/golden_set/dev_qa.csv"

TECHNICAL_QUEUES = {
    "Technical Support", "Product Support", "IT Support", "Service Outages and Maintenance",
}
INQUIRY_QUEUES = {"General Inquiry", "Customer Service", "Sales and Pre-Sales"}

# User-level access problems only — "unauthorized access" incidents are security, not this
ACCOUNT_ACCESS_PATTERN = re.compile(
    r"\b((forgot|forgotten|reset|change|expired)\w* (my |the |our )?password|locked out|"
    r"(unable to|can ?not|can't|cannot) (log|sign) ?in|login (issue|problem|error|fail\w*)|"
    r"two-factor|2fa)\b"
)
# Customer dissatisfaction — not "we received complaints about X", which is a report
COMPLAINT_PATTERN = re.compile(
    r"\b(unacceptable|disappointed|disappointing|frustrated|frustrating|dissatisfied|"
    r"fed up|terrible|worst|very unhappy)\b"
)
# The dataset's own queue labels are noisy (marketing requests filed under billing)
BILLING_PATTERN = re.compile(
    r"\b(refund\w*|invoice\w*|charge[ds]?|billing|billed|payment\w*|subscription\w*|pric\w+)\b"
)
FEATURE_PATTERN = re.compile(r"\b(feature|enhancement|functionality|capabilit\w*|add support|integrat\w*)\b")
CRITICAL_PATTERN = re.compile(
    r"\b(data breach|breach\w*|hacked|ransomware|lawsuit|legal action|lawyer|"
    r"data loss|outage|security incident|fraud\w*)\b"
)


def load_raw_tickets(path: str = RAW_PATH) -> pd.DataFrame:
    """English tickets that have both a body and an agent answer."""
    raw = pd.read_csv(path)
    raw = raw[(raw["language"] == "en") & raw["body"].notna() & raw["answer"].notna()]
    return raw.reset_index(drop=True)


def ticket_text(row) -> str:
    subject = row["subject"] if isinstance(row["subject"], str) else ""
    body = re.sub(r"<br\s*/?>", "\n", row["body"])  # some bodies contain HTML line breaks
    return f"{subject}\n\n{body}".strip()


def tags(row) -> set[str]:
    return {row[col] for col in row.index if col.startswith("tag_") and isinstance(row[col], str)}


def label_intent(row) -> str | None:
    """Map one raw ticket to an intent label, or None to drop it."""
    text = ticket_text(row).lower()
    queue, ticket_type = row["queue"], row["type"]

    if queue in {"Billing and Payments", "Returns and Exchanges"} and BILLING_PATTERN.search(text):
        return "billing"
    if ACCOUNT_ACCESS_PATTERN.search(text):
        return "account_access"
    if COMPLAINT_PATTERN.search(text):
        return "complaint"
    if ("Feature" in tags(row) and ticket_type in {"Request", "Change"}
            and FEATURE_PATTERN.search(text)):
        return "feature_request"
    if queue in TECHNICAL_QUEUES and ticket_type in {"Incident", "Problem"}:
        return "technical_issue"
    if queue in INQUIRY_QUEUES and ticket_type == "Request":
        return "general_inquiry"
    return None  # e.g. Human Resources, or a queue/type combo with no clear intent


def label_priority(row) -> str:
    priority = str(row["priority"]).lower()
    # Only live incidents escalate — a *request* to improve breach protection doesn't
    if (priority == "high" and row["type"] in {"Incident", "Problem"}
            and CRITICAL_PATTERN.search(ticket_text(row).lower())):
        return "critical"
    return priority


def to_golden_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Label every raw ticket; tickets with no clear intent are dropped."""
    rows = []
    for _, row in raw.iterrows():
        intent = label_intent(row)
        if intent is None:
            continue
        priority = label_priority(row)
        rows.append({
            "ticket_text": ticket_text(row),
            "expected_intent": intent,
            "expected_priority": priority,
            "expected_action": "escalate" if priority == "critical" else "send",
            "reference_response": "N/A - escalated to human agent" if priority == "critical" else row["answer"],
            "source": "kaggle",
        })
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS + ["source"])


def sample_golden(labeled: pd.DataFrame, n_per_intent: int = 10, n_critical: int = 8,
                  seed: int = 42) -> pd.DataFrame:
    """
    Stratified sample: n_critical escalation tickets, then n_per_intent
    non-critical tickets for every intent (fewer if an intent is rarer).
    """
    critical = labeled[labeled["expected_priority"] == "critical"]
    picked = [critical.sample(min(n_critical, len(critical)), random_state=seed)]

    rest = labeled[labeled["expected_priority"] != "critical"]
    for intent in INTENT_LABELS:
        pool = rest[rest["expected_intent"] == intent]
        picked.append(pool.sample(min(n_per_intent, len(pool)), random_state=seed))

    return pd.concat(picked).drop_duplicates("ticket_text").reset_index(drop=True)


def sample_dev_set(labeled: pd.DataFrame, golden: pd.DataFrame, n_per_intent: int = 10,
                   n_critical: int = 8, seed: int = 7) -> pd.DataFrame:
    """
    A tuning set drawn the same way as the golden set but never overlapping it.
    Prompts and thresholds are tuned on this, so the golden set stays an
    untouched test set and reported scores aren't inflated by tuning.
    """
    unseen = labeled[~labeled["ticket_text"].isin(golden["ticket_text"])]
    return sample_golden(unseen, n_per_intent, n_critical, seed)


def merge_with_hand_rows(existing: pd.DataFrame, generated: pd.DataFrame) -> pd.DataFrame:
    """Keep hand-written rows, replace any previously generated ones."""
    existing = existing.copy()
    if "source" not in existing.columns:
        existing["source"] = "hand"
    hand = existing[existing["source"].fillna("hand") == "hand"].assign(source="hand")
    # A hand-corrected Kaggle ticket wins over a fresh copy of the same ticket
    generated = generated[~generated["ticket_text"].isin(hand["ticket_text"])]
    return pd.concat([hand, generated], ignore_index=True)[REQUIRED_COLUMNS + ["source"]]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Add Kaggle tickets to the golden set")
    parser.add_argument("--n-per-intent", type=int, default=10)
    parser.add_argument("--n-critical", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-path", default=RAW_PATH)
    parser.add_argument("--dev", action="store_true",
                        help=f"write a non-overlapping tuning set to {DEV_SET_PATH} instead")
    args = parser.parse_args()

    labeled = to_golden_rows(load_raw_tickets(args.raw_path))

    if args.dev:
        dev = sample_dev_set(labeled, pd.read_csv(GOLDEN_SET_PATH), args.n_per_intent,
                             args.n_critical, seed=7 if args.seed == 42 else args.seed)
        dev.to_csv(DEV_SET_PATH, index=False)
        print(f"Wrote {len(dev)} dev rows (no overlap with the golden set) to {DEV_SET_PATH}")
        print(dev["expected_intent"].value_counts().to_string())
        raise SystemExit

    generated = sample_golden(labeled, args.n_per_intent, args.n_critical, args.seed)
    golden = merge_with_hand_rows(pd.read_csv(GOLDEN_SET_PATH), generated)
    golden.to_csv(GOLDEN_SET_PATH, index=False)

    print(f"Labeled {len(labeled)} raw tickets; wrote {len(golden)} rows to {GOLDEN_SET_PATH}")
    print(golden.groupby(["source", "expected_intent"]).size().to_string())
    print(golden["expected_priority"].value_counts().to_string())
