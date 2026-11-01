"""
Evaluation harness — runs the agent against the golden test set.

Owner: Non-technical member (golden set creation)
        + Strong technical member (metrics code)

THE GOLDEN SET (data/golden_set/golden_qa.csv) is the most important
deliverable for the evaluation. Each row should have:

| ticket_text | expected_intent | expected_priority | expected_action | reference_response |
|-------------|-----------------|-------------------|------------------|--------------------|
| "I was charged twice..." | billing | high | send | "We apologize for..." |

Aim for 50-100 rows covering all intent types and priority levels.

Run from the repo root:
    python -m src.evaluation.evaluator              # full agent only
    python -m src.evaluation.evaluator --ablation   # all 4 variants
    python -m src.evaluation.evaluator --ablation --workers 4   # gentler on rate limits
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import IO

import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm

from config.settings import CONFIDENCE_THRESHOLD, GOLDEN_SET_PATH
from src.agent.state import new_ticket_state

# Tickets are independent and the time goes on waiting for the LLM API, so
# several run at once. Lower it if the provider starts returning 429s.
DEFAULT_WORKERS = 8

REQUIRED_COLUMNS = [
    "ticket_text",
    "expected_intent",
    "expected_priority",
    "expected_action",
    "reference_response",
]


def load_golden_set(path: str | IO = GOLDEN_SET_PATH) -> pd.DataFrame:
    """Load the golden set CSV (path or uploaded file), failing loudly if a column is missing."""
    golden_set = pd.read_csv(path)

    missing = [col for col in REQUIRED_COLUMNS if col not in golden_set.columns]
    if missing:
        raise ValueError(f"Golden set is missing columns: {missing}")
    if golden_set.empty:
        raise ValueError("Golden set has no rows")

    return golden_set


def rouge_l_f1(candidate: str, reference: str) -> float:
    """ROUGE-L F1: longest common subsequence of words, as an F-score."""
    cand = re.findall(r"\w+", candidate.lower())
    ref = re.findall(r"\w+", reference.lower())
    if not cand or not ref:
        return 0.0

    # Classic LCS dynamic programme, one row at a time
    prev = [0] * (len(ref) + 1)
    for c in cand:
        curr = [0]
        for j, r in enumerate(ref, 1):
            curr.append(prev[j - 1] + 1 if c == r else max(prev[j], curr[j - 1]))
        prev = curr
    lcs = prev[-1]

    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(cand), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def is_degraded(result: dict) -> bool:
    """
    True if any tool fell back to a default (rate limit, malformed LLM output…).
    Tools catch their own errors, so without this a flaky API would just look
    like a model that answers wrong.
    """
    return any("failed" in step for step in result.get("reasoning_trace", []))


def predict_one(agent, ticket_text: str, ticket_id: str) -> dict:
    """Run the agent on one ticket and flatten the result into a prediction row."""
    try:
        result = agent.invoke(new_ticket_state(ticket_text, ticket_id=ticket_id))
        return {
            "predicted_intent": result.get("intent"),
            "predicted_priority": result.get("priority"),
            "predicted_action": result.get("final_action"),
            "drafted_response": result.get("drafted_response", ""),
            "error": None,
            "degraded": is_degraded(result),
        }
    except Exception as e:
        # Errored rows stay in the table and score as wrong — never dropped
        return {
            "predicted_intent": None,
            "predicted_priority": None,
            "predicted_action": None,
            "drafted_response": "",
            "error": f"{type(e).__name__}: {e}",
            "degraded": True,
        }


def predict(agent, golden_set: pd.DataFrame, workers: int = DEFAULT_WORKERS,
            desc: str = "Evaluating", progress: bool = True, position: int = 0) -> pd.DataFrame:
    """
    Run the agent on every golden ticket, `workers` at a time.
    One row of predictions per ticket, in golden-set order.
    """
    tickets = list(golden_set["ticket_text"])
    rows = [None] * len(tickets)
    degraded = 0

    with ThreadPoolExecutor(max_workers=workers) as pool, \
            tqdm(total=len(tickets), desc=desc, unit="ticket", position=position,
                 disable=not progress, dynamic_ncols=True) as bar:
        futures = {
            pool.submit(predict_one, agent, text, f"golden-{i}"): i
            for i, text in enumerate(tickets)
        }
        for future in as_completed(futures):
            row = future.result()
            rows[futures[future]] = row
            # Live count, so a rate-limited or broken run is obvious within seconds
            degraded += row["degraded"]
            bar.set_postfix(degraded=degraded, refresh=False)
            bar.update()

    return pd.concat([golden_set.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def score_predictions(predictions: pd.DataFrame) -> dict:
    """Compute the metrics for one variant from predict()'s output."""
    def accuracy(field):
        return float((predictions[f"predicted_{field}"] == predictions[f"expected_{field}"]).mean())

    # Escalated tickets have no reference reply ("N/A"), so only score the rest
    answerable = predictions[predictions["expected_action"] != "escalate"]
    quality = [
        rouge_l_f1(str(draft or ""), str(ref))
        for draft, ref in zip(answerable["drafted_response"], answerable["reference_response"])
    ]

    return {
        "intent_accuracy": accuracy("intent"),
        "intent_macro_f1": float(f1_score(
            predictions["expected_intent"],
            predictions["predicted_intent"].fillna("error"),
            # Average over the real classes only: errored rows still count as
            # wrong, but "error" must not become a 0-F1 class of its own
            labels=sorted(predictions["expected_intent"].unique()),
            average="macro",
            zero_division=0,
        )),
        "priority_accuracy": accuracy("priority"),
        "action_accuracy": accuracy("action"),
        "response_quality": float(sum(quality) / len(quality)) if quality else 0.0,
        "n_tickets": len(predictions),
        "n_errors": int(predictions["error"].notna().sum()),
        "n_degraded": int(predictions["degraded"].sum()),
    }


def run_evaluation(agent, golden_set: pd.DataFrame, **predict_kwargs) -> dict:
    """Run the agent on the golden set and return its metrics."""
    return score_predictions(predict(agent, golden_set, **predict_kwargs))


def default_variants(confidence_threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    """The 4 agent variants from the README's ablation table."""
    from src.agent.graph import build_agent_graph
    from src.agent.single_shot import SingleShotAgent

    return {
        "Full agent": build_agent_graph(confidence_threshold),
        "No KB retrieval": build_agent_graph(confidence_threshold, use_kb=False),
        "No priority gate": build_agent_graph(confidence_threshold, use_priority_gate=False),
        "Single-shot LLM": SingleShotAgent(confidence_threshold),
    }


def run_ablation(agent_variants: dict, golden_set: pd.DataFrame,
                 workers: int = DEFAULT_WORKERS, progress: bool = True) -> pd.DataFrame:
    """
    Evaluate every variant on the same golden set; one row per variant.
    Variants run side by side (one progress bar each), each with `workers`
    tickets in flight — so the whole ablation takes about as long as one variant.
    """
    names = list(agent_variants)
    width = max(len(name) for name in names)

    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        futures = {
            name: pool.submit(run_evaluation, agent_variants[name], golden_set, workers=workers,
                              desc=name.ljust(width), progress=progress, position=i)
            for i, name in enumerate(names)
        }
        results = {name: futures[name].result() for name in names}

    return pd.DataFrame.from_dict(results, orient="index").rename_axis("variant")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the agent on the golden set")
    parser.add_argument("--ablation", action="store_true", help="run all 4 variants")
    parser.add_argument("--golden-set", default=GOLDEN_SET_PATH)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"tickets in flight per variant (default {DEFAULT_WORKERS})")
    args = parser.parse_args()

    golden = load_golden_set(args.golden_set)
    variants = default_variants()
    if not args.ablation:
        variants = {"Full agent": variants["Full agent"]}

    table = run_ablation(variants, golden, workers=args.workers)
    print("\n" + table.round(3).to_string())
    if table["n_degraded"].any():
        print("\nWARNING: some tickets hit tool fallbacks (n_degraded > 0) — "
              "those scores understate the agent. Check rate limits / API errors.")
