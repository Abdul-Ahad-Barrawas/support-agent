import threading
import time

import pandas as pd

from src.evaluation.evaluator import predict, run_ablation

GOLDEN = pd.DataFrame({
    "ticket_text": [f"ticket {i}" for i in range(8)],
    "expected_intent": ["billing"] * 8,
    "expected_priority": ["low"] * 8,
    "expected_action": ["send"] * 8,
    "reference_response": ["ok"] * 8,
})


class SlowAgent:
    """Sleeps per ticket and records peak concurrency; earlier tickets finish LAST."""

    def __init__(self, base_delay=0.05, fail_on=None, degrade_on=None):
        self.base_delay, self.fail_on, self.degrade_on = base_delay, fail_on, degrade_on
        self.lock, self.active, self.peak = threading.Lock(), 0, 0

    def invoke(self, state):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        n = int(state["ticket_text"].split()[-1])
        time.sleep(self.base_delay * (8 - n))  # reverse finish order
        with self.lock:
            self.active -= 1
        if state["ticket_text"] == self.fail_on:
            raise RuntimeError("API down")
        trace = ["Drafting failed"] if state["ticket_text"] == self.degrade_on else []
        return {**state, "intent": "billing", "priority": "low", "final_action": "send",
                "drafted_response": state["ticket_text"], "reasoning_trace": trace}


def test_rows_stay_in_golden_order_despite_out_of_order_finish():
    out = predict(SlowAgent(), GOLDEN, workers=8, progress=False)

    assert list(out["drafted_response"]) == list(GOLDEN["ticket_text"])
    assert list(out["ticket_text"]) == list(GOLDEN["ticket_text"])


def test_tickets_really_run_concurrently():
    agent = SlowAgent(base_delay=0.05)  # 0.05..0.4s each; ~1.8s if serial

    start = time.perf_counter()
    predict(agent, GOLDEN, workers=8, progress=False)

    assert agent.peak == 8
    assert time.perf_counter() - start < 0.9


def test_workers_cap_concurrency():
    agent = SlowAgent(base_delay=0.01)

    predict(agent, GOLDEN, workers=2, progress=False)

    assert agent.peak <= 2


def test_single_worker_still_works():
    out = predict(SlowAgent(base_delay=0), GOLDEN, workers=1, progress=False)

    assert len(out) == 8 and out["error"].isna().all()


def test_failures_land_on_the_right_row():
    out = predict(SlowAgent(base_delay=0.01, fail_on="ticket 3"), GOLDEN, workers=4, progress=False)

    assert "API down" in out.loc[3, "error"]
    assert out["error"].notna().sum() == 1
    assert out.loc[3, "degraded"] and not out.loc[2, "degraded"]


def test_progress_bar_reports_count_and_degraded(capsys):
    predict(SlowAgent(base_delay=0, degrade_on="ticket 5"), GOLDEN, workers=4, desc="Full agent")

    err = capsys.readouterr().err
    assert "Full agent" in err
    assert "8/8" in err
    assert "degraded=1" in err


def test_progress_can_be_silenced(capsys):
    predict(SlowAgent(base_delay=0), GOLDEN, workers=4, progress=False)

    assert capsys.readouterr().err == ""


def test_ablation_runs_variants_side_by_side():
    variants = {"A": SlowAgent(base_delay=0.03), "B": SlowAgent(base_delay=0.03)}

    start = time.perf_counter()
    table = run_ablation(variants, GOLDEN, workers=8, progress=False)

    # each variant alone takes ~0.24s; serial would be ~0.48s
    assert time.perf_counter() - start < 0.42
    assert list(table.index) == ["A", "B"]
    assert (table["intent_accuracy"] == 1.0).all()
