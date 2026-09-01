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
"""

# TODO: Implement these functions
#
# def load_golden_set(path: str) -> pd.DataFrame:
#     - Load CSV
#     - Validate columns exist
#
# def run_evaluation(agent, golden_set: pd.DataFrame) -> dict:
#     - For each row, run agent on ticket_text
#     - Compare predicted vs expected: intent, priority, action
#     - Calculate:
#         - Intent accuracy (exact match)
#         - Priority accuracy (exact match)
#         - Action accuracy (send/escalate/flag)
#         - Response quality (LLM-as-judge or ROUGE score)
#     - Return metrics dict
#
# def run_ablation(agent_variants: dict, golden_set) -> pd.DataFrame:
#     """
#     Run multiple agent configurations and compare.
#
#     Suggested ablations:
#       1. full_agent        — all 4 tools
#       2. no_kb             — skip KB retrieval, draft from scratch
#       3. no_priority       — skip priority scoring, never escalate
#       4. single_shot       — one LLM call, no tools, just prompt
#
#     Returns a DataFrame comparing metrics across variants.
#     This table is your CV differentiator.
#     """
