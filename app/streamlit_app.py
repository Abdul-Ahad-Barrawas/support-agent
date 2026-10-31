"""
Streamlit UI for the Support Ticket Agent.

Owner: Non-technical member

Run with: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# `streamlit run` puts app/ on the path, not the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from config.settings import CONFIDENCE_THRESHOLD, GOLDEN_SET_PATH, LLM_MODEL, LLM_PROVIDER
from src.agent.graph import build_agent_graph, run_agent
from src.evaluation.evaluator import default_variants, load_golden_set, run_ablation, run_evaluation

ACTION_STYLES = {
    "send": (st.success, "✅ SEND — response is confident enough to go out"),
    "escalate": (st.error, "🚨 ESCALATE — handed off to a human agent"),
    "flag_for_review": (st.warning, "⚠️ FLAG FOR REVIEW — a human should check this draft"),
}


@st.cache_resource
def get_agent(confidence_threshold: float):
    return build_agent_graph(confidence_threshold)


st.set_page_config(page_title="Support Agent", layout="wide")
st.title("🎫 Support Ticket Triage Agent")

# --- Sidebar: settings ---
st.sidebar.header("Settings")
threshold = st.sidebar.slider(
    "Confidence threshold", 0.0, 1.0, CONFIDENCE_THRESHOLD, 0.05,
    help="Drafts below this confidence are flagged for human review",
)
st.sidebar.caption(f"LLM: `{LLM_PROVIDER}` / `{LLM_MODEL}` (set in .env)")

# --- Sidebar: evaluation dashboard ---
st.sidebar.header("Evaluation")
uploaded = st.sidebar.file_uploader("Golden set CSV (defaults to the repo's)", type="csv")

try:
    golden_set = load_golden_set(uploaded or GOLDEN_SET_PATH)
    st.sidebar.caption(f"{len(golden_set)} golden tickets loaded")
except ValueError as e:
    golden_set = None
    st.sidebar.error(str(e))

if golden_set is not None:
    if st.sidebar.button("Run evaluation"):
        with st.spinner("Running the full agent on the golden set…"):
            st.session_state["eval_metrics"] = run_evaluation(get_agent(threshold), golden_set)
    if st.sidebar.button("Run ablation", help="4 variants × every golden ticket — slow"):
        with st.spinner("Running all 4 agent variants…"):
            st.session_state["ablation"] = run_ablation(default_variants(threshold), golden_set)

if "eval_metrics" in st.session_state:
    st.sidebar.subheader("Metrics — full agent")
    st.sidebar.dataframe(
        pd.Series(st.session_state["eval_metrics"], name="value").round(3).astype(str)
    )

if "ablation" in st.session_state:
    st.subheader("Ablation study")
    st.dataframe(st.session_state["ablation"].round(3), use_container_width=True)

# --- Section 1: ticket input ---
ticket_text = st.text_area(
    "Paste a support ticket",
    height=150,
    placeholder="I was charged twice for my subscription this month. Please refund the extra charge.",
)

# Not disabled while empty: the text area only commits on blur, so a disabled
# button would swallow the first click after pasting a ticket
if st.button("Run Agent", type="primary"):
    if not ticket_text.strip():
        st.warning("Paste a ticket first.")
    else:
        with st.spinner("Agent working…"):
            st.session_state["result"] = run_agent(ticket_text.strip(), agent=get_agent(threshold))

# --- Section 2: agent trace ---
if "result" in st.session_state:
    result = st.session_state["result"]

    show_action, message = ACTION_STYLES[result["final_action"]]
    show_action(message)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Intent", result["intent"] or "—", f"{result['intent_confidence']:.0%} conf.",
                delta_color="off")
    col2.metric("Priority", result["priority"] or "—")
    col3.metric("Sentiment", f"{result['sentiment_score']:+.2f}")
    col4.metric("Response confidence", f"{result['response_confidence']:.0%}")

    # LLM output and trace lines are shaped by the customer's ticket — render them
    # as plain text so injected Markdown (e.g. tracking-pixel images) never loads
    if result["drafted_response"]:
        st.subheader("Drafted response")
        st.text(result["drafted_response"])

    with st.expander(f"Retrieved KB docs ({len(result['retrieved_docs'])})"):
        if not result["retrieved_docs"]:
            st.caption("No documents retrieved (escalated before retrieval, or KB had no match).")
        for doc, score in zip(result["retrieved_docs"], result["retrieval_scores"]):
            st.markdown(f"**Similarity {score:.2f}**")
            st.text(doc)

    with st.expander("Reasoning trace", expanded=True):
        for i, step in enumerate(result["reasoning_trace"], 1):
            st.text(f"{i}. {step}")
