"""
Streamlit UI for the Support Ticket Agent.

Owner: Non-technical member

Run with: streamlit run app/streamlit_app.py
"""

# TODO: Build this UI with these sections
#
# import streamlit as st
#
# st.set_page_config(page_title="Support Agent", layout="wide")
# st.title("🎫 Support Ticket Triage Agent")
#
# SECTION 1: Ticket Input
#   - st.text_area for pasting a support ticket
#   - st.button("Run Agent")
#
# SECTION 2: Agent Trace (shows after running)
#   - Step-by-step display of what the agent did
#   - Show: intent, confidence, priority, sentiment
#   - Show: retrieved KB docs (expandable)
#   - Show: drafted response
#   - Show: final action (send / escalate / flag)
#   - Color-code the final action (green/red/yellow)
#
# SECTION 3: Sidebar — Evaluation Dashboard
#   - Upload golden set CSV
#   - Button: "Run Evaluation"
#   - Show metrics table
#   - Button: "Run Ablation"
#   - Show ablation comparison table
#
# SECTION 4: Sidebar — Settings
#   - Confidence threshold slider
#   - LLM provider selector
#   - Model selector
