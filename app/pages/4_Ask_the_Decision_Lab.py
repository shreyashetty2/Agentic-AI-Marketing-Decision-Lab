"""A marketer-facing chat over all three models, grounded only in already-computed facts."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.chat import ask, credentials_ok  # noqa: E402
from lib.style import inject  # noqa: E402

st.set_page_config(page_title="Ask the Decision Lab", layout="wide")
inject()

col_title, col_clear = st.columns([5, 1])
col_title.title("Ask the Decision Lab")

st.caption(
    "Ask about who to target, which segments respond best, what to expect to spend, how confident a "
    "result is, or about a specific household (e.g. \"what about household 323?\"). Answers are "
    "grounded only in numbers the Modeling and Evaluation Agents already computed for Model 1 "
    "(redemption likelihood), Model 2 (spend forecast), and Model 3 (uplift and sure things) -- it "
    "will say so rather than guess at anything it wasn't given."
)

if not credentials_ok():
    st.warning(
        "No Google Cloud credentials found. Run `gcloud auth application-default login` once "
        "(same login used to deploy this project to Cloud Run) and reload this page. Answers run "
        "on Vertex AI (Gemini), billed to this project's own GCP credits."
    )
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_contexts" not in st.session_state:
    st.session_state.chat_contexts = {}

if col_clear.button("Clear", use_container_width=True, disabled=not st.session_state.chat_history):
    st.session_state.chat_history = []
    st.session_state.chat_contexts = {}
    st.rerun()

for i, turn in enumerate(st.session_state.chat_history):
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and i in st.session_state.chat_contexts:
            with st.expander("See the facts used for this answer"):
                st.text(st.session_state.chat_contexts[i])

examples = [
    "Which households should we target with Model 3?",
    "How much should we expect a mailed household to spend, per Model 2?",
    "Which population segment redeems the most, per Model 1?",
]
if not st.session_state.chat_history:
    st.caption("Try asking:")
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

question = st.chat_input("Ask about Model 1, 2, or 3...") or st.session_state.pop("pending", None)
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.spinner("Checking the numbers..."):
        try:
            answer, context_used = ask(st.session_state.chat_history[:-1], question)
        except Exception as e:
            answer, context_used = f"Couldn't reach the model ({type(e).__name__}: {e}).", None
    answer_index = len(st.session_state.chat_history)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    if context_used:
        st.session_state.chat_contexts[answer_index] = context_used
    # A fresh rerun, not just appending in place: the Clear button's `disabled` state and the
    # examples' visibility above are already rendered by the time this block runs, so without a
    # rerun they'd show stale (pre-question) state until some other interaction forced a new pass.
    st.rerun()
