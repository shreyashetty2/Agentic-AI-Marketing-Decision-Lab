"""A marketer-facing chat over Model 1 and Model 3, grounded only in already-computed facts."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.chat import ask, credentials_ok  # noqa: E402
from lib.style import inject  # noqa: E402

st.set_page_config(page_title="Ask the Decision Lab", layout="wide")
inject()
st.title("Ask the Decision Lab")
st.caption(
    "Ask about who to target, which segments respond best, or how confident a result is. Answers are "
    "grounded only in numbers the Modeling and Evaluation Agents already computed for Model 1 "
    "(redemption likelihood) and Model 3 (uplift and sure things) -- it will say so rather than guess "
    "at anything it wasn't given."
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

for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

examples = [
    "Which households should we target with Model 3?",
    "How confident are we in the Model 3 uplift result?",
    "Which population segment redeems the most, per Model 1?",
]
if not st.session_state.chat_history:
    st.caption("Try asking:")
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

question = st.chat_input("Ask about Model 1 or Model 3...") or st.session_state.pop("pending", None)
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Checking the numbers..."):
            try:
                answer = ask(st.session_state.chat_history[:-1], question)
            except Exception as e:
                answer = f"Couldn't reach the model ({type(e).__name__}: {e})."
        st.markdown(answer)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
