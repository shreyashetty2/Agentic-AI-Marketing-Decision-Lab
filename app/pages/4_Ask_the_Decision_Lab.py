"""A marketer-facing chat over all three models. Gemini reads the question; tested code computes every number."""
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
    "Ask who to mail, what a group of households would do, which products a group redeems coupons on, about one "
    "household or campaign, or how to plan a new campaign by comparing it to similar past ones. Every number is "
    "worked out by the Lab's own tested calculations on the three tools' scores -- Redemption Likelihood, Spend "
    "Forecast and Campaign Uplift -- or comes from the team's own results. If a question is unclear it asks; if the "
    "data can't answer it, it says so instead of guessing. (This data has no real calendar dates, so it can't say "
    "which month or season is best to mail.)"
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
if "chat_work" not in st.session_state:
    st.session_state.chat_work = {}   # answer index -> (how it was worked out, downloadable list or None)

if col_clear.button("Clear", use_container_width=True, disabled=not st.session_state.chat_history):
    st.session_state.chat_history = []
    st.session_state.chat_work = {}
    st.rerun()


def worked_out(turn) -> str:
    lines = []
    for name, form, result in turn.calls:
        lines.append(f"{name}({', '.join(f'{k}={v}' for k, v in form.items())}) -> {result.status}")
        lines += [f"  step: {s}" for s in result.steps]
        lines += [f"  assumed: {a}" for a in result.assumptions]
        lines += [f"  warning: {w}" for w in result.warnings]
        if result.message:
            lines.append(f"  said: {result.message}")
    return "\n".join(lines) or "Answered from the team's background facts; no calculation was needed."


for i, turn in enumerate(st.session_state.chat_history):
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and i in st.session_state.chat_work:
            how, households = st.session_state.chat_work[i]
            if households is not None:
                st.download_button(f"Download this list ({len(households):,} households)",
                                   households.to_csv(index=False), file_name="decision_lab_list.csv",
                                   mime="text/csv", key=f"download_{i}")
            with st.expander("How this was worked out"):
                st.text(how)

examples = [
    "Which 50 households are most likely to use the coupon in campaign 16?",
    "What percent of coupons actually get redeemed?",
    "Which products do people buy without using the coupon?",
    "What does campaign Type A mean?",
]
if not st.session_state.chat_history:
    st.caption("Try asking:")
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

question = st.chat_input("Ask about who to mail, a group of households, or a product...") \
    or st.session_state.pop("pending", None)
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.spinner("Working it out..."):
        try:
            turn = ask(st.session_state.chat_history[:-1], question)
            answer, work = turn.answer, (worked_out(turn), turn.households)
        except Exception as e:
            answer, work = f"Couldn't reach the model ({type(e).__name__}: {e}).", None
    answer_index = len(st.session_state.chat_history)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    if work:
        st.session_state.chat_work[answer_index] = work
    # A fresh rerun, not just appending in place: the Clear button's `disabled` state and the
    # examples' visibility above are already rendered by the time this block runs, so without a
    # rerun they'd show stale (pre-question) state until some other interaction forced a new pass.
    st.rerun()
