"""Decision Lab: a marketer-facing view of the Lead Agent's latest run. Reads only
data/processed/lead_agent/run_summary.json and the agents' own output files -- never computes a number
itself. Run from the repo root: streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.data import MODEL_BLURBS, load_run_summary, step  # noqa: E402
from lib.style import inject, status_badge  # noqa: E402

st.set_page_config(page_title="Decision Lab", page_icon=":bar_chart:", layout="wide")
inject()

summary = load_run_summary()

st.title("Decision Lab")
st.caption(f"Last run: {summary['run_at']} — status: **{summary['status']}**"
           + (f" ({summary['stopped_because']})" if summary.get("stopped_because") else ""))

st.write(
    "One engine runs the Data, Feature, Modeling and Evaluation Agents for all three models, then this "
    "page shows what it found. Nothing here is computed by the page itself — every number comes "
    "straight from that run. Use the sidebar for each model's full detail: population segments, who "
    "it would target, and why -- or ask the Decision Lab directly."
)

for model in (1, 2, 3):
    modeling = step(summary, "modeling", model)
    evaluation = step(summary, "evaluation", model)
    if modeling is None or evaluation is None:
        continue

    with st.container(border=True):
        st.subheader(f"Model {model}")
        st.caption(MODEL_BLURBS[model])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**The model**")
            st.markdown(modeling["current_model"])
            st.caption(f"Trained on {modeling['trained_on']['campaigns']} finished campaigns "
                       f"({modeling['trained_on']['rows']:,} rows)")
            if modeling.get("leans_on"):
                st.caption("Relies most on: " + "; ".join(
                    f"{c['meaning']} ({c['share']:.0%})" for c in modeling["leans_on"]))
            in_use = modeling.get("in_use")
            if isinstance(in_use, dict):
                st.info(f"Using the idea **\"{in_use.get('title')}\"**, chosen by {in_use['decided_by']} "
                        f"on {in_use['decided_on']}.\n\n{in_use.get('why', '')}")

        with col2:
            st.markdown("**How it's scoring**")
            facts = evaluation["facts"]
            st.markdown(status_badge(evaluation["verdict"]), unsafe_allow_html=True)
            st.caption(f"Score: {facts['current_score']}")
            st.caption(f"Vs. the simple rule: {facts['vs_simple_rule']}")
            st.caption(facts["forecast_check"])
            if evaluation.get("handled"):
                st.caption(evaluation["handled"])

st.caption("Pages: Model 1 (redemption), Model 2 (spend forecast), Model 3 (uplift, persuadables and "
           "population breakdowns), and Ask the Decision Lab (chat) — in the sidebar.")
