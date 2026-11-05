"""Model 3: who mailing actually persuades (uplift), plus the sure-things classifier that flags likely
wasted sends among households who were mailed."""
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.charts import population_breakdown, score_histogram  # noqa: E402
from lib.data import load_predictions, load_run_summary, load_sure_things, step, with_demographics  # noqa: E402

st.set_page_config(page_title="Model 3: Uplift", layout="wide")
st.title("Model 3: Uplift and persuadables")

summary = load_run_summary()
modeling, evaluation = step(summary, "modeling", 3), step(summary, "evaluation", 3)
facts = evaluation["facts"]

st.write(f"**{modeling['current_model']}**, trained on {modeling['trained_on']['campaigns']} finished "
         f"campaigns. Score: {facts['current_score']}. Vs. mailing at random: {facts['vs_simple_rule']}.")
st.caption(facts["forecast_check"])
in_use = modeling.get("in_use")
if isinstance(in_use, dict):
    st.info(f"In use: **\"{in_use.get('title')}\"**, chosen by {in_use['decided_by']} on "
            f"{in_use['decided_on']}.\n\n{in_use.get('why', '')}")

st.subheader(f":{'orange' if evaluation['verdict'] == 'human_needed' else 'green'}"
             f"[Verdict: {evaluation['verdict'].replace('_', ' ')}]")
if evaluation.get("handled"):
    st.caption(evaluation["handled"])

preds = with_demographics(load_predictions(3))
campaign = st.selectbox("Campaign", ["All campaigns"] + sorted(preds["campaign"].unique().tolist()))
view = preds if campaign == "All campaigns" else preds[preds["campaign"] == campaign]

st.header("Who the mailer persuades")
st.caption(f"{len(view):,} mailed households in view. Uplift score: how many extra buyers this "
           "household is estimated to contribute if mailed, vs. not mailing them.")
col1, col2 = st.columns([1, 2])
with col1:
    st.caption("Uplift score distribution")
    score_histogram(view, "score")
with col2:
    st.caption("Most persuadable (highest uplift)")
    top = view.nlargest(20, "score")[["household_key", "campaign", "score"]]
    top["score"] = top["score"].round(4)
    st.dataframe(top, hide_index=True, use_container_width=True)

st.subheader("By population segment")
population_breakdown(view, "score", "uplift score")

st.header("Sure things: likely-wasted sends")
st.caption("A separate tool, not a competing uplift model: trained only on households that were "
           "actually mailed, it flags who would likely have bought the campaign product anyway, "
           "coupon or not -- useful for trimming a mailing list even where the uplift model above "
           "is still inconclusive.")
sure = load_sure_things()
m1, m2, m3 = st.columns(3)
m1.metric("AUC", f"{sure['auc']:.3f}")
m2.metric("Top-fifth capture rate", f"{sure['top_fifth_rate']:.1%}",
          f"{sure['top_fifth_rate'] - sure['base_rate']:+.1%} vs. base rate")
m3.metric("Base rate", f"{sure['base_rate']:.1%}")
st.caption("What it relies on most:")
importances = pd.Series(sure["top_features"], name="importance").sort_values(ascending=False)
imp_df = importances.rename_axis("feature").reset_index()
chart = alt.Chart(imp_df).mark_bar().encode(
    x=alt.X("importance", title=None),
    y=alt.Y("feature", sort=imp_df["feature"].tolist(), title=None))
st.altair_chart(chart, use_container_width=True)
