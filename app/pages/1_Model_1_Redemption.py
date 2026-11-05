"""Model 1: who is likely to redeem a coupon, so a marketer can mail the most likely first."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.charts import population_breakdown, score_histogram  # noqa: E402
from lib.data import load_predictions, load_run_summary, step, with_demographics  # noqa: E402
from lib.style import inject  # noqa: E402

st.set_page_config(page_title="Model 1: Redemption", layout="wide")
inject()
st.title("Model 1: Redemption likelihood")

summary = load_run_summary()
modeling, evaluation = step(summary, "modeling", 1), step(summary, "evaluation", 1)
facts = evaluation["facts"]

st.write(f"**{modeling['current_model']}**, trained on {modeling['trained_on']['campaigns']} finished "
         f"campaigns. Score: {facts['current_score']}. Vs. the simple rule of mailing the households "
         f"that redeemed most before: {facts['vs_simple_rule']}.")

preds = with_demographics(load_predictions(1))
campaign = st.selectbox("Campaign", ["All campaigns"] + sorted(preds["campaign"].unique().tolist()))
view = preds if campaign == "All campaigns" else preds[preds["campaign"] == campaign]

with st.container(border=True):
    st.subheader(f"{len(view):,} mailed households in view")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.caption("Redemption-likelihood score")
        score_histogram(view, "score")
    with col2:
        st.caption("Most likely to redeem")
        top = view.nlargest(20, "score")[["household_key", "campaign", "score"]]
        top["score"] = top["score"].round(3)
        st.dataframe(top, hide_index=True, use_container_width=True)

with st.container(border=True):
    st.subheader("By population segment")
    population_breakdown(view, "score", "redemption-likelihood score")
