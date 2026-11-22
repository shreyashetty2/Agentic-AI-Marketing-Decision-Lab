"""Redemption Likelihood: plan a mailing for one campaign. Pick how deep to mail, see who is on the list, the
forecast with its honest range, and (back-test) what that list actually reached against the simple rule.

Numbers come from the agents' own outputs and measures (app/lib/plan.py reuses them; nothing is re-modelled
here). Scores are the model in use's forward-chained predictions on finished test campaigns."""
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.data import DEMOGRAPHIC_LABELS, load_model_table, load_predictions, load_run_summary, step, with_demographics  # noqa: E402
from lib.charts import _natural_key  # noqa: E402
from lib.plan import NO_RECORD, all_campaigns, campaign_overview, capture_curve, forecast_inflation, list_mix, plan  # noqa: E402
from lib.style import ACCENT, INK_MUTED, explainer, inject, status_badge  # noqa: E402

RULE_COLOR = "#2a78d6"   # palette slot 1 (blue): validated against the accent green, light mode, CVD-safe
REFERENCE = "#898781"    # muted ink for the random-mailing reference line
AXIS = alt.Axis(domain=False, tickColor=INK_MUTED, labelColor=INK_MUTED, gridColor="#e1e0d9")
# Short column headers for the list (the full Feature Agent meaning shows on hover).
SHORT = {"coupon_lines_per_week_26w": "Coupon lines / week", "coupon_lines_per_week_all": "Coupon lines / week (all history)",
         "coupon_commodities_26w": "Kinds of products bought with coupons", "coupon_trip_share_26w": "Trips with a coupon",
         "past_redemption_rate": "Past redemption rate", "past_redemptions": "Past redemptions",
         "campaign_product_spend_per_week_26w": "Campaign-product spend ($ / week)",
         "spend_per_week_26w": "Spend ($ / week)", "trips_per_week_26w": "Trips / week"}

st.set_page_config(page_title="Redemption Likelihood", layout="wide")
inject()
st.title("Redemption Likelihood")
explainer(1)

summary = load_run_summary()
modeling, evaluation = step(summary, "modeling", 1), step(summary, "evaluation", 1)
if modeling is None or evaluation is None:
    # The last Lead Agent run stopped before Model 1 was modeled and checked (e.g. an earlier step failed),
    # so there is nothing trustworthy to show yet.
    st.warning("Model 1 has no finished run to show yet. The last Lead Agent run stopped before Model 1 was "
               "modeled and checked" + (f": {summary['stopped_because']}" if summary.get("stopped_because") else ".")
               + "\n\nRun `python src/lead/engine.py --model all`, then reload this page.")
    st.stop()
facts = evaluation["facts"]

with st.container(border=True):
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.markdown(f"**Model in use:** {modeling['current_model']}, trained on "
                    f"{modeling['trained_on']['campaigns']} finished campaigns.")
        st.caption(f"Score: {facts['current_score']}. Vs. mailing the households that redeemed before: "
                   f"{facts['vs_simple_rule']}.")
        if modeling.get("leans_on"):
            st.caption("Relies most on: " + "; ".join(
                f"{c['meaning'].rstrip('.')} ({c['share']:.0%})" for c in modeling["leans_on"]))
    with col_b:
        st.markdown(status_badge(evaluation["verdict"]), unsafe_allow_html=True)

preds = load_predictions(1)
overview = campaign_overview(preds)
phis = forecast_inflation(preds)
st.caption(f"Scores below are a back-test: each of the {len(overview)} test campaigns was scored by the model "
           "trained only on campaigns that had finished before it, and its outcomes are known. So you can see "
           "what a list would actually have reached.")

# ---- plan a mailing -----------------------------------------------------------------------------------------
st.header("Plan a mailing")
labels = {r.campaign: f"Campaign {r.campaign}  ·  Type {r.campaign_type}  ·  {r.mailed:,} mailed"
          for r in overview.itertuples()}
c1, c2 = st.columns([2, 3])
with c1:
    campaign = st.selectbox("Campaign", list(labels), index=len(labels) - 1, format_func=labels.get,
                            help="Most recent test campaign first selected. Model 1 ranks households "
                                 "within one campaign's list, the way it is mailed.")
with c2:
    pct = st.slider("Mail the top % of this campaign's list", 5, 100, 20, 5, format="%d%%")
share = pct / 100

view = preds[preds["campaign"] == campaign]
p = plan(view, share, phis[campaign])

def tile(col, label: str, value: str, note: str, help: str | None = None) -> None:
    with col:
        st.metric(label, value, help=help)
        st.caption(note)


m1, m2, m3, m4 = st.columns(4)
tile(m1, "Households to mail", f"{p['k']:,}", f"of {p['n']:,} mailed in this campaign")
tile(m2, "Forecast", f"{p['forecast']:.0f}", f"redemptions, likely {p['low']:.0f} to {p['high']:.0f}",
     help="Sum of each listed household's chance of redeeming. The range is the Evaluation Agent's honest "
          "range, widened by how far earlier campaigns missed their forecasts.")
tile(m3, "Actually redeemed", f"{p['reached']} of {p['redeemers']}",
     "back-test: " + ("inside the forecast range" if p["low"] <= p["reached"] <= p["high"] else "outside the forecast range"))
if p["redeemers"]:
    tile(m4, "Redeemers reached", f"{p['capture']:.0%}",
         f"\"redeemed before\" rule: {p['capture_rule']:.0%}; random mailing: about {p['capture_random']:.0%}",
         help="Same list size for all three. Of everyone in this campaign who redeemed, the share on the list.")
else:
    tile(m4, "Redeemers reached", "n/a", "no redeemers in this campaign")

curve = capture_curve(view)
if not curve.empty:
    long = curve.melt("share", var_name="ranking", value_name="reached")
    order = ["Model in use", '"Redeemed before" rule', "Random mailing"]
    color = alt.Color("ranking:N", sort=order, title=None,
                      scale=alt.Scale(domain=order, range=[ACCENT, RULE_COLOR, REFERENCE]),
                      legend=alt.Legend(orient="top", labelColor=INK_MUTED))
    base = alt.Chart(long).encode(
        x=alt.X("share:Q", title="Share of this campaign's list mailed", axis=alt.Axis(format="%", **AXIS.to_dict()),
                scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("reached:Q", title="Share of redeemers reached", axis=alt.Axis(format="%", **AXIS.to_dict()),
                scale=alt.Scale(domain=[0, 1])),
        color=color,
        strokeDash=alt.StrokeDash("ranking:N", sort=order, legend=None,
                                  scale=alt.Scale(domain=order, range=[[1, 0], [1, 0], [4, 4]])),
        tooltip=[alt.Tooltip("ranking:N", title=" "), alt.Tooltip("share:Q", title="Mailed", format=".0%"),
                 alt.Tooltip("reached:Q", title="Redeemers reached", format=".0%")])
    lines = base.mark_line(strokeWidth=2) + base.mark_point(size=40, filled=True, opacity=0.9)
    marker = alt.Chart(pd.DataFrame({"share": [share]})).mark_rule(color=INK_MUTED, strokeDash=[2, 3]).encode(x="share:Q")
    st.altair_chart((lines + marker).properties(height=300).configure_view(strokeWidth=0)
                    .configure_axis(labelFontSize=11, titleFontSize=11, titleColor=INK_MUTED),
                    use_container_width=True)
    st.caption("Each point: mail the top part of this list and see what share of its real redeemers it reaches. "
               "The thin vertical line marks the depth chosen above.")

# ---- who is on the list ---------------------------------------------------------------------------------------
st.subheader(f"Who to mail: the top {pct}% of campaign {campaign}")
drivers = [c["column"] for c in modeling.get("leans_on", [])]
meanings = {c["column"]: c["meaning"].rstrip(".") for c in modeling.get("leans_on", [])}
headers = {c: SHORT.get(c, c) for c in drivers}
table = load_model_table(1)
listed = p["list"].merge(table[["household_key", "campaign", *drivers]], on=["household_key", "campaign"], how="left")
shown = pd.DataFrame({"Rank": listed["rank"], "Household": listed["household_key"],
                      "Chance of redeeming (%)": (100 * listed["score"]).round(1)})
for col in drivers:
    shown[headers[col]] = listed[col]
shown["Redeemed? (back-test)"] = listed["label"].map({1: "Yes", 0: "No"})
st.dataframe(shown, hide_index=True, use_container_width=True, height=320,
             column_config={"Chance of redeeming (%)": st.column_config.ProgressColumn(
                 format="%.0f%%", min_value=0, max_value=100)} | {
                 headers[c]: st.column_config.NumberColumn(format="%.2f", help=meanings[c]) for c in drivers})
st.download_button(f"Download this list ({p['k']:,} households, CSV)", shown.to_csv(index=False),
                   file_name=f"campaign_{campaign}_top_{pct}pct.csv", mime="text/csv")
if drivers:
    st.caption("The middle columns are what the model leans on most, so each household's score can be read "
               "from its own shopping history.")

# ---- who the list leaves out ------------------------------------------------------------------------------------
st.subheader("Who this list over- or under-represents")
options = {c: l for c, l in DEMOGRAPHIC_LABELS.items()}
group_col = st.selectbox("Compare by", list(options), format_func=options.get)
mix = list_mix(with_demographics(view), with_demographics(p["list"]), group_col)
long_mix = mix.melt("group", var_name="set", value_name="share")
sets = ["On this list", "Everyone mailed"]
mix_chart = alt.Chart(long_mix).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
    x=alt.X("group:N", title=None, sort=sorted(mix["group"], key=lambda g: (g == NO_RECORD, _natural_key(g))),
            axis=alt.Axis(labelAngle=0, labelLimit=140, **AXIS.to_dict())),
    xOffset=alt.XOffset("set:N", sort=sets),
    y=alt.Y("share:Q", title=None, axis=alt.Axis(format="%", **AXIS.to_dict())),
    color=alt.Color("set:N", sort=sets, title=None, scale=alt.Scale(domain=sets, range=[ACCENT, "#c3c2b7"]),
                    legend=alt.Legend(orient="top", labelColor=INK_MUTED)),
    tooltip=[alt.Tooltip("group:N", title="Group"), alt.Tooltip("set:N", title=" "),
             alt.Tooltip("share:Q", title="Share", format=".1%")])
st.altair_chart(mix_chart.properties(height=260).configure_view(strokeWidth=0).configure_axis(labelFontSize=11),
                use_container_width=True)
st.caption("The model does not use demographic labels; this only checks who the list leaves out. Households "
           "with no demographic record on file are their own group.")

# ---- every test campaign ------------------------------------------------------------------------------------------
with st.expander(f"All {len(overview)} test campaigns at the top {pct}%"):
    tbl, pooled = all_campaigns(preds, share)
    st.markdown(f"Mailing the top {pct}% of every campaign's list reaches **{pooled['model']:.1%}** of their "
                f"redeemers, against {pooled['rule']:.1%} for the \"redeemed before\" rule and "
                f"{pooled['random']:.1%} for random mailing. Forecast {pooled['forecast']:.0f} redemptions, "
                f"actual {pooled['reached']}; the actual count fell inside the forecast range for "
                f"{tbl['in_range'].sum()} of {len(tbl)} campaigns.")
    st.dataframe(pd.DataFrame({
        "Campaign": tbl["campaign"], "Type": tbl["type"], "Mailed": tbl["mailed"], "On the list": tbl["list"],
        "Forecast": tbl["forecast"].round(1),
        "Likely range": [f"{lo:.0f} to {hi:.0f}" for lo, hi in zip(tbl["low"], tbl["high"])],
        "Actually redeemed": tbl["reached"], "All redeemers": tbl["redeemers"]}),
        hide_index=True, use_container_width=True)

st.caption("Have a question about these results? Ask the Decision Lab, in the sidebar.")
