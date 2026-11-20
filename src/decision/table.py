"""The Decision Lab's one clean table: one row per household per campaign, holding every model's score from
the model actually in use, the household's traits and the campaign's details. Every question the Lab answers
reads this table and nothing else (docs/agent_design/decision_lab_design.md).

What it reads, all already written by the agents:
- each model's predictions.parquet -- the test campaigns only, each scored by a model trained on campaigns that
  ended before it. Held-back campaigns (locked_predictions.parquet) are never read; they stay a fair test.
- only the rows marked as the model in use (is_default -- the Modeling Agent sets it from the owner's default or
  src/lead/models_in_use.yaml), never the agent's per-campaign experimental picks (chosen).
- Model 2 only scores households that redeemed, so its fitted model in use is applied here to every mailed
  household's usual spend (Model 2's one input), giving spend-if-they-redeem for all of them.
"""
import json
import re
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data/processed"

KEY = ["household_key", "campaign"]


class TableError(RuntimeError):
    """The agents' outputs don't fit together; the Lab must not answer from them."""


def in_use_rows(model: int, modeling_dir: Path) -> pd.DataFrame:
    """One model's test-campaign rows, from the model in use only, one row per household per campaign."""
    path = modeling_dir / f"model{model}/predictions.parquet"
    df = pd.read_parquet(path)
    if df["locked"].any():
        raise TableError(f"{path} holds held-back campaigns; the Lab never reads those")
    rows = df[df["is_default"]]
    if rows.empty:
        raise TableError(f"no rows marked as the model in use in {path}")
    if rows["candidate_id"].nunique() != 1:
        raise TableError(f"model {model}: more than one model marked in use: {sorted(rows['candidate_id'].unique())}")
    if rows.duplicated(KEY).any():
        raise TableError(f"model {model}: a household is scored more than once in the same campaign")
    return rows


def usual_spend(rows: pd.DataFrame) -> pd.Series:
    """Model 2's one input, same formula as src/modeling/model2_baseline.py: weekly spend on the campaign's
    products over the 26 weeks before it, times the campaign's length in weeks."""
    return rows["campaign_product_spend_per_week_26w"] * rows["duration_days"] / 7


def spend_if_redeem(m1: pd.DataFrame, processed: Path, modeling_dir: Path) -> pd.Series:
    """Model 2's in-use model applied to every household Model 1 scores (mailed, test campaigns), but only in
    campaigns Model 2 was built for: its table leaves out Type A campaigns, where one coupon book spans hundreds
    of categories and the line on usual spend stretches far past anything it learned from (up to ~$750).
    Blank elsewhere, so the Lab says Spend Forecast doesn't cover that campaign instead of extrapolating."""
    final = joblib.load(modeling_dir / "model2/final_model.joblib")
    in_use = in_use_rows(2, modeling_dir)["candidate_id"].iloc[0]
    if final["candidate_id"] != in_use:
        raise TableError(f"Model 2's saved model is `{final['candidate_id']}` but the model in use is `{in_use}`")
    if final["fitted"].features != ["usual_spend"]:
        raise TableError(f"Model 2's model in use needs {final['fitted'].features}; the Lab only supplies usual_spend")
    inputs = pd.read_parquet(processed / "model1_table.parquet",
                             columns=[*KEY, "campaign_product_spend_per_week_26w", "duration_days"])
    rows = m1[KEY].merge(inputs, on=KEY, how="left", validate="one_to_one")
    if rows["duration_days"].isna().any():
        raise TableError("some mailed households have no usual spend to forecast from")
    rows["usual_spend"] = usual_spend(rows)
    spend = pd.Series(final["fitted"].predict(rows)["score"], index=m1.index)
    covered = set(pd.read_parquet(processed / "model2_table.parquet", columns=["campaign"])["campaign"])
    return spend.where(m1["campaign"].isin(covered).to_numpy())


def campaigns(processed: Path) -> pd.DataFrame:
    """Every campaign's type and timing. Days are the dataset's own day numbers (1-711); it has no calendar dates."""
    desc = pd.read_parquet(processed / "campaign_desc.parquet")
    return pd.DataFrame({"campaign": desc["CAMPAIGN"], "campaign_type": desc["DESCRIPTION"].str.removeprefix("Type"),
                         "start_day": desc["START_DAY"], "end_day": desc["END_DAY"],
                         "length_days": desc["END_DAY"] - desc["START_DAY"] + 1}).sort_values("campaign")


def campaign_products(processed: Path) -> pd.DataFrame:
    """For each campaign, how many of its coupon products fall in each product category, most first."""
    coupons = pd.read_parquet(processed / "coupon.parquet").drop_duplicates(["CAMPAIGN", "PRODUCT_ID"])
    products = pd.read_parquet(processed / "product.parquet", columns=["PRODUCT_ID", "COMMODITY_DESC"])
    joined = coupons.merge(products, on="PRODUCT_ID", how="inner")
    out = joined.groupby(["CAMPAIGN", "COMMODITY_DESC"]).size().rename("products").reset_index()
    out = out.rename(columns={"CAMPAIGN": "campaign", "COMMODITY_DESC": "category"})
    return out.sort_values(["campaign", "products", "category"], ascending=[True, False, True], ignore_index=True)


def category_redemptions(processed: Path) -> pd.DataFrame:
    """One row per household per product category it ever redeemed a coupon in (any campaign), with how many
    coupon redemptions that was. A redeemed coupon counts toward every category its products fall in."""
    redeemed = pd.read_parquet(processed / "coupon_redempt.parquet")
    coupons = pd.read_parquet(processed / "coupon.parquet")
    products = pd.read_parquet(processed / "product.parquet", columns=["PRODUCT_ID", "COMMODITY_DESC"])
    cats = coupons.merge(products, on="PRODUCT_ID")[["COUPON_UPC", "CAMPAIGN", "COMMODITY_DESC"]].drop_duplicates()
    joined = redeemed.merge(cats, on=["COUPON_UPC", "CAMPAIGN"])
    joined = joined.drop_duplicates(["household_key", "DAY", "COUPON_UPC", "CAMPAIGN", "COMMODITY_DESC"])
    out = joined.groupby(["household_key", "COMMODITY_DESC"]).size().rename("redemptions").reset_index()
    return out.rename(columns={"COMMODITY_DESC": "category"})


def build(processed: Path = PROCESSED, modeling_dir: Path | None = None) -> pd.DataFrame:
    """The table. Rows: every household-campaign any model scores. A score is blank where its model doesn't
    cover that household or campaign -- Redemption Likelihood and Spend Forecast score mailed households only;
    Campaign Uplift scores every household, mailed or not, in the campaigns it covers."""
    modeling_dir = modeling_dir or processed / "modeling_agent"
    m1 = in_use_rows(1, modeling_dir)
    m1 = m1.assign(spend_if_redeem=spend_if_redeem(m1, processed, modeling_dir))
    m1 = m1[[*KEY, "label", "score", "spend_if_redeem"]].rename(columns={"label": "redeemed", "score": "p_redeem"})
    m3 = in_use_rows(3, modeling_dir)[[*KEY, "treatment", "label", "score", "p_control"]]
    m3 = m3.rename(columns={"treatment": "mailed", "label": "bought", "score": "uplift", "p_control": "buy_anyway"})

    table = m3.merge(m1, on=KEY, how="outer", validate="one_to_one", indicator=True)
    if ((table["_merge"] == "both") & (table["mailed"] != 1)).any():
        raise TableError("Redemption Likelihood scores a household that Campaign Uplift says was not mailed")
    table["mailed"] = table["mailed"].fillna(1).astype(bool)   # Model 1-only rows are mailed households
    table = table.drop(columns="_merge")
    table["expected_revenue"] = table["p_redeem"] * table["spend_if_redeem"]

    table = table.merge(campaigns(processed), on="campaign", how="left", validate="many_to_one")
    demo = pd.read_parquet(processed / "hh_demographic.parquet")
    table = table.merge(demo, on="household_key", how="left", validate="many_to_one")
    table["has_demographics"] = table["homeowner_desc"].notna()
    return table.sort_values(KEY, ignore_index=True)


def models_in_use(modeling_dir: Path = PROCESSED / "modeling_agent") -> dict[int, str]:
    return {m: in_use_rows(m, modeling_dir)["candidate_id"].iloc[0] for m in (1, 2, 3)}


def cautions(processed: Path = PROCESSED) -> dict[int, str]:
    """For each model the team's Evaluation Agent hasn't approved, a plain warning built from its own verdict
    (lead_agent/run_summary.json), so the Lab never presents those numbers as settled."""
    summary = json.loads((processed / "lead_agent/run_summary.json").read_text())
    names = {1: "Redemption Likelihood", 2: "Spend Forecast", 3: "Campaign Uplift"}
    out = {}
    for step in summary["steps"]:
        if not step["step"].startswith("evaluation:model") or step.get("verdict") == "approve":
            continue
        model = int(step["step"].removeprefix("evaluation:model"))
        facts = step.get("facts") or {}
        reason = re.sub(r"^[A-Z]{3}-\d+: ", "", step.get("handled") or "")   # drop the check's code, e.g. "EVA-02: "
        if reason.startswith("no rule covers"):                            # the agent's own wording, not a reason
            reason = ""
        out[model] = (f"{names[model]} hasn't been signed off by the team's checks yet"
                      + (f": {reason}" if reason else "")
                      + (f". Compared with a simple rule it scores {facts['vs_simple_rule']}" if facts.get("vs_simple_rule") else "")
                      + ". Treat its numbers as a rough guide.")
    return out

