"""Shared loaders for the Decision Lab interface. Reads only what the Lead Agent engine already wrote --
never recomputes a number itself (same rule the lead-agent skill follows)."""
from pathlib import Path

import json
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data/processed"
sys.path.insert(0, str(ROOT / "src"))   # the Lab's engine (src/decision) and the models it loads (src/modeling)
from decision.table import in_use_rows  # noqa: E402

MODEL_NAMES = {
    1: "Redemption Likelihood",
    2: "Spend Forecast",
    3: "Campaign Uplift",
}

MODEL_BLURBS = {
    1: "Ranks mailed households by how likely they are to redeem a coupon, so a marketer can mail the "
       "most likely first and forecast redemptions.",
    2: "Forecasts the dollars a mailed household will spend on the campaign's products, so the Decision "
       "Lab can weigh sales against the cost of the discounts.",
    3: "Estimates how much mailing changes a household's buying, so a marketer can mail the households "
       "the mailer persuades, not the ones who would have bought anyway.",
}


@st.cache_data(ttl=60)
def load_run_summary() -> dict:
    return json.loads((PROCESSED / "lead_agent/run_summary.json").read_text())


def step(summary: dict, name: str, model: int) -> dict | None:
    return next((s for s in summary["steps"] if s["step"] == f"{name}:model{model}"), None)


@st.cache_data(ttl=60)
def load_predictions(model: int) -> pd.DataFrame:
    """One row per household x campaign, from the model in use only -- the rows marked is_default, which the
    Modeling Agent sets from the owner's default or src/lead/models_in_use.yaml. Not `chosen`: that marks the
    agent's per-campaign experimental pick, which mixed seven different models across Model 1's 13 campaigns."""
    return in_use_rows(model, PROCESSED / "modeling_agent")


@st.cache_data(ttl=60)
def load_model_table(model: int) -> pd.DataFrame:
    """The Feature Agent's table for one model (the household x campaign columns the model learns from)."""
    return pd.read_parquet(PROCESSED / f"model{model}_table.parquet")


@st.cache_data(ttl=60)
def load_demographics() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "hh_demographic.parquet")


def with_demographics(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.merge(load_demographics(), on="household_key", how="left")


@st.cache_data(ttl=60)
def load_sure_things() -> dict:
    return json.loads((PROCESSED / "model3_sure_things.json").read_text())


@st.cache_data(ttl=60)
def load_campaign_desc() -> pd.DataFrame:
    """Real campaign metadata (not a model output): each campaign's type (A/B/C) and how many days it
    ran. The Type label itself has a real, documented meaning (see docs/agent_design/data_agent_design.md
    section 4.6, confirmed against coupon.parquet's own coupon counts): TypeA campaigns mail a *pool* of
    ~180-210 possible coupons and each household gets 16 chosen from it (which 16 is not recorded), so a
    TypeA campaign's product list is broad/storewide, not any one household's actual mailing. TypeB/C
    campaigns are narrower (TypeB 2-33 coupons, TypeC 1-34) and every mailed household gets ALL of that
    campaign's coupons, so their product list is exactly what that household received. START_DAY/END_DAY
    are the dataset's own anonymised, sequential day-numbers, not real calendar dates -- there is no month,
    season, or actual date anywhere in this dataset (checked: neither campaign_desc nor transaction_data
    carries one), so duration and type are the only real facts about timing we have."""
    df = pd.read_parquet(PROCESSED / "campaign_desc.parquet")
    df = df.rename(columns={"CAMPAIGN": "campaign", "DESCRIPTION": "campaign_type"})
    # Raw file spells this "TypeA"/"TypeB"/"TypeC"; the modeling agent's own predictions already
    # normalise it to a bare "A"/"B"/"C" -- match that so the chatbot never sees two spellings of
    # the same thing.
    df["campaign_type"] = df["campaign_type"].str.removeprefix("Type")
    df["duration_days"] = df["END_DAY"] - df["START_DAY"]
    return df[["campaign", "campaign_type", "duration_days"]]


@st.cache_data(ttl=60)
def load_model_features(model: int) -> list[str]:
    """The literal input columns of the model actually in use (not a generic count) -- straight from
    the Modeling Agent's own manifest, the candidate it marked "in_use" (or its own "default" when
    nobody has picked one yet, same fallback load_predictions() uses)."""
    manifest = json.loads((PROCESSED / f"modeling_agent/model{model}/manifest.json").read_text())
    in_use = manifest.get("in_use")
    candidate_id = in_use["id"] if isinstance(in_use, dict) else manifest["default"]
    return manifest["candidates"][candidate_id]["features"]


@st.cache_data(ttl=60)
def load_campaign_products() -> pd.DataFrame:
    """Real product categories tied to each campaign's coupons (coupon.parquet links campaign -> product,
    product.parquet names its real department/commodity/brand) -- not a model output, just what the
    campaign actually covers. Keeps both the fine-grained commodity and the coarser department, so a
    broad/storewide campaign (see load_campaign_desc's Type explanation) can be summarised at the
    department level instead of listing 100+ commodities."""
    coupon = pd.read_parquet(PROCESSED / "coupon.parquet")
    product = pd.read_parquet(PROCESSED / "product.parquet")
    merged = coupon.merge(product, on="PRODUCT_ID", how="left")
    return merged.rename(columns={
        "CAMPAIGN": "campaign", "COMMODITY_DESC": "commodity", "DEPARTMENT": "department",
    })[["campaign", "department", "commodity"]].dropna(subset=["commodity"])


@st.cache_data(ttl=60)
def load_product_redemption_summary() -> pd.DataFrame:
    """Real products, already analysed by Model 3's own product work (model3_product_redemption_summary.csv):
    for each product seen across campaigns, how often a household that got the coupon never redeemed it."""
    return pd.read_csv(PROCESSED / "model3_product_redemption_summary.csv")


@st.cache_data(ttl=60)
def load_dataset_facts() -> dict:
    """Real, countable facts about the mailing data itself (not a model output): how many households
    were ever mailed, and how many mailings were actually redeemed, overall and by campaign Type. Read
    straight from campaign_table (who was mailed which campaign) and coupon_redempt (who redeemed),
    both small files; a mailing counts as redeemed if that household redeemed any coupon in that
    campaign. Matches the 889 of 7,208 (about 1 in 8) the team already reports."""
    mailed = pd.read_parquet(PROCESSED / "campaign_table.parquet")
    redeemed = pd.read_parquet(PROCESSED / "coupon_redempt.parquet")[["household_key", "CAMPAIGN"]].drop_duplicates()
    redeemed = redeemed.assign(redeemed=1)
    m = mailed.merge(redeemed, on=["household_key", "CAMPAIGN"], how="left").fillna({"redeemed": 0})
    m["campaign_type"] = m["DESCRIPTION"].str.removeprefix("Type")
    by_type = m.groupby("campaign_type")["redeemed"].agg(["sum", "count"])
    return {
        "mailings": int(len(m)), "households_mailed": int(m["household_key"].nunique()),
        "campaigns": int(m["CAMPAIGN"].nunique()), "redeemed": int(m["redeemed"].sum()),
        "by_type": {t: (int(r["sum"]), int(r["count"])) for t, r in by_type.iterrows()},
    }


@st.cache_data
def load_type_a_product_example() -> pd.DataFrame:
    """The team's own product drill-down for campaign 8 (src/modeling/model3_product_lookup.py
    --campaign 8), top 15 products by purchase volume, saved as a small reference file because the
    drill-down itself needs the 2.6M-row transaction table, which the hosted app never loads. Campaign
    8 is Type A, so these per-product results are an illustration, not individually reliable."""
    return pd.read_csv(Path(__file__).resolve().parent / "ref" / "campaign8_product_example.csv")


# Exact wording from src/features/feature_2a_spec.yaml -- the team anonymised and deliberately did not
# claim what classification_2/3/4/5 represent (e.g. never assumed classification_3 means income), so
# this page doesn't either.
DEMOGRAPHIC_LABELS = {
    "classification_1": "Demographic label (anonymised; ordered age groups)",
    "classification_2": "Demographic label (anonymised; X / Y / Z)",
    "classification_3": "Demographic label (anonymised; ordered levels 1-12)",
    "classification_4": "Demographic label (anonymised; 1 to 5+)",
    "classification_5": "Demographic label (anonymised; ordered groups 1-6)",
    "homeowner_desc": "Homeowner status",
    "kid_category_desc": "Kids in the household",
}
