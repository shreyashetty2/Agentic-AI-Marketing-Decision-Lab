"""Shared loaders for the Decision Lab interface. Reads only what the Lead Agent engine already wrote --
never recomputes a number itself (same rule the lead-agent skill follows)."""
from pathlib import Path

import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data/processed"

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
    """One row per household x campaign, kept to the candidate actually used for scoring (falls back to
    the owner's own default when no candidate has been chosen yet)."""
    df = pd.read_parquet(PROCESSED / f"modeling_agent/model{model}/predictions.parquet")
    chosen = df[df["chosen"]] if "chosen" in df.columns and df["chosen"].any() else df[df["is_default"]]
    return chosen.drop_duplicates(["household_key", "campaign"])


@st.cache_data(ttl=60)
def load_demographics() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "hh_demographic.parquet")


def with_demographics(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.merge(load_demographics(), on="household_key", how="left")


@st.cache_data(ttl=60)
def load_sure_things() -> dict:
    return json.loads((PROCESSED / "model3_sure_things.json").read_text())


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
