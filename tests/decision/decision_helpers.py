"""Shared helpers for the Decision Lab tests. A unique module name, not `conftest`, so a test can import it
without picking up another test folder's conftest."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


PROCESSED = ROOT / "data/processed"
needs_data = pytest.mark.skipif(not (PROCESSED / "modeling_agent/model1/predictions.parquet").exists(),
                                reason="needs the agents' outputs in data/processed")


def toy_table(n: int = 40) -> pd.DataFrame:
    """Campaign 1: n mailed households 1..n, scored so household 1 is best. Every 4th redeemed. Households 1..10
    are homeowners with 1 kid, 11..20 renters, the rest have no demographic record. Campaign 2 is only scored by
    Campaign Uplift (Redemption Likelihood doesn't cover it)."""
    keys = np.arange(1, n + 1)
    c1 = pd.DataFrame({
        "household_key": keys, "campaign": 1, "mailed": True,
        "redeemed": (keys % 4 == 0).astype(float),
        "p_redeem": 1 - keys / 100, "spend_if_redeem": 10.0 + keys, "uplift": keys / 1000, "buy_anyway": 0.5,
        "bought": 0.0, "campaign_type": "B", "start_day": 100, "end_day": 129, "length_days": 30,
    })
    c1["expected_revenue"] = c1["p_redeem"] * c1["spend_if_redeem"]
    c2 = c1.assign(campaign=2, redeemed=np.nan, p_redeem=np.nan, spend_if_redeem=np.nan, expected_revenue=np.nan,
                   mailed=False)
    t = pd.concat([c1, c2], ignore_index=True)
    t["homeowner_desc"] = np.where(t["household_key"] <= 10, "Homeowner",
                                   np.where(t["household_key"] <= 20, "Renter", None))
    t["kid_category_desc"] = np.where(t["household_key"] <= 10, "1", np.where(t["household_key"] <= 20, "None/Unknown", None))
    for c in ("classification_1", "classification_2", "classification_3", "classification_4", "classification_5"):
        t[c] = np.where(t["household_key"] <= 20, {"classification_1": "Age Group1", "classification_2": "X",
                                                   "classification_3": "Level1", "classification_4": "1",
                                                   "classification_5": "Group1"}[c], None)
    t["has_demographics"] = t["homeowner_desc"].notna()
    return t


def toy_products() -> pd.DataFrame:
    return pd.DataFrame({"campaign": [1, 1, 2, 3], "category": ["YOGURT", "BEEF", "YOGURT", "BEEF"],
                         "products": [5, 2, 3, 1]})


def toy_redemptions() -> pd.DataFrame:
    """Households 1-5 (homeowners) redeemed YOGURT, household 25 (no record) did too; 11 (renter) redeemed BEEF."""
    rows = [(k, "YOGURT", 1) for k in (1, 2, 3, 4, 5, 25)] + [(11, "BEEF", 2), (12, "NO COMMODITY DESCRIPTION", 1)]
    return pd.DataFrame(rows, columns=["household_key", "category", "redemptions"])
