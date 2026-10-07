"""Profile columns: what each household did BEFORE its row's cutoff_day.

Every value here is computed from rows with day < cutoff_day only. compute_profiles() is the one
implementation; features() is the same thing for a single household, used for scoring.
"""
import numpy as np
import pandas as pd

from features.data import Tables

DEMOGRAPHIC_COLUMNS = ["classification_1", "classification_2", "classification_3", "classification_4",
                       "classification_5", "homeowner_desc", "kid_category_desc"]
KEYS = ["household_key", "campaign", "cutoff_day"]


def compute_profiles(tables: Tables, rows: pd.DataFrame) -> pd.DataFrame:
    """rows: household_key, campaign, cutoff_day. Returns rows plus every profile and flag column, same index."""
    parts = [_at_cutoff(tables, group, int(cutoff)) for cutoff, group in rows.groupby("cutoff_day", sort=False)]
    out = pd.concat(parts).loc[rows.index]
    demographics = tables.demographics.set_index("household_key")[DEMOGRAPHIC_COLUMNS]
    return out.join(demographics, on="household_key")


def features(tables: Tables, household: int, campaign: int, cutoff_day: int) -> pd.Series:
    """Profile of one household for one campaign as of cutoff_day (any day, not only campaign cutoffs)."""
    rows = pd.DataFrame({"household_key": [household], "campaign": [campaign], "cutoff_day": [cutoff_day]})
    return compute_profiles(tables, rows).iloc[0].drop(KEYS)


def _at_cutoff(tables: Tables, rows: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    tx = tables.transactions
    before = tx[(tx["day"] < cutoff) & tx["household_key"].isin(rows["household_key"])]
    hh = rows["household_key"]
    out = rows.copy()

    def per_household(frame: pd.DataFrame, column: str, how: str) -> pd.Series:
        """Aggregate frame per household and line it up with rows; households with no rows get 0."""
        return hh.map(frame.groupby("household_key")[column].agg(how)).fillna(0).astype(float)

    # History: days from the first purchase before the cutoff to the cutoff (0 = none).
    history_days = cutoff - hh.map(before.groupby("household_key")["day"].min())
    has_history = history_days.notna()
    history_days = history_days.fillna(0)
    out["history_weeks_before_cutoff"] = history_days / 7

    def weeks_available(window_weeks: int) -> pd.Series:
        """Weeks of history inside the window; blank when there is no history, so per-week values are blank too."""
        return (np.minimum(7 * window_weeks, history_days) / 7).where(has_history)

    window = {w: before[before["day"] >= cutoff - 7 * w] for w in (8, 26)}
    for w, frame in window.items():
        out[f"spend_per_week_{w}w"] = per_household(frame, "sales_value", "sum") / weeks_available(w)
        out[f"trips_per_week_{w}w"] = per_household(frame, "basket_id", "nunique") / weeks_available(w)

    w26 = window[26]
    spend_26w = per_household(w26, "sales_value", "sum")
    has_spend = spend_26w > 0
    out["avg_basket_value_26w"] = (spend_26w / per_household(w26, "basket_id", "nunique")).where(has_spend)
    out["days_since_last_trip"] = (cutoff - hh.map(before.groupby("household_key")["day"].max())).astype(float)
    departments = w26.merge(tables.products, on="product_id", how="left")
    out["n_departments_26w"] = per_household(departments, "department", "nunique").where(has_history)

    # Deal habits
    on_discount = per_household(w26[w26["retail_disc"] < 0], "sales_value", "sum")
    out["share_spend_on_discount_26w"] = (on_discount / spend_26w).where(has_spend)
    coupon_lines = per_household(w26[w26["coupon_disc"] < 0], "day", "size")
    out["coupon_lines_per_week_26w"] = coupon_lines / weeks_available(26)

    # Coupon history: redemptions by their own date; campaigns only once they have ended.
    redemptions = tables.redemptions[tables.redemptions["day"] < cutoff]
    out["past_redemptions"] = per_household(redemptions, "day", "size").astype(int)
    mailings = tables.mailings()
    ended = mailings[mailings["end_day"] < cutoff]
    redeemed_pairs = redemptions[["household_key", "campaign"]].drop_duplicates()
    ended_redeemed = ended.merge(redeemed_pairs, on=["household_key", "campaign"])
    received = per_household(ended, "campaign", "size").astype(int)
    redeemed = per_household(ended_redeemed, "campaign", "size").astype(int)
    out["past_campaigns_received"] = received
    out["past_campaigns_redeemed"] = redeemed
    out["past_redemption_rate"] = redeemed / received.where(received > 0)

    # Interest in this row's campaign products
    campaign_products = tables.campaign_products()
    campaign_spend = pd.Series(0.0, index=rows.index)
    for campaign, index in rows.groupby("campaign").groups.items():
        products = campaign_products.loc[campaign_products["campaign"] == campaign, "product_id"]
        spend = w26[w26["product_id"].isin(products)].groupby("household_key")["sales_value"].sum()
        campaign_spend.loc[index] = hh.loc[index].map(spend).fillna(0)
    out["campaign_product_spend_per_week_26w"] = campaign_spend / weeks_available(26)
    out["campaign_product_share_26w"] = (campaign_spend / spend_26w).where(has_spend)

    # Other campaigns already running for this household on the cutoff day
    live = mailings[(mailings["start_day"] < cutoff) & (mailings["end_day"] >= cutoff)]
    pairs = (rows[["household_key", "campaign"]].assign(row=rows.index)
             .merge(live[["household_key", "campaign"]].rename(columns={"campaign": "other"}), on="household_key"))
    pairs = pairs[pairs["other"] != pairs["campaign"]]
    out["other_campaigns_live_at_cutoff"] = pairs.groupby("row").size().reindex(rows.index, fill_value=0)

    out["short_history_flag"] = (history_days < 182).astype(int)
    out["no_history_flag"] = (~has_history).astype(int)
    return out
