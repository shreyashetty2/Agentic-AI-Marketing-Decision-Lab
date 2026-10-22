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

# The same labels as numbers, so models can use them. Ordered labels keep their order (1 = first level);
# 0 = the household has no demographic record (has_demographics says which), so these are never blank.
DEMOGRAPHIC_CODES = {
    "classification_1_code": ("classification_1", {f"Age Group{i}": i for i in range(1, 7)}),
    "classification_3_code": ("classification_3", {f"Level{i}": i for i in range(1, 13)}),
    "classification_4_code": ("classification_4", {"1": 1, "2": 2, "3": 3, "4": 4, "5+": 5}),
    "classification_5_code": ("classification_5", {f"Group{i}": i for i in range(1, 7)}),
    "kid_count_code": ("kid_category_desc", {"None/Unknown": 0, "1": 1, "2": 2, "3+": 3}),
    "classification_2_x": ("classification_2", {"X": 1}),
    "classification_2_y": ("classification_2", {"Y": 1}),
    "classification_2_z": ("classification_2", {"Z": 1}),
    "homeowner_flag": ("homeowner_desc", {"Homeowner": 1, "Probable Owner": 1}),
    "renter_flag": ("homeowner_desc", {"Renter": 1, "Probable Renter": 1}),
}


def demographic_codes(demographics: pd.DataFrame) -> pd.DataFrame:
    """One numeric column per DEMOGRAPHIC_CODES entry, indexed by household_key. Unlisted labels -> 0."""
    d = demographics.set_index("household_key")
    return pd.DataFrame({name: d[src].astype(str).map(codes).fillna(0).astype(int)
                         for name, (src, codes) in DEMOGRAPHIC_CODES.items()}, index=d.index)


def compute_profiles(tables: Tables, rows: pd.DataFrame) -> pd.DataFrame:
    """rows: household_key, campaign, cutoff_day. Returns rows plus every profile and flag column, same index."""
    parts = [_at_cutoff(tables, group, int(cutoff)) for cutoff, group in rows.groupby("cutoff_day", sort=False)]
    out = pd.concat(parts).loc[rows.index]
    demographics = tables.demographics.set_index("household_key")[DEMOGRAPHIC_COLUMNS]
    out = out.join(demographics, on="household_key").join(demographic_codes(tables.demographics), on="household_key")
    out[list(DEMOGRAPHIC_CODES)] = out[list(DEMOGRAPHIC_CODES)].fillna(0).astype(int)
    return out


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
    # Coupon habit in more detail: recent (8 weeks), all history, how many trips use one, how many kinds of product
    w8 = window[8]
    out["coupon_lines_per_week_8w"] = per_household(w8[w8["coupon_disc"] < 0], "day", "size") / weeks_available(8)
    out["coupon_lines_per_week_all"] = (per_household(before[before["coupon_disc"] < 0], "day", "size")
                                        / (history_days / 7).where(has_history))
    trips_26w = per_household(w26, "basket_id", "nunique")
    coupon_trips = per_household(w26[w26["coupon_disc"] < 0], "basket_id", "nunique")
    out["coupon_trip_share_26w"] = (coupon_trips / trips_26w).where(trips_26w > 0)
    out["coupon_commodities_26w"] = per_household(departments[departments["coupon_disc"] < 0], "commodity_desc",
                                                  "nunique").where(has_history)

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
    # Did the household redeem the campaign(s) it received that ended most recently?
    last = ended.assign(redeemed=ended.merge(redeemed_pairs.assign(r=1), on=["household_key", "campaign"],
                                             how="left")["r"].fillna(0).to_numpy())
    last = last[last["end_day"] == last.groupby("household_key")["end_day"].transform("max")]
    out["redeemed_last_campaign"] = hh.map(last.groupby("household_key")["redeemed"].max()).where(received > 0)

    # Interest in this row's campaign products, and coupon use on them and on their kinds of product (commodity)
    campaign_products = tables.campaign_products()
    commodity_of = tables.products.set_index("product_id")["commodity_desc"]
    upc_commodities = (tables.coupons[["coupon_upc", "product_id"]].drop_duplicates()
                       .assign(commodity_desc=lambda d: d["product_id"].map(commodity_of))
                       [["coupon_upc", "commodity_desc"]].dropna().drop_duplicates())
    redeemed_commodities = redemptions.merge(upc_commodities, on="coupon_upc")
    w26_coupon = departments[departments["coupon_disc"] < 0]
    campaign_spend = pd.Series(0.0, index=rows.index)
    product_coupon_lines, commodity_coupon_lines, redeemed_in_commodities = (pd.Series(0.0, index=rows.index)
                                                                             for _ in range(3))
    for campaign, index in rows.groupby("campaign").groups.items():
        products = campaign_products.loc[campaign_products["campaign"] == campaign, "product_id"]
        commodities = set(commodity_of.reindex(products).dropna())
        spend = w26[w26["product_id"].isin(products)].groupby("household_key")["sales_value"].sum()
        campaign_spend.loc[index] = hh.loc[index].map(spend).fillna(0)
        lines = w26_coupon[w26_coupon["product_id"].isin(products)].groupby("household_key").size()
        product_coupon_lines.loc[index] = hh.loc[index].map(lines).fillna(0)
        lines = w26_coupon[w26_coupon["commodity_desc"].isin(commodities)].groupby("household_key").size()
        commodity_coupon_lines.loc[index] = hh.loc[index].map(lines).fillna(0)
        coupons = (redeemed_commodities[redeemed_commodities["commodity_desc"].isin(commodities)]
                   .groupby("household_key")["coupon_upc"].nunique())
        redeemed_in_commodities.loc[index] = hh.loc[index].map(coupons).fillna(0)
    out["campaign_product_spend_per_week_26w"] = campaign_spend / weeks_available(26)
    out["campaign_product_share_26w"] = (campaign_spend / spend_26w).where(has_spend)
    out["campaign_product_coupon_lines_per_week_26w"] = product_coupon_lines / weeks_available(26)
    out["campaign_commodity_coupon_lines_per_week_26w"] = commodity_coupon_lines / weeks_available(26)
    out["past_redemptions_in_campaign_commodities"] = redeemed_in_commodities.astype(int)

    # Other campaigns already running for this household on the cutoff day
    live = mailings[(mailings["start_day"] < cutoff) & (mailings["end_day"] >= cutoff)]
    pairs = (rows[["household_key", "campaign"]].assign(row=rows.index)
             .merge(live[["household_key", "campaign"]].rename(columns={"campaign": "other"}), on="household_key"))
    pairs = pairs[pairs["other"] != pairs["campaign"]]
    out["other_campaigns_live_at_cutoff"] = pairs.groupby("row").size().reindex(rows.index, fill_value=0)

    out["short_history_flag"] = (history_days < 182).astype(int)
    out["no_history_flag"] = (~has_history).astype(int)
    return out
