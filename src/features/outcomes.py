"""Outcome and filter columns: what happened DURING each campaign window (start_day to end_day inclusive).

Runs in DuckDB so the 37M-row display/flyer file is queried in place instead of loaded into memory.
"""
from pathlib import Path

import duckdb
import pandas as pd

from features.data import Tables

MEMORY_LIMIT = "2GB"


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
    return con


def _causal_source(con: duckdb.DuckDBPyConnection, causal: pd.DataFrame | Path) -> str:
    """A SQL source for the display/flyer data: an in-memory frame (tests) or a csv/parquet file."""
    if isinstance(causal, pd.DataFrame):
        con.register("causal_frame", causal)
        return "causal_frame"
    if Path(causal).suffix == ".parquet":
        return f"read_parquet('{causal}')"
    return f"read_csv('{causal}', types={{'display': 'VARCHAR', 'mailer': 'VARCHAR'}})"


def _create_promoted(con: duckdb.DuckDBPyConnection, causal: pd.DataFrame | Path, products: str | None) -> None:
    """One row per (product, store, week); promoted if ANY causal row for that key says so."""
    only = f"WHERE product_id IN (SELECT product_id FROM {products})" if products else ""
    con.execute(f"""
        CREATE TABLE promoted AS
        SELECT product_id, store_id, week_no,
               max((display <> '0')::INT) AS on_display,
               max((mailer  <> '0')::INT) AS in_flyer
        FROM {_causal_source(con, causal)} {only}
        GROUP BY ALL""")


def promoted_keys(causal: pd.DataFrame | Path) -> pd.DataFrame:
    con = _connect()
    _create_promoted(con, causal, products=None)
    return con.execute("SELECT * FROM promoted ORDER BY ALL").df()


def compute_outcomes(tables: Tables, causal: pd.DataFrame | Path) -> pd.DataFrame:
    """One row per spine row: the four outcome columns plus the Model 3 overlap filter."""
    con = _connect()
    con.register("spine", tables.spine)
    con.register("tx", tables.transactions)
    con.register("cprod", tables.campaign_products())
    con.register("mailings", tables.mailings())
    con.register("redemptions", tables.redemptions)
    con.register("coupons", tables.coupons)
    con.execute("""
        CREATE TABLE campaigns AS SELECT DISTINCT campaign, campaign_type, start_day, end_day FROM spine;
        CREATE TABLE bc_products AS
            SELECT DISTINCT product_id FROM cprod JOIN campaigns USING (campaign) WHERE campaign_type <> 'A';
        CREATE TABLE buys AS   -- every purchase of a campaign's product inside that campaign's window
            SELECT t.household_key, k.campaign, k.campaign_type, t.product_id, t.store_id,
                   round(t.sales_value * 100)::BIGINT AS cents,   -- whole cents: sums are exact in any order
                   ceil((t.day + 2) / 7.0)::INT AS week_no
            FROM tx t
            JOIN cprod p USING (product_id)
            JOIN campaigns k ON k.campaign = p.campaign AND t.day BETWEEN k.start_day AND k.end_day;
    """)
    _create_promoted(con, causal, products="bc_products")
    return con.execute("""
        WITH spend AS (
            SELECT household_key, campaign, sum(cents) / 100 AS v FROM buys GROUP BY ALL),
        redeemed_products AS (
            SELECT DISTINCT r.household_key, r.campaign, c.product_id
            FROM redemptions r JOIN coupons c ON c.campaign = r.campaign AND c.coupon_upc = r.coupon_upc),
        redeemed_spend AS (
            SELECT b.household_key, b.campaign, sum(b.cents) / 100 AS v
            FROM buys b JOIN redeemed_products USING (household_key, campaign, product_id) GROUP BY ALL),
        model3 AS (
            SELECT b.household_key, b.campaign,
                   max(CASE WHEN coalesce(p.on_display, 0) = 0 AND coalesce(p.in_flyer, 0) = 0 THEN 1 ELSE 0 END) AS excl_both,
                   max(CASE WHEN coalesce(p.on_display, 0) = 0 THEN 1 ELSE 0 END) AS excl_display,
                   max(CASE WHEN coalesce(p.on_display, 0) = 0 AND coalesce(p.in_flyer, 0) = 0
                             AND rp.product_id IS NOT NULL THEN 1 ELSE 0 END) AS via_redemption
            FROM buys b
            LEFT JOIN promoted p USING (product_id, store_id, week_no)
            LEFT JOIN redeemed_products rp
                ON rp.household_key = b.household_key AND rp.campaign = b.campaign AND rp.product_id = b.product_id
            WHERE b.campaign_type <> 'A' GROUP BY ALL),
        overlap AS (
            SELECT DISTINCT s.household_key, s.campaign, 1 AS flag
            FROM spine s JOIN mailings m
              ON m.household_key = s.household_key AND m.campaign <> s.campaign
             AND m.start_day <= s.end_day AND m.end_day >= s.start_day)
        SELECT s.household_key, s.campaign,
               coalesce(spend.v, 0)::DOUBLE AS campaign_product_spend_during,
               CASE WHEN s.redeemed_flag = 1 THEN coalesce(redeemed_spend.v, 0) END::DOUBLE
                   AS redeemed_coupon_product_spend_during,
               CASE WHEN s.campaign_type <> 'A' THEN coalesce(model3.excl_both, 0) END::DOUBLE
                   AS bought_campaign_product_excl_display_flyer,
               CASE WHEN s.campaign_type <> 'A' THEN coalesce(model3.excl_display, 0) END::DOUBLE
                   AS bought_campaign_product_excl_display,
               CASE WHEN s.campaign_type <> 'A' THEN coalesce(model3.excl_both, 0) - coalesce(model3.via_redemption, 0) END::DOUBLE
                   AS bought_campaign_product_without_redemption,
               coalesce(overlap.flag, 0)::INT AS mailed_overlapping_campaign
        FROM spine s
        LEFT JOIN spend USING (household_key, campaign)
        LEFT JOIN redeemed_spend USING (household_key, campaign)
        LEFT JOIN model3 USING (household_key, campaign)
        LEFT JOIN overlap USING (household_key, campaign)
        ORDER BY s.household_key, s.campaign
    """).df()
