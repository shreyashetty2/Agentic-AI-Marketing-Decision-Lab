"""Model 3 product-level drill-down: for a given campaign (and optionally one product or coupon
within it), show the actual product name, its observed uplift, and whether treated households
bought it without ever redeeming the coupon.

model3_uplift.py and model3_redemption_analysis.py report Model 3's results aggregated across a
whole campaign's eligible products. This answers the finer-grained question: for THIS product, in
THIS campaign, what happened? Computed directly from transaction_data + causal_data + coupon
(same confounder logic as Feature Agent 2A's compute_outcomes, src/features/outcomes.py, but kept
separate by product instead of aggregated across a campaign's whole product list).

Run from the repo folder:
    python src/modeling/model3_product_lookup.py --campaign 29                    # top products in a campaign
    python src/modeling/model3_product_lookup.py --campaign 29 --product 1133018  # one specific product
    python src/modeling/model3_product_lookup.py --campaign 29 --coupon 10000089073  # one specific coupon

Input : data/processed/*.parquet (from src/preprocessing/ingest.py)
Output: prints a table to the console
"""
import argparse
from pathlib import Path

import duckdb
import pandas as pd

MEMORY_LIMIT = "2GB"


def _connect(processed: Path) -> duckdb.DuckDBPyConnection:
    """ingest.py keeps each raw file's original column casing (mostly upper-case) in the parquet
    files it writes, same as features/data.py notes; lower-case them here the same way so this
    script's SQL can use consistent lower-case column names throughout."""
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
    for name in ("transaction_data", "campaign_table", "campaign_desc", "coupon",
                 "coupon_redempt", "product"):
        df = pd.read_parquet(processed / f"{name}.parquet")
        df.columns = [c.lower() for c in df.columns]
        con.register(name, df)
    return con


def product_lookup(con: duckdb.DuckDBPyConnection, causal: Path, campaign_id: int,
                    product_id: int | None, coupon_upc: str | None, top_n: int) -> pd.DataFrame:
    campaign = con.execute(
        "SELECT campaign, description AS campaign_type, start_day, end_day FROM campaign_desc "
        "WHERE campaign = ?", [campaign_id]).df()
    if campaign.empty:
        raise SystemExit(f"[STOP] campaign {campaign_id} not found")
    start_day, end_day = int(campaign.iloc[0]["start_day"]), int(campaign.iloc[0]["end_day"])
    campaign_type = campaign.iloc[0]["campaign_type"]
    if campaign_type == "TypeA":
        print(f"WARNING: campaign {campaign_id} is TypeA -- a household's actual 16 coupons out "
              f"of this pool are unknown, so per-product results here are not individually "
              f"reliable (same reason Model 3 excludes TypeA from its main outcome).")

    products_filter = "AND product_id = ?" if product_id else ""
    coupon_filter = "AND coupon_upc = ?" if coupon_upc else ""
    params = [campaign_id] + ([product_id] if product_id else []) + ([coupon_upc] if coupon_upc else [])
    eligible = con.execute(f"""
        SELECT DISTINCT product_id FROM coupon
        WHERE campaign = ? {products_filter} {coupon_filter}
    """, params).df()
    if eligible.empty:
        raise SystemExit(f"[STOP] no eligible products found for campaign {campaign_id} "
                          f"with the given product/coupon filter")

    con.register("eligible_products", eligible)
    con.execute(f"""
        CREATE OR REPLACE TABLE buys AS
            SELECT t.household_key, t.product_id, t.store_id,
                   round(t.sales_value * 100)::BIGINT AS cents,
                   ceil((t.day + 2) / 7.0)::INT AS week_no
            FROM transaction_data t
            JOIN eligible_products e ON e.product_id = t.product_id
            WHERE t.day BETWEEN {start_day} AND {end_day};
        CREATE OR REPLACE TABLE promoted AS
            SELECT product_id, store_id, week_no,
                   max((display <> '0')::INT) AS on_display, max((mailer <> '0')::INT) AS in_flyer
            FROM read_parquet('{causal}') c
            WHERE c.product_id IN (SELECT product_id FROM eligible_products) GROUP BY ALL;
    """)
    mailed = con.execute("SELECT DISTINCT household_key FROM campaign_table WHERE campaign = ?",
                          [campaign_id]).df()
    # Clean control, same definition as the rest of Model 3 (src/features/outcomes.py): exclude
    # households mailed any OTHER campaign whose window overlaps this one's, so "control" isn't
    # secretly contaminated by a different, concurrently-running campaign.
    overlapping = con.execute("""
        SELECT campaign FROM campaign_desc
        WHERE campaign != ? AND start_day <= ? AND end_day >= ?
    """, [campaign_id, end_day, start_day]).df()
    all_households = con.execute("SELECT DISTINCT household_key FROM transaction_data").df()
    contaminated = con.execute(
        "SELECT DISTINCT household_key FROM campaign_table WHERE campaign IN "
        f"({','.join(str(c) for c in overlapping['campaign'])})"
    ).df() if not overlapping.empty else pd.DataFrame({"household_key": []})
    clean_control_households = (set(all_households["household_key"]) - set(mailed["household_key"])
                                 - set(contaminated["household_key"]))
    redeemed_products = con.execute("""
        SELECT DISTINCT r.household_key, c.product_id
        FROM coupon_redempt r JOIN coupon c ON c.campaign = r.campaign AND c.coupon_upc = r.coupon_upc
        WHERE r.campaign = ?
    """, [campaign_id]).df()
    con.register("mailed", mailed)
    con.register("clean_control", pd.DataFrame({"household_key": list(clean_control_households)}))
    con.register("redeemed_products", redeemed_products)

    result = con.execute("""
        WITH pure_buys AS (
            SELECT b.household_key, b.product_id
            FROM buys b LEFT JOIN promoted p USING (product_id, store_id, week_no)
            WHERE coalesce(p.on_display, 0) = 0 AND coalesce(p.in_flyer, 0) = 0
            GROUP BY ALL),
        per_product AS (
            SELECT e.product_id,
                   count(DISTINCT CASE WHEN m.household_key IS NOT NULL THEN pb.household_key END) AS n_treated_bought,
                   count(DISTINCT CASE WHEN cc.household_key IS NOT NULL THEN pb.household_key END) AS n_control_bought,
                   count(DISTINCT CASE WHEN m.household_key IS NOT NULL AND rp.household_key IS NULL
                         THEN pb.household_key END) AS n_without_redemption
            FROM eligible_products e
            LEFT JOIN pure_buys pb ON pb.product_id = e.product_id
            LEFT JOIN mailed m ON m.household_key = pb.household_key
            LEFT JOIN clean_control cc ON cc.household_key = pb.household_key
            LEFT JOIN redeemed_products rp ON rp.household_key = pb.household_key AND rp.product_id = pb.product_id
            GROUP BY e.product_id)
        SELECT pp.product_id, pr.commodity_desc, pr.sub_commodity_desc, pr.brand, pr.department,
               pp.n_treated_bought, pp.n_control_bought, pp.n_without_redemption
        FROM per_product pp JOIN product pr USING (product_id)
        ORDER BY pp.n_treated_bought + pp.n_control_bought DESC
    """).df()

    n_treated_mailed, n_control_mailed = len(mailed), len(clean_control_households)
    result["treated_rate"] = (result["n_treated_bought"] / n_treated_mailed).round(4)
    result["control_rate"] = (result["n_control_bought"] / n_control_mailed).round(4)
    result["observed_uplift"] = (result["treated_rate"] - result["control_rate"]).round(4)
    bought_float = result["n_treated_bought"].astype(float).replace(0, float("nan"))
    result["pct_without_redemption"] = (result["n_without_redemption"] / bought_float * 100).round(1)
    return result if product_id or coupon_upc else result.head(top_n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=int, required=True)
    parser.add_argument("--product", type=int, default=None, help="Show only this product_id")
    parser.add_argument("--coupon", type=str, default=None, help="Show only products under this coupon_upc")
    parser.add_argument("--top", type=int, default=15, help="How many products to show when not filtered (default 15)")
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--causal", type=Path, default=Path("data/processed/causal_data.parquet"))
    args = parser.parse_args()

    con = _connect(args.processed)
    result = product_lookup(con, args.causal, args.campaign, args.product, args.coupon, args.top)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 30)
    cols = ["product_id", "commodity_desc", "sub_commodity_desc", "brand", "n_treated_bought",
            "n_control_bought", "observed_uplift", "n_without_redemption", "pct_without_redemption"]
    print(f"\nCampaign {args.campaign}"
          + (f", product {args.product}" if args.product else "")
          + (f", coupon {args.coupon}" if args.coupon else "")
          + (f" (top {args.top} products by purchase volume)" if not (args.product or args.coupon) else ""))
    print(result[cols].to_string(index=False))


if __name__ == "__main__":
    main()
