"""
check_integrity.py  (Data Agent, Step 2)

Checks the clean tables against each other, looks for data quality
problems, and writes EVERY finding to a log. It only observes: it does not
change any data. Step 3 reads this log and decides what to fix, flag or stop on.

Run from the folder that contains data/processed:
    python check_integrity.py
or point it at the clean tables:
    python check_integrity.py --processed path/to/data/processed

Add --skip-causal to skip the large causal_data checks (needs duckdb otherwise).

Input : data/processed/*.parquet  (written by ingest.py)
Output: data/processed/findings.csv
        data/processed/findings.json
Exit code is 1 if any integrity check fails (status ERROR), otherwise 0.

Each finding has a kind:
  integrity  something that should hold exactly. Status OK or ERROR.
  quality    a data quality issue. Status OK (none found) or FLAG.
  info       a fact worth recording. Status INFO.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--processed", default="data/processed", help="folder with the clean parquet tables")
parser.add_argument("--skip-causal", action="store_true", help="skip the large causal_data checks")
args = parser.parse_args()
P = Path(args.processed)

# Numbers confirmed on the real data / in the proposal
EXPECTED_MAILED_PAIRS = 7208
EXPECTED_REDEEMED_PAIRS = 889
EXPECTED_HOUSEHOLDS = 2500
OUTLIER_QUANTITY = 100        # quantity above this is flagged as an outlier
SHORT_HISTORY_DAYS = 182      # 26 weeks
POSITIVE_DISCOUNT_TOLERANCE = 0.005   # half a cent; smaller positive values are floating-point noise (keep in sync with apply_decisions.py)
DISPLAY_CODES = ["0", "1", "2", "3", "4", "5", "6", "7", "9", "A"]
MAILER_CODES = ["0", "A", "C", "D", "F", "H", "J", "L", "P", "X", "Z"]

findings = []


def ex(values, k=5):
    """Up to k distinct example values, as text."""
    out = []
    for v in pd.unique(pd.Series(values).dropna()):
        out.append(str(v))
        if len(out) >= k:
            break
    return out


def add(check_id, kind, table, description, n, total=None, expected=0, examples=None):
    n = int(n)
    if kind == "integrity":
        status = "OK" if n == expected else "ERROR"
    elif kind == "quality":
        status = "OK" if n == 0 else "FLAG"
    else:
        status = "INFO"
    findings.append({
        "check_id": check_id,
        "status": status,
        "kind": kind,
        "table": table,
        "description": description,
        "n_rows": n,
        "expected": int(expected) if kind == "integrity" else None,
        "pct_of_table": round(100 * n / total, 4) if total else None,
        "examples": examples or [],
    })


def read(name, required=True):
    path = P / f"{name}.parquet"
    if not path.exists():
        if required:
            print(f"[FAIL] missing {path}. Run ingest.py first.")
            sys.exit(1)
        return None
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    return df


print("Reading clean tables...")
T = {name: read(name) for name in
     ["transaction_data", "hh_demographic", "product", "campaign_table",
      "campaign_desc", "coupon", "coupon_redempt"]}
tx, demo, prod, ct, cd, cp, cr = (T[n] for n in
    ["transaction_data", "hh_demographic", "product", "campaign_table",
     "campaign_desc", "coupon", "coupon_redempt"])
for df in (ct, cr, cp):
    if "coupon_upc" in df.columns:
        df["coupon_upc"] = df["coupon_upc"].astype(str)

universe = set(tx["household_key"])

# ---------------------------------------------------------------- integrity
key_cols = {
    "transaction_data": ["household_key", "basket_id", "day", "product_id"],
    "hh_demographic": ["household_key"],
    "product": ["product_id"],
    "campaign_table": ["household_key", "campaign"],
    "campaign_desc": ["campaign", "start_day", "end_day"],
    "coupon": ["coupon_upc", "product_id", "campaign"],
    "coupon_redempt": ["household_key", "day", "coupon_upc", "campaign"],
}
null_detail = {}
for t, cols in key_cols.items():
    for c in cols:
        if c in T[t].columns:
            n_null = int(T[t][c].isna().sum())
            if n_null:
                null_detail[f"{t}.{c}"] = n_null
add("X0", "integrity", "all tables", "Null values in key columns",
    sum(null_detail.values()), examples=[f"{k}={v}" for k, v in null_detail.items()])

add("X1", "integrity", "transaction_data", "Distinct households in transaction_data",
    len(universe), expected=EXPECTED_HOUSEHOLDS)

for cid, t in [("X2a", "campaign_table"), ("X2b", "coupon_redempt"), ("X2c", "hh_demographic")]:
    bad = ~T[t]["household_key"].isin(universe)
    add(cid, "integrity", t, "Rows whose household does not appear in transaction_data",
        bad.sum(), len(T[t]), examples=ex(T[t].loc[bad, "household_key"]))

bad = ~tx["product_id"].isin(prod["product_id"])
add("X3", "integrity", "transaction_data", "Rows whose product is missing from the product table",
    bad.sum(), len(tx), examples=ex(tx.loc[bad, "product_id"]))

bad = ~ct["campaign"].isin(cd["campaign"])
add("X4", "integrity", "campaign_table", "Rows whose campaign is missing from campaign_desc",
    bad.sum(), len(ct), examples=ex(ct.loc[bad, "campaign"]))

bad = ~cp["campaign"].isin(cd["campaign"])
add("X5", "integrity", "coupon", "Rows whose campaign is missing from campaign_desc",
    bad.sum(), len(cp), examples=ex(cp.loc[bad, "campaign"]))

m = ct.merge(cd[["campaign", "description"]], on="campaign", how="left", suffixes=("", "_desc"))
bad = m["description_desc"].notna() & (m["description"] != m["description_desc"])
add("X6", "integrity", "campaign_table", "Rows whose campaign type disagrees with campaign_desc",
    bad.sum(), len(ct), examples=ex(m.loc[bad, "campaign"]))

mailed = ct[["household_key", "campaign"]].drop_duplicates()
m = cr.merge(mailed.assign(_mailed=1), on=["household_key", "campaign"], how="left")
bad = m["_mailed"].isna()
add("X7", "integrity", "coupon_redempt", "Redemptions by a household that was not mailed that campaign",
    bad.sum(), len(cr),
    examples=ex(m.loc[bad, "household_key"].astype(str) + "/" + m.loc[bad, "campaign"].astype(str)))

m = cr.merge(cd[["campaign", "start_day", "end_day"]], on="campaign", how="inner")
bad = (m["day"] < m["start_day"]) | (m["day"] > m["end_day"])
add("X8", "integrity", "coupon_redempt", "Redemptions dated outside their campaign's start/end days",
    bad.sum(), len(cr), examples=ex(m.loc[bad, "household_key"]))

cp_keys = cp[["campaign", "coupon_upc"]].drop_duplicates().assign(_in=1)
m = cr.merge(cp_keys, on=["campaign", "coupon_upc"], how="left")
bad = m["_in"].isna()
add("X9", "integrity", "coupon_redempt", "Redemptions whose (campaign, coupon) is not in the coupon table",
    bad.sum(), len(cr), examples=ex(m.loc[bad, "coupon_upc"]))

add("X10a", "integrity", "campaign_table", "Mailings (rows in campaign_table)",
    len(ct), expected=EXPECTED_MAILED_PAIRS)
add("X10b", "integrity", "campaign_table", "Duplicate (household, campaign) pairs",
    ct.duplicated(["household_key", "campaign"]).sum(), len(ct))

red_pairs = cr[["household_key", "campaign"]].drop_duplicates()
n_pos = len(mailed.merge(red_pairs, on=["household_key", "campaign"], how="inner"))
add("X11", "integrity", "campaign_table + coupon_redempt",
    "Mailed pairs with at least one redemption",
    n_pos, len(mailed), expected=EXPECTED_REDEEMED_PAIRS)

expected_week = np.ceil((tx["day"] + 2) / 7)
bad = tx["week_no"] != expected_week
add("X12", "integrity", "transaction_data", "Rows where week_no is not ceil((day + 2) / 7)",
    bad.sum(), len(tx), examples=ex(tx.loc[bad, "day"]))

add("X13", "integrity", "product", "Duplicate product_id values",
    prod["product_id"].duplicated().sum(), len(prod))
add("X14", "integrity", "hh_demographic", "Duplicate household_key values",
    demo["household_key"].duplicated().sum(), len(demo))

# ------------------------------------------------------------------ quality
dups = cp.duplicated()
add("Q1", "quality", "coupon", "Exact duplicate rows (safe to remove)",
    dups.sum(), len(cp), examples=ex(cp.loc[dups, "coupon_upc"]))
add("Q2", "quality", "coupon_redempt", "Exact duplicate rows",
    cr.duplicated().sum(), len(cr))
add("Q3", "quality", "transaction_data", "Exact duplicate rows (do not remove without review)",
    tx.duplicated().sum(), len(tx))

for cid, col in [("Q4a", "retail_disc"), ("Q4b", "coupon_disc"), ("Q4c", "coupon_match_disc")]:
    bad = tx[col] > POSITIVE_DISCOUNT_TOLERANCE
    add(cid, "quality", "transaction_data",
        f"Rows where {col} is positive by more than half a cent (discounts should be 0 or negative)",
        bad.sum(), len(tx), examples=ex(tx.loc[bad, "household_key"]))
noise = pd.Series(False, index=tx.index)
for col in ["retail_disc", "coupon_disc", "coupon_match_disc"]:
    noise |= (tx[col] > 0) & (tx[col] <= POSITIVE_DISCOUNT_TOLERANCE)
add("I5", "info", "transaction_data",
    "Rows where a discount is positive only by rounding noise (half a cent or less); ignored by Q4",
    noise.sum(), len(tx))

bad = tx["quantity"] <= 0
add("Q5", "quality", "transaction_data", "Rows with quantity of 0 or less",
    bad.sum(), len(tx), examples=ex(tx.loc[bad, "product_id"]))
bad = tx["sales_value"] <= 0
add("Q6", "quality", "transaction_data", "Rows with sales_value of 0 or less",
    bad.sum(), len(tx), examples=ex(tx.loc[bad, "product_id"]))
bad = tx["quantity"] > OUTLIER_QUANTITY
add("Q7", "quality", "transaction_data", f"Rows with quantity above {OUTLIER_QUANTITY} (outliers)",
    bad.sum(), len(tx), examples=ex(tx.loc[bad, "quantity"].sort_values(ascending=False)))

last_day = int(tx["day"].max())
late = cd[cd["end_day"] > last_day]
add("Q8", "quality", "campaign_desc", f"Campaigns ending after the last transaction day ({last_day})",
    len(late), len(cd), examples=ex(late["campaign"]))

# --------------------------------------------------------------------- info
first_day = tx.groupby("household_key")["day"].min().rename("first_day").reset_index()
h = (mailed.merge(cd[["campaign", "start_day"]], on="campaign", how="inner")
           .merge(first_day, on="household_key", how="inner"))
add("I1", "info", "campaign_table",
    f"Mailed pairs with less than {SHORT_HISTORY_DAYS} days of purchase history before the campaign",
    ((h["start_day"] - h["first_day"]) < SHORT_HISTORY_DAYS).sum(), len(h))

camps = cd[["campaign", "start_day", "end_day"]].drop_duplicates().sort_values("campaign").to_dict("records")
n_overlap = sum(1 for i in range(len(camps)) for j in range(i + 1, len(camps))
                if camps[j]["start_day"] <= camps[i]["end_day"] and camps[i]["start_day"] <= camps[j]["end_day"])
add("I2", "info", "campaign_desc", "Pairs of campaigns whose dates overlap",
    n_overlap, len(camps) * (len(camps) - 1) // 2)

reused = cp.groupby("coupon_upc")["campaign"].nunique()
add("I3", "info", "coupon", "coupon_upc values used in more than one campaign (join on campaign + coupon_upc)",
    (reused > 1).sum(), len(reused))

add("I4", "info", "hh_demographic", "Households (of those in transaction_data) with demographic details",
    demo["household_key"].isin(universe).sum(), len(universe))

# ------------------------------------------------------------- causal_data
causal_path = P / "causal_data.parquet"
if args.skip_causal:
    print("Skipping causal_data checks (--skip-causal).")
elif not causal_path.exists():
    print("causal_data.parquet not found, skipping its checks.")
else:
    try:
        import duckdb
        src = f"read_parquet('{causal_path.as_posix()}')"
        n_total = duckdb.sql(f"select count(*) from {src}").fetchone()[0]
        print(f"Checking causal_data ({n_total:,} rows)...")
        n_dup = duckdb.sql(
            f"select count(*) from (select product_id, store_id, week_no from {src} "
            f"group by 1, 2, 3 having count(*) > 1)").fetchone()[0]
        add("Q9", "quality", "causal_data",
            "(product, store, week) keys that appear more than once (values may conflict)",
            n_dup, n_total)
        disp = ", ".join(f"'{c}'" for c in DISPLAY_CODES)
        mail = ", ".join(f"'{c}'" for c in MAILER_CODES)
        n_bad = duckdb.sql(
            f"select count(*) from {src} where cast(display as varchar) not in ({disp}) "
            f"or cast(mailer as varchar) not in ({mail})").fetchone()[0]
        add("X15", "integrity", "causal_data", "Rows with a display or mailer code outside the documented codes",
            n_bad, n_total)
    except ImportError:
        print("duckdb is not installed (pip install duckdb), skipping causal_data checks.")

# ----------------------------------------------------------------- output
order = {"ERROR": 0, "FLAG": 1, "INFO": 2, "OK": 3}
print("\n" + "=" * 100)
print(f"{'STATUS':<7} {'ID':<5} {'TABLE':<30} {'ROWS':>12}  DESCRIPTION")
print("=" * 100)
for f in sorted(findings, key=lambda x: (order[x["status"]], x["check_id"])):
    exp = f" (expected {f['expected']:,})" if f["status"] == "ERROR" and f["expected"] else ""
    print(f"{f['status']:<7} {f['check_id']:<5} {f['table'][:30]:<30} {f['n_rows']:>12,}  {f['description']}{exp}")

counts = {s: sum(1 for f in findings if f["status"] == s) for s in order}
print("\n" + "-" * 100)
print(f"{len(findings)} checks: {counts['OK']} OK, {counts['FLAG']} flagged, "
      f"{counts['ERROR']} errors, {counts['INFO']} info")

csv_rows = [{**f, "examples": "; ".join(f["examples"])} for f in findings]
pd.DataFrame(csv_rows).to_csv(P / "findings.csv", index=False)
(P / "findings.json").write_text(json.dumps({"summary": counts, "findings": findings}, indent=2))
print(f"Saved the log to {P / 'findings.csv'} and {P / 'findings.json'}")

sys.exit(1 if counts["ERROR"] else 0)
