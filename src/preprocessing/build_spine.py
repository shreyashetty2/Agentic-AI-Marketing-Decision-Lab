"""
build_spine.py  (Data Agent, Step 1)

Builds the household x campaign table (the "spine") from the clean tables
that ingest.py already wrote to data/processed/.

One row per household x campaign, for ALL households and ALL campaigns
(2,500 x 30 = 75,000 rows). Rows where mailed_flag = 1 are the 7,208
mailings used for Model 1.

Run from the repo folder:
    python agents/data_agent/build_spine.py
or from anywhere:
    python build_spine.py --processed path/to/data/processed

Input : data/processed/{transaction_data, campaign_table, campaign_desc,
        coupon_redempt, hh_demographic}.parquet
Output: data/processed/spine_household_campaign.parquet
        data/processed/spine_check.json
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--processed", default="data/processed", help="folder with the clean parquet tables")
args = parser.parse_args()
P = Path(args.processed)

# Numbers we expect from the project proposal (confirmed on the real data earlier)
EXPECTED = {"mailed_pairs": 7208, "redeemed_pairs": 889, "households": 2500, "campaigns": 30}


def read(name, columns=None):
    path = P / f"{name}.parquet"
    if not path.exists():
        print(f"[FAIL] missing {path}. Run ingest.py first.")
        sys.exit(1)
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]   # works with older mixed-case files too
    return df[columns] if columns else df


print("Reading clean tables...")
tx = read("transaction_data", columns=["household_key", "day"])
ct = read("campaign_table")
cd = read("campaign_desc")
cr = read("coupon_redempt")
demo = read("hh_demographic", columns=["household_key"])

# --- Households and campaigns (the full grid) --------------------------------
households = pd.DataFrame({"household_key": sorted(tx["household_key"].unique())})
campaigns = cd.rename(columns={"description": "campaign_type"}).copy()
campaigns["campaign_type"] = campaigns["campaign_type"].str.replace("Type", "", regex=False)

# Campaign facts: order in time, length, overlaps, truncated window
last_tx_day = int(tx["day"].max())
campaigns = campaigns.sort_values(["start_day", "end_day", "campaign"]).reset_index(drop=True)
campaigns["chrono_rank"] = campaigns.index + 1
campaigns["duration_days"] = campaigns["end_day"] - campaigns["start_day"] + 1


def count_overlaps(row):
    others = campaigns[campaigns["campaign"] != row["campaign"]]
    return int(((others["start_day"] <= row["end_day"]) & (others["end_day"] >= row["start_day"])).sum())


campaigns["n_overlapping_campaigns"] = campaigns.apply(count_overlaps, axis=1)
campaigns["window_truncated_flag"] = (campaigns["end_day"] > last_tx_day).astype(int)

# --- Spine: every household x every campaign ---------------------------------
spine = households.merge(campaigns, how="cross")

# mailed or not
mailed = ct[["household_key", "campaign"]].drop_duplicates().assign(mailed_flag=1)
spine = spine.merge(mailed, on=["household_key", "campaign"], how="left")
spine["mailed_flag"] = spine["mailed_flag"].fillna(0).astype(int)

# redemptions (outcome columns: these must NEVER be used as features)
cr = cr.merge(campaigns[["campaign", "start_day", "end_day"]], on="campaign", how="left")
cr["in_window"] = ((cr["day"] >= cr["start_day"]) & (cr["day"] <= cr["end_day"])).astype(int)
red = cr.groupby(["household_key", "campaign"]).agg(
    n_redemptions=("day", "size"),
    n_redemptions_in_window=("in_window", "sum"),
).reset_index()
spine = spine.merge(red, on=["household_key", "campaign"], how="left")
spine[["n_redemptions", "n_redemptions_in_window"]] = (
    spine[["n_redemptions", "n_redemptions_in_window"]].fillna(0).astype(int)
)
spine["redeemed_flag"] = (spine["n_redemptions"] >= 1).astype(int)

# household context available BEFORE the campaign starts
first_day = tx.groupby("household_key")["day"].min().rename("first_txn_day").reset_index()
spine = spine.merge(first_day, on="household_key", how="left")
spine["history_days_before_start"] = spine["start_day"] - spine["first_txn_day"]
spine["has_demographics"] = spine["household_key"].isin(demo["household_key"]).astype(int)

cols = ["household_key", "campaign", "campaign_type", "start_day", "end_day", "chrono_rank",
        "duration_days", "n_overlapping_campaigns", "window_truncated_flag",
        "mailed_flag", "has_demographics", "first_txn_day", "history_days_before_start",
        "n_redemptions", "n_redemptions_in_window", "redeemed_flag"]
spine = spine[cols].sort_values(["household_key", "chrono_rank"]).reset_index(drop=True)

# --- Checks (the acceptance tests) --------------------------------------------
mailed_rows = spine[spine["mailed_flag"] == 1]
checks = {
    "rows = households x campaigns": (len(spine), len(households) * len(campaigns)),
    "households": (spine["household_key"].nunique(), EXPECTED["households"]),
    "campaigns": (spine["campaign"].nunique(), EXPECTED["campaigns"]),
    "key is unique (duplicate rows)": (int(spine.duplicated(["household_key", "campaign"]).sum()), 0),
    "mailed pairs": (len(mailed_rows), EXPECTED["mailed_pairs"]),
    "mailed pairs with a redemption": (int(mailed_rows["redeemed_flag"].sum()), EXPECTED["redeemed_pairs"]),
    "redeemed pairs that were NOT mailed": (int(((spine["redeemed_flag"] == 1) & (spine["mailed_flag"] == 0)).sum()), 0),
    "redemptions outside the campaign dates": (int((spine["n_redemptions"] - spine["n_redemptions_in_window"]).sum()), 0),
}

print("\nChecks (actual vs expected):")
all_ok = True
results = {}
for name, (actual, expected) in checks.items():
    ok = actual == expected
    all_ok &= ok
    results[name] = {"actual": int(actual), "expected": int(expected), "pass": bool(ok)}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {actual:,} (expected {expected:,})")

rate = mailed_rows["redeemed_flag"].mean()
print(f"\nRedemption rate among mailed pairs: {rate:.2%}")
print("By campaign type:")
for t, g in mailed_rows.groupby("campaign_type"):
    print(f"  Type{t}: {int(g['redeemed_flag'].sum())} of {len(g):,} = {g['redeemed_flag'].mean():.1%}")
print(f"\nMailed pairs with less than 26 weeks (182 days) of history: "
      f"{int((mailed_rows['history_days_before_start'] < 182).sum()):,}")
print(f"Campaigns whose window runs past the last transaction day ({last_tx_day}): "
      f"{campaigns.loc[campaigns['window_truncated_flag'] == 1, 'campaign'].tolist()}")

# --- Save -----------------------------------------------------------------------
out = P / "spine_household_campaign.parquet"
spine.to_parquet(out, index=False)
(P / "spine_check.json").write_text(json.dumps({"all_pass": bool(all_ok), "checks": results}, indent=2))
print(f"\nSaved {len(spine):,} rows to {out}")
print("RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED (see above)")
sys.exit(0 if all_ok else 1)
