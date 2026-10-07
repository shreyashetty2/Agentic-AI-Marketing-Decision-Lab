"""
verify_assumptions.py

Checks the assumptions in the Data Agent guideline against the real
dunnhumby raw files. Runs locally; no data leaves your machine.

Usage:
    python verify_assumptions.py --raw data/raw
    python verify_assumptions.py --raw data/raw --causal    # also checks the big causal_data file (needs duckdb)

Output:
    Prints every check, and saves the same text to verification_results.txt

Status labels:
    OK     matches what the guideline expected
    CHECK  differs from the expectation, or needs a human look
    INFO   just a number to record
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--raw", default="data/raw", help="folder containing the 8 raw CSV files")
parser.add_argument("--causal", action="store_true", help="also check causal_data (large; needs duckdb)")
parser.add_argument("--out", default="verification_results.txt")
args = parser.parse_args()

RAW = Path(args.raw)
lines = []


def log(text=""):
    print(text)
    lines.append(text)


def result(check_id, status, message):
    log(f"[{status:5}] {check_id}: {message}")


def section(title):
    log("")
    log("=" * 78)
    log(title)
    log("=" * 78)


def find_file(name):
    """Find <name>.csv in the raw folder, ignoring upper/lower case."""
    for p in RAW.glob("*"):
        if p.stem.lower() == name.lower() and p.suffix.lower() == ".csv":
            return p
    return None


def load(name):
    path = find_file(name)
    if path is None:
        log(f"!! Missing file for table '{name}' in {RAW}")
        sys.exit(1)
    header = pd.read_csv(path, nrows=0).columns
    dtypes = {c: str for c in header if c.lower() == "coupon_upc"}
    df = pd.read_csv(path, dtype=dtypes)
    df.columns = [c.lower() for c in df.columns]
    return df, list(header)


# ---------------------------------------------------------------------------
# Load all tables
# ---------------------------------------------------------------------------
section("0. LOADING FILES")
tx, tx_header = load("transaction_data")
demo, demo_header = load("hh_demographic")
prod, prod_header = load("product")
ct, ct_header = load("campaign_table")
cd, cd_header = load("campaign_desc")
cp, cp_header = load("coupon")
cr, cr_header = load("coupon_redempt")
for name, df in [("transaction_data", tx), ("hh_demographic", demo), ("product", prod),
                 ("campaign_table", ct), ("campaign_desc", cd), ("coupon", cp),
                 ("coupon_redempt", cr)]:
    result("load", "INFO", f"{name}: {len(df):,} rows, columns = {list(df.columns)}")

# ---------------------------------------------------------------------------
# C1. hh_demographic real column names (the PDF's page-4 table looks wrong)
# ---------------------------------------------------------------------------
section("C1. hh_demographic column names vs the PDF")
pdf_wrong_names = {"basket_id", "day", "product_id", "quantity", "sales_value", "store_id"}
real_expected = {"classification_1", "classification_2", "classification_3", "classification_4",
                 "classification_5", "homeowner_desc", "kid_category_desc", "household_key"}
actual = set(demo.columns)
result("C1", "OK" if actual == real_expected else "CHECK",
       f"actual columns = {sorted(actual)}")
result("C1", "OK" if not (actual & pdf_wrong_names) else "CHECK",
       "PDF page-4 names (basket_id, day, ...) " +
       ("do NOT appear in the file, so the PDF table is a documentation error" if not (actual & pdf_wrong_names)
        else "DO appear in the file, so re-read the guide"))

# ---------------------------------------------------------------------------
# C2. Universe and counts
# ---------------------------------------------------------------------------
section("C2. Household universe and headline counts")
households_tx = tx["household_key"].nunique()
result("C2a", "OK" if households_tx == 2500 else "CHECK",
       f"distinct households in transaction_data = {households_tx:,} (expected 2,500)")
result("C2b", "OK" if demo["household_key"].nunique() == 801 else "CHECK",
       f"households with demographics = {demo['household_key'].nunique():,} (expected 801)")
result("C2c", "OK" if demo["household_key"].is_unique else "CHECK",
       f"hh_demographic unique per household = {demo['household_key'].is_unique}")
result("C2d", "OK" if len(ct) == 7208 else "CHECK",
       f"campaign_table rows = {len(ct):,} (expected 7,208)")
dup_pairs = ct.duplicated(["household_key", "campaign"]).sum()
result("C2e", "OK" if dup_pairs == 0 else "CHECK",
       f"duplicate (household, campaign) pairs in campaign_table = {dup_pairs}")
n_hh_mailed = ct["household_key"].nunique()
result("C2f", "INFO",
       f"households that appear in campaign_table = {n_hh_mailed:,}; "
       f"never mailed any campaign = {households_tx - n_hh_mailed:,}")
result("C2g", "OK" if cd["campaign"].nunique() == 30 else "CHECK",
       f"campaigns in campaign_desc = {cd['campaign'].nunique()} (expected 30)")
result("C2h", "INFO", f"campaign types = {sorted(cd['description'].unique())}; "
       f"mailings per type = {ct['description'].value_counts().to_dict()}")
demo_not_in_universe = len(set(demo["household_key"]) - set(tx["household_key"]))
result("C2i", "OK" if demo_not_in_universe == 0 else "CHECK",
       f"demographic households not in transaction_data = {demo_not_in_universe}")
ct_type = ct.merge(cd[["campaign", "description"]], on="campaign", suffixes=("", "_desc"))
type_mismatch = (ct_type["description"] != ct_type["description_desc"]).sum()
result("C2j", "OK" if type_mismatch == 0 else "CHECK",
       f"campaign_table rows whose type disagrees with campaign_desc = {type_mismatch}")

# ---------------------------------------------------------------------------
# C3. Discount sign convention
# ---------------------------------------------------------------------------
section("C3. Discount columns: are they stored as negative numbers?")
for col in ["retail_disc", "coupon_disc", "coupon_match_disc"]:
    s = tx[col]
    result("C3", "OK" if (s > 0).sum() == 0 else "CHECK",
           f"{col}: min={s.min():.2f}, max={s.max():.2f}, "
           f"rows > 0 = {(s > 0).sum():,}, rows < 0 = {(s < 0).sum():,}")

# ---------------------------------------------------------------------------
# C4. Net revenue formula: SALES_VALUE - COUPON_MATCH_DISC
# ---------------------------------------------------------------------------
section("C4. Net revenue formula: what does SALES_VALUE - COUPON_MATCH_DISC do?")
match_rows = tx[tx["coupon_match_disc"] < 0]
result("C4a", "INFO", f"transaction lines with a coupon match discount = {len(match_rows):,}")
if len(match_rows):
    literal = match_rows["sales_value"] - match_rows["coupon_match_disc"]
    result("C4b", "INFO",
           "on those lines, sales_value - coupon_match_disc is HIGHER than sales_value in "
           f"{(literal > match_rows['sales_value']).mean():.1%} of rows "
           "(if ~100%, the literal formula adds the discount back instead of subtracting a cost)")
    log("    sample rows (household_key, sales_value, retail_disc, coupon_disc, coupon_match_disc):")
    sample = match_rows[["household_key", "sales_value", "retail_disc", "coupon_disc",
                         "coupon_match_disc"]].head(8)
    for _, r in sample.iterrows():
        log(f"      {int(r.household_key):>5}  {r.sales_value:>7.2f}  {r.retail_disc:>7.2f}  "
            f"{r.coupon_disc:>7.2f}  {r.coupon_match_disc:>7.2f}")
    log("    -> compare with the guide's example: sales 2.89, coupon_disc -0.55, match -0.45")

# ---------------------------------------------------------------------------
# C5. Transaction quality and day/week relationship
# ---------------------------------------------------------------------------
section("C5. Transaction quality and the day vs week_no relationship")
result("C5a", "INFO", f"transaction rows = {len(tx):,}; distinct products = {tx['product_id'].nunique():,}")
result("C5b", "INFO", f"day range = {tx['day'].min()} to {tx['day'].max()}; "
       f"week_no range = {tx['week_no'].min()} to {tx['week_no'].max()} (expected weeks 1-102)")
result("C5c", "INFO", f"rows with quantity <= 0 = {(tx['quantity'] <= 0).sum():,}; "
       f"sales_value <= 0 = {(tx['sales_value'] <= 0).sum():,}; "
       f"sales_value < 0 = {(tx['sales_value'] < 0).sum():,}")
result("C5d", "INFO", f"max quantity = {tx['quantity'].max():,}; "
       f"rows with quantity > 100 = {(tx['quantity'] > 100).sum():,}")
result("C5e", "OK" if tx["trans_time"].between(0, 2359).all() else "CHECK",
       "trans_time within 0-2359 for all rows = " + str(bool(tx["trans_time"].between(0, 2359).all())))

pairs = tx[["day", "week_no"]].drop_duplicates()
weeks_per_day = pairs.groupby("day")["week_no"].nunique()
result("C5f", "OK" if (weeks_per_day > 1).sum() == 0 else "CHECK",
       f"days that map to more than one week_no = {(weeks_per_day > 1).sum()} of {len(weeks_per_day)}")
plain = np.ceil(tx["day"] / 7).astype(int)
result("C5g", "OK" if (plain == tx["week_no"]).all() else "CHECK",
       f"rows where week_no == ceil(day/7): {(plain == tx['week_no']).mean():.2%}")
best = max(range(0, 7), key=lambda k: (np.ceil((tx["day"] + k) / 7).astype(int) == tx["week_no"]).mean())
best_rate = (np.ceil((tx["day"] + best) / 7).astype(int) == tx["week_no"]).mean()
result("C5h", "INFO", f"best simple offset: week = ceil((day + {best}) / 7) matches {best_rate:.2%} of rows")
missing_products = len(set(tx["product_id"]) - set(prod["product_id"]))
result("C5i", "OK" if missing_products == 0 else "CHECK",
       f"transaction product_ids missing from the product table = {missing_products:,}")
result("C5j", "OK" if prod["product_id"].is_unique else "CHECK",
       f"product table unique per product_id = {prod['product_id'].is_unique}")

# ---------------------------------------------------------------------------
# C6. Campaign timing
# ---------------------------------------------------------------------------
section("C6. Campaign windows: order and overlap")
cd_sorted = cd.sort_values("campaign")
result("C6a", "OK" if (cd["start_day"] <= cd["end_day"]).all() else "CHECK",
       f"start_day <= end_day for all campaigns = {bool((cd['start_day'] <= cd['end_day']).all())}")
chrono = cd_sorted["start_day"].is_monotonic_increasing
result("C6b", "OK" if not chrono else "CHECK",
       f"campaign IDs are in chronological order = {chrono} "
       "(guideline claims they are NOT, e.g. campaign 26 before campaign 8)")
log("    campaigns by start day (campaign, type, start, end):")
for _, r in cd.sort_values("start_day").iterrows():
    log(f"      {int(r.campaign):>3}  {r.description:<6} {int(r.start_day):>4} - {int(r.end_day):>4}")
overlaps = []
rows = cd.sort_values("start_day").to_dict("records")
for i in range(len(rows)):
    for j in range(i + 1, len(rows)):
        if rows[j]["start_day"] <= rows[i]["end_day"]:
            overlaps.append((rows[i]["campaign"], rows[j]["campaign"]))
result("C6c", "OK" if overlaps else "CHECK",
       f"overlapping campaign pairs = {len(overlaps)} (guideline claims some overlap); pairs: {overlaps[:10]}")
result("C6d", "INFO", f"campaign day range = {cd['start_day'].min()} to {cd['end_day'].max()}; "
       f"transaction day range = {tx['day'].min()} to {tx['day'].max()}")

# ---------------------------------------------------------------------------
# C7. Coupon table
# ---------------------------------------------------------------------------
section("C7. Coupon table")
result("C7a", "INFO", f"coupon rows = {len(cp):,}; exact duplicate rows = {cp.duplicated().sum():,}")
upc_campaigns = cp.groupby("coupon_upc")["campaign"].nunique()
result("C7b", "INFO",
       f"coupon_upc values that appear under more than one campaign = {(upc_campaigns > 1).sum():,} "
       f"of {len(upc_campaigns):,} (if > 0, join on (campaign, coupon_upc), not coupon_upc alone)")
missing_camp = len(set(cp["campaign"]) - set(cd["campaign"]))
result("C7c", "OK" if missing_camp == 0 else "CHECK", f"coupon campaigns missing from campaign_desc = {missing_camp}")
cp_type = cp.merge(cd[["campaign", "description"]], on="campaign")
per_campaign = cp_type.groupby(["description", "campaign"])["coupon_upc"].nunique().reset_index()
log("    distinct coupons per campaign, by type:")
for t, g in per_campaign.groupby("description"):
    log(f"      {t}: min={g['coupon_upc'].min()}, median={g['coupon_upc'].median():.0f}, max={g['coupon_upc'].max()}")
a_max = per_campaign[per_campaign["description"] == "TypeA"]["coupon_upc"].max()
result("C7d", "OK" if a_max > 16 else "CHECK",
       f"largest TypeA coupon pool = {a_max} (> 16 supports the claim that TypeA households got 16 out of a larger pool)")

# ---------------------------------------------------------------------------
# C8. Redemptions
# ---------------------------------------------------------------------------
section("C8. Redemptions: do they line up with mailings, windows and coupons?")
result("C8a", "INFO", f"redemption rows = {len(cr):,}; identical duplicate rows = {cr.duplicated().sum():,} "
       "(guideline says these are real repeat redemptions, not errors)")
mailed_pairs = ct[["household_key", "campaign"]].drop_duplicates()
m = cr.merge(mailed_pairs.assign(_mailed=1), on=["household_key", "campaign"], how="left")
orphans = m["_mailed"].isna().sum()
result("C8b", "OK" if orphans == 0 else "CHECK",
       f"redemption rows whose (household, campaign) was NOT mailed = {orphans:,}")
w = cr.merge(cd[["campaign", "start_day", "end_day"]], on="campaign", how="left")
out_window = ((w["day"] < w["start_day"]) | (w["day"] > w["end_day"])).sum()
result("C8c", "OK" if out_window == 0 else "CHECK",
       f"redemption rows outside the campaign's start/end days = {out_window:,}")
cp_keys = cp[["campaign", "coupon_upc"]].drop_duplicates().assign(_in_coupon=1)
k = cr.merge(cp_keys, on=["campaign", "coupon_upc"], how="left")
result("C8d", "OK" if k["_in_coupon"].notna().all() else "CHECK",
       f"redemption rows whose (campaign, coupon_upc) is not in the coupon table = {k['_in_coupon'].isna().sum():,}")
result("C8e", "OK" if set(cr["household_key"]) <= set(tx["household_key"]) else "CHECK",
       "all redeeming households exist in transaction_data = "
       f"{set(cr['household_key']) <= set(tx['household_key'])}")

# ---------------------------------------------------------------------------
# C9. Reproduce the proposal's headline numbers
# ---------------------------------------------------------------------------
section("C9. Reproduce the proposal: 7,208 mailings, 889 with at least one redemption")
redeemed_pairs = cr[["household_key", "campaign"]].drop_duplicates()
both = mailed_pairs.merge(redeemed_pairs.assign(_r=1), on=["household_key", "campaign"], how="left")
n_mailed = len(mailed_pairs)
n_pos = int(both["_r"].notna().sum())
result("C9a", "OK" if n_mailed == 7208 else "CHECK", f"distinct mailed pairs = {n_mailed:,} (expected 7,208)")
result("C9b", "OK" if n_pos == 889 else "CHECK",
       f"mailed pairs with at least one redemption = {n_pos:,} (expected 889); rate = {n_pos / n_mailed:.2%} (expected 12.3%)")
all_redeemed_pairs = len(redeemed_pairs)
result("C9c", "INFO", f"distinct redeemed (household, campaign) pairs in coupon_redempt = {all_redeemed_pairs:,}; "
       f"of these not mailed = {all_redeemed_pairs - n_pos:,}")
by_type = both.merge(cd[["campaign", "description"]], on="campaign").groupby("description").agg(
    mailings=("_r", "size"), redeemed=("_r", lambda s: s.notna().sum()))
by_type["rate"] = (by_type["redeemed"] / by_type["mailings"]).round(3)
log("    redemption rate by campaign type:")
for t, r in by_type.iterrows():
    log(f"      {t}: {int(r.redeemed)} of {int(r.mailings)} = {r.rate:.1%}")

# ---------------------------------------------------------------------------
# C10. History available before each campaign
# ---------------------------------------------------------------------------
section("C10. How much purchase history exists before each mailed campaign?")
first_day = tx.groupby("household_key")["day"].min().rename("first_txn_day")
h = mailed_pairs.merge(cd[["campaign", "start_day"]], on="campaign").merge(first_day, on="household_key")
h["history_days"] = h["start_day"] - h["first_txn_day"]
result("C10a", "INFO", f"earliest campaign start day = {cd['start_day'].min()}")
result("C10b", "INFO", f"mailed pairs with no prior history (start_day <= first purchase day) = "
       f"{(h['history_days'] <= 0).sum():,} of {len(h):,}")
result("C10c", "INFO", f"mailed pairs with less than 8 weeks (56 days) of history = "
       f"{(h['history_days'] < 56).sum():,}; less than 26 weeks (182 days) = {(h['history_days'] < 182).sum():,}")
log(f"    history days before campaign start: median={h['history_days'].median():.0f}, "
    f"min={h['history_days'].min()}, max={h['history_days'].max()}")

# ---------------------------------------------------------------------------
# C11. Demographic labels
# ---------------------------------------------------------------------------
section("C11. Demographic categories (needed to build the ordinal mapping)")
for col in demo.columns:
    if col == "household_key":
        continue
    counts = demo[col].value_counts(dropna=False)
    log(f"    {col} ({len(counts)} values): " + "; ".join(f"{k}={v}" for k, v in counts.items()))

# ---------------------------------------------------------------------------
# C12. causal_data (optional)
# ---------------------------------------------------------------------------
if args.causal:
    section("C12. causal_data (display and mailer codes)")
    causal_path = find_file("causal_data")
    if causal_path is None:
        result("C12", "CHECK", "causal_data.csv not found")
    else:
        try:
            import duckdb
            q = f"read_csv_auto('{causal_path.as_posix()}', all_varchar=true)"
            n = duckdb.sql(f"select count(*) from {q}").fetchone()[0]
            result("C12a", "INFO", f"causal_data rows = {n:,}")
            disp = sorted(r[0] for r in duckdb.sql(f"select distinct display from {q}").fetchall())
            mail = sorted(r[0] for r in duckdb.sql(f"select distinct mailer from {q}").fetchall())
            result("C12b", "OK" if set(disp) <= set("0123456789A") else "CHECK", f"display codes found = {disp}")
            result("C12c", "OK" if set(mail) <= set("0ACDFHJLPXZ") else "CHECK", f"mailer codes found = {mail}")
            dup = duckdb.sql(f"select count(*) from (select product_id, store_id, week_no, count(*) c "
                             f"from {q} group by 1,2,3 having c > 1)").fetchone()[0]
            result("C12d", "OK" if dup == 0 else "CHECK", f"duplicate (product, store, week) keys = {dup:,}")
        except ImportError:
            result("C12", "CHECK", "duckdb not installed (pip install duckdb), skipped")
else:
    log("")
    log("(causal_data skipped; run with --causal to check it)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
section("SUMMARY")
flagged = [l for l in lines if l.startswith("[CHECK")]
log(f"{len(flagged)} item(s) differ from the guideline's expectation:")
for l in flagged:
    log("  " + l)

Path(args.out).write_text("\n".join(lines), encoding="utf-8")
print(f"\nSaved to {args.out}")
