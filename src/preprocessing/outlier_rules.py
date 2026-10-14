"""
Data Agent: data-derived outlier rules (replaces the fixed "quantity > 100" cut-off).

WHY THIS VERSION EXISTS
    On the real data, 99.99% of rows with quantity > 100 turned out to be gas-kiosk and
    miscellaneous-transaction lines (unit price about $0.002, so "quantity" is not an item
    count there). Those rows are not unusual shoppers; they are a different kind of product.
    So the agent first separates NON-MERCHANDISE products using the product table, then fits
    outlier cut-offs on ordinary merchandise only.

Step 1  Flag non-merchandise rows (department in NON_MERCH_DEPARTMENTS or
        commodity in NON_MERCH_COMMODITIES). These rows get is_non_merchandise = True and
        are left out of the outlier fitting. Their sales_value is real money, so whether
        they count as "household spend" is a team decision, not made here.
Step 2  For each numeric field, compute cut-offs from that field's own distribution using
        the IQR method on a LOG scale (these fields are heavily right-skewed):
            fence_mild    = Q3 + 1.5 * IQR
            fence_extreme = Q3 + 3.0 * IQR
        If the IQR is 0 (e.g. most quantities are exactly 1) it falls back to percentiles
        (99th / 99.9th). A percentile fallback flags a fixed share of rows by construction,
        so the output says which method was used for each field.
        Discount columns only use the extreme level: coupon values are set face amounts,
        so a larger coupon is legitimate, not noise.

Fields checked (merchandise rows only)
    quantity            rows with quantity > 0
    sales_value         rows with sales_value > 0
    unit_price          sales_value / quantity, rows with both > 0   (both sides)
    retail_disc         size of the discount, rows with a discount < 0
    coupon_disc         same
    coupon_match_disc   same

Rows are FLAGGED, never removed.

Reads   data/processed/transaction_data.parquet, product.parquet
Writes  data/processed/outlier_thresholds.json
        data/processed/outlier_sensitivity.csv
        data/processed/outlier_flags.parquet   row_id, is_non_merchandise, no_department,
                                               and a level per field (0 none, 1 mild, 2 extreme)

Usage:  python agents/data_agent/outlier_rules.py [--processed data/processed]
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

OLD_FIXED_QUANTITY_CUTOFF = 100   # the rule this replaces, kept only for comparison
K_MILD = 1.5
K_EXTREME = 3.0
DISCOUNT_FIELDS = ["retail_disc", "coupon_disc", "coupon_match_disc"]

# Found on the real data (see the quantity > 100 investigation). Edit if the team agrees on more.
NON_MERCH_DEPARTMENTS = ["KIOSK-GAS", "MISC SALES TRAN"]
NON_MERCH_COMMODITIES = ["COUPON/MISC ITEMS"]


def fences(x, k_mild, k_extreme):
    """IQR fences on the log scale, converted back to original units. x must be > 0.

    If the middle half of the data is a single value (IQR = 0), the IQR method would flag
    everything above that value, so fall back to percentiles and record that we did.
    """
    lx = np.log(x)
    q1, q3 = np.percentile(lx, [25, 75])
    iqr = q3 - q1
    p = {pc: float(np.percentile(x, pc)) for pc in (0.1, 1, 99, 99.9)}
    out = {"q1": float(np.exp(q1)), "q3": float(np.exp(q3)), "iqr_log": float(iqr), "p99_9": p[99.9]}
    if iqr > 1e-9:
        out.update({
            "method": "iqr_log",
            "mild_low": float(np.exp(q1 - k_mild * iqr)), "mild_high": float(np.exp(q3 + k_mild * iqr)),
            "extreme_low": float(np.exp(q1 - k_extreme * iqr)), "extreme_high": float(np.exp(q3 + k_extreme * iqr)),
        })
    else:
        out.update({
            "method": "percentile_fallback (IQR was 0)",
            "mild_low": p[1], "mild_high": p[99],
            "extreme_low": p[0.1], "extreme_high": p[99.9],
        })
    return out


def level_high(values, eligible, f, use_mild=True):
    """0 none, 1 mild, 2 extreme, upper tail only. use_mild=False skips the mild level."""
    lvl = np.zeros(len(values), dtype=np.int8)
    if use_mild:
        lvl[eligible & (values > f["mild_high"])] = 1
    lvl[eligible & (values > f["extreme_high"])] = 2
    return lvl


def level_low(values, eligible, f):
    lvl = np.zeros(len(values), dtype=np.int8)
    lvl[eligible & (values < f["mild_low"])] = 1
    lvl[eligible & (values < f["extreme_low"])] = 2
    return lvl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--k-mild", type=float, default=K_MILD)
    ap.add_argument("--k-extreme", type=float, default=K_EXTREME)
    args = ap.parse_args()
    proc = Path(args.processed)
    for name in ("transaction_data.parquet", "product.parquet"):
        if not (proc / name).exists():
            sys.exit(f"{proc / name} not found. Run ingest.py first.")

    tx = pd.read_parquet(proc / "transaction_data.parquet")
    tx.columns = [c.lower() for c in tx.columns]
    tx = tx[["product_id", "quantity", "sales_value", "retail_disc", "coupon_disc", "coupon_match_disc"]].copy()
    n = len(tx)
    print(f"Loaded {n:,} transaction rows.")

    pr = pd.read_parquet(proc / "product.parquet")
    pr.columns = [c.lower() for c in pr.columns]
    pr = pr[["product_id", "department", "commodity_desc"]].drop_duplicates("product_id")
    tx = tx.merge(pr, on="product_id", how="left")
    if len(tx) != n:
        sys.exit("Merging the product table changed the row count; product_id is not unique. Stopping.")

    dept = tx["department"].fillna("").astype(str).str.strip()
    comm = tx["commodity_desc"].fillna("").astype(str).str.strip()
    is_nm = (dept.isin(NON_MERCH_DEPARTMENTS) | comm.isin(NON_MERCH_COMMODITIES)).to_numpy()
    no_dept = (dept == "").to_numpy()
    merch = ~is_nm

    qty = tx["quantity"].to_numpy(dtype="float64")
    sales = tx["sales_value"].to_numpy(dtype="float64")

    thresholds = {}
    flags = {"row_id": np.arange(n, dtype=np.int64),
             "is_non_merchandise": is_nm, "no_department": no_dept}
    sens = []

    def add_sens(field, rule, n_flagged, n_eligible):
        sens.append({"field": field, "rule": rule, "rows_flagged": int(n_flagged),
                     "pct_of_eligible": round(100 * n_flagged / max(n_eligible, 1), 4),
                     "n_eligible": int(n_eligible)})

    # ---- quantity ----
    pos = qty > 0
    elig = pos & merch
    f = fences(qty[elig], args.k_mild, args.k_extreme)
    thresholds["quantity"] = {"applies_to": "quantity > 0, merchandise rows only", "n_eligible": int(elig.sum()), **f}
    lvl = level_high(qty, elig, f)
    flags["out_quantity"] = lvl
    old_all = pos & (qty > OLD_FIXED_QUANTITY_CUTOFF)
    old_merch = old_all & merch
    add_sens("quantity", f"old fixed rule (> {OLD_FIXED_QUANTITY_CUTOFF}), ALL rows", int(old_all.sum()), pos.sum())
    add_sens("quantity", f"old fixed rule (> {OLD_FIXED_QUANTITY_CUTOFF}), merchandise only", int(old_merch.sum()), elig.sum())
    add_sens("quantity", f"mild fence (> {f['mild_high']:.1f}), merchandise", int((lvl >= 1).sum()), elig.sum())
    add_sens("quantity", f"extreme fence (> {f['extreme_high']:.1f}), merchandise", int((lvl >= 2).sum()), elig.sum())
    add_sens("quantity", f"99.9th percentile (> {f['p99_9']:.1f}), merchandise", int((elig & (qty > f['p99_9'])).sum()), elig.sum())

    # ---- sales_value ----
    elig = (sales > 0) & merch
    f = fences(sales[elig], args.k_mild, args.k_extreme)
    thresholds["sales_value"] = {"applies_to": "sales_value > 0, merchandise rows only", "n_eligible": int(elig.sum()), **f}
    lvl = level_high(sales, elig, f)
    flags["out_sales_value"] = lvl
    add_sens("sales_value", f"mild fence (> {f['mild_high']:.2f})", int((lvl >= 1).sum()), elig.sum())
    add_sens("sales_value", f"extreme fence (> {f['extreme_high']:.2f})", int((lvl >= 2).sum()), elig.sum())
    add_sens("sales_value", f"99.9th percentile (> {f['p99_9']:.2f})", int((elig & (sales > f['p99_9'])).sum()), elig.sum())

    # ---- unit price (both tails) ----
    elig = (qty > 0) & (sales > 0) & merch
    price = np.full(n, np.nan)
    price[elig] = sales[elig] / qty[elig]
    f = fences(price[elig], args.k_mild, args.k_extreme)
    thresholds["unit_price"] = {"applies_to": "quantity > 0 and sales_value > 0, merchandise rows only",
                                "n_eligible": int(elig.sum()), **f}
    hi = level_high(np.nan_to_num(price), elig, f)
    lo = level_low(np.nan_to_num(price, nan=np.inf), elig, f)
    flags["out_unit_price_high"] = hi
    flags["out_unit_price_low"] = lo
    add_sens("unit_price", f"mild fence high (> {f['mild_high']:.2f})", int((hi >= 1).sum()), elig.sum())
    add_sens("unit_price", f"extreme fence high (> {f['extreme_high']:.2f})", int((hi >= 2).sum()), elig.sum())
    add_sens("unit_price", f"mild fence low (< {f['mild_low']:.3f})", int((lo >= 1).sum()), elig.sum())
    add_sens("unit_price", f"extreme fence low (< {f['extreme_low']:.3f})", int((lo >= 2).sum()), elig.sum())

    # ---- discounts: size of the discount, extreme level only ----
    for col in DISCOUNT_FIELDS:
        v = tx[col].to_numpy(dtype="float64")
        elig = (v < 0) & merch
        size = np.abs(v)
        if elig.sum() < 20:
            thresholds[col] = {"applies_to": f"{col} < 0", "n_eligible": int(elig.sum()),
                               "method": "n/a", "note": "too few rows to fit"}
            flags[f"out_{col}"] = np.zeros(n, dtype=np.int8)
            continue
        f = fences(size[elig], args.k_mild, args.k_extreme)
        thresholds[col] = {"applies_to": f"{col} < 0 (size of the discount), merchandise rows only; extreme level only",
                           "n_eligible": int(elig.sum()), **f}
        lvl = level_high(size, elig, f, use_mild=False)
        flags[f"out_{col}"] = lvl
        mild_n = int((elig & (size > f["mild_high"])).sum())
        add_sens(col, f"mild fence (size > {f['mild_high']:.2f})  [reported, NOT used as a flag]", mild_n, elig.sum())
        add_sens(col, f"extreme fence (size > {f['extreme_high']:.2f})", int((lvl >= 2).sum()), elig.sum())
        add_sens(col, f"99.9th percentile (size > {f['p99_9']:.2f})", int((elig & (size > f['p99_9'])).sum()), elig.sum())

    # ---- write outputs ----
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "method": "separate non-merchandise products, then IQR on log scale (percentile fallback if IQR is 0); flag only, never remove",
        "k_mild": args.k_mild, "k_extreme": args.k_extreme,
        "non_merchandise_departments": NON_MERCH_DEPARTMENTS,
        "non_merchandise_commodities": NON_MERCH_COMMODITIES,
        "non_merchandise_rows": int(is_nm.sum()),
        "no_department_rows": int(no_dept.sum()),
        "replaces": f"fixed quantity cut-off of {OLD_FIXED_QUANTITY_CUTOFF}",
        "rows": n,
    }
    (proc / "outlier_thresholds.json").write_text(json.dumps({"meta": meta, "fields": thresholds}, indent=2))
    sens_df = pd.DataFrame(sens)
    sens_df.to_csv(proc / "outlier_sensitivity.csv", index=False)
    pd.DataFrame(flags).to_parquet(proc / "outlier_flags.parquet", index=False)

    # ---- print ----
    print("\nNON-MERCHANDISE ROWS (gas kiosk and miscellaneous lines)")
    print("=" * 100)
    print(f"rows: {int(is_nm.sum()):,}  ({100 * is_nm.mean():.2f}% of all transactions)")
    print(f"rows with no department in the product table: {int(no_dept.sum()):,}")
    by = tx.loc[is_nm].groupby(["department", "commodity_desc"]).size().sort_values(ascending=False).head(6)
    for (d, c), k in by.items():
        print(f"  {d:<18}{c:<22}{k:>10,}")
    print(f"quantity > {OLD_FIXED_QUANTITY_CUTOFF}: {int(old_all.sum()):,} rows, of which non-merchandise: "
          f"{int((old_all & is_nm).sum()):,}  ->  left after separating them: {int(old_merch.sum()):,}")

    print("\nTHRESHOLDS (computed from the data, merchandise rows only)")
    print("=" * 100)
    print(f"{'field':<20}{'rows used':>12}{'Q1 .. Q3':>20}{'mild fence':>14}{'extreme fence':>16}  method")
    for name, t in thresholds.items():
        if "mild_high" not in t:
            print(f"{name:<20}{t['n_eligible']:>12,}   {t['note']}")
            continue
        iq = f"{t['q1']:.2f} .. {t['q3']:.2f}"
        print(f"{name:<20}{t['n_eligible']:>12,}{iq:>20}{t['mild_high']:>14.2f}{t['extreme_high']:>16.2f}  {t['method']}")

    print("\nSENSITIVITY: how many rows each rule would flag")
    print("=" * 100)
    print(f"{'field':<20}{'rule':<62}{'rows':>10}{'% of eligible':>15}")
    last = None
    for _, r in sens_df.iterrows():
        if last and r["field"] != last:
            print()
        last = r["field"]
        print(f"{r['field']:<20}{r['rule']:<62}{r['rows_flagged']:>10,}{r['pct_of_eligible']:>14.3f}%")

    print("\nSaved:")
    for fn in ("outlier_thresholds.json", "outlier_sensitivity.csv", "outlier_flags.parquet"):
        print(f"  {proc / fn}")


if __name__ == "__main__":
    main()
