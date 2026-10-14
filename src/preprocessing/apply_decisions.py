"""
Data Agent, Step 3: fix / flag / escalate / halt decisions + decision log.

Reads  data/processed/findings.json         (written by check_integrity.py)
       data/processed/outlier_flags.parquet (written by outlier_rules.py; optional but recommended)
Writes data/processed/decision_log.csv / decision_log.json
       data/processed/coupon_clean.parquet        (FIX: exact duplicates removed)
       data/processed/transaction_flags.parquet   (FLAG: one row per transaction, flag columns only)

Policy
  HALT     : an integrity check failed (tables disagree with each other). Nothing is written except the log.
  FIX      : the problem is unambiguous and removing it loses no information (exact duplicate rows).
  FLAG     : the problem may be real data or an error. Keep the rows, add a flag column, let later agents decide.
             The agent decides this on its own when a written rule covers the case (decided_by = rule).
  ESCALATE : a small leftover the rules cannot explain, or a policy question for the team (decided_by = human_needed).
  NOTE     : a fact later agents must respect (e.g. short history), no data change.

What the agent resolves by itself (found by looking at the real data):
  - discounts that are positive only by floating-point noise (below half a cent) are ignored
  - positive discounts on lines with no quantity are flagged as adjustment lines
  - lines with no quantity AND no sales are flagged as zero lines
  - lines with a quantity but no sales are flagged as free / fully discounted items
  - gas-kiosk and miscellaneous lines are flagged by product type (their "quantity" is not an item count)
  - odd values that look like typos are flagged by rule: a positive discount that becomes a normal discount when the
    sign is flipped, a negative quantity whose price is normal, a zero quantity whose sales value is normal.
    "Normal" is measured from the data (mild IQR fence), never a fixed number. Corrected values go in new columns.
What it hands to a person: odd values that do NOT look like typos (far from normal), and whether gas/misc spend
counts as household spend.

Usage:  python agents/data_agent/apply_decisions.py [--processed data/processed]
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

POSITIVE_DISCOUNT_TOLERANCE = 0.005   # half a cent; keep in sync with check_integrity.py
OLD_FIXED_QUANTITY_CUTOFF = 100
K_TYPO = 1.5                 # mild IQR fence used to decide what counts as "close to normal"
MIN_REFERENCE_LINES = 5      # a product needs this many normal lines to have a reference price

# check_id -> (action, decided_by, rationale, output, downstream)   [static rules]
RULES = {
    "Q1": ("FIX", "rule",
           "Exact duplicate rows in coupon carry no extra information; keeping them would double-count coupons per campaign.",
           "coupon_clean.parquet",
           "Feature Agent: use coupon_clean, not coupon."),
    "Q2": ("NOTE", "rule", "No duplicates found; nothing to do.", "", ""),
    "Q3": ("NOTE", "rule",
           "No exact duplicate transactions. If any appear later, flag for review rather than delete (repeat purchases can look identical).",
           "", ""),
    "Q4b": ("NOTE", "rule", "Sign is as documented.", "", ""),
    "Q4c": ("NOTE", "rule", "Sign is as documented.", "", ""),
    "Q8": ("FLAG", "rule",
           "A campaign window that runs past the last transaction day cannot be fully observed. Already carried into the spine table as window_truncated_flag.",
           "spine_household_campaign.parquet (window_truncated_flag)",
           "Modeling Agent: consider excluding or weighting truncated campaigns."),
    "Q9": ("FLAG", "human_needed",
           "Duplicate (product, store, week) keys in causal_data with conflicting display/mailer values. Cannot tell which row is right.",
           "none (documented only)",
           "Not used for Model 1. Revisit before using in-store display features."),
    "I1": ("NOTE", "rule",
           "Households with under 182 days of history before a campaign have thinner features. Flagged in the spine as history_days_before_start.",
           "spine_household_campaign.parquet (history_days_before_start)",
           "Feature Agent: lookback windows must respect this."),
    "I2": ("NOTE", "rule",
           "Overlapping campaigns mean a purchase can belong to more than one campaign window. Recorded as n_overlapping_campaigns in the spine.",
           "spine_household_campaign.parquet (n_overlapping_campaigns)",
           "Modeling Agent: account for overlap in train/test splits."),
    "I3": ("NOTE", "rule",
           "coupon_upc is reused across campaigns.",
           "", "Always join coupon on (campaign, coupon_upc)."),
    "I4": ("NOTE", "rule",
           "Only some households have demographics.",
           "", "Feature Agent: treat demographics as optional and add a has_demographics flag."),
    "I5": ("NOTE", "rule",
           "Positive discounts smaller than half a cent are floating-point rounding noise (values like 0.0000000000000004). They are ignored by Q4 and recorded in transaction_flags (flag_discount_rounding_noise).",
           "transaction_flags.parquet (flag_discount_rounding_noise)", ""),
}

TX_DEPENDENT = {"Q4a", "Q5", "Q6", "Q7"}

# plain-language label for each flag column, used in the printed summary
FLAG_LABELS = {
    "flag_positive_retail_disc": "positive discount (above half a cent)",
    "flag_discount_rounding_noise": "discount positive by rounding noise only (ignored)",
    "flag_adjustment_line": "adjustment-type lines (no quantity, positive discount)",
    "flag_discount_sign_typo": "positive discount that looks like a sign typo (flagged by rule)",
    "flag_positive_discount_unexplained": "positive discount, unexplained (escalated)",
    "flag_qty_sign_typo": "negative quantity with a normal price (probable sign typo)",
    "flag_qty_missing_money_ok": "quantity 0 with a normal sales value (quantity probably missing)",
    "flag_nonpositive_qty": "quantity of 0 or less (all)",
    "flag_nonpositive_sales": "sales value of 0 or less (all)",
    "flag_zero_line": "zero lines (no quantity AND no sales)",
    "flag_free_item": "free / fully discounted items (quantity but no sales)",
    "flag_qty_nonpos_with_sales": "no quantity but money recorded (escalated)",
    "flag_sales_rounding_noise": "sales positive only by rounding noise, no quantity (ignored)",
    "flag_qty_outlier": "quantity above 100 (old fixed rule)",
    "is_non_merchandise": "non-merchandise lines (gas kiosk, miscellaneous)",
    "no_department": "products with no department",
}


def _upper_fence(x, k=K_TYPO):
    """Mild upper IQR fence of a non-negative sample: how far 'normal' variation reaches."""
    q1, q3 = np.percentile(x, [25, 75])
    return float(q3 + k * (q3 - q1))


def _typo_tests(tx, qty, sales, rd, tol, unexp_all, qnp_all):
    """Decide which odd rows look like a simple typo, by comparing them with normal rows.

    The rule (from the mentor): if the odd value is close to what a normal value would be, it is
    probably a typo; if it is far from normal, it probably means something, so a person looks at it.
    'Close' is never a fixed number. Each limit is the mild IQR fence of how far normal lines
    vary from their own reference.

    1. Positive discount, normal quantity: flip the sign. Does the discount then take a normal share of the
       shelf price (discount / (sales + discount))? The limit is the fence of that share on normal discounted lines.
    2. Quantity below 0, positive sales: flip the quantity. Does sales / |quantity| match the product's typical unit price?
    3. Quantity 0, positive sales: does the sales value match the product's typical sales value per line?
    Reference prices come from that product's own normal lines (quantity > 0, sales > 0); products with fewer than
    MIN_REFERENCE_LINES such lines have no reference and stay escalated.
    """
    n = len(tx)
    out = {"disc_typo": np.zeros(n, bool), "qty_sign_typo": np.zeros(n, bool),
           "qty_missing_ok": np.zeros(n, bool), "limits": {}}

    # 1. discount sign
    normal_disc = (rd < -tol) & (qty > 0) & (sales > 0)
    if unexp_all.any() and normal_disc.sum() >= 100:
        share_norm = (-rd[normal_disc]) / (sales[normal_disc] - rd[normal_disc])
        limit = min(_upper_fence(share_norm), 1.0)
        denom = sales + rd                      # shelf price if the sign is flipped: sales + |discount|
        share = np.where(denom > 0, rd / np.where(denom > 0, denom, 1.0), np.inf)
        out["disc_typo"] = unexp_all & (share <= limit)
        out["limits"]["discount_share"] = limit

    # 2 and 3. quantity
    if qnp_all.any() and "product_id" in tx.columns:
        pid = tx["product_id"].to_numpy()
        ref = (qty > 0) & (sales > 0)
        r = pd.DataFrame({"p": pid[ref], "unit": sales[ref] / qty[ref], "sales": sales[ref]})
        g = r.groupby("p")
        med_unit, med_sales, cnt = g["unit"].median(), g["sales"].median(), g["unit"].size()
        mu = pd.Series(pid).map(med_unit).to_numpy(dtype="float64")
        ms = pd.Series(pid).map(med_sales).to_numpy(dtype="float64")
        ok_ref = pd.Series(pid).map(cnt).fillna(0).to_numpy() >= MIN_REFERENCE_LINES

        rp = r["p"].map(med_unit).to_numpy(dtype="float64")
        d_unit = np.abs(np.log(r["unit"].to_numpy() / rp))
        rp2 = r["p"].map(med_sales).to_numpy(dtype="float64")
        d_sales = np.abs(np.log(r["sales"].to_numpy() / rp2))
        lim_unit, lim_sales = _upper_fence(d_unit), _upper_fence(d_sales)

        neg = qnp_all & (qty < 0) & ok_ref
        zero = qnp_all & (qty == 0) & ok_ref
        with np.errstate(divide="ignore", invalid="ignore"):
            d_neg = np.abs(np.log((sales / np.abs(qty)) / mu))
            d_zero = np.abs(np.log(sales / ms))
        out["qty_sign_typo"] = neg & np.isfinite(d_neg) & (d_neg <= lim_unit)
        out["qty_missing_ok"] = zero & np.isfinite(d_zero) & (d_zero <= lim_sales)
        out["limits"]["unit_price_log_ratio"] = lim_unit
        out["limits"]["line_sales_log_ratio"] = lim_sales
    return out


def build_transaction_flags(tx, outlier_path):
    """Return (flags DataFrame, stats dict). One row per transaction, flag columns only."""
    tol = POSITIVE_DISCOUNT_TOLERANCE
    qty = tx["quantity"].to_numpy(dtype="float64")
    sales = tx["sales_value"].to_numpy(dtype="float64")
    rd = tx["retail_disc"].to_numpy(dtype="float64")
    cd = tx["coupon_disc"].to_numpy(dtype="float64")
    md = tx["coupon_match_disc"].to_numpy(dtype="float64")

    pos = rd > tol
    noise = ((rd > 0) & (rd <= tol)) | ((cd > 0) & (cd <= tol)) | ((md > 0) & (md <= tol))
    adj = pos & (qty <= 0)
    unexp_all = pos & (qty > 0)                              # positive discount on a normal line
    zero_line = (qty <= 0) & (sales <= 0)
    free_item = (qty > 0) & (sales <= 0)
    adj_with_sales = adj & (sales > 0)
    qnp_noise = (qty <= 0) & (sales > 0) & (sales <= tol) & ~adj   # sales positive only by rounding noise (e.g. 1e-16)
    qnp_all = (qty <= 0) & (sales > tol) & ~adj              # real money; adjustment lines are already explained by rule

    # ---- typo test: does the odd value look normal once the sign (or a missing quantity) is fixed? ----
    typo = _typo_tests(tx, qty, sales, rd, tol, unexp_all, qnp_all)
    disc_typo = typo["disc_typo"]
    qty_sign_typo = typo["qty_sign_typo"]
    qty_missing_ok = typo["qty_missing_ok"]
    unexp = unexp_all & ~disc_typo                           # still unexplained -> escalated
    qty_np_sales = qnp_all & ~qty_sign_typo & ~qty_missing_ok

    flags = pd.DataFrame({"row_id": np.arange(len(tx), dtype=np.int64)})
    flags["flag_positive_retail_disc"] = pos
    flags["flag_discount_rounding_noise"] = noise
    flags["flag_adjustment_line"] = adj
    flags["flag_discount_sign_typo"] = disc_typo
    flags["flag_positive_discount_unexplained"] = unexp
    flags["flag_nonpositive_qty"] = qty <= 0
    flags["flag_nonpositive_sales"] = sales <= 0
    flags["flag_zero_line"] = zero_line
    flags["flag_free_item"] = free_item
    flags["flag_qty_sign_typo"] = qty_sign_typo
    flags["flag_qty_missing_money_ok"] = qty_missing_ok
    flags["flag_qty_nonpos_with_sales"] = qty_np_sales
    flags["flag_sales_rounding_noise"] = qnp_noise
    # corrected values live in new columns; the original columns are never overwritten
    flags["retail_disc_corrected"] = np.where(disc_typo, -np.abs(rd), np.nan)
    flags["quantity_corrected"] = np.where(qty_sign_typo, np.abs(qty), np.nan)

    stats = {
        "n": len(tx),
        "pos": int(pos.sum()), "noise": int(noise.sum()), "adj": int(adj.sum()),
        "unexp_all": int(unexp_all.sum()), "disc_typo": int(disc_typo.sum()), "unexp": int(unexp.sum()),
        "qty_np": int((qty <= 0).sum()), "sales_np": int((sales <= 0).sum()),
        "zero_line": int(zero_line.sum()), "free_item": int(free_item.sum()),
        "qnp_noise": int(qnp_noise.sum()),
        "qnp_all": int(qnp_all.sum()), "qty_sign_typo": int(qty_sign_typo.sum()),
        "qty_missing_ok": int(qty_missing_ok.sum()), "qty_np_sales": int(qty_np_sales.sum()),
        "adj_with_sales": int(adj_with_sales.sum()),
        "typo_limits": typo["limits"],
        "big_qty": int((qty > OLD_FIXED_QUANTITY_CUTOFF).sum()),
        "have_outliers": False,
    }

    if outlier_path.exists():
        of = pd.read_parquet(outlier_path)
        if len(of) == len(tx) and (of["row_id"].to_numpy() == flags["row_id"].to_numpy()).all():
            flags = pd.concat([flags, of.drop(columns="row_id").reset_index(drop=True)], axis=1)
            nm = of["is_non_merchandise"].to_numpy()
            nd = of["no_department"].to_numpy()
            out_cols = [c for c in of.columns if c.startswith("out_")]
            extreme_any = (of[out_cols].to_numpy() == 2).any(axis=1)
            stats.update({
                "have_outliers": True,
                "non_merch": int(nm.sum()),
                "big_qty_non_merch": int(((qty > OLD_FIXED_QUANTITY_CUTOFF) & nm).sum()),
                "no_dept": int(nd.sum()),
                "no_dept_zero_line": int((nd & zero_line).sum()),
                "extreme_any": int(extreme_any.sum()),
            })
        else:
            print("WARNING: outlier_flags.parquet does not line up with transaction_data; ignoring it. Rerun outlier_rules.py.")
    if not stats["have_outliers"]:
        flags["flag_qty_outlier"] = qty > OLD_FIXED_QUANTITY_CUTOFF
        print("NOTE: outlier_flags.parquet not found. Run outlier_rules.py first for the product-type and data-derived outlier flags. "
              "Falling back to the old fixed rule (quantity > 100), which needs a person to confirm.")
    return flags, stats


def dynamic_rules(S):
    """Rules whose wording depends on what the data showed. Returns {id: rule tuple} and extra rows."""
    rules = {}
    extra = []   # (id, description, n_rows, rule tuple)
    n = S["n"]

    # Q4a: positive retail discounts
    rules["Q4a"] = ("FLAG", "rule",
        f"{S['pos']:,} rows have a retail discount above half a cent. {S['adj']:,} of them are lines with no quantity "
        f"(a pattern consistent with price-adjustment lines) and are flagged by rule; {S['unexp_all']:,} have a normal quantity "
        f"({S['disc_typo']:,} look like sign typos and are flagged by rule, {S['unexp']:,} stay unexplained and are escalated below). "
        f"A further {S['noise']:,} rows are positive only by floating-point rounding noise (below half a cent) and are ignored.",
        "transaction_flags.parquet (flag_positive_retail_disc, flag_adjustment_line, flag_discount_rounding_noise)",
        "Feature Agent: exclude flagged rows from discount features or treat adjustment lines separately.")
    if S["disc_typo"]:
        lim = S["typo_limits"].get("discount_share")
        extra.append(("Q4a-3", "Positive discount that looks like a sign typo", S["disc_typo"],
            ("FLAG", "rule",
             f"With the sign flipped, the discount is a normal share of the shelf price (at most {lim:.0%}, the mild IQR fence of discount share on normal discounted lines). "
             "So it is probably a sign typo, not a real value. The original column is untouched; the flipped value is in retail_disc_corrected.",
             "transaction_flags.parquet (flag_discount_sign_typo, retail_disc_corrected)",
             "Feature Agent: use retail_disc_corrected for these rows. Probable typo, not confirmed.")))
    if S["unexp"]:
        extra.append(("Q4a-2", "Positive discount on a normal line that does not look like a typo", S["unexp"],
            ("ESCALATE", "human_needed",
             "A positive discount on a line with a normal quantity. Flipping the sign would not give a normal discount, so it may mean something else. A person should look at these rows.",
             "transaction_flags.parquet (flag_positive_discount_unexplained)",
             "Review the listed rows; decide whether to exclude or correct the sign.")))

    # Q5 / Q6: zero and negative lines
    rules["Q5"] = ("FLAG", "rule",
        f"{S['qty_np']:,} rows have a quantity of 0 or less. {S['zero_line']:,} of them also have no sales value (zero lines: "
        f"nothing was bought, likely voided or placeholder lines) and are flagged by rule. {S['adj_with_sales']:,} are the adjustment "
        f"lines described under Q4a. {S['qnp_noise']:,} have a sales value that is positive only by floating-point noise "
        f"(half a cent or less) and are ignored. {S['qnp_all']:,} have real money despite no quantity: "
        f"{S['qty_sign_typo'] + S['qty_missing_ok']:,} look like typos and are flagged by rule, {S['qty_np_sales']:,} stay unexplained "
        f"and are escalated below.",
        "transaction_flags.parquet (flag_nonpositive_qty, flag_zero_line)",
        "Feature Agent: exclude zero lines from purchase counts.")
    if S["qnp_noise"]:
        extra.append(("I6", "Sales value positive only by rounding noise (half a cent or less), no quantity", S["qnp_noise"],
            ("NOTE", "rule",
             "Sales values like 0.0000000000000002 on lines with no quantity are floating-point noise, the same effect as I5 for discounts. "
             "They are effectively zero, so they are not escalated.",
             "transaction_flags.parquet (flag_sales_rounding_noise)",
             "Feature Agent: treat these like zero lines.")))
    typo_q = S["qty_sign_typo"] + S["qty_missing_ok"]
    if typo_q:
        lu = S["typo_limits"].get("unit_price_log_ratio")
        ls = S["typo_limits"].get("line_sales_log_ratio")
        parts = []
        if S["qty_sign_typo"]:
            parts.append(f"{S['qty_sign_typo']:,} have a negative quantity and sales / |quantity| matches the product's typical unit price "
                         f"(within a factor of {np.exp(lu):.1f}, the mild IQR fence), so the quantity sign is probably a typo; |quantity| is in quantity_corrected")
        if S["qty_missing_ok"]:
            parts.append(f"{S['qty_missing_ok']:,} have quantity 0 and a sales value that matches the product's typical sales per line "
                         f"(within a factor of {np.exp(ls):.1f}), so the quantity is probably just missing and the money is kept")
        extra.append(("Q5-3", "Quantity of 0 or less with money, but the money looks normal", typo_q,
            ("FLAG", "rule",
             "; ".join(parts) + ". Originals are untouched.",
             "transaction_flags.parquet (flag_qty_sign_typo, flag_qty_missing_money_ok, quantity_corrected)",
             "Feature Agent: keep the sales value; use quantity_corrected where present. Probable typo, not confirmed.")))
    if S["qty_np_sales"]:
        extra.append(("Q5-2", "Quantity of 0 or less but a positive sales value that does not look normal", S["qty_np_sales"],
            ("ESCALATE", "human_needed",
             "The row says no items were bought but money was recorded, and the amount is far from the product's usual price, or the product has too few normal lines to compare. A person should decide.",
             "transaction_flags.parquet (flag_qty_nonpos_with_sales)",
             "Review a sample; decide whether to count the sales value.")))
    rules["Q6"] = ("FLAG", "rule",
        f"{S['sales_np']:,} rows have a sales value of 0 or less. {S['zero_line']:,} are the zero lines above (nothing bought) and "
        f"{S['free_item']:,} are items with a real quantity but no money (free or fully discounted), flagged by rule.",
        "transaction_flags.parquet (flag_nonpositive_sales, flag_zero_line, flag_free_item)",
        "Feature Agent: exclude zero lines from average-basket features; decide how to treat free items.")

    # Q7: large quantities
    if S["have_outliers"]:
        rest = S["big_qty"] - S["big_qty_non_merch"]
        rules["Q7"] = ("FLAG", "rule",
            f"{S['big_qty_non_merch']:,} of the {S['big_qty']:,} rows with quantity above {OLD_FIXED_QUANTITY_CUTOFF} are gas-kiosk or "
            f"miscellaneous lines, where the unit price is a fraction of a cent, so 'quantity' is not an item count. They are flagged by "
            f"product type, not by a numeric cut-off. {rest:,} merchandise row(s) remain.",
            "transaction_flags.parquet (is_non_merchandise, out_quantity)",
            "Feature Agent: exclude non-merchandise rows from quantity features.")
        extra.append(("Q7b", "Do gas-kiosk and miscellaneous lines count as household spend?", S["non_merch"],
            ("ESCALATE", "human_needed",
             "These lines carry real money but are not grocery shopping. Counting them changes every spend-based feature. This is a team decision, not a data fact.",
             "transaction_flags.parquet (is_non_merchandise)",
             "Team: decide whether non-merchandise spend counts toward household spend.")))
        extra.append(("O1", "Unusually large values on ordinary merchandise (data-derived cut-offs)", S["extreme_any"],
            ("FLAG", "rule",
             "Cut-offs come from each field's own distribution on merchandise rows (log IQR; percentiles when the IQR is 0). "
             "Only the extreme level is used as a flag; the mild level is informational. These are watch flags, not errors.",
             "transaction_flags.parquet (out_* columns); cut-offs in outlier_thresholds.json",
             "Feature Agent: cap or winsorize flagged values before aggregating.")))
        if S["no_dept"]:
            extra.append(("D1", "Products with no department or commodity in the product table", S["no_dept"],
                ("FLAG", "rule",
                 f"{S['no_dept']:,} rows belong to products with no department or commodity. {S['no_dept_zero_line']:,} of them are zero lines, "
                 f"so these look like placeholder products.",
                 "transaction_flags.parquet (no_department)",
                 "Feature Agent: do not count these rows as purchases.")))
    else:
        rules["Q7"] = ("FLAG", "human_needed",
            "Quantity above 100 could be bulk buying, a data error, or non-merchandise lines. Run outlier_rules.py to separate them by product type.",
            "transaction_flags.parquet (flag_qty_outlier)",
            "Run outlier_rules.py, then rerun this script.")
    return rules, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    args = ap.parse_args()
    proc = Path(args.processed)

    fpath = proc / "findings.json"
    if not fpath.exists():
        sys.exit(f"{fpath} not found. Run check_integrity.py first.")
    findings = json.loads(fpath.read_text())["findings"]
    pct_by_id = {f["check_id"]: f.get("pct_of_table") for f in findings}
    run_time = datetime.now().isoformat(timespec="seconds")

    log = []

    def record(check_id, finding, n, action, decided_by, rationale, output, downstream, executed, rows_affected, pct=None):
        log.append({
            "run_time": run_time, "check_id": check_id, "finding": finding, "n_rows": int(n),
            "pct_of_table": None if pct is None else round(float(pct), 4),
            "action": action, "decided_by": decided_by, "rationale": rationale,
            "output": output, "downstream_note": downstream,
            "executed": bool(executed), "rows_affected": int(rows_affected),
        })

    # ---- HALT rule: any failed integrity check stops everything ----
    errors = [f for f in findings if f["status"] == "ERROR"]
    if errors:
        for f in errors:
            record(f["check_id"], f["description"], f["n_rows"], "HALT", "rule",
                   "Tables disagree with each other. Cleaning on top of a broken join would hide the problem.",
                   "", "Fix the source or the loader, then rerun check_integrity.py.", False, 0)
        _save(log, proc)
        print(f"HALT: {len(errors)} integrity check(s) failed. No data was changed. See decision_log.csv.")
        sys.exit(1)

    # ---- Which findings need a decision? ----
    pending = []
    for f in findings:
        cid, status = f["check_id"], f["status"]
        if cid.startswith("X") or status == "OK":
            continue
        pending.append(f)
    pending_ids = {f["check_id"] for f in pending}

    # ---- Execute FIX: dedupe coupon ----
    fix_rows = 0
    if "Q1" in pending_ids:
        cp = pd.read_parquet(proc / "coupon.parquet")
        cp.columns = [c.lower() for c in cp.columns]
        clean = cp.drop_duplicates()
        fix_rows = len(cp) - len(clean)
        clean.to_parquet(proc / "coupon_clean.parquet", index=False)

    # ---- Execute FLAG: transaction flag columns ----
    S, rules, extra, flags = None, dict(RULES), [], None
    if pending_ids & TX_DEPENDENT:
        tx = pd.read_parquet(proc / "transaction_data.parquet")
        tx.columns = [c.lower() for c in tx.columns]
        flags, S = build_transaction_flags(tx, proc / "outlier_flags.parquet")
        flags.to_parquet(proc / "transaction_flags.parquet", index=False)
        dyn, extra = dynamic_rules(S)
        rules.update(dyn)

    # ---- Record a decision for every pending finding ----
    for f in pending:
        cid = f["check_id"]
        rule = rules.get(cid)
        if rule is None:
            record(cid, f["description"], f["n_rows"], "ESCALATE", "human_needed",
                   "No rule exists for this check. A person must decide.", "", "", False, 0, f.get("pct_of_table"))
            continue
        action, who, why, out, down = rule
        if action == "FIX":
            affected, executed = fix_rows, True
        elif S is not None and cid in TX_DEPENDENT:
            affected = {"Q4a": S["pos"], "Q5": S["qty_np"], "Q6": S["sales_np"], "Q7": S["big_qty"]}[cid]
            executed = True
        else:
            affected = f["n_rows"] if action == "FLAG" else 0
            executed = action in ("FLAG", "NOTE")
        record(cid, f["description"], f["n_rows"], action, who, why, out, down, executed, affected, pct_by_id.get(cid))

    for cid, desc, n_rows, (action, who, why, out, down) in extra:
        record(cid, desc, n_rows, action, who, why, out, down, action in ("FLAG", "NOTE"), n_rows if action == "FLAG" else 0,
               100 * n_rows / S["n"])

    _save(log, proc)

    df = pd.DataFrame(log)
    print("\nDECISION LOG")
    print("=" * 100)
    print(f"{'ACTION':<9}{'ID':<7}{'ROWS':>9}  {'DECIDED_BY':<13} FINDING")
    print("-" * 100)
    for _, r in df.iterrows():
        print(f"{r['action']:<9}{r['check_id']:<7}{r['n_rows']:>9,}  {r['decided_by']:<13} {r['finding'][:58]}")
    print("-" * 100)
    print(df["action"].value_counts().to_string())
    human = df[df["decided_by"] == "human_needed"]
    print(f"\n{len(human)} decision(s) need a human to confirm (decided_by = human_needed): {', '.join(human['check_id'])}")
    if fix_rows:
        print(f"coupon_clean.parquet written ({fix_rows:,} duplicate rows removed).")
    if S is not None:
        print("\nFLAG COLUMNS WRITTEN TO transaction_flags.parquet (rows are flagged, none deleted)")
        print("=" * 100)
        print(f"{'COLUMN':<38}{'ROWS':>10}  MEANING")
        print("-" * 100)
        for col in flags.columns:
            if col == "row_id":
                continue
            if flags[col].dtype == bool:
                print(f"{col:<38}{int(flags[col].sum()):>10,}  {FLAG_LABELS.get(col, '')}")
        for col in flags.columns:
            if col.startswith("out_"):
                print(f"{col:<38}{int((flags[col] == 2).sum()):>10,}  extreme level (watch flag, not an error)")
    print(f"Saved: {proc / 'decision_log.csv'}")


def _save(log, proc):
    pd.DataFrame(log).to_csv(proc / "decision_log.csv", index=False)
    (proc / "decision_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
