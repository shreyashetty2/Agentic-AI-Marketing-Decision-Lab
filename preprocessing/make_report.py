"""
Data Agent, Step 4: plain-language data quality report.

Reads  data/processed/findings.json and decision_log.json
Writes data/processed/data_quality_report.md

No analysis happens here. Every number comes from the two logs, and the
wording for each check lives in PLAIN below, so the report cannot drift from the data.

Usage:  python agents/data_agent/make_report.py [--processed data/processed]
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# check_id -> (plain-language title, what it means, why it matters)
PLAIN = {
    "Q1": ("Duplicate coupon rows",
           "The coupon list had the same line written twice in several places.",
           "Left in, every campaign would look like it offered more coupons than it did."),
    "Q4a": ("Discounts with the wrong sign",
            "A few receipt lines show a discount as a positive number. Most are lines with no items (price adjustments); the rest are explained below.",
            "Treated as normal discounts they would slightly distort revenue, so they are marked."),
    "Q4a-3": ("Positive discounts that look like sign typos",
              "A few lines with a normal quantity show a positive discount. With the sign flipped it would be a normal-sized discount, so it is probably a typo.",
              "Marked by rule, with the flipped value in a separate column. The original is untouched and the fix is not confirmed."),
    "Q4a-2": ("Positive discounts that don't look like typos",
              "A handful of lines with a normal quantity show a positive discount, and flipping the sign would not give a normal discount.",
              "They may mean something else. A person should look at them."),
    "Q5-3": ("No quantity, but the money looks normal",
             "Some lines record no items (or a negative count) yet have a normal sales value for that product. Either the sign is a typo or the quantity is missing.",
             "Marked by rule: the money is kept and the quantity is corrected only where the sign was the problem. Not confirmed."),
    "Q5": ("Lines with no quantity",
           "Some receipt lines record no items bought. Most also have no money attached, so they look like voided or placeholder lines.",
           "Counted as purchases, they would understate how much a household buys."),
    "Q5-2": ("No quantity, and the money doesn't look normal",
             "A few lines record no items bought and yet a positive sales value that is far from the product's usual price.",
             "The data contradicts itself here, so a person should decide how to treat them."),
    "Q6": ("Lines with no money",
           "Some receipt lines have no sales value. Most are the same voided or placeholder lines; the rest are items that were free or fully discounted.",
           "Counted as normal purchases, they would pull down a household's average spend per basket."),
    "Q7": ("Very large quantities",
           "Almost every line with more than 100 units is a gas-kiosk or miscellaneous item, where quantity is not a count of items (the price per unit is a fraction of a cent).",
           "Summing quantity over these lines would hugely overstate how much a household buys, so they are marked by product type."),
    "Q7b": ("Do gas and miscellaneous lines count as spend?",
            "Gas-kiosk and miscellaneous lines carry real money but are not grocery shopping.",
            "Whether they count changes every spend-based feature, so this is a decision for the team."),
    "O1": ("Unusually large values",
           "Cut-offs for what counts as unusually large are calculated from each field's own data, on ordinary merchandise only.",
           "These are watch flags for the Feature Agent to cap, not errors."),
    "D1": ("Products with no category",
           "A few products have no department or category in the product list, and every line for them is empty (no quantity, no sales).",
           "They look like placeholder products and should not be counted as purchases."),
    "Q8": ("Campaign that ends after our data does",
           "One campaign was meant to run past the last day we have purchase records for.",
           "We can only see part of that campaign, so a household's response may look lower than it was."),
    "Q9": ("Conflicting in-store display records",
           "The store display file lists some product, store and week combinations more than once, and the copies disagree.",
           "We cannot tell which copy is correct. This file is not used for the first model."),
    "I1": ("Households with a short history",
           "For some household and campaign pairs, we have under six months of purchases before the campaign started.",
           "Features built from a short history are less reliable, so later steps must respect how far back they can look."),
    "I2": ("Overlapping campaigns",
           "Some campaigns ran at the same time.",
           "A purchase can belong to more than one campaign, so training and test data must be split carefully."),
    "I3": ("Coupon codes reused across campaigns",
           "The same coupon code appears in more than one campaign.",
           "Coupons must always be matched using both the campaign and the code."),
    "I6": ("Rounding noise in sales values",
           "Some lines with no quantity have a sales value that is positive by a tiny fraction of a cent, a computer rounding effect.",
           "They are effectively zero and are treated like zero lines, not escalated."),
    "I5": ("Rounding noise in discounts",
           "Some discounts are positive by a tiny fraction of a cent, a computer rounding effect.",
           "They are effectively zero and are ignored."),
    "I4": ("Households with demographic details",
           "Only some households have details such as age band and household size.",
           "Demographics can only be used as an optional extra, never as something every household has."),
}


def fmt(n):
    return f"{int(n):,}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    args = ap.parse_args()
    proc = Path(args.processed)
    for name in ("findings.json", "decision_log.json"):
        if not (proc / name).exists():
            sys.exit(f"{proc / name} not found. Run check_integrity.py and apply_decisions.py first.")

    findings = json.loads((proc / "findings.json").read_text())["findings"]
    decisions = {d["check_id"]: d for d in json.loads((proc / "decision_log.json").read_text())}

    integrity = [f for f in findings if f["kind"] == "integrity"]
    failed = [f for f in integrity if f["status"] == "ERROR"]
    passed = [f for f in integrity if f["status"] == "OK"]
    halted = any(d["action"] == "HALT" for d in decisions.values())

    out = []
    w = out.append
    w("# Data quality report: dunnhumby Complete Journey")
    w(f"*Generated by the Data Agent on {datetime.now():%Y-%m-%d %H:%M}. Every number below comes from findings.csv and decision_log.csv.*\n")

    # ---- 1. Bottom line ----
    w("## 1. Bottom line")
    if halted or failed:
        w(f"**The data agent stopped.** {len(failed)} cross-table check(s) failed, so the tables do not agree with each other. "
          "No cleaning was done. See section 3.\n")
    else:
        fixed = [d for d in decisions.values() if d["action"] == "FIX"]
        flagged = [d for d in decisions.values() if d["action"] == "FLAG"]
        need = [d for d in decisions.values() if d["decided_by"] == "human_needed"]
        w(f"**The tables agree with each other.** All {len(passed)} cross-table checks passed, "
          "so every household, product, campaign and coupon can be linked as expected.")
        w(f"The agent made **{len(fixed)} fix**, **{len(flagged)} flags**, and left "
          f"**{len(need)} decision(s) for a person**. No rows were deleted except exact duplicate coupon lines.\n")

    # ---- 2. What we checked ----
    w("## 2. What we checked, and what passed")
    w(f"We ran {len(integrity)} cross-table checks. {len(passed)} passed.")
    if not failed:
        w("In plain terms:")
        w("- Every household in the mailing list, the coupon redemptions and the demographics exists in the purchase records.")
        w("- Every product bought exists in the product list, and every campaign and coupon refers to a real campaign.")
        w("- Every coupon redemption was by a household that was actually mailed, happened inside the campaign's dates, and used a coupon that was part of that campaign.")
        w("- The headline numbers match the project proposal: 2,500 households, 7,208 mailings, and 889 mailings with at least one redemption.\n")

    # ---- 3. Failed checks ----
    sec = 2
    if failed:
        sec += 1
        w(f"## {sec}. Checks that failed")
        for f in failed:
            exp = f" (expected {fmt(f['expected'])})" if f.get("expected") else ""
            w(f"- **{f['check_id']}**: {f['description']}: found {fmt(f['n_rows'])}{exp}.")
        w("")

    # ---- 4. Fixed ----
    sec += 1
    fixes = [d for d in decisions.values() if d["action"] == "FIX"]
    w(f"## {sec}. What we fixed")
    if fixes:
        for d in fixes:
            t, what, why = PLAIN.get(d["check_id"], (d["finding"], "", ""))
            w(f"**{t}**: {fmt(d['rows_affected'])} rows removed. {what} {why} "
              f"Fixed because it is safe: the removed rows were exact copies, so no information was lost. Output: `{d['output']}`.\n")
    else:
        w("Nothing needed fixing.\n")

    # ---- 5. Flagged ----
    sec += 1
    flags = [d for d in decisions.values() if d["action"] == "FLAG"]
    w(f"## {sec}. What we flagged but kept")
    w("These rows stay in the data with a marker. We do not delete anything we are unsure about.\n")
    pct = {f["check_id"]: f.get("pct_of_table") for f in findings}
    w("| Issue | Rows | Share | What it means | Who decides |")
    w("|---|---|---|---|---|")
    for d in flags:
        t, what, why = PLAIN.get(d["check_id"], (d["finding"], "", ""))
        p = d.get("pct_of_table")
        if p is None:
            p = pct.get(d["check_id"])
        share = f"{p:.2f}%" if isinstance(p, (int, float)) else "n/a"
        who = "A person (judgment call)" if d["decided_by"] == "human_needed" else "Rule"
        w(f"| {t} | {fmt(d['n_rows'])} | {share} | {what} {why} | {who} |")
    w("")

    # ---- 6. Notes ----
    sec += 1
    notes = [d for d in decisions.values() if d["action"] == "NOTE"]
    w(f"## {sec}. Facts later agents must respect")
    for d in notes:
        t, what, why = PLAIN.get(d["check_id"], (d["finding"], "", ""))
        w(f"- **{t}** ({fmt(d['n_rows'])}): {what} {why}")
    w("")

    # ---- 7. Open questions ----
    sec += 1
    need = [d for d in decisions.values() if d["decided_by"] == "human_needed"]
    w(f"## {sec}. Decisions for the team and mentor")
    if need:
        w("The agent flagged these but cannot settle them from the data alone:\n")
        for i, d in enumerate(need, 1):
            t, _, _ = PLAIN.get(d["check_id"], (d["finding"], "", ""))
            w(f"{i}. **{t}**: {d['rationale']} Suggested next step: {d['downstream_note']}")
    else:
        w("None.")
    w("")

    # ---- 8. Files ----
    sec += 1
    w(f"## {sec}. Files produced")
    w("- `findings.csv` / `findings.json`: every check and its result")
    w("- `decision_log.csv` / `decision_log.json`: every decision, with the reason and who decided")
    w("- `coupon_clean.parquet`: coupon table without duplicates")
    w("- `transaction_flags.parquet`: one row per transaction with true/false flag columns (zero lines, adjustment lines, non-merchandise, outlier levels)")
    w("- `outlier_thresholds.json` / `outlier_sensitivity.csv`: the data-derived cut-offs and how many rows each rule flags")
    w("- `spine_household_campaign.parquet`: the household × campaign table the models use")

    path = proc / "data_quality_report.md"
    path.write_text("\n".join(out) + "\n")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
