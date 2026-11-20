"""Check: how accurate is the Decision Lab's expected revenue?

The Decision Lab (src/decision/table.py, PR #46) answers marketers' dollar questions with

    expected revenue = P(redeem)  x  spend if they redeem
                       Model 1        Model 2 (line on usual spend, applied to every mailed household)

This script does not change the Lab. It measures how close that number comes to what households
actually spent, so the Lab's answers can say how much to trust it.

Two versions of the number are scored on the same households:
  - Fair version: both models trained only on campaigns that ENDED before each test campaign's
    feature cutoff (start day - cutoff_gap_days). Model 1 is the model in use (owner_forest);
    Model 2's line is refit on the redeemed TypeB/C rows of those same campaigns. This is how the
    number would have looked when the campaign was planned.
  - As the Lab computes it: the same P(redeem), times Model 2's line fit on all 254 rows. That fit
    includes campaigns that started after the test campaign, so it has seen later spending.

Scope
  - TypeB/C mailed households only: Model 2 has no TypeA estimate (a TypeA household's 16 coupons
    are not recorded), and the Lab leaves TypeA blank too.
  - Open test campaigns only: Model 1's forward chaining (from first_test_day), minus each model's
    locked campaigns (modeling_spec.yaml locked_test_campaigns). The held-back campaigns stay unread.

"Actual revenue" = spend on the campaign's products during the campaign by households that redeemed
(Model 2's target), and $0 for households that did not redeem: the quantity the formula forecasts.

Checks
  1. Totals   forecast vs. actual revenue per campaign and pooled, split into its two parts:
              redemptions (sum of P vs. actual count) and spend per redeemer.
  2. Ranking  share of actual revenue reached by mailing the top X% of each campaign's list, ranked
              by expected revenue vs. by P(redeem) alone vs. random; at 20% and averaged over
              5-50% (Model 1's main measure). 95% range for the gap from resampling households.

Run from the repo folder (after src/features/build_2b.py):
    python src/modeling/check_expected_revenue.py --processed data/processed

Input : data/processed/model1_table.parquet, model2_table.parquet (Feature Agent 2B)
Output: data/processed/expected_revenue_check.md
        data/processed/expected_revenue_check.parquet   (one row per scored household x campaign)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # lets `features.*` / `modeling.*` import as a script

from modeling import model2_value  # noqa: E402
from modeling.agent import load_modeling_spec  # noqa: E402
from modeling.model1_baseline import FEATURES as M1_FEATURES  # noqa: E402
from modeling.model1_baseline import LABEL as M1_LABEL  # noqa: E402
from modeling.model1_evaluation import (  # noqa: E402
    CURVE_SHARES, bootstrap_capture, forward_chain_folds, make_model, prepare_table,
    random_expected_capture, random_expected_capture_avg, within_campaign_capture, within_campaign_capture_avg,
)

SPEND = model2_value.LABEL            # campaign_product_spend_during
USUAL = model2_value.USUAL            # usual_spend


# --------------------------------------------------------------------------- data
def usual_spend(table: pd.DataFrame) -> pd.Series:
    """Model 2's input for any row. Blank weekly spend only means no history / no spend in the 26
    weeks (feature_2a_spec.yaml), where 0 is the correct reading -- same handling as model2_value."""
    weekly = table["campaign_product_spend_per_week_26w"].fillna(0)
    return weekly * table["duration_days"] / 7


def attach_spend(model1: pd.DataFrame, model2: pd.DataFrame) -> pd.DataFrame:
    """Model 1's table carries only its own answer column, so the spend of each redeemed TypeB/C row comes from
    Model 2's table (exactly those rows). Rows that did not redeem get no spend; they count as $0 revenue."""
    keys = ["household_key", "campaign"]
    out = model1.merge(model2[keys + [SPEND]], on=keys, how="left", validate="one_to_one")
    redeemed_bc = (out[M1_LABEL] == 1) & (out["campaign_type"] != "A")
    if out.loc[redeemed_bc, SPEND].isna().any() or out.loc[~redeemed_bc, SPEND].notna().any():
        raise ValueError("Model 2's rows should be exactly Model 1's redeemed TypeB/C rows; rebuild with build_2b.py")
    return out


def locked_campaigns(spec: dict, model1: pd.DataFrame, model2: pd.DataFrame) -> set:
    """Each model's last locked_test_campaigns campaigns (by start day), as its agent locks them. Both are
    left out so no held-back campaign is read again."""
    locked = set()
    for table, cfg in ((model1, spec["models"][1]), (model2, spec["models"][2])):
        n = cfg.get("locked_test_campaigns", 0)
        if n:
            order = table[["campaign", "start_day"]].drop_duplicates().sort_values(["start_day", "campaign"])
            locked |= set(order["campaign"].iloc[-n:].astype(int))
    return locked


def open_folds(model1: pd.DataFrame, first_test_day: int, locked: set, min_train_rows: int,
               min_train_positives: int) -> list[dict]:
    """Model 1's forward-chaining folds (same minimums as its agent), TypeB/C test campaigns only, not locked."""
    folds = []
    for f in forward_chain_folds(model1, first_test_day):
        train = model1[model1["campaign"].isin(f["train_campaigns"])]
        big_enough = len(train) >= min_train_rows and train[M1_LABEL].sum() >= min_train_positives
        if f["campaign_type"] != "A" and int(f["test_campaign"]) not in locked and big_enough:
            folds.append(f)
    return folds


# --------------------------------------------------------------------------- scoring
def score_folds(model1: pd.DataFrame, model2: pd.DataFrame, folds: list[dict],
                min_m2_rows: int = model2_value.MIN_TRAIN_ROWS) -> tuple[pd.DataFrame, list[dict]]:
    """Score every mailed TypeB/C household of each test campaign with both models in use, each trained on
    the campaigns that ended before that campaign's cutoff."""
    parts, info = [], []
    for f in folds:
        m1_train = model1[model1["campaign"].isin(f["train_campaigns"])]
        m2_train = model2[model2["campaign"].isin(f["train_campaigns"])]
        if len(m2_train) < min_m2_rows:
            info.append({"campaign": int(f["test_campaign"]), "skipped": f"Model 2 has {len(m2_train)} training rows"})
            continue
        m1 = make_model("rf").fit(m1_train[M1_FEATURES], m1_train[M1_LABEL])
        m2 = model2_value.LinearUsualSpend().fit(m2_train)
        test = model1[model1["campaign"] == f["test_campaign"]]
        p = m1.predict_proba(test[M1_FEATURES])[:, 1]
        spend = m2.predict(test)
        redeemed = test[M1_LABEL].to_numpy()
        parts.append(pd.DataFrame({
            "household_key": test["household_key"].to_numpy(), "campaign": test["campaign"].to_numpy(),
            "campaign_type": test["campaign_type"].to_numpy(), USUAL: test[USUAL].to_numpy(),
            "p_redeem": p, "expected_spend": spend, "expected_revenue": p * spend,
            "redeemed": redeemed, "actual_revenue": np.where(redeemed == 1, test[SPEND].to_numpy(), 0.0),
        }))
        info.append({"campaign": int(f["test_campaign"]), "m1_train_rows": len(m1_train),
                     "m2_train_rows": len(m2_train), "slope": m2.slope, "intercept": m2.intercept})
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), info


# --------------------------------------------------------------------------- checks
def totals(preds: pd.DataFrame) -> pd.DataFrame:
    """Per campaign and pooled: forecast vs. actual revenue, and its two parts."""
    def row(d: pd.DataFrame) -> dict:
        red = d[d["redeemed"] == 1]
        return {"mailed": len(d), "redeemers": int(d["redeemed"].sum()),
                "forecast_redemptions": d["p_redeem"].sum(),
                "actual_revenue": d["actual_revenue"].sum(), "forecast_revenue": d["expected_revenue"].sum(),
                "spend_ratio": red["expected_spend"].sum() / red["actual_revenue"].sum() if len(red) else np.nan}
    out = pd.DataFrame([{"campaign": c, **row(d)} for c, d in preds.groupby("campaign")]
                       + [{"campaign": "All", **row(preds)}])
    if "lab_expected_revenue" in preds:
        out["lab_forecast_revenue"] = [d["lab_expected_revenue"].sum() for _, d in preds.groupby("campaign")] \
            + [preds["lab_expected_revenue"].sum()]
        out["lab_revenue_ratio"] = out["lab_forecast_revenue"] / out["actual_revenue"]
    out["redemption_ratio"] = out["forecast_redemptions"] / out["redeemers"]
    out["revenue_ratio"] = out["forecast_revenue"] / out["actual_revenue"]
    return out


def ranking(preds: pd.DataFrame, reps: int = 300) -> dict:
    """Share of actual revenue reached by mailing the top of each campaign's list."""
    c, y = preds["campaign"].to_numpy(), preds["actual_revenue"].to_numpy()
    rev, p = preds["expected_revenue"].to_numpy(), preds["p_redeem"].to_numpy()
    out = {"random_20": random_expected_capture(c, y, 0.2), "random_avg": random_expected_capture_avg(c, y)}
    for name, score in (("revenue", rev), ("p_only", p)):
        out[f"{name}_20"] = within_campaign_capture(c, y, score, 0.2)
        out[f"{name}_avg"] = within_campaign_capture_avg(c, y, score)
    if "lab_expected_revenue" in preds:
        lab = preds["lab_expected_revenue"].to_numpy()
        out["lab_20"], out["lab_avg"] = within_campaign_capture(c, y, lab, 0.2), within_campaign_capture_avg(c, y, lab)
    out["gap_20_ci"] = bootstrap_capture(c, y, rev, p, reps=reps)["diff_ci"]
    out["gap_avg_ci"] = _bootstrap_avg_gap(c, y, rev, p, reps=reps)
    out["depth_curve"] = pd.DataFrame({
        "mail share": [f"{s:.0%}" for s in CURVE_SHARES],
        "By expected revenue": [within_campaign_capture(c, y, rev, s) for s in CURVE_SHARES],
        "By P(redeem) only": [within_campaign_capture(c, y, p, s) for s in CURVE_SHARES],
        "Random": [random_expected_capture(c, y, s) for s in CURVE_SHARES],
    })
    return out


def _bootstrap_avg_gap(campaign, y, a, b, reps: int = 300, seed: int = 1) -> np.ndarray:
    """95% range for capture_avg(a) - capture_avg(b), resampling households within each campaign
    (the same scheme as model1_evaluation.bootstrap_capture)."""
    groups = list(pd.Series(np.arange(len(campaign))).groupby(campaign).indices.values())
    rng, diffs = np.random.default_rng(seed), []
    for r in range(reps):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        diffs.append(within_campaign_capture_avg(campaign[idx], y[idx], a[idx], n_seeds=1, seed=r)
                     - within_campaign_capture_avg(campaign[idx], y[idx], b[idx], n_seeds=1, seed=r))
    return np.nanpercentile(diffs, [2.5, 97.5])


# --------------------------------------------------------------------------- report
def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def render_report(preds, tot, rank, info, locked, fitted) -> str:
    pooled = tot[tot["campaign"] == "All"].iloc[0]
    skipped = [i for i in info if "skipped" in i]
    per = tot.assign(**{
        "actual revenue ($)": tot["actual_revenue"].round(0).astype(int),
        "fair forecast ($)": tot["forecast_revenue"].round(0).astype(int),
        "revenue ratio (fair)": tot["revenue_ratio"].round(2),
        "revenue ratio (as in the Lab)": tot["lab_revenue_ratio"].round(2),
        "redemptions (forecast / actual)": [f"{f:.1f} / {a}" for f, a in zip(tot["forecast_redemptions"], tot["redeemers"])],
        "spend ratio": tot["spend_ratio"].round(2),
    })[["campaign", "mailed", "actual revenue ($)", "fair forecast ($)", "revenue ratio (fair)", "revenue ratio (as in the Lab)",
        "redemptions (forecast / actual)", "spend ratio"]]
    folds = pd.DataFrame([i for i in info if "skipped" not in i])
    folds = folds.assign(line=[f"${b:.2f} + {a:.2f} x usual" for a, b in zip(folds["slope"], folds["intercept"])])[
        ["campaign", "m1_train_rows", "m2_train_rows", "line"]].rename(
        columns={"m1_train_rows": "Model 1 training rows", "m2_train_rows": "Model 2 training rows"})
    curve = rank["depth_curve"].copy()
    for col in curve.columns[1:]:
        curve[col] = curve[col].map(lambda v: f"{v:.1%}")
    g20, gavg = rank["gap_20_ci"], rank["gap_avg_ci"]
    return f"""# Check: how accurate is the Decision Lab's expected revenue?

The Lab (src/decision/table.py) answers with expected revenue = P(redeem) (Model 1 in use) x spend if they redeem
(Model 2 in use). This check scores that number against what households actually spent.

- **Fair version:** both models trained only on campaigns that ended before each test campaign's cutoff.
- **As in the Lab:** the same P(redeem) x Model 2's line fit on all {fitted['rows']} rows
  (**${fitted['intercept']:.2f} + {fitted['slope']:.2f} x usual spend**), which includes later campaigns.

## Scope

- {len(preds):,} mailed TypeB/C household x campaign rows from {preds['campaign'].nunique()} open test campaigns
  ({int(preds['redeemed'].sum())} redeemed), each scored by both models trained only on campaigns that ended before its cutoff.
- Locked campaigns left out (not read): {sorted(locked)}.{' Skipped (too little Model 2 history): ' + ', '.join(str(s['campaign']) for s in skipped) + '.' if skipped else ''}
- Actual revenue = spend on the campaign's products by households that redeemed ($0 otherwise).

## 1. Totals

Pooled, fair version: forecast **${pooled['forecast_revenue']:,.0f}** vs. actual **${pooled['actual_revenue']:,.0f}**
(ratio **{pooled['revenue_ratio']:.2f}**). As in the Lab: **${pooled['lab_forecast_revenue']:,.0f}** (ratio **{pooled['lab_revenue_ratio']:.2f}**).
In the fair version, the two parts: redemptions {pooled['forecast_redemptions']:.1f} forecast vs.
{int(pooled['redeemers'])} actual (ratio {pooled['redemption_ratio']:.2f}); spend per redeemer ratio {pooled['spend_ratio']:.2f}.

{_md(per)}

## 2. Ranking: revenue reached by mailing the top of each list

| Ranked by | Top 20% | Averaged over 5-50% |
|---|---|---|
| Expected revenue, fair version | {rank['revenue_20']:.1%} | {rank['revenue_avg']:.1%} |
| Expected revenue, as in the Lab | {rank['lab_20']:.1%} | {rank['lab_avg']:.1%} |
| P(redeem) only (Model 1 alone) | {rank['p_only_20']:.1%} | {rank['p_only_avg']:.1%} |
| Random | {rank['random_20']:.1%} | {rank['random_avg']:.1%} |

Fair expected revenue minus P(redeem) only: top 20% {rank['revenue_20'] - rank['p_only_20']:+.1%} (95% range {g20[0]:+.1%} to {g20[1]:+.1%});
averaged {rank['revenue_avg'] - rank['p_only_avg']:+.1%} (95% range {gavg[0]:+.1%} to {gavg[1]:+.1%}).

By mailing depth:

{_md(curve)}

## Model 2 in each fold

The line is refit for each test campaign on the redeemed TypeB/C rows of campaigns that ended before its cutoff.

{_md(folds)}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--reps", type=int, default=300, help="bootstrap resamples for the ranking gap")
    args = ap.parse_args()

    spec = load_modeling_spec()
    model1 = prepare_table(args.processed)
    model2 = model2_value.load_table(args.processed)
    model1 = attach_spend(model1.assign(**{USUAL: usual_spend(model1)}), model2)
    locked = locked_campaigns(spec, model1, model2)
    fc = spec["models"][1]["forward_chain"]
    folds = open_folds(model1, spec["defaults"]["first_test_day"], locked, fc["min_train_rows"], fc["min_train_positives"])
    preds, info = score_folds(model1, model2, folds)
    if preds.empty:
        raise SystemExit("no open TypeB/C test campaign had enough training data")

    full = model2_value.LinearUsualSpend().fit(model2)   # what the Decision Lab uses: all 254 rows
    fitted = {"rows": len(model2), "slope": full.slope, "intercept": full.intercept}
    preds["lab_expected_spend"] = full.predict(preds)
    preds["lab_expected_revenue"] = preds["p_redeem"] * preds["lab_expected_spend"]
    tot, rank = totals(preds), ranking(preds, reps=args.reps)

    preds.to_parquet(args.processed / "expected_revenue_check.parquet", index=False)
    report = render_report(preds, tot, rank, info, locked, fitted)
    (args.processed / "expected_revenue_check.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
