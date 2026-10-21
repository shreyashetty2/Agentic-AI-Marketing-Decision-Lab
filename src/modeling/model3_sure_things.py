"""Model 3: "sure things" classifier.

A T-learner answers "how much does mailing change the chance of buying" -- it needs both treated
and control households to compute a difference. This script answers a narrower, directly actionable
question instead: among households the retailer is ALREADY mailing, which ones were going to buy the
product regardless, without ever touching the coupon? That's a plain classification problem on
treated households only, using bought_campaign_product_without_redemption (Feature Agent 2B) as the
label -- not a second uplift model (control households can never redeem a coupon they never
received, so that label never varies for them; a T-learner's Model C side can't be trained on it).

Run from the repo folder:
    python src/modeling/model3_sure_things.py --processed data/processed

Input : data/processed/model3_table.parquet (from src/features/build_2b.py), treated rows only
Output: prints results to the console and writes data/processed/model3_sure_things_report.md
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Kept identical to model3_uplift.py (not imported across scripts, same convention as
# model3_redemption_analysis.py's own copies of these constants).
WITHOUT_REDEMPTION = "bought_campaign_product_without_redemption"
TREATMENT = "mailed_flag"
TRAIN_FRACTION = 0.7
FEATURES = [
    "spend_per_week_8w", "spend_per_week_26w", "avg_basket_value_26w",
    "trips_per_week_26w", "coupon_lines_per_week_26w", "share_spend_on_discount_26w",
    "past_redemption_rate", "campaign_product_spend_per_week_26w", "campaign_product_share_26w",
    "history_weeks_before_cutoff", "duration_days",
]


def load_table(processed: Path) -> pd.DataFrame:
    table = pd.read_parquet(processed / "model3_table.parquet")
    return table.assign(**{c: table[c].fillna(0) for c in FEATURES if table[c].isna().any()})


def time_split(table: pd.DataFrame, train_fraction: float = TRAIN_FRACTION):
    campaigns = table[["campaign", "start_day"]].drop_duplicates().sort_values("start_day")
    cutoff = int(len(campaigns) * train_fraction)
    train_campaigns = set(campaigns["campaign"].iloc[:cutoff])
    test_campaigns = set(campaigns["campaign"].iloc[cutoff:])
    return table[table["campaign"].isin(train_campaigns)], table[table["campaign"].isin(test_campaigns)]


def train_and_evaluate(table: pd.DataFrame) -> dict:
    train, test = time_split(table)
    train_t, test_t = train[train[TREATMENT] == 1], test[test[TREATMENT] == 1]

    # Same reasoning as train_t_learner() in model3_uplift.py: no class_weight rebalancing --
    # it distorts predict_proba's calibration, and there's enough natural signal without it.
    model = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42)
    model.fit(train_t[FEATURES], train_t[WITHOUT_REDEMPTION])

    pred = model.predict_proba(test_t[FEATURES])[:, 1]
    auc = roc_auc_score(test_t[WITHOUT_REDEMPTION], pred)

    ranked = test_t.assign(predicted=pred).sort_values("predicted", ascending=False)
    top_n = max(1, int(len(ranked) * 0.2))
    top_fifth = ranked.iloc[:top_n]
    capture = {
        "n": len(ranked),
        "top_n": top_n,
        "base_rate": float(test_t[WITHOUT_REDEMPTION].mean()),
        "top_fifth_rate": float(top_fifth[WITHOUT_REDEMPTION].mean()),
    }

    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)

    return {
        "n_train": len(train_t), "n_test": len(test_t), "auc": auc,
        "capture": capture, "importances": importances, "model": model,
    }


def render_report(result: dict) -> str:
    imp = result["importances"].head(5)
    importance_lines = "\n".join(f"- {name}: {value:.3f}" for name, value in imp.items())
    c = result["capture"]
    return f"""# Model 3: "sure things" classifier

Trained only on treated (mailed) households -- not a T-learner, since control households can never
redeem a coupon they never received, so bought_campaign_product_without_redemption never varies for
them. Predicts, among households already being mailed, which ones will buy the eligible product
without ever redeeming the coupon: a direct, actionable flag for households the mailer is wasted on,
separate from the uplift model's ranking.

## Setup

- Train: {result['n_train']:,} treated rows (earliest 70% of campaigns). Test: {result['n_test']:,}
  treated rows (latest 30%) -- same out-of-time split as model3_uplift.py.
- RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20), natural class
  distribution, no class_weight rebalancing (same reasoning as the uplift model).

## Held-out performance

- AUC: {result['auc']:.3f}
- Base rate (share of treated test households who buy without redeeming): {c['base_rate']:.1%}
- Among the top 20% most confidently predicted "sure things" ({c['top_n']:,} households):
  {c['top_fifth_rate']:.1%} actually bought without redeeming -- {c['top_fifth_rate'] / c['base_rate']:.1f}x the base rate.

## What the model relies on most

{importance_lines}

## How to use this

Feed the predicted probability into targeting as a deprioritization signal: a household with a high
score here is likely to buy regardless of the mailer, independent of how high their raw purchase
probability looks. This directly operationalizes the spend-tier and redemption findings in
model3_uplift.py and model3_redemption_analysis.py -- heavy spenders score high here precisely
because they're the "sure things" those findings already identified.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    table = load_table(args.processed)
    result = train_and_evaluate(table)

    print(f"Train: {result['n_train']:,} treated rows, Test: {result['n_test']:,} treated rows")
    print(f"AUC: {result['auc']:.3f}")
    c = result["capture"]
    print(f"Base rate: {c['base_rate']:.1%}, top-20% rate: {c['top_fifth_rate']:.1%} "
          f"({c['top_fifth_rate'] / c['base_rate']:.1f}x)")
    print("\nTop features:")
    print(result["importances"].head(5))

    report_path = args.processed / "model3_sure_things_report.md"
    report_path.write_text(render_report(result))

    # A small JSON alongside the written report, for the Lead Agent to read without parsing markdown
    # -- the same pattern as the Modeling/Evaluation Agents' manifest.json / verdict.json.
    facts_path = args.processed / "model3_sure_things.json"
    facts_path.write_text(json.dumps({
        "n_train": result["n_train"], "n_test": result["n_test"], "auc": round(result["auc"], 4),
        "base_rate": round(result["capture"]["base_rate"], 4),
        "top_fifth_rate": round(result["capture"]["top_fifth_rate"], 4),
        "top_features": result["importances"].head(5).round(4).to_dict(),
    }, indent=2))

    print(f"\nReport: {report_path}")
    print(f"Facts: {facts_path}")


if __name__ == "__main__":
    main()
