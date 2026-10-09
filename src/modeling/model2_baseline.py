"""Model 2 (Expected Value) baseline: a regressor trained on Feature Agent 2B's model2_table.parquet.

This is a BASELINE to demonstrate end-to-end feasibility on the real, shared pipeline output
(Data Agent -> Feature Agent 2A -> Feature Agent 2B) -- not the final Modeling Agent. It mirrors
model3_baseline.py's pattern (same features, same out-of-time split, same model settings) and
answers, concretely: does a model trained on 2B's table predict campaign-product spend better
than the household's usual spend on those products?

Run from the repo folder:
    python src/modeling/model2_baseline.py --processed data/processed

Input : data/processed/model2_table.parquet (from src/features/build_2b.py)
Output: prints an out-of-time evaluation to the console and writes model2_baseline_report.md
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Same profile/context columns as model3_baseline.py (feature_2a_spec.yaml roles "profile" and
# "context" -- never "outcome"/"filter"/"flag"), plus a TypeB flag: Model 2's rows are TypeB/C
# only (feature_2b_spec.yaml), so TypeC is the reference level.
BASE_FEATURES = [
    "spend_per_week_26w", "trips_per_week_26w", "coupon_lines_per_week_26w",
    "share_spend_on_discount_26w", "past_redemption_rate",
    "campaign_product_spend_per_week_26w", "campaign_product_share_26w",
    "history_weeks_before_cutoff", "duration_days",
]
FEATURES = BASE_FEATURES + ["is_type_b"]
LABEL = "campaign_product_spend_during"
TRAIN_FRACTION = 0.7


def load_table(processed: Path) -> pd.DataFrame:
    table = pd.read_parquet(processed / "model2_table.parquet")
    # Same blank handling as model3_baseline.py: past_redemption_rate is blank only when
    # never_received, the rest only for no_history / no_spend_26w -- 0 is the correct reading
    # for all of them (see feature_2a_spec.yaml).
    table = table.assign(**{c: table[c].fillna(0) for c in BASE_FEATURES if table[c].isna().any()})
    return table.assign(is_type_b=(table["campaign_type"] == "B").astype(int))


def time_split(table: pd.DataFrame, train_fraction: float = TRAIN_FRACTION):
    """Out-of-time split: earliest campaigns (by start_day) train, latest test -- same discipline
    as model3_baseline.py and Feature Agent 2A's cutoff rule."""
    campaigns = table[["campaign", "start_day"]].drop_duplicates().sort_values("start_day")
    cutoff = int(len(campaigns) * train_fraction)
    train_campaigns = set(campaigns["campaign"].iloc[:cutoff])
    test_campaigns = set(campaigns["campaign"].iloc[cutoff:])
    return table[table["campaign"].isin(train_campaigns)], table[table["campaign"].isin(test_campaigns)]


def train_models(train: pd.DataFrame) -> dict:
    """Random forest (same settings as model3_baseline.py) is the main model; linear regression
    (README's suggested dollar model) is trained alongside it as a simpler comparison."""
    models = {
        "Random forest": RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42),
        "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
    }
    for model in models.values():
        model.fit(train[FEATURES], train[LABEL])
    return models


def usual_spend(table: pd.DataFrame) -> pd.Series:
    """README evaluation baseline ("vs. usual spend"): the household's average weekly spend on
    this campaign's products over the 26 weeks before cutoff, scaled to the campaign's length."""
    return table["campaign_product_spend_per_week_26w"] * table["duration_days"] / 7


def evaluate(models: dict, test: pd.DataFrame, train_mean: float) -> dict:
    test = test.copy()
    predictions = {name: m.predict(test[FEATURES]) for name, m in models.items()}
    test["predicted_spend"] = predictions["Random forest"]
    test["usual_spend"] = usual_spend(test)
    y = test[LABEL]

    test["quintile"] = pd.qcut(test["predicted_spend"], 5, labels=False, duplicates="drop")
    rows = []
    for q in sorted(test["quintile"].dropna().unique()):
        bucket = test[test["quintile"] == q]
        rows.append({"quintile": int(q), "n": len(bucket),
                     "mean_predicted": round(bucket["predicted_spend"].mean(), 2),
                     "mean_actual": round(bucket[LABEL].mean(), 2)})
    quintiles = pd.DataFrame(rows)
    rho, p = spearmanr(quintiles["quintile"], quintiles["mean_actual"])

    def errors(pred) -> dict:
        pred = np.broadcast_to(np.asarray(pred, dtype=float), y.shape)
        return {"mae": mean_absolute_error(y, pred), "median_ae": float(np.median(np.abs(y - pred))),
                "rank_rho": spearmanr(pred, y)[0] if np.ptp(pred) > 0 else float("nan")}

    comparison = {name: errors(pred) for name, pred in predictions.items()}
    comparison["Usual spend (26-week weekly spend on campaign products x weeks in campaign)"] = errors(test["usual_spend"])
    comparison[f"Training-set mean (${train_mean:.2f} for everyone)"] = errors(train_mean)
    return {
        "test": test, "quintiles": quintiles, "spearman_rho": rho, "spearman_p": p,
        "comparison": comparison, "train_mean": train_mean,
        "actual_mean": y.mean(), "actual_median": y.median(),
    }


def render_report(train_df, result: dict) -> str:
    q = result["quintiles"].to_string(index=False)
    rows = "\n".join(
        f"| {name} | {e['mae']:.2f} | {e['median_ae']:.2f} | "
        + ("n/a" if np.isnan(e["rank_rho"]) else f"{e['rank_rho']:.3f}") + " |"
        for name, e in result["comparison"].items())
    return f"""# Model 2 baseline: expected-value regressor run report

Baseline, not the final Modeling Agent -- checks whether a spend model trained on
`model2_table.parquet` (Feature Agent 2B) beats "usual spend" on an out-of-time split.

## Setup

- Train: {len(train_df):,} redeemed TypeB/C rows (earliest {TRAIN_FRACTION:.0%} of campaigns by start_day)
- Test: {len(result['test']):,} rows; actual spend mean ${result['actual_mean']:.2f}, median ${result['actual_median']:.2f}
- Main model: RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=20), same as Model 3
- Comparison model: LinearRegression on standardized features (README's suggested dollar model)
- Features: {", ".join(FEATURES)}

## Held-out comparison (README evaluation: average dollar error vs. usual spend)

| Prediction | Mean abs. error ($) | Median abs. error ($) | Rank correlation with actual |
|---|---|---|---|
{rows}

## Held-out quintile check (random forest, ranked by predicted spend)

```
{q}
```

Spearman rank correlation (quintile vs. actual mean spend): rho={result['spearman_rho']:.3f}, p={result['spearman_p']:.3f}

## Caveat

Model 2 has only 254 rows in total (TypeB/C redeemers), so the test set is small and these numbers
will move noticeably with the split. Treat them as a feasibility check, not a final estimate.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    table = load_table(args.processed)
    train, test = time_split(table)
    print(f"Train: {train['campaign'].nunique()} campaigns, {len(train):,} rows | "
          f"Test: {test['campaign'].nunique()} campaigns, {len(test):,} rows")

    models = train_models(train)
    for name, model in models.items():
        print(f"{name} train MAE: ${mean_absolute_error(train[LABEL], model.predict(train[FEATURES])):.2f}")

    result = evaluate(models, test, train_mean=float(train[LABEL].mean()))
    print("\nHeld-out quintile check:")
    print(result["quintiles"].to_string(index=False))
    print(f"\nSpearman rho={result['spearman_rho']:.3f}, p={result['spearman_p']:.3f}")
    print("\nHeld-out mean absolute error ($):")
    for name, e in result["comparison"].items():
        print(f"  {e['mae']:8.2f}  {name}")

    report_path = args.processed / "model2_baseline_report.md"
    report_path.write_text(render_report(train, result))
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
