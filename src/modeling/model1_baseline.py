"""Model 1 (Response) baseline: a classifier trained on Feature Agent 2B's model1_table.parquet.

This is a BASELINE to demonstrate end-to-end feasibility on the real, shared pipeline output
(Data Agent -> Feature Agent 2A -> Feature Agent 2B) -- not the final Modeling Agent. It mirrors
model3_baseline.py's pattern (same features, same out-of-time split, same model settings) and
answers, concretely: does a model trained on 2B's table rank future redeemers better than the
simple "redeemed before" rule?

Run from the repo folder:
    python src/modeling/model1_baseline.py --processed data/processed

Input : data/processed/model1_table.parquet (from src/features/build_2b.py)
Output: prints an out-of-time evaluation to the console and writes model1_baseline_report.md
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Same profile/context columns as model3_baseline.py (feature_2a_spec.yaml roles "profile" and
# "context" -- never "outcome"/"filter"/"flag"), plus campaign type: Model 1 covers all 30
# campaigns, and redemption rates differ sharply by type (README: TypeA 16.0% vs. TypeB/C ~7.8%).
BASE_FEATURES = [
    "spend_per_week_26w", "trips_per_week_26w", "coupon_lines_per_week_26w",
    "share_spend_on_discount_26w", "past_redemption_rate",
    "campaign_product_spend_per_week_26w", "campaign_product_share_26w",
    "history_weeks_before_cutoff", "duration_days",
]
TYPE_FEATURES = ["is_type_a", "is_type_b"]   # TypeC is the reference level
FEATURES = BASE_FEATURES + TYPE_FEATURES
LABEL = "redeemed_flag"
TRAIN_FRACTION = 0.7
TOP_SHARE = 0.2   # README evaluation: "redeemers caught in the top 20%"


def load_table(processed: Path) -> pd.DataFrame:
    table = pd.read_parquet(processed / "model1_table.parquet")
    # Same blank handling as model3_baseline.py: past_redemption_rate is blank only when
    # never_received, the rest only for no_history / no_spend_26w -- 0 is the correct reading
    # for all of them (see feature_2a_spec.yaml).
    table = table.assign(**{c: table[c].fillna(0) for c in BASE_FEATURES if table[c].isna().any()})
    return table.assign(is_type_a=(table["campaign_type"] == "A").astype(int),
                        is_type_b=(table["campaign_type"] == "B").astype(int))


def time_split(table: pd.DataFrame, train_fraction: float = TRAIN_FRACTION):
    """Out-of-time split: earliest campaigns (by start_day) train, latest test -- same discipline
    as model3_baseline.py and Feature Agent 2A's cutoff rule."""
    campaigns = table[["campaign", "start_day"]].drop_duplicates().sort_values("start_day")
    cutoff = int(len(campaigns) * train_fraction)
    train_campaigns = set(campaigns["campaign"].iloc[:cutoff])
    test_campaigns = set(campaigns["campaign"].iloc[cutoff:])
    return table[table["campaign"].isin(train_campaigns)], table[table["campaign"].isin(test_campaigns)]


def train_models(train: pd.DataFrame) -> dict:
    """Random forest (same settings as model3_baseline.py) is the main model; logistic regression
    (PR #5 / README's suggested yes/no model) is trained alongside it as a simpler comparison."""
    models = {
        "Random forest": RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42),
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    }
    for model in models.values():
        model.fit(train[FEATURES], train[LABEL])
    return models


def top_share_capture(scores: np.ndarray, labels: np.ndarray, share: float = TOP_SHARE, seed: int = 42) -> float:
    """Share of all redeemers that land in the top `share` of rows ranked by score.
    Ties are broken at random (fixed seed) -- this matters for the "redeemed before" rule,
    where most households share the same past_redemption_rate (often 0)."""
    tiebreak = np.random.default_rng(seed).random(len(scores))
    order = np.lexsort((tiebreak, -np.asarray(scores, dtype=float)))   # highest score first
    k = int(round(len(scores) * share))
    return float(np.asarray(labels)[order[:k]].sum() / np.asarray(labels).sum())


def evaluate(models: dict, test: pd.DataFrame) -> dict:
    test = test.copy()
    y = test[LABEL].to_numpy()
    scores = {name: m.predict_proba(test[FEATURES])[:, 1] for name, m in models.items()}
    scores['"Redeemed before" rule (`past_redemption_rate`)'] = test["past_redemption_rate"].to_numpy()
    comparison = pd.DataFrame([{
        "ranking": name, "top_capture": top_share_capture(s, y),
        "roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s),
    } for name, s in scores.items()])
    test["p_redeem"] = scores["Random forest"]

    test["quintile"] = pd.qcut(test["p_redeem"], 5, labels=False, duplicates="drop")
    rows = []
    for q in sorted(test["quintile"].dropna().unique()):
        bucket = test[test["quintile"] == q]
        rows.append({"quintile": int(q), "n": len(bucket), "n_redeemed": int(bucket[LABEL].sum()),
                     "mean_predicted": round(bucket["p_redeem"].mean(), 4),
                     "actual_rate": round(bucket[LABEL].mean(), 4)})
    quintiles = pd.DataFrame(rows)
    rho, p = spearmanr(quintiles["quintile"], quintiles["actual_rate"])

    return {
        "test": test, "quintiles": quintiles, "spearman_rho": rho, "spearman_p": p,
        "base_rate": y.mean(), "n_redeemed": int(y.sum()), "comparison": comparison,
        "by_type": test.groupby("campaign_type")[LABEL].agg(["size", "sum", "mean"]).round(4),
    }


def render_report(train_df, result: dict) -> str:
    q = result["quintiles"].to_string(index=False)
    rows = "\n".join(f"| {r.ranking} | {r.top_capture:.3f} | {r.roc_auc:.3f} | {r.pr_auc:.3f} |"
                     for r in result["comparison"].itertuples())
    by_type = result["by_type"].rename(columns={"size": "rows", "sum": "redeemed", "mean": "rate"}).to_string()
    return f"""# Model 1 baseline: response classifier run report

Baseline, not the final Modeling Agent -- checks whether a response model trained on
`model1_table.parquet` (Feature Agent 2B) beats the "redeemed before" rule on an out-of-time split.

## Setup

- Train: {len(train_df):,} mailed rows, {int(train_df[LABEL].sum()):,} redeemed (earliest {TRAIN_FRACTION:.0%} of campaigns by start_day)
- Test: {len(result['test']):,} mailed rows, {result['n_redeemed']:,} redeemed (base rate {result['base_rate']:.3f})
- Main model: RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20), same as Model 3
- Comparison model: LogisticRegression on standardized features (PR #5's suggested yes/no model)
- Features: {", ".join(FEATURES)}

## Held-out comparison (README evaluation: redeemers caught in the top {TOP_SHARE:.0%})

| Ranking | Redeemers caught in top {TOP_SHARE:.0%} | ROC-AUC | PR-AUC |
|---|---|---|---|
{rows}
| Random targeting | {TOP_SHARE:.3f} | 0.500 | {result['base_rate']:.3f} |

## Held-out quintile check (random forest, ranked by predicted probability)

```
{q}
```

Spearman rank correlation (quintile vs. actual redemption rate): rho={result['spearman_rho']:.3f}, p={result['spearman_p']:.3f}

## Test set by campaign type

```
{by_type}
```

Model 1 keeps TypeA in scope (Oct 2 mentor meeting decision); its personalized coupon selection may
add some bias, so check whether the gain over the rule holds within each type, not just overall.
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
        print(f"{name} train AUC: {roc_auc_score(train[LABEL], model.predict_proba(train[FEATURES])[:, 1]):.3f}")

    result = evaluate(models, test)
    print("\nHeld-out quintile check:")
    print(result["quintiles"].to_string(index=False))
    print(f"\nSpearman rho={result['spearman_rho']:.3f}, p={result['spearman_p']:.3f}")
    print(f"\nHeld-out comparison (random targeting catches {TOP_SHARE:.0%} in the top {TOP_SHARE:.0%}):")
    print(result["comparison"].round(3).to_string(index=False))

    report_path = args.processed / "model1_baseline_report.md"
    report_path.write_text(render_report(train, result))
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
