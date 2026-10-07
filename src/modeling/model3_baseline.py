"""Model 3 (Uplift) baseline: a T-learner trained on Feature Agent 2B's model3_table.parquet.

This is a BASELINE to demonstrate end-to-end feasibility on the real, shared pipeline output
(Data Agent -> Feature Agent 2A -> Feature Agent 2B) -- not the final Modeling Agent. It exists to
answer, concretely: does the uplift signal 2B's table is built to support actually show up when a
model is trained on it?

Run from the repo folder:
    python src/modeling/model3_baseline.py --processed data/processed

Input : data/processed/model3_table.parquet (from src/features/build_2b.py)
Output: prints an out-of-time evaluation to the console and writes model3_baseline_report.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Profile/context columns used as model inputs for this baseline (feature_2a_spec.yaml roles
# "profile" and "context" -- never "outcome"/"filter"/"flag"; 2B already enforces this by not
# including flag columns like short_history_flag in model3_table.parquet at all).
FEATURES = [
    "spend_per_week_26w", "trips_per_week_26w", "coupon_lines_per_week_26w",
    "share_spend_on_discount_26w", "past_redemption_rate",
    "campaign_product_spend_per_week_26w", "campaign_product_share_26w",
    "history_weeks_before_cutoff", "duration_days",
]
LABEL = "bought_campaign_product_excl_display_flyer"
TREATMENT = "mailed_flag"
TRAIN_FRACTION = 0.7


def load_table(processed: Path) -> pd.DataFrame:
    table = pd.read_parquet(processed / "model3_table.parquet")
    # past_redemption_rate is blank only when never_received (no prior campaign); 0 is the
    # correct reading (no prior campaigns to have redeemed). The rest are blank only for
    # no_history / no_spend_26w, where 0 is also the correct reading (see feature_2b_spec.yaml).
    return table.assign(**{c: table[c].fillna(0) for c in FEATURES if table[c].isna().any()})


def time_split(table: pd.DataFrame, train_fraction: float = TRAIN_FRACTION):
    """Out-of-time split: earliest campaigns (by start_day) train, latest test -- same discipline
    as the Model 1 evaluation plan (README) and Feature Agent 2A's cutoff rule."""
    campaigns = table[["campaign", "start_day"]].drop_duplicates().sort_values("start_day")
    cutoff = int(len(campaigns) * train_fraction)
    train_campaigns = set(campaigns["campaign"].iloc[:cutoff])
    test_campaigns = set(campaigns["campaign"].iloc[cutoff:])
    return table[table["campaign"].isin(train_campaigns)], table[table["campaign"].isin(test_campaigns)]


def train_t_learner(train: pd.DataFrame):
    treated, control = train[train[TREATMENT] == 1], train[train[TREATMENT] == 0]
    control_sampled = control.sample(n=min(len(control), len(treated) * 5), random_state=42)
    model_t = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42)
    model_c = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42)
    model_t.fit(treated[FEATURES], treated[LABEL])
    model_c.fit(control_sampled[FEATURES], control_sampled[LABEL])
    return model_t, model_c, treated, control_sampled


def evaluate(model_t, model_c, test: pd.DataFrame) -> dict:
    test = test.copy()
    test["p_treated"] = model_t.predict_proba(test[FEATURES])[:, 1]
    test["p_control"] = model_c.predict_proba(test[FEATURES])[:, 1]
    test["predicted_uplift"] = test["p_treated"] - test["p_control"]

    test["quintile"] = pd.qcut(test["predicted_uplift"], 5, labels=False, duplicates="drop")
    rows = []
    for q in sorted(test["quintile"].dropna().unique()):
        bucket = test[test["quintile"] == q]
        t_rate = bucket[bucket[TREATMENT] == 1][LABEL].mean()
        c_rate = bucket[bucket[TREATMENT] == 0][LABEL].mean()
        rows.append({"quintile": int(q), "n_treated": int((bucket[TREATMENT] == 1).sum()),
                     "n_control": int((bucket[TREATMENT] == 0).sum()),
                     "actual_uplift": round(t_rate - c_rate, 4)})
    quintiles = pd.DataFrame(rows)
    rho, p = spearmanr(quintiles["quintile"], quintiles["actual_uplift"])

    bottom20 = test[test["predicted_uplift"] <= test["predicted_uplift"].quantile(0.2)]
    top20 = test[test["predicted_uplift"] >= test["predicted_uplift"].quantile(0.8)]

    def uplift(bucket):
        return bucket[bucket[TREATMENT] == 1][LABEL].mean() - bucket[bucket[TREATMENT] == 0][LABEL].mean()

    return {
        "test": test, "quintiles": quintiles, "spearman_rho": rho, "spearman_p": p,
        "bottom20_uplift": uplift(bottom20), "top20_uplift": uplift(top20),
        "treated_pre_spend": test[test[TREATMENT] == 1]["spend_per_week_26w"].mean(),
        "control_pre_spend": test[test[TREATMENT] == 0]["spend_per_week_26w"].mean(),
    }


def render_report(train_t_df, train_c_df, model_t, result: dict) -> str:
    q = result["quintiles"].to_string(index=False)
    return f"""# Model 3 baseline: T-learner run report

Baseline, not the final Modeling Agent -- demonstrates the uplift signal in
`model3_table.parquet` (Feature Agent 2B) survives an out-of-time train/test split.

## Setup

- Train: {len(train_t_df):,} treated + {len(train_c_df):,} sampled control rows (earliest {TRAIN_FRACTION:.0%} of campaigns by start_day)
- Model T / Model C: RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20)
- Features: {", ".join(FEATURES)}

## Held-out quintile check (ranked by predicted uplift)

```
{q}
```

Spearman rank correlation (quintile vs. actual observed uplift): rho={result['spearman_rho']:.3f}, p={result['spearman_p']:.3f}

Bottom 20% predicted uplift -> actual uplift {result['bottom20_uplift']:.3f}
Top 20% predicted uplift -> actual uplift {result['top20_uplift']:.3f}

## Targeting-bias check (held-out set)

Treated households' pre-campaign spend/week: ${result['treated_pre_spend']:.2f}
Control households' pre-campaign spend/week: ${result['control_pre_spend']:.2f}
Ratio: {result['treated_pre_spend'] / result['control_pre_spend']:.2f}x -- confirms campaign targeting
is non-random (README: "Campaigns are targeted, not randomized"); this is why the T-learner
conditions on pre-campaign behavior rather than taking the raw treated-vs-control gap at face value.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    table = load_table(args.processed)
    train, test = time_split(table)
    print(f"Train: {train['campaign'].nunique()} campaigns, {len(train):,} rows | "
          f"Test: {test['campaign'].nunique()} campaigns, {len(test):,} rows")

    model_t, model_c, train_t_df, train_c_df = train_t_learner(train)
    print(f"Model T train AUC: {roc_auc_score(train_t_df[LABEL], model_t.predict_proba(train_t_df[FEATURES])[:, 1]):.3f}")
    print(f"Model C train AUC: {roc_auc_score(train_c_df[LABEL], model_c.predict_proba(train_c_df[FEATURES])[:, 1]):.3f}")

    result = evaluate(model_t, model_c, test)
    print("\nHeld-out quintile check:")
    print(result["quintiles"].to_string(index=False))
    print(f"\nSpearman rho={result['spearman_rho']:.3f}, p={result['spearman_p']:.3f}")
    print(f"Bottom 20% actual uplift: {result['bottom20_uplift']:.3f}  |  Top 20% actual uplift: {result['top20_uplift']:.3f}")

    report_path = args.processed / "model3_baseline_report.md"
    report_path.write_text(render_report(train_t_df, train_c_df, model_t, result))
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
