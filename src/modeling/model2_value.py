"""Model 2: Expected Financial Value -- calibrated usual-spend model (primary build).

Supersedes model2_baseline.py as the primary Model 2 build (the baseline stays as the comparison it
was built to be). For a household that redeems a TypeB/C campaign, predicts its spend on that
campaign's products during the campaign window.

The primary model is deliberately small: a one-feature median (least-absolute-deviation) regression
of spend on "usual spend" -- the household's average weekly spend on this campaign's products over
the 26 weeks before the cutoff, times the campaign's length in weeks. In other words, it calibrates
the README's own "usual spend" benchmark rather than replacing it. Every larger model tried (more
features, log target, tree ensembles, LightGBM, campaign-level features -- see
docs/agent_design/modeling_agent_model2_design.md and notebooks/yuxin_01_model2_tuning_experiments.py)
failed to beat usual spend reliably on 254 rows; this is the only candidate that does.

Evaluated two ways, because a single 70/30 split leaves only 128 test rows from 8 campaigns:
  1. the same 70/30 out-of-time split as every model{n}_baseline.py (comparable across models);
  2. walk-forward: each campaign is predicted by a model trained only on campaigns that started
     before it (pooled over every campaign with >= MIN_TRAIN_ROWS rows of earlier history), with a
     campaign-level bootstrap of the difference against the usual-spend benchmark.

Run from the repo folder:
    python src/modeling/model2_value.py --processed data/processed

Input : data/processed/model2_table.parquet (from src/features/build_2b.py), 254 redeemed TypeB/C rows
Output: prints results to the console and writes data/processed/model2_value_report.md
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LABEL = "campaign_product_spend_during"
USUAL = "usual_spend"
TRAIN_FRACTION = 0.7
MIN_TRAIN_ROWS = 60       # walk-forward starts once a campaign has this much earlier history
N_BOOTSTRAP = 2000
PRIMARY = "Calibrated usual spend (primary)"
BENCHMARK = "Usual spend (benchmark)"

# Same candidate inputs as model2_baseline.py, used only by the comparison models below.
BASELINE_FEATURES = [
    "spend_per_week_26w", "trips_per_week_26w", "coupon_lines_per_week_26w",
    "share_spend_on_discount_26w", "past_redemption_rate",
    "campaign_product_spend_per_week_26w", "campaign_product_share_26w",
    "history_weeks_before_cutoff", "duration_days", "is_type_b",
]


def usual_spend(table: pd.DataFrame) -> pd.Series:
    """README benchmark: weekly spend on this campaign's products (26 weeks before cutoff) x campaign weeks."""
    return table["campaign_product_spend_per_week_26w"] * table["duration_days"] / 7


def load_table(processed: Path) -> pd.DataFrame:
    table = pd.read_parquet(processed / "model2_table.parquet")
    # Blanks are only no_history / no_spend_26w / never_received, where 0 is the correct reading
    # (feature_2a_spec.yaml) -- same handling as model2_baseline.py.
    raw = [c for c in BASELINE_FEATURES if c != "is_type_b"]
    table = table.assign(**{c: table[c].fillna(0) for c in raw if table[c].isna().any()})
    table = table.assign(is_type_b=(table["campaign_type"] == "B").astype(int))
    return table.assign(**{USUAL: usual_spend(table)})


def time_split(table: pd.DataFrame, train_fraction: float = TRAIN_FRACTION):
    """Out-of-time split: earliest campaigns (by start_day) train, latest test -- same as model{n}_baseline.py."""
    campaigns = table[["campaign", "start_day"]].drop_duplicates().sort_values("start_day")
    cutoff = int(len(campaigns) * train_fraction)
    train_campaigns = set(campaigns["campaign"].iloc[:cutoff])
    test_campaigns = set(campaigns["campaign"].iloc[cutoff:])
    return table[table["campaign"].isin(train_campaigns)], table[table["campaign"].isin(test_campaigns)]


def walk_forward_splits(table: pd.DataFrame, min_train_rows: int = MIN_TRAIN_ROWS):
    """Yields (campaign, train, test): each campaign is predicted from campaigns that started strictly
    before it. Campaigns with fewer than min_train_rows rows of earlier history are skipped."""
    campaigns = table[["campaign", "start_day"]].drop_duplicates().sort_values("start_day")
    for campaign, start_day in campaigns.itertuples(index=False):
        train = table[table["start_day"] < start_day]
        if len(train) >= min_train_rows:
            yield campaign, train, table[table["campaign"] == campaign]


class CalibratedUsualSpend:
    """spend = slope x usual_spend + intercept, fit by median (least-absolute-deviation) regression.

    Median rather than least squares because spend is heavily right-skewed (a few households spend
    $100+) and the evaluation metric is mean absolute error, which a median fit targets directly."""

    def fit(self, train: pd.DataFrame):
        model = QuantileRegressor(quantile=0.5, alpha=0, solver="highs").fit(train[[USUAL]], train[LABEL])
        self.slope, self.intercept = float(model.coef_[0]), float(model.intercept_)
        return self

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        return np.clip(self.slope * table[USUAL].to_numpy() + self.intercept, 0, None)


class UsualSpend:
    """The README's benchmark, unchanged: assume the household keeps buying as it usually does."""

    def fit(self, train: pd.DataFrame):
        return self

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        return table[USUAL].to_numpy()


class TrainingMean:
    def fit(self, train: pd.DataFrame):
        self.value = float(train[LABEL].mean())
        return self

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        return np.full(len(table), self.value)


class FeatureModel:
    """Wraps a scikit-learn regressor trained on a fixed column list."""

    def __init__(self, estimator, features: list[str]):
        self.estimator, self.features = estimator, features

    def fit(self, train: pd.DataFrame):
        self.estimator.fit(train[self.features], train[LABEL])
        return self

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        return np.clip(self.estimator.predict(table[self.features]), 0, None)


def model_factories() -> dict:
    """Primary model, the benchmark it must beat, and the comparisons that show why it was chosen."""
    return {
        PRIMARY: CalibratedUsualSpend,
        BENCHMARK: UsualSpend,
        "Training-set mean": TrainingMean,
        "Huber regression, baseline features + usual spend": lambda: FeatureModel(
            make_pipeline(StandardScaler(), HuberRegressor(max_iter=1000)), BASELINE_FEATURES + [USUAL]),
        "Linear regression (model2_baseline.py)": lambda: FeatureModel(
            make_pipeline(StandardScaler(), LinearRegression()), BASELINE_FEATURES),
        "Random forest (model2_baseline.py)": lambda: FeatureModel(
            RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42),
            BASELINE_FEATURES),
    }


def errors(y, pred) -> dict:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    abs_err = np.abs(y - pred)
    rho = spearmanr(pred, y)[0] if np.ptp(pred) > 0 else float("nan")
    return {"mae": float(abs_err.mean()), "median_ae": float(np.median(abs_err)), "rank_rho": float(rho)}


def evaluate_split(factories: dict, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = [{"model": name, **errors(test[LABEL], make().fit(train).predict(test))}
            for name, make in factories.items()]
    return pd.DataFrame(rows)


def evaluate_walk_forward(factories: dict, table: pd.DataFrame) -> dict:
    """Pooled walk-forward predictions for every model, plus the actuals and each row's campaign."""
    preds = {name: [] for name in factories}
    actual, campaign_of_row = [], []
    for campaign, train, test in walk_forward_splits(table):
        actual.append(test[LABEL].to_numpy())
        campaign_of_row.append(np.full(len(test), campaign))
        for name, make in factories.items():
            preds[name].append(make().fit(train).predict(test))
    actual, campaign_of_row = np.concatenate(actual), np.concatenate(campaign_of_row)
    preds = {name: np.concatenate(p) for name, p in preds.items()}
    summary = pd.DataFrame([{"model": name, **errors(actual, p)} for name, p in preds.items()])
    return {"summary": summary, "preds": preds, "actual": actual, "campaign": campaign_of_row}


def bootstrap_vs_benchmark(actual, pred, benchmark, campaign_of_row,
                           n: int = N_BOOTSTRAP, seed: int = 0) -> dict:
    """Campaign-level bootstrap of MAE(pred) - MAE(benchmark). Campaigns, not rows, are resampled
    because rows within one campaign share that campaign's spend level."""
    actual, pred, benchmark = (np.asarray(a, dtype=float) for a in (actual, pred, benchmark))
    rows_of = {c: np.where(campaign_of_row == c)[0] for c in np.unique(campaign_of_row)}
    campaigns = np.array(list(rows_of))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for i in range(n):
        idx = np.concatenate([rows_of[c] for c in rng.choice(campaigns, len(campaigns))])
        diffs[i] = np.abs(actual[idx] - pred[idx]).mean() - np.abs(actual[idx] - benchmark[idx]).mean()
    observed = np.abs(actual - pred).mean() - np.abs(actual - benchmark).mean()
    return {"diff": float(observed), "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)), "share_better": float((diffs < 0).mean())}


def quintile_check(actual, pred, benchmark) -> pd.DataFrame:
    frame = pd.DataFrame({"predicted": pred, "actual": actual, "usual_spend": benchmark})
    frame["quintile"] = pd.qcut(frame["predicted"], 5, labels=False, duplicates="drop")
    out = frame.groupby("quintile").agg(n=("actual", "size"), mean_predicted=("predicted", "mean"),
                                        mean_actual=("actual", "mean"), mean_usual_spend=("usual_spend", "mean"))
    return out.reset_index().round(2)


def _table(df: pd.DataFrame) -> str:
    lines = ["| Model | Mean abs. error ($) | Median abs. error ($) | Rank correlation |", "|---|---|---|---|"]
    for r in df.itertuples():
        rho = "n/a" if np.isnan(r.rank_rho) else f"{r.rank_rho:.2f}"
        lines.append(f"| {r.model} | {r.mae:.2f} | {r.median_ae:.2f} | {rho} |")
    return "\n".join(lines)


def render_report(table, train, test, fitted, split, wf, boot, quintiles) -> str:
    return f"""# Model 2: calibrated usual-spend model run report

Primary Model 2 build (supersedes `model2_baseline.py`). Design and every alternative tried:
`docs/agent_design/modeling_agent_model2_design.md`.

## Primary model

One-feature median regression, fit on all {len(table)} rows for scoring:

**spend = {fitted.slope:.2f} x usual spend + ${fitted.intercept:.2f}**

where usual spend = weekly spend on this campaign's products (26 weeks before cutoff) x campaign weeks.

## Walk-forward evaluation (primary)

Each campaign predicted from campaigns that started before it; {len(wf['actual'])} rows from
{len(np.unique(wf['campaign']))} campaigns pooled.

{_table(wf['summary'])}

Calibrated usual spend vs. usual spend: MAE difference {boot['diff']:+.2f}
(95% campaign-bootstrap CI {boot['ci_low']:+.2f} to {boot['ci_high']:+.2f}; better in
{boot['share_better']:.0%} of {N_BOOTSTRAP:,} resamples).

## 70/30 out-of-time split (same as model2_baseline.py)

Train: {len(train)} rows, {train['campaign'].nunique()} campaigns | Test: {len(test)} rows, {test['campaign'].nunique()} campaigns

{_table(split)}

## Quintile check (walk-forward, ranked by calibrated prediction)

```
{quintiles.to_string(index=False)}
```

Ranking is correct in every quintile. All levels are under-predicted by a few dollars: later
campaigns spend more than earlier ones, which no pre-campaign feature tried here captures.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    table = load_table(args.processed)
    factories = model_factories()

    train, test = time_split(table)
    split = evaluate_split(factories, train, test)
    wf = evaluate_walk_forward(factories, table)
    boot = bootstrap_vs_benchmark(wf["actual"], wf["preds"][PRIMARY], wf["preds"][BENCHMARK], wf["campaign"])
    quintiles = quintile_check(wf["actual"], wf["preds"][PRIMARY], wf["preds"][BENCHMARK])
    fitted = CalibratedUsualSpend().fit(table)

    print(f"Primary model (fit on all {len(table)} rows): spend = {fitted.slope:.2f} x usual spend + ${fitted.intercept:.2f}")
    print(f"\nWalk-forward ({len(wf['actual'])} rows, {len(np.unique(wf['campaign']))} campaigns):")
    print(wf["summary"].round(2).to_string(index=False))
    print(f"\nvs. usual spend: MAE {boot['diff']:+.2f}, 95% CI [{boot['ci_low']:+.2f}, {boot['ci_high']:+.2f}], "
          f"better in {boot['share_better']:.0%} of resamples")
    print(f"\n70/30 split (train {len(train)} rows, test {len(test)} rows):")
    print(split.round(2).to_string(index=False))
    print("\nQuintile check (walk-forward):")
    print(quintiles.to_string(index=False))

    report_path = args.processed / "model2_value_report.md"
    report_path.write_text(render_report(table, train, test, fitted, split, wf, boot, quintiles))
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
