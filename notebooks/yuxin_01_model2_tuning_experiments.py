"""Model 2 tuning experiments: every candidate tried before choosing src/modeling/model2_value.py.

Trial script (notebooks/ naming convention: name + number), kept so the "what was tried and why it
didn't help" section of docs/agent_design/modeling_agent_model2_design.md is reproducible. Uses the
exact same splits and metrics as model2_value.py by importing them.

Run from the repo folder (after build_2b.py; needs ingest.py's transaction_data and coupon parquet
files for the campaign-level features):
    python notebooks/yuxin_01_model2_tuning_experiments.py --processed data/processed
"""
import argparse
import sys
import warnings
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from modeling.model2_value import (  # noqa: E402
    BASELINE_FEATURES, LABEL, USUAL, CalibratedUsualSpend, TrainingMean, UsualSpend,
    errors, load_table, time_split, walk_forward_splits,
)

warnings.filterwarnings("ignore")

EXTRA_FEATURES = [
    "spend_per_week_8w", "trips_per_week_8w", "avg_basket_value_26w", "days_since_last_trip",
    "n_departments_26w", "past_redemptions", "past_campaigns_received", "past_campaigns_redeemed",
    "other_campaigns_live_at_cutoff", "n_overlapping_campaigns",
]
CAMPAIGN_FEATURES = ["camp_n_products", "camp_n_coupons", "camp_pop_spend_per_hh_week", "camp_n_buyers_26w"]


def campaign_features(processed: Path) -> pd.DataFrame:
    """Campaign-level inputs known before mailing (no data after cutoff = start_day - 7): the size of
    the campaign's product and coupon list, and how popular those products were across all 2,500
    households in the 26 weeks before cutoff."""
    p = processed.as_posix()
    return duckdb.sql(f"""
        WITH camp AS (SELECT DISTINCT campaign, start_day - 7 AS cutoff FROM read_parquet('{p}/model2_table.parquet')),
        prods AS (SELECT DISTINCT CAMPAIGN AS campaign, PRODUCT_ID AS product_id FROM read_parquet('{p}/coupon.parquet')),
        design AS (SELECT CAMPAIGN AS campaign, count(DISTINCT PRODUCT_ID) AS camp_n_products,
                          count(DISTINCT COUPON_UPC) AS camp_n_coupons
                   FROM read_parquet('{p}/coupon.parquet') GROUP BY 1),
        pop AS (SELECT c.campaign, sum(t.SALES_VALUE) / 2500.0 / 26 AS camp_pop_spend_per_hh_week,
                       count(DISTINCT t.household_key) AS camp_n_buyers_26w
                FROM camp c JOIN prods USING (campaign)
                JOIN read_parquet('{p}/transaction_data.parquet') t
                  ON t.PRODUCT_ID = prods.product_id AND t.DAY < c.cutoff AND t.DAY >= c.cutoff - 182
                GROUP BY 1)
        SELECT c.campaign, d.camp_n_products, d.camp_n_coupons,
               coalesce(pop.camp_pop_spend_per_hh_week, 0) AS camp_pop_spend_per_hh_week,
               coalesce(pop.camp_n_buyers_26w, 0) AS camp_n_buyers_26w
        FROM camp c JOIN design d USING (campaign) LEFT JOIN pop USING (campaign)""").df()


class Model:
    """kind: 'raw' fits spend; 'log' fits log1p(spend); 'ratio' fits log1p(spend) - log1p(usual spend)."""

    def __init__(self, estimator, features, kind="raw"):
        self.estimator, self.features, self.kind = estimator, features, kind

    def fit(self, d):
        y = d[LABEL].to_numpy()
        if self.kind == "log":
            y = np.log1p(y)
        elif self.kind == "ratio":
            y = np.log1p(y) - np.log1p(d[USUAL].to_numpy())
        self.estimator.fit(d[self.features], y)
        return self

    def predict(self, d):
        p = self.estimator.predict(d[self.features])
        if self.kind == "log":
            p = np.expm1(p)
        elif self.kind == "ratio":
            p = np.expm1(p + np.log1p(d[USUAL].to_numpy()))
        return np.clip(p, 0, None)


class TrainingMedian(TrainingMean):
    def fit(self, train):
        self.value = float(train[LABEL].median())
        return self


def rf(leaf, criterion="squared_error"):
    return RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=leaf, criterion=criterion,
                                 random_state=42, n_jobs=-1)


def lgbm(objective="regression"):
    return lgb.LGBMRegressor(objective=objective, n_estimators=300, learning_rate=0.03, num_leaves=7,
                             min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             verbose=-1, random_state=42)


def scaled(estimator):
    return make_pipeline(StandardScaler(), estimator)


def gbm_l1():
    return GradientBoostingRegressor(loss="absolute_error", n_estimators=200, max_depth=2, learning_rate=0.05,
                                     subsample=0.8, random_state=42)


def candidates() -> dict:
    f0 = BASELINE_FEATURES
    f1 = f0 + [USUAL]
    f1_log = f1 + ["log_usual"]
    f2 = f1 + EXTRA_FEATURES
    f3 = f1 + CAMPAIGN_FEATURES
    return {
        # benchmarks and the chosen model
        "Training mean": lambda: TrainingMean(),
        "Training median": lambda: TrainingMedian(),
        "Usual spend (benchmark)": lambda: UsualSpend(),
        "Calibrated usual spend (chosen)": lambda: CalibratedUsualSpend(),
        # model2_baseline.py, as merged
        "Random forest, leaf 20 (baseline)": lambda: Model(rf(20), f0),
        "Linear regression (baseline)": lambda: Model(scaled(LinearRegression()), f0),
        # 1. less conservative trees, usual spend as an explicit input, absolute-error loss
        "Random forest, leaf 5": lambda: Model(rf(5), f0),
        "Random forest, leaf 5 + usual spend": lambda: Model(rf(5), f1),
        "Random forest, leaf 5 + usual spend, absolute error": lambda: Model(rf(5, "absolute_error"), f1),
        # 2. log target
        "Random forest, leaf 5 + usual spend, log target": lambda: Model(rf(5), f1, "log"),
        "Random forest, leaf 10 + usual spend, log target": lambda: Model(rf(10), f1, "log"),
        "Linear + usual spend, log target": lambda: Model(scaled(LinearRegression()), f1_log, "log"),
        "Ridge + usual spend, log target": lambda: Model(scaled(Ridge(10)), f1_log, "log"),
        # 3. robust / linear with usual spend
        "Linear + usual spend": lambda: Model(scaled(LinearRegression()), f1),
        "Huber + usual spend": lambda: Model(scaled(HuberRegressor(max_iter=1000)), f1),
        # 4. predict the ratio to usual spend
        "Ratio to usual spend, random forest": lambda: Model(rf(10), f1, "ratio"),
        "Ratio to usual spend, ridge": lambda: Model(scaled(Ridge(10)), f1_log, "ratio"),
        # 5. boosting
        "LightGBM + usual spend, L1 loss": lambda: Model(lgbm("l1"), f1),
        "LightGBM + usual spend, log target": lambda: Model(lgbm(), f1_log, "log"),
        "Gradient boosting + usual spend, absolute error": lambda: Model(gbm_l1(), f1),
        # 6. more household features
        "Random forest, all household features, log target": lambda: Model(rf(5), f2, "log"),
        "LightGBM, all household features, log target": lambda: Model(lgbm(), f2 + ["log_usual"], "log"),
        # 7. campaign-level features
        "Linear + usual spend + campaign features": lambda: Model(scaled(LinearRegression()), f3),
        "Huber + usual spend + campaign features": lambda: Model(scaled(HuberRegressor(max_iter=1000)), f3),
        "Random forest + campaign features": lambda: Model(rf(10), f3),
        "Random forest + campaign features, absolute error": lambda: Model(rf(5, "absolute_error"), f3),
        "LightGBM + campaign features, L1 loss": lambda: Model(lgbm("l1"), f3),
        "Gradient boosting + campaign features, absolute error": lambda: Model(gbm_l1(), f3),
        "Ratio to usual spend, random forest + campaign features": lambda: Model(rf(10), f3, "ratio"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    table = load_table(args.processed).merge(campaign_features(args.processed), on="campaign", how="left")
    table["log_usual"] = np.log1p(table[USUAL])
    extra = [c for c in EXTRA_FEATURES if table[c].isna().any()]
    table = table.assign(**{c: table[c].fillna(0) for c in extra})
    train, test = time_split(table)

    rows = []
    for name, make in candidates().items():
        split = errors(test[LABEL], make().fit(train).predict(test))
        actual, pred = [], []
        for _, tr, te in walk_forward_splits(table):
            actual.append(te[LABEL].to_numpy())
            pred.append(make().fit(tr).predict(te))
        wf = errors(np.concatenate(actual), np.concatenate(pred))
        rows.append({"model": name, "walk_forward_MAE": wf["mae"], "walk_forward_rho": wf["rank_rho"],
                     "split_MAE": split["mae"], "split_rho": split["rank_rho"]})
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
