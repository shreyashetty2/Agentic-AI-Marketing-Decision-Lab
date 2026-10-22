"""Model 1 feature investigation: which features add something BEYOND coupon use?

Background: on forward-chaining tests, ranking households by coupon use alone (coupon_lines_per_week_26w)
catches as many redeemers as the owner's 11-column random forest, so coupon use is the bar any new feature
must beat. This script tests candidate features against that bar, on the Modeling Agent's OPEN test
campaigns only (the 5 locked campaigns, 21-25, are never read here).

Candidates: the 18 columns Feature Agent 2A builds for this (coupon habit in more detail, coupon use on the
campaign's products and commodities, redemption history in more detail, demographics as numbers), plus the
ideas that were tested and NOT adopted, which this script builds itself from pre-cutoff data so the result
can be reproduced (more coupon-habit variants, offer-fit variants, recency of redemption, private-label share,
the main store's coupon culture, campaign size, and in-store display/flyer habits from causal_data).

What it measures (all on the open test campaigns):
  A. no model: does each column still sort redeemers among households with the SAME coupon habit?
     (AUC inside each campaign x coupon-use fifth, weighted by redeemers; 0.5 = adds nothing)
  B. random forests with the owner's settings, changing only the columns: coupon + one group,
     owner + one group, owner + one column (--quick skips this), all columns, all minus one group
  C. the learner: forest, slow LightGBM, logistic regression on log-scaled columns
Every comparison is a paired gap with a 95% interval from resampling whole households, on within-campaign
AUC and top-20% capture, plus how many of the 13 campaigns improved and the gap in the earlier vs later ones.

Run from the repo folder (after src/features/build_2b.py; about 45 minutes, --quick about 15):
    python src/modeling/model1_feature_search.py --processed data/processed --causal data/raw/causal_data.parquet

Output: data/processed/model1_feature_search/ results_a.csv, results_b.csv, report.md
"""
import argparse
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # lets `features.*` / `modeling.*` import as a script

from features.data import load_tables  # noqa: E402
from modeling.learners import signed_log1p  # noqa: E402
from modeling.model1_baseline import FEATURES as OWNER, LABEL, load_table  # noqa: E402
from modeling.model1_evaluation import (CUTOFF_GAP, forward_chain_folds, md_table,  # noqa: E402
                                        within_campaign_auc, within_campaign_capture)

ROOT = Path(__file__).resolve().parents[2]
COUPON = "coupon_lines_per_week_26w"
FIRST_TEST_DAY, LOCKED, MIN_TRAIN_ROWS, MIN_TRAIN_POS = 412, 5, 500, 50   # as modeling_spec.yaml, Model 1
FOREST = dict(n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42, n_jobs=2)   # the owner's settings
EXTENDED_EXTRA = ["spend_per_week_8w", "trips_per_week_8w", "days_since_last_trip", "avg_basket_value_26w",
                  "n_departments_26w", "past_redemptions", "past_campaigns_received", "past_campaigns_redeemed",
                  "other_campaigns_live_at_cutoff", "has_demographics"]
BUILT_GROUPS = {   # already-built 2A columns, as the coupon_plus_* sets in modeling_spec.yaml
    "shopping": ["spend_per_week_26w", "trips_per_week_26w", "history_weeks_before_cutoff", "spend_per_week_8w",
                 "trips_per_week_8w", "days_since_last_trip", "avg_basket_value_26w", "n_departments_26w"],
    "deal": ["share_spend_on_discount_26w"],
    "history": ["past_redemption_rate", "past_redemptions", "past_campaigns_received", "past_campaigns_redeemed"],
    "offer fit": ["campaign_product_spend_per_week_26w", "campaign_product_share_26w"],
    "campaign": ["duration_days", "is_type_a", "is_type_b", "other_campaigns_live_at_cutoff"],
}
ADOPTED = {   # built by 2A since the investigation
    "coupon habit, refined": ["coupon_lines_per_week_8w", "coupon_lines_per_week_all", "coupon_trip_share_26w",
                              "coupon_commodities_26w"],
    "offer fit, refined": ["campaign_product_coupon_lines_per_week_26w", "campaign_commodity_coupon_lines_per_week_26w"],
    "redemption history, refined": ["redeemed_last_campaign", "past_redemptions_in_campaign_commodities"],
    "demographics": ["classification_1_code", "classification_3_code", "classification_4_code", "classification_5_code",
                     "kid_count_code", "classification_2_x", "classification_2_y", "classification_2_z",
                     "homeowner_flag", "renter_flag"],
}
NOT_ADOPTED = {   # built here only
    "coupon habit, refined": ["coupon_trend_8w_minus_26w", "days_since_last_coupon", "coupon_line_share_26w",
                              "coupon_savings_per_week_26w", "coupon_match_lines_per_week_26w", "coupon_days_26w"],
    "offer fit, refined": ["campaign_commodity_share_26w", "campaign_products_bought_26w", "campaign_product_trip_share_26w",
                           "campaign_coupons_matched", "campaign_coupons_matched_share"],
    "redemption history, refined": ["days_since_last_redemption", "redemptions_in_live_campaigns",
                                    "redeemed_this_campaigns_coupons_before"],
    "brand and store": ["private_label_share_26w", "main_store_coupon_line_share"],
    "campaign size": ["campaign_n_coupons", "campaign_n_products"],
    "in-store promotion habit": ["flyer_spend_share_26w", "display_spend_share_26w"],
}
NEW_GROUPS = {g: ADOPTED.get(g, []) + NOT_ADOPTED.get(g, []) + (["has_demographics"] if g == "demographics" else [])
              for g in list(ADOPTED) + [g for g in NOT_ADOPTED if g not in ADOPTED]}
EXTENDED = OWNER + EXTENDED_EXTRA
EXTENDED_PLUS = EXTENDED + [c for g in ADOPTED.values() for c in g]          # = extended_plus in modeling_spec.yaml
ALL = EXTENDED + [c for g in NEW_GROUPS.values() for c in g if c not in EXTENDED]
LEARNERS = {
    "forest": lambda: RandomForestClassifier(**FOREST),
    "lightgbm_slow": lambda: LGBMClassifier(n_estimators=600, learning_rate=0.01, num_leaves=7, min_child_samples=60,
                                            subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0,
                                            random_state=42, n_jobs=2, verbose=-1),
    "logistic_log": lambda: make_pipeline(FunctionTransformer(signed_log1p), StandardScaler(),
                                          LogisticRegression(C=0.05, max_iter=3000)),
}


# --------------------------------------------------------------------------- candidate columns not built by 2A
def build_not_adopted(processed: Path, causal: Path, rows: pd.DataFrame) -> pd.DataFrame:
    """rows: household_key, campaign, start_day (Model 1's). Every value uses only days before the cutoff."""
    t = load_tables(processed)
    brand = pd.read_parquet(processed / "product.parquet")
    brand.columns = [c.lower() for c in brand.columns]
    brand = brand.set_index("product_id")["brand"]
    commodity_of = t.products.set_index("product_id")["commodity_desc"]
    con = duckdb.connect()
    con.register("tx", t.transactions)
    tx = con.execute(f"""
        WITH p AS (SELECT product_id, store_id, week_no, max((display <> '0')::INT) d, max((mailer <> '0')::INT) m
                   FROM read_parquet('{causal}') GROUP BY ALL)
        SELECT tx.*, coalesce(p.d, 0) AS on_display, coalesce(p.m, 0) AS in_flyer
        FROM tx LEFT JOIN p USING (product_id, store_id, week_no)""").df()
    tx["commodity_desc"] = tx["product_id"].map(commodity_of)
    tx["private"] = tx["product_id"].map(brand).eq("Private")
    mailings, red, coupons = t.mailings(), t.redemptions, t.coupons
    out = []
    for camp, g in rows.groupby("campaign"):
        cutoff = int(g["start_day"].iloc[0]) - CUTOFF_GAP
        hh = g["household_key"]
        before_all = tx[tx["day"] < cutoff]
        before = before_all[before_all["household_key"].isin(hh)]
        hist_days = (cutoff - hh.map(before.groupby("household_key")["day"].min())).fillna(0)
        wk26, wk8 = np.minimum(182, hist_days) / 7, np.minimum(56, hist_days) / 7

        def agg(frame, col, how):
            return hh.map(frame.groupby("household_key")[col].agg(how)).fillna(0).astype(float)

        def ratio(num, den):
            return (num / den.where(den > 0)).fillna(0)

        w26, w8 = before[before["day"] >= cutoff - 182], before[before["day"] >= cutoff - 56]
        c26 = w26[w26["coupon_disc"] < 0]
        spend26, trips26, lines26 = agg(w26, "sales_value", "sum"), agg(w26, "basket_id", "nunique"), agg(w26, "day", "size")
        o = g[["household_key", "campaign"]].copy()
        o["coupon_trend_8w_minus_26w"] = (ratio(agg(w8[w8["coupon_disc"] < 0], "day", "size"), wk8)
                                          - ratio(agg(c26, "day", "size"), wk26))
        last_cpn = hh.map(before[before["coupon_disc"] < 0].groupby("household_key")["day"].max())
        o["days_since_last_coupon"] = (cutoff - last_cpn).fillna(cutoff).astype(float)     # never: as long as possible
        o["coupon_line_share_26w"] = ratio(agg(c26, "day", "size"), lines26)
        o["coupon_savings_per_week_26w"] = ratio(-agg(c26, "coupon_disc", "sum"), wk26)
        o["coupon_match_lines_per_week_26w"] = ratio(agg(w26[w26["coupon_match_disc"] < 0], "day", "size"), wk26)
        o["coupon_days_26w"] = agg(c26, "day", "nunique")
        cp = coupons[coupons["campaign"] == camp]
        prods = set(cp["product_id"])
        comms = set(commodity_of.reindex(list(prods)).dropna())
        in_p, in_c = w26[w26["product_id"].isin(prods)], w26[w26["commodity_desc"].isin(comms)]
        o["campaign_commodity_share_26w"] = ratio(agg(in_c, "sales_value", "sum"), spend26)
        o["campaign_products_bought_26w"] = agg(in_p, "product_id", "nunique")
        o["campaign_product_trip_share_26w"] = ratio(agg(in_p, "basket_id", "nunique"), trips26)
        matched = (in_p[["household_key", "product_id"]].drop_duplicates()
                   .merge(cp[["coupon_upc", "product_id"]].drop_duplicates(), on="product_id"))
        o["campaign_coupons_matched"] = agg(matched, "coupon_upc", "nunique")
        o["campaign_coupons_matched_share"] = o["campaign_coupons_matched"] / cp["coupon_upc"].nunique()
        rb = red[(red["day"] < cutoff) & red["household_key"].isin(hh)]
        o["days_since_last_redemption"] = (cutoff - hh.map(rb.groupby("household_key")["day"].max())).fillna(cutoff).astype(float)
        live = mailings[(mailings["start_day"] < cutoff) & (mailings["end_day"] >= cutoff)]
        o["redemptions_in_live_campaigns"] = agg(rb.merge(live[["household_key", "campaign"]]), "day", "size")
        o["redeemed_this_campaigns_coupons_before"] = agg(rb[rb["coupon_upc"].isin(set(cp["coupon_upc"]))], "coupon_upc", "nunique")
        o["private_label_share_26w"] = ratio(agg(w26[w26["private"]], "sales_value", "sum"), spend26)
        st = before_all[before_all["day"] >= cutoff - 182].assign(cpn=lambda d: d["coupon_disc"] < 0)
        store = st.groupby("store_id").agg(lines=("day", "size"), cpn=("cpn", "sum"))
        own = st[st["household_key"].isin(hh)].groupby(["household_key", "store_id"]).agg(lines=("day", "size"), cpn=("cpn", "sum"))
        main = hh.map(w26.groupby("household_key")["store_id"].agg(lambda s: s.value_counts().index[0]))
        key = list(zip(hh, main))
        own_l = pd.Series([own["lines"].get(k, 0) for k in key], index=hh.index)
        own_c = pd.Series([own["cpn"].get(k, 0) for k in key], index=hh.index)
        o["main_store_coupon_line_share"] = ratio((main.map(store["cpn"]) - own_c).astype(float),   # leave the household out
                                                  (main.map(store["lines"]) - own_l).astype(float))
        o["campaign_n_coupons"] = float(cp["coupon_upc"].nunique())
        o["campaign_n_products"] = float(len(prods))
        o["flyer_spend_share_26w"] = ratio(agg(w26[w26["in_flyer"] == 1], "sales_value", "sum"), spend26)
        o["display_spend_share_26w"] = ratio(agg(w26[w26["on_display"] == 1], "sales_value", "sum"), spend26)
        out.append(o)
    return pd.concat(out)


def load(processed: Path, causal: Path) -> pd.DataFrame:
    """The owner's Model 1 table (same rows, blank filling and type flags) plus the not-adopted candidates.
    2A blanks in the new columns all mean "none" (no history / no spend / never received), so 0, as the
    Modeling Agent fills them."""
    t = load_table(processed)
    extra = build_not_adopted(processed, causal, t[["household_key", "campaign", "start_day"]])
    t = t.merge(extra, on=["household_key", "campaign"], how="left", validate="1:1")
    cols = sorted(set(ALL))
    t[cols] = t[cols].fillna(0)
    return t


def open_folds(t: pd.DataFrame) -> list[dict]:
    """The Modeling Agent's outer folds minus its locked last five."""
    folds = [f for f in forward_chain_folds(t, FIRST_TEST_DAY)
             if (tr := t[t["campaign"].isin(f["train_campaigns"])]).shape[0] >= MIN_TRAIN_ROWS
             and tr[LABEL].sum() >= MIN_TRAIN_POS]
    return folds[:-LOCKED]


# --------------------------------------------------------------------------- measures
def same_habit_auc(d: pd.DataFrame, col: str, by: str = "coupon_fifth") -> float:
    """AUC of `col` among households in the same campaign AND the same coupon-use fifth, weighted by redeemers.
    0.5 = it adds nothing once coupon habit is known; far from 0.5 either way = it does."""
    num = den = 0.0
    for _, g in d.groupby(["campaign", by]):
        y = g[LABEL]
        if 0 < y.sum() < len(y) and g[col].nunique() > 1:
            num += roc_auc_score(y, g[col]) * y.sum()
            den += y.sum()
    return num / den if den else np.nan


def paired_gap(test: pd.DataFrame, a, b, reps: int = 200, seed: int = 7) -> dict:
    """a minus b on within-campaign AUC and top-20% capture; 95% interval from resampling whole households;
    campaigns where a's AUC beat b's; and the AUC gap on the earlier vs later half of the test campaigns."""
    d = pd.DataFrame({"household_key": test["household_key"].to_numpy(), "campaign": test["campaign"].to_numpy(),
                      "label": test[LABEL].to_numpy(), "a": np.asarray(a), "b": np.asarray(b)})

    def stat(x, seeds):
        c, y = x["campaign"], x["label"]
        return np.array([within_campaign_auc(c, y, x["a"]) - within_campaign_auc(c, y, x["b"]),
                         within_campaign_capture(c, y, x["a"], n_seeds=seeds)
                         - within_campaign_capture(c, y, x["b"], n_seeds=seeds)])
    point = stat(d, 20)
    rows_of = [idx.to_numpy() for idx in d.groupby("household_key").groups.values()]
    rng = np.random.default_rng(seed)
    draws = np.array([stat(d.iloc[np.concatenate([rows_of[k] for k in rng.integers(0, len(rows_of), len(rows_of))])], 1)
                      for _ in range(reps)])
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    per = d.groupby("campaign")[["label", "a", "b"]].apply(
        lambda g: roc_auc_score(g["label"], g["a"]) - roc_auc_score(g["label"], g["b"]) if 0 < g["label"].sum() < len(g) else np.nan)
    order = list(dict.fromkeys(d["campaign"]))
    half = len(order) // 2 + 1
    halves = {}
    for name, cs in (("early", order[:half]), ("late", order[half:])):
        x = d[d["campaign"].isin(cs)]
        halves[f"auc_gap_{name}"] = within_campaign_auc(x["campaign"], x["label"], x["a"]) - within_campaign_auc(x["campaign"], x["label"], x["b"])
    return {"auc_gap": point[0], "auc_low": lo[0], "auc_high": hi[0], "capture_gap": point[1], "capture_low": lo[1],
            "capture_high": hi[1], "campaigns_better": int((per > 0).sum()), "campaigns": int(per.notna().sum()), **halves}


class Runner:
    """Forward-chaining predictions on the open campaigns, cached by (learner, columns)."""
    def __init__(self, table: pd.DataFrame, folds: list[dict]):
        self.t, self.folds, self.cache = table, folds, {}
        self.test = pd.concat([table[table["campaign"] == f["test_campaign"]] for f in folds])

    def predict(self, cols: list[str], learner: str = "forest") -> np.ndarray:
        key = (learner, tuple(sorted(cols)))
        if key not in self.cache:
            parts = []
            for f in self.folds:
                tr, te = self.t[self.t["campaign"].isin(f["train_campaigns"])], self.t[self.t["campaign"] == f["test_campaign"]]
                parts.append(LEARNERS[learner]().fit(tr[list(key[1])], tr[LABEL]).predict_proba(te[list(key[1])])[:, 1])
            self.cache[key] = np.concatenate(parts)
        return self.cache[key]

    def score(self, s) -> dict:
        c, y = self.test["campaign"], self.test[LABEL]
        return {"auc": within_campaign_auc(c, y, s), "capture": within_campaign_capture(c, y, s, n_seeds=20)}


# --------------------------------------------------------------------------- run
def run(processed: Path, causal: Path, quick: bool, out_dir: Path) -> None:
    t0 = time.time()
    t = load(processed, causal)
    folds = open_folds(t)
    r = Runner(t, folds)
    locked = [f["test_campaign"] for f in forward_chain_folds(t, FIRST_TEST_DAY)][-LOCKED:]
    print(f"open test campaigns {[f['test_campaign'] for f in folds]}; locked, not read: {locked}")

    # A. no model
    d = t[t["campaign"].isin([f["test_campaign"] for f in folds])].copy()
    d["coupon_fifth"] = d.groupby("campaign")[COUPON].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    group_of = {c: g for g, cols in {**BUILT_GROUPS, **NEW_GROUPS}.items() for c in cols}
    a = pd.DataFrame([{"column": c, "group": group_of.get(c, "owner"), "adopted_in_2a": c in EXTENDED_PLUS,
                       "auc_alone": within_campaign_auc(d["campaign"], d[LABEL], d[c]),
                       "auc_same_coupon_habit": same_habit_auc(d, c)}
                      for c in ALL if c != COUPON and d[c].nunique() > 1])
    a["beyond_coupon"] = (a["auc_same_coupon_habit"] - 0.5).abs()
    a = a.sort_values("beyond_coupon", ascending=False)
    print(f"A done {time.time() - t0:.0f}s")

    # B / C. models
    owner, coupon_forest, coupon_rule = r.predict(OWNER), r.predict([COUPON]), r.test[COUPON].to_numpy()
    rows = []

    def add(name, kind, cols, learner="forest", vs="owner forest"):
        s = r.predict(cols, learner)
        base = {"owner forest": owner, "coupon-only forest": coupon_forest, "coupon rule": coupon_rule,
                "all-columns forest": r.predict(ALL)}[vs]
        rows.append({"comparison": name, "kind": kind, "learner": learner, "n_columns": len(cols), "vs": vs,
                     **r.score(s), **paired_gap(r.test, s, base)})
        x = rows[-1]
        print(f"{time.time() - t0:5.0f}s {name:52s} vs {vs:18s} AUC {x['auc_gap']:+.4f} [{x['auc_low']:+.4f},{x['auc_high']:+.4f}] "
              f"capture {x['capture_gap']:+.4f} [{x['capture_low']:+.4f},{x['capture_high']:+.4f}] {x['campaigns_better']}/{x['campaigns']}",
              flush=True)

    bars = pd.DataFrame([{"ranking": k, **r.score(v)} for k, v in {
        '"Redeemed before" rule': r.test["past_redemption_rate"].to_numpy(), "Coupon use alone (no model)": coupon_rule,
        "Coupon-only forest": coupon_forest, "Owner forest (11 columns)": owner,
        "Extended forest (21)": r.predict(EXTENDED)}.items()])
    rows.append({"comparison": "coupon rule (no model)", "kind": "bar", "learner": "none", "n_columns": 1,
                 "vs": "owner forest", **r.score(coupon_rule), **paired_gap(r.test, coupon_rule, owner)})
    for g, cols in BUILT_GROUPS.items():
        add(f"coupon + {g}", "coupon + built group", [COUPON] + cols, vs="coupon-only forest")
    for g, cols in NEW_GROUPS.items():
        add(f"coupon + {g} (new)", "coupon + new group", [COUPON] + cols, vs="coupon-only forest")
    for g, cols in NEW_GROUPS.items():
        add(f"owner + {g} (new)", "owner + new group", OWNER + [c for c in cols if c not in OWNER])
    if not quick:
        for c in [c for cols in NEW_GROUPS.values() for c in cols if c not in OWNER]:
            add(f"owner + {c}", "owner + one column", OWNER + [c])
    for learner in LEARNERS:
        add(f"all {len(ALL)} columns, {learner}", "all columns", ALL, learner)
        add(f"extended_plus ({len(EXTENDED_PLUS)} columns), {learner}", "adopted set", EXTENDED_PLUS, learner)
        add(f"extended_plus ({len(EXTENDED_PLUS)} columns), {learner}", "adopted set vs coupon rule", EXTENDED_PLUS, learner,
            vs="coupon rule")
    add("owner columns, logistic_log", "learner only", OWNER, "logistic_log")
    for g, cols in {**{f"built: {k}": v for k, v in BUILT_GROUPS.items()},
                    **{f"new: {k}": v for k, v in NEW_GROUPS.items()}}.items():
        add(f"all minus {g}", "drop one group", [c for c in ALL if c not in cols or c == COUPON], vs="all-columns forest")
    add("all minus coupon_lines_per_week_26w", "drop one group", [c for c in ALL if c != COUPON], vs="all-columns forest")

    b = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    a.to_csv(out_dir / "results_a.csv", index=False)
    b.to_csv(out_dir / "results_b.csv", index=False)
    bars.to_csv(out_dir / "bars.csv", index=False)
    (out_dir / "report.md").write_text(render(a, b, bars, folds, locked, time.time() - t0))
    print(f"done in {time.time() - t0:.0f}s -> {out_dir}")


def render(a, b, bars, folds, locked, runtime) -> str:
    pct = lambda v: f"{100 * v:.1f}%"
    pts = lambda v: f"{100 * v:+.1f}"
    def gap_table(sub):
        return md_table(pd.DataFrame({
            "comparison": sub["comparison"], "vs": sub["vs"], "columns": sub["n_columns"],
            "AUC gap (pts)": [f"{pts(g)} [{pts(lo)}, {pts(hi)}]" for g, lo, hi in zip(sub.auc_gap, sub.auc_low, sub.auc_high)],
            "capture gap (pts)": [f"{pts(g)} [{pts(lo)}, {pts(hi)}]" for g, lo, hi in zip(sub.capture_gap, sub.capture_low, sub.capture_high)],
            "campaigns better": [f"{x}/{n}" for x, n in zip(sub.campaigns_better, sub.campaigns)],
            "AUC gap early / late": [f"{pts(e)} / {pts(l)}" for e, l in zip(sub.auc_gap_early, sub.auc_gap_late)]}))
    lines = [
        "# Model 1 feature search: results", "",
        f"Open test campaigns ({len(folds)}): {[f['test_campaign'] for f in folds]}. Locked campaigns {locked} were not read. "
        "Gaps are in percentage points with 95% intervals from resampling whole households; AUC is within-campaign, "
        "capture is redeemers caught in each campaign's top 20%. All models are random forests with the owner's settings "
        "unless a learner is named.", "",
        "## Bars", "", md_table(bars.assign(auc=bars.auc.round(4), capture=bars.capture.map(pct))), "",
        "## A. Signal beyond coupon use (no model)", "",
        "AUC among households in the same campaign and the same coupon-use fifth (0.5 = adds nothing beyond coupon use).", "",
        md_table(a.assign(**{c: a[c].round(3) for c in ("auc_alone", "auc_same_coupon_habit", "beyond_coupon")})), "",
    ]
    for kind, title in (("bar", "Coupon use alone vs. the owner's forest"),
                        ("coupon + built group", "Coupon use + one already-built group (vs. coupon-only forest)"),
                        ("coupon + new group", "Coupon use + one new group (vs. coupon-only forest)"),
                        ("owner + new group", "Owner's 11 + one new group (vs. owner forest)"),
                        ("owner + one column", "Owner's 11 + one new column (vs. owner forest)"),
                        ("all columns", "All columns, three learners (vs. owner forest)"),
                        ("adopted set", "Adopted set: extended_plus (vs. owner forest)"),
                        ("adopted set vs coupon rule", "Adopted set: extended_plus (vs. coupon use alone)"),
                        ("learner only", "Owner's columns, logistic on log-scaled columns (vs. owner forest)"),
                        ("drop one group", "Drop one group from all columns (vs. all-columns forest)")):
        sub = b[b["kind"] == kind]
        if len(sub):
            lines += [f"## {title}", "", gap_table(sub), ""]
    lines.append(f"Runtime {runtime / 60:.0f} min.")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--processed", type=Path, default=ROOT / "data/processed")
    p.add_argument("--causal", type=Path, default=ROOT / "data/raw/causal_data.parquet")
    p.add_argument("--quick", action="store_true", help="skip the one-column-at-a-time screen")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    run(args.processed, args.causal, args.quick, args.out or args.processed / "model1_feature_search")


if __name__ == "__main__":
    main()
