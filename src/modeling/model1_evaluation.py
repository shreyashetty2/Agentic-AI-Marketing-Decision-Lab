"""Model 1 (Response): validate the baseline, understand its features, then compare one change at a time.

Builds on Anna's baseline (src/modeling/model1_baseline.py, PR #8) without changing it. That baseline
showed a clear signal on one out-of-time split: the random forest catches 61% of held-out redeemers in
its top 20%, vs. 52% for the "redeemed before" rule. This script asks how well that holds up, and sets
up a fair way to improve it (following the mentors' guidance: validate and understand the current
features before adding complexity, and keep feature gains separate from model gains).

  Step 1  Validate    a) Anna's exact split, scored two ways: pooled (as reported) and WITHIN each
                         campaign (one target list per campaign, which is how a marketer uses it).
                      b) Forward-chaining: every campaign starting on/after --first-test-day is tested
                         on its own, trained only on campaigns that ENDED before its feature cutoff
                         (start_day - cutoff_gap_days, from feature_2a_spec.yaml). Results by type,
                         per campaign, with bootstrap intervals.
                      c) Calibration: do predicted probabilities match observed redemption rates?
                         (The Decision Lab sums them to forecast redemptions.)
  Step 2  Understand  which of the 11 baseline features the random forest relies on: permutation
                      importance, single-feature rankings, and drop-one-group retraining.
  Step 3  Compare     {random forest, LightGBM} x {baseline 11 features, + already-built 2A features},
                      same forward-chaining folds, so any gain is traceable to features, model, or both.
                      No hyperparameter tuning here: settings are fixed up front so test folds are
                      never used to choose them.

Run from the repo folder (after src/features/build_2b.py):
    python src/modeling/model1_evaluation.py --processed data/processed
    python src/modeling/model1_evaluation.py --quick      # fewer repeats, for a fast check

Input : data/processed/model1_table.parquet (Feature Agent 2B)
Output: data/processed/model1_evaluation_report.md
        data/processed/model1_evaluation_predictions.parquet   (every fold's held-out scores)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # lets `features.*` / `modeling.*` import as a script

from features.spec import column_roles, load_spec  # noqa: E402
from modeling.model1_baseline import (  # noqa: E402
    BASE_FEATURES, FEATURES, LABEL, TOP_SHARE, load_table, time_split, top_share_capture, train_models,
)

SPEC = load_spec()
CUTOFF_GAP = SPEC["settings"]["cutoff_gap_days"]
RULE = "past_redemption_rate"          # the "redeemed before" rule, same as the baseline
FIRST_TEST_DAY = 412                   # campaign 8: first point with 2 Type A campaigns already finished

# Already built and leak-tested by Feature Agent 2A, but not used by the baseline. Each has a reason it
# should carry signal and is known by the feature cutoff. Deliberately left out for now:
#   n_overlapping_campaigns  counts campaigns that START after the cutoff (not known at decision time)
#   classification_* / homeowner_desc / kid_category_desc   blank for 68% of households; separate test later
EXTRA_FEATURES = {
    "spend_per_week_8w": "recent spend; with the 26-week version, shows ramping up vs. slowing down",
    "trips_per_week_8w": "recent visit frequency",
    "days_since_last_trip": "recency: lapsed shoppers are unlikely to redeem",
    "avg_basket_value_26w": "big-basket shoppers have more chances to use a coupon",
    "n_departments_26w": "broad shoppers are more likely to buy whatever the coupons cover",
    "past_redemptions": "count of past redemptions (the rate hides how many campaigns it is based on)",
    "past_campaigns_received": "how much campaign history the rate is based on",
    "past_campaigns_redeemed": "number of campaigns redeemed before, a sturdier signal than the rate",
    "other_campaigns_live_at_cutoff": "competing campaigns in the household's hands at the same time",
    "has_demographics": "known before the campaign; may reflect how engaged the loyalty member is",
}
EXTENDED_FEATURES = FEATURES + list(EXTRA_FEATURES)

FEATURE_GROUPS = {   # for drop-one-group retraining (covers all 11 baseline features)
    "shopping activity": ["spend_per_week_26w", "trips_per_week_26w", "history_weeks_before_cutoff"],
    "deal seeking": ["coupon_lines_per_week_26w", "share_spend_on_discount_26w"],
    "redemption history": ["past_redemption_rate"],
    "offer fit": ["campaign_product_spend_per_week_26w", "campaign_product_share_26w"],
    "campaign attributes": ["duration_days", "is_type_a", "is_type_b"],
}
MODEL_LABELS = {"rf": "Random forest", "lgbm": "LightGBM", "lr": "Logistic regression"}


# --------------------------------------------------------------------------- data
def prepare_table(processed: Path) -> pd.DataFrame:
    """Anna's load_table (same blank handling and type flags), plus a role check on every feature."""
    table = load_table(processed)
    roles = column_roles(SPEC)
    for col in BASE_FEATURES + list(EXTRA_FEATURES):
        if roles.get(col) not in ("profile", "context"):
            raise ValueError(f"{col} has role {roles.get(col)!r} in feature_2a_spec.yaml; only profile/context may be inputs")
    blanks = [c for c in EXTRA_FEATURES if table[c].isna().any()]
    if blanks:   # none expected on the real data; stop rather than guess what a blank means
        raise ValueError(f"unexpected blanks in {blanks}; check feature_2a_spec.yaml blank_when before filling")
    return table


def campaign_calendar(table: pd.DataFrame) -> pd.DataFrame:
    return (table[["campaign", "campaign_type", "start_day", "end_day"]].drop_duplicates()
            .sort_values(["start_day", "end_day", "campaign"]).reset_index(drop=True))


def forward_chain_folds(table: pd.DataFrame, first_test_day: int = FIRST_TEST_DAY, gap: int = CUTOFF_GAP) -> list[dict]:
    """One fold per campaign starting on/after first_test_day. Train only on campaigns that ended
    before the test campaign's feature cutoff, so no training label comes from after the decision."""
    cal = campaign_calendar(table)
    folds = []
    for row in cal.itertuples():
        if row.start_day < first_test_day:
            continue
        cutoff = row.start_day - gap
        train = cal.loc[cal["end_day"] < cutoff, "campaign"].tolist()
        folds.append({"test_campaign": row.campaign, "campaign_type": row.campaign_type, "cutoff_day": cutoff,
                      "train_campaigns": train,
                      "n_train_type_a": int(cal.loc[cal["campaign"].isin(train), "campaign_type"].eq("A").sum())})
    return folds


# --------------------------------------------------------------------------- metrics
def _groups(campaign) -> dict:
    return pd.Series(np.arange(len(campaign))).groupby(np.asarray(campaign)).indices


def within_campaign_capture(campaign, y, score, share: float = TOP_SHARE, n_seeds: int = 20, seed: int = 0) -> float:
    """Mail the top `share` of EACH campaign's list; return the share of all redeemers reached.
    Ties are broken at random and averaged over n_seeds (matters for the rule, where most rates are 0)."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    total = y.sum()
    if total == 0:
        return np.nan
    groups, rng, caught = _groups(campaign), np.random.default_rng(seed), 0.0
    for _ in range(n_seeds):
        tiebreak = rng.random(len(y))
        for idx in groups.values():
            k = max(1, int(round(len(idx) * share)))
            caught += y[idx[np.lexsort((tiebreak[idx], -score[idx]))[:k]]].sum()
    return caught / (n_seeds * total)


def random_expected_capture(campaign, y, share: float = TOP_SHARE) -> float:
    """What random targeting reaches with the same per-campaign list sizes (about 20%)."""
    y = np.asarray(y)
    hit = sum(max(1, int(round(len(idx) * share))) / len(idx) * y[idx].sum() for idx in _groups(campaign).values())
    return hit / y.sum()


def within_campaign_auc(campaign, y, score) -> float:
    """Average per-campaign AUC, weighted by campaign size (campaigns with one class only are skipped)."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    vals, weights = [], []
    for idx in _groups(campaign).values():
        if 0 < y[idx].sum() < len(idx):
            vals.append(roc_auc_score(y[idx], score[idx]))
            weights.append(len(idx))
    return float(np.average(vals, weights=weights)) if vals else np.nan


def bootstrap_capture(campaign, y, score_a, score_b, reps: int = 300, seed: int = 1) -> dict:
    """Resample households within each campaign; 95% intervals for capture(a) and capture(a) - capture(b)."""
    campaign, y = np.asarray(campaign), np.asarray(y)
    score_a, score_b = np.asarray(score_a, dtype=float), np.asarray(score_b, dtype=float)
    groups, rng = list(_groups(campaign).values()), np.random.default_rng(seed)
    caps, diffs = [], []
    for r in range(reps):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        a = within_campaign_capture(campaign[idx], y[idx], score_a[idx], n_seeds=1, seed=r)
        b = within_campaign_capture(campaign[idx], y[idx], score_b[idx], n_seeds=1, seed=r)
        caps.append(a)
        diffs.append(a - b)
    return {"capture_ci": np.nanpercentile(caps, [2.5, 97.5]), "diff_ci": np.nanpercentile(diffs, [2.5, 97.5])}


def summarize(preds: pd.DataFrame, score_col: str, n_seeds: int) -> dict:
    out = {}
    for name, part in (("all", preds), ("A", preds[preds["campaign_type"] == "A"]),
                       ("B/C", preds[preds["campaign_type"] != "A"])):
        out[f"capture_{name}"] = within_campaign_capture(part["campaign"], part[LABEL], part[score_col], n_seeds=n_seeds)
    out["random_all"] = random_expected_capture(preds["campaign"], preds[LABEL])
    out["pooled_capture"] = top_share_capture(preds[score_col].to_numpy(), preds[LABEL].to_numpy())
    out["pooled_auc"] = roc_auc_score(preds[LABEL], preds[score_col])
    out["within_auc"] = within_campaign_auc(preds["campaign"], preds[LABEL], preds[score_col])
    return out


def calibration(preds: pd.DataFrame, score_col: str, bins: int = 10) -> dict:
    p, y = preds[score_col].to_numpy(), preds[LABEL].to_numpy()
    deciles = (preds.assign(bin=pd.qcut(p, bins, labels=False, duplicates="drop"))
               .groupby("bin").agg(n=(LABEL, "size"), predicted=(score_col, "mean"), actual=(LABEL, "mean")).reset_index())
    per_campaign = preds.groupby(["campaign", "campaign_type"]).agg(
        n=(LABEL, "size"), expected=(score_col, "sum"), actual=(LABEL, "sum")).reset_index()
    return {"brier": brier_score_loss(y, p), "brier_base": brier_score_loss(y, np.full_like(p, y.mean(), dtype=float)),
            "deciles": deciles, "per_campaign": per_campaign,
            "expected_total": per_campaign["expected"].sum(), "actual_total": int(per_campaign["actual"].sum())}


# --------------------------------------------------------------------------- models
def make_model(name: str):
    if name == "rf":    # Anna's settings, unchanged
        return RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42, n_jobs=-1)
    if name == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if name == "lgbm":  # conservative fixed settings for ~5k rows; tuning comes later with inner validation
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=42, n_jobs=2, verbose=-1)
    raise ValueError(name)


def run_forward_chain(table: pd.DataFrame, folds: list[dict], model_name: str, features: list[str]):
    """Train one model per fold; return every fold's held-out scores and the fitted models."""
    parts, models = [], []
    for fold in folds:
        train = table[table["campaign"].isin(fold["train_campaigns"])]
        test = table[table["campaign"] == fold["test_campaign"]]
        model = make_model(model_name).fit(train[features], train[LABEL])
        parts.append(test[["campaign", "campaign_type", "household_key", LABEL, RULE]]
                     .assign(score=model.predict_proba(test[features])[:, 1]))
        models.append(model)
    return pd.concat(parts, ignore_index=True), models


# --------------------------------------------------------------------------- step 2
def permutation_importance(table, folds, models, features, base_capture, repeats, n_seeds, seed=0) -> pd.DataFrame:
    """Shuffle one feature inside each test campaign; how much within-campaign capture is lost?"""
    rng, rows = np.random.default_rng(seed), []
    tests = [table[table["campaign"] == f["test_campaign"]] for f in folds]
    for feat in features:
        drops = []
        for _ in range(repeats):
            parts = []
            for test, model in zip(tests, models):
                shuffled = test.assign(**{feat: rng.permutation(test[feat].to_numpy())})
                parts.append(test[["campaign", LABEL]].assign(score=model.predict_proba(shuffled[features])[:, 1]))
            p = pd.concat(parts)
            drops.append(base_capture - within_campaign_capture(p["campaign"], p[LABEL], p["score"], n_seeds=n_seeds))
        rows.append({"feature": feat, "capture_drop": float(np.mean(drops)), "sd": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("capture_drop", ascending=False)


def single_feature_rankings(table, folds, features, n_seeds) -> pd.DataFrame:
    """Rank each test campaign by ONE feature. Direction (high vs. low = more likely) is set on the
    campaigns that finished before the first test campaign, never on test data."""
    early = table[table["campaign"].isin(folds[0]["train_campaigns"])]
    tests = table[table["campaign"].isin([f["test_campaign"] for f in folds])]
    rows = []
    for feat in features:
        corr = early[feat].corr(early[LABEL], method="spearman")
        sign = 1.0 if (pd.isna(corr) or corr >= 0) else -1.0
        rows.append({"feature": feat, "direction": "higher" if sign > 0 else "lower",
                     "capture": within_campaign_capture(tests["campaign"], tests[LABEL], sign * tests[feat], n_seeds=n_seeds)})
    return pd.DataFrame(rows).sort_values("capture", ascending=False)


# --------------------------------------------------------------------------- report
def md_table(df: pd.DataFrame, pct: tuple = (), dec: tuple = ()) -> str:
    def fmt(col, v):
        if isinstance(v, float) and np.isnan(v):
            return "n/a"
        if col in pct:
            return f"{v:.1%}"
        if col in dec:
            return f"{v:.3f}"
        return f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(fmt(c, v) for c, v in zip(df.columns, row)) + " |"
                            for row in df.itertuples(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--first-test-day", type=int, default=FIRST_TEST_DAY)
    parser.add_argument("--quick", action="store_true", help="fewer repeats/bootstraps and no drop-one-group retraining")
    args = parser.parse_args()
    seeds, reps, perm_reps = (5, 100, 2) if args.quick else (20, 300, 5)
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)  # noqa: E731

    table = prepare_table(args.processed)
    folds = forward_chain_folds(table, args.first_test_day)
    log(f"{len(table):,} rows; {len(folds)} forward-chaining folds "
        f"({sum(f['campaign_type'] == 'A' for f in folds)} Type A); cutoff gap {CUTOFF_GAP} days")

    # ---- Step 1a: Anna's split, scored pooled vs. within campaign
    train, test = time_split(table)
    anna = []
    scores = {name: m.predict_proba(test[FEATURES])[:, 1] for name, m in train_models(train).items()}
    scores['"Redeemed before" rule'] = test[RULE].to_numpy()
    for name, s in scores.items():
        anna.append({"ranking": name,
                     "pooled (as reported)": top_share_capture(s, test[LABEL].to_numpy()),
                     "within campaign": within_campaign_capture(test["campaign"], test[LABEL], s, n_seeds=seeds)})
    anna = pd.DataFrame(anna)
    anna_random = random_expected_capture(test["campaign"], test[LABEL])
    log("step 1a done (baseline split re-scored)")

    # ---- Step 1b/3: forward-chaining grid
    runs = {}
    for model_name, feat_name, feats in (("rf", "baseline 11", FEATURES), ("lr", "baseline 11", FEATURES),
                                         ("lgbm", "baseline 11", FEATURES), ("rf", "extended", EXTENDED_FEATURES),
                                         ("lgbm", "extended", EXTENDED_FEATURES)):
        preds, models = run_forward_chain(table, folds, model_name, feats)
        runs[(model_name, feat_name)] = {"preds": preds, "models": models, "summary": summarize(preds, "score", seeds)}
        log(f"forward-chaining: {MODEL_LABELS[model_name]} / {feat_name}")

    base = runs[("rf", "baseline 11")]
    rule_summary = summarize(base["preds"], RULE, seeds)
    boot = bootstrap_capture(base["preds"]["campaign"], base["preds"][LABEL], base["preds"]["score"], base["preds"][RULE], reps=reps)
    log("bootstrap done")

    per_campaign = []
    for c, part in base["preds"].groupby("campaign", sort=False):
        per_campaign.append({"campaign": int(c), "type": part["campaign_type"].iat[0], "mailed": len(part),
                             "redeemers": int(part[LABEL].sum()),
                             "random forest": within_campaign_capture(part["campaign"], part[LABEL], part["score"], n_seeds=seeds),
                             "rule": within_campaign_capture(part["campaign"], part[LABEL], part[RULE], n_seeds=seeds)})
    per_campaign = pd.DataFrame(per_campaign)

    cal = {m: calibration(runs[(m, "baseline 11")]["preds"], "score") for m in ("rf", "lr", "lgbm")}
    log("calibration done")

    # ---- Step 2: understand the baseline features (random forest, baseline 11)
    base_cap = within_campaign_capture(base["preds"]["campaign"], base["preds"][LABEL], base["preds"]["score"], n_seeds=seeds)
    perm = permutation_importance(table, folds, base["models"], FEATURES, base_cap, perm_reps, seeds)
    log("permutation importance done")
    single = single_feature_rankings(table, folds, FEATURES, seeds)
    ablation = []
    if not args.quick:
        for group, cols in FEATURE_GROUPS.items():
            feats = [f for f in FEATURES if f not in cols]
            p, _ = run_forward_chain(table, folds, "rf", feats)
            ablation.append({"dropped group": group, "features dropped": ", ".join(cols),
                             "capture without it": within_campaign_capture(p["campaign"], p[LABEL], p["score"], n_seeds=seeds)})
        ablation = pd.DataFrame(ablation).assign(change=lambda d: d["capture without it"] - base_cap)
        log("drop-one-group retraining done")

    # ---- write outputs
    all_preds = pd.concat([r["preds"].assign(model=m, features=f) for (m, f), r in runs.items()], ignore_index=True)
    all_preds.to_parquet(args.processed / "model1_evaluation_predictions.parquet", index=False)

    grid = pd.DataFrame([{"model": MODEL_LABELS[m], "features": f, "capture (all)": r["summary"]["capture_all"],
                          "Type A": r["summary"]["capture_A"], "Type B/C": r["summary"]["capture_B/C"],
                          "AUC within campaign": r["summary"]["within_auc"]} for (m, f), r in runs.items()])
    grid = pd.concat([grid, pd.DataFrame([{"model": '"Redeemed before" rule', "features": RULE,
                                           "capture (all)": rule_summary["capture_all"], "Type A": rule_summary["capture_A"],
                                           "Type B/C": rule_summary["capture_B/C"], "AUC within campaign": rule_summary["within_auc"]}])])
    s = base["summary"]
    lines = [
        "# Model 1 evaluation: validating and extending the baseline",
        "",
        f"Builds on `model1_baseline.py` (PR #8). Generated by `src/modeling/model1_evaluation.py`"
        f"{' (--quick)' if args.quick else ''}.",
        "",
        "**Capture** = share of all redeemers reached by mailing the top 20% of households. "
        "**Within campaign** = the top 20% of *each* campaign's list (one target list per campaign, as a marketer would use it). "
        "Ranking all campaigns together lets campaign type alone push Type A rows to the top, which no single list can use.",
        "",
        "## Step 1a. The baseline split, scored two ways",
        f"Same split as the baseline (earliest 70% of campaigns train, latest 9 test; 1 of the 9 is Type A). "
        f"Random targeting reaches {anna_random:.1%} within campaign.",
        "",
        md_table(anna, pct=("pooled (as reported)", "within campaign")),
        "",
        "## Step 1b. Forward-chaining by campaign",
        f"{len(folds)} test campaigns ({sum(f['campaign_type'] == 'A' for f in folds)} Type A), each trained only on campaigns "
        f"that ended before its feature cutoff (start day minus {CUTOFF_GAP}). Random targeting reaches {s['random_all']:.1%}.",
        "",
        f"- Random forest (baseline settings and features): **{s['capture_all']:.1%}** within campaign "
        f"(95% interval {boot['capture_ci'][0]:.1%} to {boot['capture_ci'][1]:.1%}); Type A {s['capture_A']:.1%}, Type B/C {s['capture_B/C']:.1%}",
        f"- \"Redeemed before\" rule: **{rule_summary['capture_all']:.1%}**; Type A {rule_summary['capture_A']:.1%}, Type B/C {rule_summary['capture_B/C']:.1%}",
        f"- Model minus rule: {s['capture_all'] - rule_summary['capture_all']:+.1%} "
        f"(95% interval {boot['diff_ci'][0]:+.1%} to {boot['diff_ci'][1]:+.1%}; an interval above 0 means the model reliably beats the rule)",
        f"- For reference, pooled across all {len(folds)} test campaigns: random forest {s['pooled_capture']:.1%}, AUC {s['pooled_auc']:.3f} "
        f"(within-campaign AUC {s['within_auc']:.3f})",
        "",
        "Per test campaign (small B/C campaigns are noisy; read the totals above first):",
        "",
        md_table(per_campaign, pct=("random forest", "rule")),
        "",
        "## Step 1c. Calibration (forward-chaining predictions)",
        "Expected redemptions = sum of predicted probabilities; the Decision Lab relies on this matching reality.",
        "",
        md_table(pd.DataFrame([{"model": MODEL_LABELS[m], "Brier score": c["brier"], "Brier, always predict base rate": c["brier_base"],
                                "expected redemptions": round(c["expected_total"], 1), "actual redemptions": c["actual_total"]}
                               for m, c in cal.items()]), dec=("Brier score", "Brier, always predict base rate")),
        "",
        "Random forest, by predicted-probability decile:",
        "",
        md_table(cal["rf"]["deciles"], dec=("predicted", "actual")),
        "",
        "## Step 2. Understanding the baseline features (random forest, forward-chaining)",
        "Permutation importance: capture lost when one feature is shuffled within each test campaign (bigger = relied on more).",
        "",
        md_table(perm, pct=("capture_drop", "sd")),
        "",
        "Each feature alone as a ranking rule (direction set on campaigns before the first test campaign):",
        "",
        md_table(single, pct=("capture",)),
        "",
    ]
    if len(ablation):
        lines += ["Drop one group of features and retrain:", "", md_table(ablation, pct=("capture without it", "change")), ""]
    lines += [
        "## Step 3. One change at a time (same folds, fixed settings, no tuning)",
        "Read across a row for the effect of features, down a column for the effect of the model.",
        "",
        md_table(grid, pct=("capture (all)", "Type A", "Type B/C"), dec=("AUC within campaign",)),
        "",
        "Features added in the extended set (already built and leak-tested by Feature Agent 2A):",
        "",
        *[f"- `{k}`: {v}" for k, v in EXTRA_FEATURES.items()],
        "",
        "## Caveats",
        "- Not causal: a high score means likely to redeem, not that the coupon caused it (Model 3's question).",
        "- Only 3 Type A test campaigns and many small B/C campaigns, so differences of a few points may be noise; use the intervals.",
        "- Offer-fit features for Type A use the whole coupon pool (each household saw only 16 coupons).",
        f"- Runtime {time.time() - t0:.0f}s.",
    ]
    report = args.processed / "model1_evaluation_report.md"
    report.write_text("\n".join(lines))
    log(f"report: {report}")


if __name__ == "__main__":
    main()
