"""Model 1 wrap-up analysis: which new columns carry the gain, which kinds of model use them best, and whether
the result holds by campaign type, mailing depth and forecast totals.

Follows reports/model1_feature_investigation.md. Mentor questions (10/23): is one or a few of the 18 new 2A columns
driving the benefit, and do other kinds of model do better than logistic regression on log-scaled columns?

Everything runs on the Modeling Agent's OPEN test campaigns (8-20); the locked campaigns (21-25) are never read.
Each test campaign is predicted by a model trained only on campaigns that ended before its cutoff.

  A. Importance of the 18 new columns, for the log-scaled logistic regression and the random forest:
     - total gain of the 18 over the 21 already-built columns
     - each new group added alone, and dropped from all 39
     - each new column dropped from all 39 (retrained), with a 95% interval
     - out-of-time permutation: shuffle one column inside each test campaign and re-predict with the fold's model
     - logistic only: add the new columns one at a time in order of importance (how many are needed?)
  B. Other kinds of model on the same 39 columns (vs. the baseline forest, and vs. the log-scaled logistic)
  C. Robustness: gains by campaign type, at other mailing depths (10/20/30/50%), predicted vs. actual totals
  D. Optional (--compare-processed DIR): the same models on a second set of tables, e.g. 2A rebuilt with
     --exclude-non-merchandise

Gaps are in AUC points (within-campaign ranking) and capture points (redeemers caught in each campaign's top 20%),
with 95% intervals from resampling whole households (model1_feature_search.paired_gap).

Run from the repo folder (about 20 minutes; --quick skips the one-column-at-a-time retrains, about 8):
    python src/modeling/model1_importance.py
    python src/modeling/model1_importance.py --compare-processed data/processed_nonmerch

Output: data/processed/model1_importance/ report.md and one CSV per table
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, SplineTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # lets `modeling.*` import as a script

from modeling.learners import signed_log1p  # noqa: E402
from modeling.model1_baseline import FEATURES as BASELINE, LABEL, load_table  # noqa: E402
from modeling.model1_evaluation import md_table, within_campaign_auc, within_campaign_capture  # noqa: E402
from modeling.model1_feature_search import ADOPTED, EXTENDED, EXTENDED_PLUS, open_folds, paired_gap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
NEW = [c for cols in ADOPTED.values() for c in cols]
# The 3 new columns that carry the gain for the log-scaled logistic regression (section A of this script's report);
# EXTENDED + these = the `lean` feature set in modeling_spec.yaml.
LEAN_ADDED = ["coupon_commodities_26w", "coupon_trip_share_26w", "coupon_lines_per_week_all"]
LEAN = EXTENDED + LEAN_ADDED
FOREST = dict(n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42, n_jobs=2)   # the baseline's settings
DEPTHS = (0.1, 0.2, 0.3, 0.5)


def log_logistic(C: float = 0.05, l1: bool = False):
    """Logistic regression on log-scaled columns, as the Modeling Agent's `logistic_log` learner."""
    if not l1:
        lr = LogisticRegression(C=C, max_iter=5000)
    elif tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8):   # `penalty` is deprecated from 1.8
        lr = LogisticRegression(C=C, l1_ratio=1.0, solver="liblinear", max_iter=5000)
    else:
        lr = LogisticRegression(C=C, penalty="l1", solver="liblinear", max_iter=5000)
    return make_pipeline(FunctionTransformer(signed_log1p), StandardScaler(), lr)


LEARNERS = {
    "forest": lambda: RandomForestClassifier(**FOREST),
    "logistic_log": lambda: log_logistic(0.05),
    "logistic_log, C=0.01": lambda: log_logistic(0.01),
    "logistic_log, C=0.2": lambda: log_logistic(0.2),
    "logistic_log, C=1": lambda: log_logistic(1.0),
    "logistic_log, L1": lambda: log_logistic(0.05, l1=True),
    # GAM-like: a smooth curve per column (cubic, 4 knots) before one logistic formula
    "spline logistic": lambda: make_pipeline(FunctionTransformer(signed_log1p), StandardScaler(),
                                             SplineTransformer(n_knots=4, degree=3), LogisticRegression(C=0.05, max_iter=5000)),
    "LightGBM, slow": lambda: LGBMClassifier(n_estimators=600, learning_rate=0.01, num_leaves=7, min_child_samples=60,
                                             subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0,
                                             random_state=42, n_jobs=2, verbose=-1),
    "HistGradientBoosting": lambda: HistGradientBoostingClassifier(learning_rate=0.03, max_iter=300, max_leaf_nodes=8,
                                                                   min_samples_leaf=40, l2_regularization=1.0, random_state=42),
}


class Runner:
    """Forward-chaining predictions on the open campaigns, cached by (learner, columns, separate-by-type)."""

    def __init__(self, table: pd.DataFrame, folds: list[dict]):
        self.t, self.folds, self.cache, self.models = table, folds, {}, {}
        self.test = pd.concat([table[table["campaign"] == f["test_campaign"]] for f in folds]).reset_index(drop=True)

    def predict(self, cols: list[str], learner: str = "forest", by_type: bool = False) -> np.ndarray:
        key = (learner, tuple(cols), by_type)
        if key not in self.cache:
            parts, models = [], []
            for f in self.folds:
                tr = self.t[self.t["campaign"].isin(f["train_campaigns"])]
                te = self.t[self.t["campaign"] == f["test_campaign"]]
                if by_type:   # one model for Type A campaigns, one for Type B/C
                    tr = tr[(tr["campaign_type"] == "A") == (te["campaign_type"].iloc[0] == "A")]
                m = LEARNERS[learner]().fit(tr[cols], tr[LABEL])
                parts.append(m.predict_proba(te[cols])[:, 1])
                models.append(m)
            self.cache[key], self.models[key] = np.concatenate(parts), models
        return self.cache[key]

    def score(self, s: np.ndarray, mask: np.ndarray | None = None) -> dict:
        d, s = (self.test, s) if mask is None else (self.test[mask], s[mask])
        out = {"auc": within_campaign_auc(d["campaign"], d[LABEL], s)}
        for share in DEPTHS:
            out[f"capture_{int(share * 100)}"] = within_campaign_capture(d["campaign"], d[LABEL], s, share=share, n_seeds=20)
        return out

    def gap(self, a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None, reps: int = 200) -> dict:
        d = self.test if mask is None else self.test[mask].reset_index(drop=True)
        return paired_gap(d, a if mask is None else a[mask], b if mask is None else b[mask], reps=reps)


def permutation_importance(r: Runner, cols: list[str], learner: str, reps: int, seed: int = 0) -> pd.DataFrame:
    """Shuffle one column INSIDE each test campaign (so campaign-level columns stay put), re-predict with that
    fold's own model, and record the drop in within-campaign AUC. Reliance, not added value: two columns that
    carry the same information can each look unimportant."""
    key = (learner, tuple(cols), False)
    r.predict(cols, learner)
    base = r.score(r.cache[key])["auc"]
    rng = np.random.default_rng(seed)
    rows = []
    for col in cols:
        drops = []
        for _ in range(reps):
            preds = []
            for f, m in zip(r.folds, r.models[key]):
                te = r.t[r.t["campaign"] == f["test_campaign"]][cols].copy()
                te[col] = rng.permutation(te[col].to_numpy())
                preds.append(m.predict_proba(te)[:, 1])
            drops.append(base - r.score(np.concatenate(preds))["auc"])
        rows.append({"column": col, "learner": learner, "new": col in NEW, "auc_drop": float(np.mean(drops))})
    return pd.DataFrame(rows).sort_values("auc_drop", ascending=False)


def gap_row(name: str, kind: str, vs: str, r: Runner, a, b, mask=None, reps: int = 200) -> dict:
    g = r.gap(a, b, mask, reps)
    return {"comparison": name, "kind": kind, "vs": vs, **r.score(a, mask), **g}


def run(processed: Path, out_dir: Path, quick: bool, compare: Path | None) -> None:
    t0 = time.time()
    say = lambda m: print(f"{time.time() - t0:6.0f}s {m}", flush=True)
    t = load_table(processed)
    t[EXTENDED_PLUS] = t[EXTENDED_PLUS].fillna(0)   # 2A blanks here all mean "none", as the Modeling Agent fills them
    r = Runner(t, open_folds(t))
    baseline = r.predict(BASELINE, "forest")
    rows: list[dict] = []

    # ---------------- A. importance of the 18 new columns
    drop_one = []
    for learner in ("logistic_log", "forest"):
        full, built = r.predict(EXTENDED_PLUS, learner), r.predict(EXTENDED, learner)
        rows.append(gap_row(f"21 built + 18 new vs. 21 built, {learner}", "A total gain of the 18", f"21 built, {learner}",
                            r, full, built))
        for g, cols in ADOPTED.items():
            rows.append(gap_row(f"21 built + {g} only, {learner}", "A one new group added", f"21 built, {learner}",
                                r, r.predict(EXTENDED + cols, learner), built))
            rows.append(gap_row(f"all 39 minus {g}, {learner}", "A one new group dropped", f"all 39, {learner}",
                                r, r.predict([c for c in EXTENDED_PLUS if c not in cols], learner), full))
        if not quick:
            for c in NEW:
                g = r.gap(r.predict([x for x in EXTENDED_PLUS if x != c], learner), full, reps=100)
                drop_one.append({"column": c, "learner": learner, "auc_lost": -g["auc_gap"], "low": -g["auc_high"],
                                 "high": -g["auc_low"], "capture_lost": -g["capture_gap"],
                                 "campaigns_better_without": g["campaigns_better"]})
        say(f"A: {learner} done")
    drop_one = pd.DataFrame(drop_one)
    perm = pd.concat([permutation_importance(r, EXTENDED_PLUS, "logistic_log", reps=5),
                      permutation_importance(r, EXTENDED_PLUS, "forest", reps=3)])
    order = (drop_one[drop_one["learner"] == "logistic_log"].sort_values("auc_lost", ascending=False)["column"].tolist()
             if len(drop_one) else perm[(perm["learner"] == "logistic_log") & perm["new"]]["column"].tolist())
    curve = pd.DataFrame([{"new_columns_added": k, "last_added": order[k - 1] if k else "(21 built only)",
                           **r.score(r.predict(EXTENDED + order[:k], "logistic_log"))} for k in range(len(order) + 1)])
    rows.append(gap_row("21 built + top 3 coupon columns (lean), logistic_log", "A lean set", "baseline forest",
                        r, r.predict(LEAN, "logistic_log"), baseline))
    say("A done")

    # ---------------- B. other kinds of model, all on the 39 columns
    lg39 = r.predict(EXTENDED_PLUS, "logistic_log")
    for learner in LEARNERS:
        s = r.predict(EXTENDED_PLUS, learner)
        rows.append(gap_row(f"{learner}, 39 columns", "B model type vs. baseline", "baseline forest", r, s, baseline))
        if learner != "logistic_log":
            rows.append(gap_row(f"{learner}, 39 columns", "B model type vs. logistic_log", "logistic_log, 39", r, s, lg39, reps=100))
    by_campaign = r.test["campaign"].to_numpy()
    rank = lambda v: pd.Series(v).groupby(by_campaign).rank(pct=True).to_numpy()
    blend = (rank(lg39) + rank(r.predict(EXTENDED_PLUS, "forest"))) / 2
    rows.append(gap_row("average of logistic_log and forest rankings, 39", "B model type vs. baseline", "baseline forest",
                        r, blend, baseline))
    rows.append(gap_row("average of logistic_log and forest rankings, 39", "B model type vs. logistic_log", "logistic_log, 39",
                        r, blend, lg39, reps=100))
    for learner in ("logistic_log", "forest"):
        rows.append(gap_row(f"separate Type A / Type B-C models, {learner}, 39", "B separate by type", f"one model, {learner}",
                            r, r.predict(EXTENDED_PLUS, learner, by_type=True), r.predict(EXTENDED_PLUS, learner), reps=100))
    rows.append(gap_row("logistic_log, baseline's 11 columns", "B learner only", "baseline forest",
                        r, r.predict(BASELINE, "logistic_log"), baseline))
    rows.append(gap_row("logistic_log, 21 built columns", "B learner only", "baseline forest",
                        r, r.predict(EXTENDED, "logistic_log"), baseline))
    say("B done")

    # ---------------- C. robustness
    finalists = {"baseline forest (11)": baseline, "forest (39)": r.predict(EXTENDED_PLUS, "forest"),
                 "logistic_log (39)": lg39, "logistic_log, lean (24)": r.predict(LEAN, "logistic_log"),
                 "coupon use alone (rule)": r.test["coupon_lines_per_week_26w"].fillna(0).to_numpy(),
                 '"redeemed before" rule': r.test["past_redemption_rate"].fillna(0).to_numpy()}
    masks = {"Type A": (r.test["campaign_type"] == "A").to_numpy(), "Type B/C": (r.test["campaign_type"] != "A").to_numpy(),
             "all": np.ones(len(r.test), bool)}
    depth = pd.DataFrame([{"ranking": n, "campaigns": typ, "rows": int(m.sum()), "redeemers": int(r.test.loc[m, LABEL].sum()),
                           **r.score(s, m)} for n, s in finalists.items() for typ, m in masks.items()])
    for typ in ("Type A", "Type B/C"):
        for n in ("forest (39)", "logistic_log (39)", "logistic_log, lean (24)"):
            rows.append(gap_row(f"{n}, {typ}", "C by campaign type", "baseline forest", r, finalists[n], baseline, masks[typ]))
    totals = (r.test[["campaign", "campaign_type", LABEL]]
              .assign(**{n: s for n, s in finalists.items() if "rule" not in n})
              .groupby(["campaign", "campaign_type"]).sum().reset_index())
    say("C done")

    # ---------------- D. optional second set of tables
    compare_rows = []
    if compare is not None:
        t2 = load_table(compare)
        t2[EXTENDED_PLUS] = t2[EXTENDED_PLUS].fillna(0)
        r2 = Runner(t2, open_folds(t2))
        for name, cols, learner in (("baseline forest (11)", BASELINE, "forest"), ("forest (39)", EXTENDED_PLUS, "forest"),
                                    ("logistic_log (39)", EXTENDED_PLUS, "logistic_log"), ("logistic_log, lean (24)", LEAN, "logistic_log")):
            a, b = r.score(r.predict(cols, learner)), r2.score(r2.predict(cols, learner))
            compare_rows.append({"model": name, "auc": a["auc"], "auc_compare": b["auc"],
                                 "capture": a["capture_20"], "capture_compare": b["capture_20"]})
        say("D done")

    # ---------------- outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "comparisons.csv", index=False)
    drop_one.to_csv(out_dir / "drop_one_column.csv", index=False)
    perm.to_csv(out_dir / "permutation.csv", index=False)
    curve.to_csv(out_dir / "logistic_add_new_columns.csv", index=False)
    depth.to_csv(out_dir / "by_type_and_depth.csv", index=False)
    totals.to_csv(out_dir / "totals_by_campaign.csv", index=False)
    if compare_rows:
        pd.DataFrame(compare_rows).to_csv(out_dir / "compare_processed.csv", index=False)
    (out_dir / "report.md").write_text(render(results, drop_one, perm, curve, depth, totals, compare_rows, compare, r,
                                              time.time() - t0))
    say(f"done -> {out_dir}")


def render(results, drop_one, perm, curve, depth, totals, compare_rows, compare, r, runtime) -> str:
    pts = lambda v: f"{100 * v:+.1f}"
    pct = lambda v: f"{100 * v:.1f}%"

    def gaps(kind: str) -> str:
        sub = results[results["kind"] == kind]
        return md_table(pd.DataFrame({
            "comparison": sub["comparison"], "vs.": sub["vs"],
            "AUC gap (pts)": [f"{pts(g)} [{pts(lo)}, {pts(hi)}]" for g, lo, hi in zip(sub.auc_gap, sub.auc_low, sub.auc_high)],
            "capture gap (pts)": [f"{pts(g)} [{pts(lo)}, {pts(hi)}]" for g, lo, hi in
                                  zip(sub.capture_gap, sub.capture_low, sub.capture_high)],
            "campaigns better": [f"{b}/{n}" for b, n in zip(sub.campaigns_better, sub.campaigns)],
            "capture": sub["capture_20"].map(pct)}))

    folds = [f["test_campaign"] for f in r.folds]
    lines = ["# Model 1 wrap-up analysis: results", "",
             f"Open test campaigns ({len(folds)}): {folds}; the locked campaigns were not read. Gaps in points with 95% "
             "intervals from resampling whole households. AUC = within-campaign ranking; capture = redeemers caught in each "
             "campaign's top 20%. `forest` = random forest with the baseline's settings; `logistic_log` = logistic "
             "regression on log-scaled columns (C=0.05). 21 built = the baseline's 11 + 10 already-built 2A columns; "
             "39 = those + the 18 new 2A columns; lean = 21 built + 3 new coupon columns.", ""]
    for kind, title in (("A total gain of the 18", "A. What the 18 new columns add"),
                        ("A one new group added", "A. One new group added to the 21 built columns"),
                        ("A one new group dropped", "A. One new group dropped from all 39"),
                        ("A lean set", "A. The lean set")):
        lines += [f"## {title}", "", gaps(kind), ""]
    if len(drop_one):
        d = drop_one.sort_values(["learner", "auc_lost"], ascending=[True, False])
        lines += ["## A. Each new column dropped from all 39 (retrained)", "",
                  "AUC points lost without the column; positive = the model needed it.", "",
                  md_table(d.assign(auc_lost=(100 * d.auc_lost).round(2), low=(100 * d.low).round(2),
                                    high=(100 * d.high).round(2), capture_lost=(100 * d.capture_lost).round(1))), ""]
    p = perm.assign(auc_drop=(100 * perm.auc_drop).round(2))
    lines += ["## A. Permutation importance (top 12 per learner, all 39 columns)", "",
              "AUC points lost when the column is shuffled inside each test campaign. Reliance, not added value.", "",
              md_table(pd.concat([p[p.learner == lr].head(12) for lr in ("logistic_log", "forest")])), "",
              "## A. Logistic regression: adding the new columns one at a time (most important first)", "",
              md_table(curve.assign(auc=curve.auc.round(4), capture_20=curve.capture_20.map(pct))
                       [["new_columns_added", "last_added", "auc", "capture_20"]]), ""]
    for kind, title in (("B model type vs. baseline", "B. Kinds of model on the 39 columns, vs. the baseline forest"),
                        ("B model type vs. logistic_log", "B. The same, vs. logistic_log on 39"),
                        ("B separate by type", "B. Separate models for Type A and Type B/C"),
                        ("B learner only", "B. Log-scaled logistic on fewer columns"),
                        ("C by campaign type", "C. Gains by campaign type")):
        lines += [f"## {title}", "", gaps(kind), ""]
    dd = depth.assign(**{c: depth[c].map(pct) for c in depth if c.startswith("capture")}, auc=depth.auc.round(3))
    lines += ["## C. By campaign type and mailing depth (share of redeemers caught when mailing the top 10/20/30/50%)", "",
              md_table(dd), "", "## C. Predicted vs. actual redemptions per campaign", "",
              md_table(pd.concat([totals, totals.drop(columns=["campaign", "campaign_type"]).sum().to_frame().T
                                  .assign(campaign="all", campaign_type="")]).round(1)), ""]
    if compare_rows:
        lines += [f"## D. Same models on `{compare}`", "", md_table(pd.DataFrame(compare_rows).round(4)), ""]
    lines.append(f"Runtime {runtime / 60:.0f} min.")
    return "\n".join(lines)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--processed", type=Path, default=ROOT / "data/processed")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--quick", action="store_true", help="skip the one-column-at-a-time retrains")
    p.add_argument("--compare-processed", type=Path, default=None,
                   help="a second processed folder (e.g. 2A rebuilt with --exclude-non-merchandise) to score the same models on")
    args = p.parse_args()
    run(args.processed, args.out or args.processed / "model1_importance", args.quick, args.compare_processed)


if __name__ == "__main__":
    main()
