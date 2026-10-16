"""Scores used by the Modeling Agent (to choose) and the Evaluation Agent (to judge).

Existing, already-validated metric code is imported, not copied:
  within_campaign_capture, within_campaign_auc   John, src/modeling/model1_evaluation.py
  qini_points                                     Shreya, src/modeling/model3_uplift.py (pooled Qini)
New here:
  mae                    Model 2's average dollar error
  stratified_qini        Model 3's Qini computed WITHIN each campaign, optionally propensity-weighted
  fit_propensity         chance a household was mailed, from its pre-campaign columns (never sees the outcome)
  balance                whether weighting really made mailed and unmailed households comparable
  cluster_bootstrap      resamples whole households, since one household appears in many campaigns
  forecast_coverage      share of campaigns whose actual count lands inside an honest forecast range

Every metric reads a predictions frame with columns campaign, household_key, label, score and, for
uplift, treatment / ipw_weight / in_overlap.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from modeling.model1_evaluation import within_campaign_auc, within_campaign_capture

__all__ = ["within_campaign_auc", "within_campaign_capture", "mae", "stratified_qini", "fit_propensity",
           "balance", "cluster_bootstrap", "forecast_coverage", "METRICS", "metric"]


def mae(label, score) -> float:
    return float(np.mean(np.abs(np.asarray(label, dtype=float) - np.asarray(score, dtype=float))))


def _campaign_qini(y, t, score, w, grid) -> tuple[np.ndarray, float]:
    """One campaign: incremental buyers among the top fraction f (for each f in grid), and its treated weight."""
    order = np.argsort(-score, kind="stable")
    y, t, w = y[order], t[order], w[order]
    wt, wc = w * t, w * (1 - t)
    n_t, n_c = wt.sum(), wc.sum()
    if n_t == 0 or n_c == 0:
        return np.zeros(len(grid)), 0.0
    cum_t, cum_c = np.concatenate([[0], np.cumsum(wt * y)]), np.concatenate([[0], np.cumsum(wc * y)])
    k = np.ceil(grid * len(y)).astype(int)
    return cum_t[k] - cum_c[k] * (n_t / n_c), n_t


def stratified_qini(df: pd.DataFrame, weighted: bool = False, grid_points: int = 101) -> float:
    """Model 3's ranking score, within each campaign (the counterpart of Model 1's within-campaign capture).

    Each campaign is ranked on its own; its Qini curve (incremental buyers vs. share targeted) is put on
    a common 0-1 grid, the curves are added up, and the area above the random-targeting line is divided
    by the total treated weight: extra buyers per mailed household gained by targeting in this order
    instead of at random. 0 = no better than random; higher is better.
    weighted=True uses inverse-propensity weights on rows inside the overlap region, which corrects
    for the retailer having mailed heavier spenders (raw Qini otherwise mostly measures who got mailed)."""
    grid = np.linspace(0, 1, grid_points)
    data = df[df["in_overlap"]] if weighted else df
    total_curve, total_t = np.zeros(grid_points), 0.0
    for _, g in data.groupby("campaign"):
        w = g["ipw_weight"].to_numpy(float) if weighted else np.ones(len(g))
        curve, n_t = _campaign_qini(g["label"].to_numpy(float), g["treatment"].to_numpy(float),
                                    g["score"].to_numpy(float), w, grid)
        total_curve += curve
        total_t += n_t
    if total_t == 0:
        return np.nan
    gain = total_curve - grid * total_curve[-1]
    return float(np.trapezoid(gain, grid) / total_t)


def fit_propensity(table: pd.DataFrame, features: list[str], treatment: str, folds: int = 5,
                   clip: tuple[float, float] = (0.01, 0.99), seed: int = 0) -> np.ndarray:
    """Chance each row was mailed, from its pre-campaign columns plus which campaign it is.
    Boosted trees, because mailing depends on spend in a strongly non-linear way: a straight-line
    (logistic) model left mailed and unmailed households far apart on spend even after weighting
    (balance 0.79; boosted trees 0.14-0.23 -- see balance()). Cross-fitted by household (a household's
    rows are never scored by a model that trained on them). Takes no outcome column at all."""
    X = table[features].reset_index(drop=True).assign(campaign=table["campaign"].to_numpy())
    t = table[treatment].to_numpy()
    e = np.zeros(len(table))
    for train_idx, test_idx in GroupKFold(n_splits=folds).split(X, t, groups=table["household_key"]):
        model = LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=50,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed,
                               n_jobs=4, verbose=-1)
        e[test_idx] = model.fit(X.iloc[train_idx], t[train_idx]).predict_proba(X.iloc[test_idx])[:, 1]
    return np.clip(e, *clip)


def balance(df: pd.DataFrame, columns: list[str], weight: str, treatment: str = "treatment",
            min_treated: int = 20) -> dict[str, float]:
    """After weighting, do mailed and unmailed households look alike? For each column: the gap between
    the two groups' weighted averages in units of spread (standardized mean difference), measured inside
    each campaign with at least min_treated mailed rows, then averaged. 0 = identical; under 0.1 is the
    usual standard for "balanced"; a large value means the weighted comparison is still confounded."""
    gaps = []
    for _, g in df.groupby("campaign"):
        m = g[treatment] == 1
        if m.sum() < min_treated or (~m).sum() == 0:
            continue
        row = {}
        for c in columns:
            x = g[c].astype(float)
            spread = np.sqrt((x[m].var() + x[~m].var()) / 2)
            row[c] = (abs(np.average(x[m], weights=g.loc[m, weight]) - np.average(x[~m], weights=g.loc[~m, weight]))
                      / spread if spread > 0 else 0.0)
        gaps.append(row)
    return pd.DataFrame(gaps).mean().to_dict() if gaps else {c: np.nan for c in columns}


def cluster_bootstrap(df: pd.DataFrame, stat, reps: int, seed: int, cluster: str = "household_key") -> np.ndarray:
    """stat(sample) over `reps` resamples of whole households (each keeps all of its rows)."""
    rows_of = {k: idx.to_numpy() for k, idx in df.reset_index(drop=True).groupby(cluster).groups.items()}
    keys = np.array(list(rows_of))
    base = df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        pick = rng.choice(keys, len(keys))
        out.append(stat(base.iloc[np.concatenate([rows_of[k] for k in pick])]))
    return np.asarray(out, dtype=float)


def forecast_coverage(per_campaign: pd.DataFrame, z: float = 1.96) -> dict:
    """per_campaign: one row per test campaign in time order, with expected, chance_var (sum p(1-p)),
    actual, start_day, end_day. Each campaign's range is widened by how much the campaigns that had
    FINISHED before it was planned missed by (phi = mean squared miss in units of chance, at least 1)."""
    pc = per_campaign.sort_values("start_day").reset_index(drop=True)
    z2 = (pc["actual"] - pc["expected"]) ** 2 / pc["chance_var"]
    inside, phis = [], []
    for i, row in pc.iterrows():
        past = z2[pc["end_day"] < row["start_day"] - 7]
        phi = max(1.0, float(past.mean())) if len(past) else 1.0
        half = z * np.sqrt(phi * row["chance_var"])
        inside.append(row["expected"] - half <= row["actual"] <= row["expected"] + half)
        phis.append(phi)
    return {"coverage": float(np.mean(inside)), "n": len(pc), "inside": inside, "phi": phis}


# name -> (function(predictions) -> float, higher_is_better)
METRICS = {
    "within_campaign_auc": (lambda d: within_campaign_auc(d["campaign"], d["label"], d["score"]), True),
    "within_campaign_capture": (lambda d: within_campaign_capture(d["campaign"], d["label"], d["score"]), True),
    "mae": (lambda d: mae(d["label"], d["score"]), False),
    "ipw_stratified_qini": (lambda d: stratified_qini(d, weighted=True), True),
    "raw_stratified_qini": (lambda d: stratified_qini(d, weighted=False), True),
}


def metric(name: str):
    if name not in METRICS:
        raise ValueError(f"unknown metric {name!r}; known: {sorted(METRICS)}")
    return METRICS[name]
