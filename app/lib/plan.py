"""Mailing-plan helpers for the Redemption Likelihood page: mail the top part of one campaign's list and see
what that list would have reached.

Pure functions with no Streamlit, so they are unit-tested (tests/app/test_plan.py). Every measure reuses the
agents' own code instead of re-deriving it, so the page and the agents always agree:
  - list size and redeemers reached: modeling.model1_evaluation, Model 1's measure in the Evaluation Agent
    (the top `share` of EACH campaign's list, k = max(1, round(n * share)))
  - the forecast range: evaluation.metrics.forecast_coverage, the Evaluation Agent's honest range (expected count
    +/- 1.96 sd, widened by how far campaigns that finished before this one was planned missed their forecast)

Predictions are the Modeling Agent's forward-chained back-test: each test campaign was scored by a model trained
only on campaigns that ended before it, and its outcomes (`label`) are known, so the page can show what a list
would actually have reached.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.metrics import forecast_coverage  # noqa: E402
from modeling.model1_evaluation import random_expected_capture, within_campaign_capture  # noqa: E402

SHARES = tuple(round(0.05 * k, 2) for k in range(1, 21))   # mailing depths 5%, 10%, ..., 100%
NO_RECORD = "No record"


def list_size(n: int, share: float) -> int:
    """How many households 'the top `share` of a list of n' means: the Evaluation Agent's own rule."""
    return max(1, int(round(n * share)))


def ranked(view: pd.DataFrame) -> pd.DataFrame:
    """One campaign's households, highest score first, numbered 1..n. Ties keep a fixed order (household id) so
    the list never reshuffles between page loads."""
    out = view.sort_values(["score", "household_key"], ascending=[False, True]).reset_index(drop=True)
    return out.assign(rank=np.arange(1, len(out) + 1))


def campaign_overview(preds: pd.DataFrame) -> pd.DataFrame:
    """One row per test campaign, in time order: type, households mailed, redeemers, start day."""
    return (preds.groupby("campaign")
            .agg(campaign_type=("campaign_type", "first"), mailed=("household_key", "size"),
                 redeemers=("label", "sum"), start_day=("start_day", "first"))
            .reset_index().sort_values("start_day").reset_index(drop=True))


def forecast_inflation(preds: pd.DataFrame) -> dict:
    """Each campaign's range widening (phi >= 1), exactly as the Evaluation Agent's forecast check computes it:
    from how far the campaigns that had finished before it was planned missed their forecasts."""
    p = preds["score"]
    pc = (preds.assign(var=p * (1 - p)).groupby("campaign")
          .agg(expected=("score", "sum"), chance_var=("var", "sum"), actual=("label", "sum"),
               start_day=("start_day", "first"), end_day=("end_day", "first"))
          .reset_index().sort_values("start_day").reset_index(drop=True))
    return dict(zip(pc["campaign"], forecast_coverage(pc)["phi"]))


def plan(view: pd.DataFrame, share: float, phi: float = 1.0) -> dict:
    """Mail the top `share` of ONE campaign's list (`view`): who is on it, the forecast with its range, and
    (back-test) what it actually reached, next to the "redeemed before" rule and random mailing."""
    if view["campaign"].nunique() != 1:
        raise ValueError("plan() works on one campaign's list at a time")
    lst = ranked(view)
    k = list_size(len(lst), share)
    top = lst.head(k)
    p = top["score"].to_numpy(dtype=float)
    forecast, half = float(p.sum()), 1.96 * float(np.sqrt(phi * (p * (1 - p)).sum()))
    total = int(lst["label"].sum())
    c, y = lst["campaign"], lst["label"]
    return {
        "list": top, "k": k, "n": len(lst),
        "forecast": forecast, "low": max(0.0, forecast - half), "high": forecast + half,
        "reached": int(top["label"].sum()), "redeemers": total,
        # the list shown on the page, so the share always matches "reached of redeemers" (equals the agents'
        # measure unless households tie on score right at the cut-off, where the agents average random tie-breaks)
        "capture": top["label"].sum() / total if total else np.nan,
        "capture_rule": within_campaign_capture(c, y, lst["baseline_score"], share) if total else np.nan,
        "capture_random": random_expected_capture(c, y, share) if total else np.nan,
    }


def capture_curve(view: pd.DataFrame, shares=SHARES) -> pd.DataFrame:
    """Share of redeemers reached at each mailing depth, for the model, the rule and random mailing.
    Works on one campaign or on many (each campaign's own list, then pooled, like the Evaluation Agent)."""
    c, y = view["campaign"], view["label"]
    if y.sum() == 0:
        return pd.DataFrame(columns=["share", "Model in use", '"Redeemed before" rule', "Random mailing"])
    return pd.DataFrame({
        "share": list(shares),
        "Model in use": [within_campaign_capture(c, y, view["score"], s) for s in shares],
        '"Redeemed before" rule': [within_campaign_capture(c, y, view["baseline_score"], s) for s in shares],
        "Random mailing": [random_expected_capture(c, y, s) for s in shares],
    })


def all_campaigns(preds: pd.DataFrame, share: float) -> tuple[pd.DataFrame, dict]:
    """The same plan for every test campaign at one depth, plus the pooled totals across them."""
    phis = forecast_inflation(preds)
    rows = []
    for _, r in campaign_overview(preds).iterrows():
        pl = plan(preds[preds["campaign"] == r["campaign"]], share, phis[r["campaign"]])
        rows.append({"campaign": r["campaign"], "type": r["campaign_type"], "mailed": r["mailed"],
                     "list": pl["k"], "forecast": pl["forecast"], "low": pl["low"], "high": pl["high"],
                     "reached": pl["reached"], "redeemers": pl["redeemers"],
                     "in_range": pl["low"] <= pl["reached"] <= pl["high"]})
    table = pd.DataFrame(rows)
    c, y = preds["campaign"], preds["label"]
    pooled = {"model": within_campaign_capture(c, y, preds["score"], share),
              "rule": within_campaign_capture(c, y, preds["baseline_score"], share),
              "random": random_expected_capture(c, y, share),
              "forecast": float(table["forecast"].sum()), "reached": int(table["reached"].sum())}
    return table, pooled


def list_mix(view: pd.DataFrame, top: pd.DataFrame, column: str) -> pd.DataFrame:
    """Each group's share of the mailing list next to its share of everyone mailed. Households with no
    demographic record are their own group, never dropped, so the shares add up to 100%."""
    def shares(df):
        return df[column].fillna(NO_RECORD).astype(str).value_counts(normalize=True)
    out = pd.DataFrame({"On this list": shares(top), "Everyone mailed": shares(view)}).fillna(0.0)
    return out.rename_axis("group").reset_index()
