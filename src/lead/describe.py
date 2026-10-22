"""Plain words for the lead agent's run summary: what each model is, and what each score means.

The front door explains a run to someone who did not build the system, using only facts the engine measured.
These helpers turn the agents' own records (a candidate's family and settings, a metric's name and value) into
the words it uses, so every message names models the same way.
"""
from pathlib import Path

import yaml

FEATURE_SPEC = Path(__file__).resolve().parents[1] / "features/feature_2a_spec.yaml"
# Inputs that do not come from the Feature Agent's profile columns: the Data Agent's spine (listed in the
# feature spec without a description) and columns the owners' scripts add.
DERIVED = {"is_type_a": "Whether the campaign is Type A.", "is_type_b": "Whether the campaign is Type B.",
           "usual_spend": "What the household usually spends on the campaign's products.",
           "duration_days": "How many days the campaign runs.",
           "has_demographics": "Whether the household's age, income and family details are known (for about a third)."}

SCORES = {   # metric -> (what it measures, how to show a value)
    "within_campaign_capture": ("redeemers caught in the top 20% of each campaign's list", lambda v: f"{v:.1%}"),
    "mae": ("average dollar error per household", lambda v: f"${v:.2f}"),
    "ipw_stratified_qini": ("extra buyers per 100 mailed households, compared with mailing at random",
                            lambda v: f"{100 * v:+.1f}"),
}
GAP = {"within_campaign_capture": lambda v: f"{100 * v:+.1f} points", "mae": lambda v: f"{'+' if v >= 0 else '-'}${abs(v):.2f}",
       "ipw_stratified_qini": lambda v: f"{100 * v:+.1f} per 100"}


def model_words(candidate: dict, task: str) -> str:
    """e.g. 'random forest (300 trees, 6 levels deep) on 11 columns'."""
    family, p = candidate["family"], candidate.get("params", {})
    kind = {"forest": f"random forest ({p.get('n_estimators', 100)} trees, {p.get('max_depth', 'any number of')} levels deep)",
            "lightgbm": f"LightGBM ({p.get('n_estimators', 100)} trees built one after another, each fixing the last)",
            "logistic": "logistic regression (one weighted formula, turned into a chance)",
            "logistic_log": ("logistic regression on log-scaled columns (one weighted formula, after shrinking very "
                             "large counts and dollars so a few heavy coupon users do not dominate it)"),
            "linear": "linear regression (one weighted formula, giving dollars)",
            "median_linear": "calibrated usual spend (a straight line on usual spend, fit through the middle of the answers)",
            }.get(family, family)
    if task == "uplift":
        kind = f"pair of {kind.replace('random forest', 'random forests')}, one for mailed and one for unmailed households"
    n = len(candidate.get("features", []))
    return f"{kind} on {n} column{'s' if n != 1 else ''}" if n else kind


def score_words(metric: str, value: float) -> str:
    what, show = SCORES.get(metric, (metric, lambda v: f"{v:.4f}"))
    return f"{show(value)} ({what})"


def gap_words(metric: str, comparison: dict) -> str:
    """A comparison {gap, low, high, status} in plain words; positive gap = the first is better."""
    show = GAP.get(metric, lambda v: f"{v:+.4f}")
    verdict = {"pass": "a real difference", "fail": "really worse", "inconclusive": "could be luck"}[comparison["status"]]
    return f"{show(comparison['gap'])} (likely range {show(comparison['low'])} to {show(comparison['high'])}: {verdict})"


def column_meanings(spec_path: Path = FEATURE_SPEC) -> dict[str, str]:
    """Each model input's plain meaning, from the Feature Agent's own spec (plus the few columns the owners'
    scripts add themselves)."""
    spec = yaml.safe_load(spec_path.read_text())
    return {**DERIVED, **{c: v["desc"] for section in ("passthrough", "profile", "context") for c, v in (spec.get(section) or {}).items()
                          if isinstance(v, dict) and "desc" in v}}
