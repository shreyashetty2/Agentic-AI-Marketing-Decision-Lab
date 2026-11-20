"""Tests for the Decision Lab's predictions loader and the Redemption Likelihood mailing plan (app/lib)."""
import numpy as np
import pandas as pd
import pytest

from lib.plan import all_campaigns, capture_curve, forecast_inflation, list_mix, list_size, plan
from modeling.model1_evaluation import within_campaign_capture


def fake_predictions(seed: int = 0, campaigns=(8, 9, 10), n: int = 200) -> pd.DataFrame:
    """Back-test predictions shaped like the Modeling Agent's: a useful score, a weaker rule, known outcomes."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, c in enumerate(campaigns):
        habit = rng.gamma(1.0, 1.0, n)
        p = 1 / (1 + np.exp(-(habit - 2.5)))
        rows.append(pd.DataFrame({
            "household_key": np.arange(n) + 1000 * i, "campaign": c, "campaign_type": "B",
            "start_day": 100 + 60 * i, "end_day": 140 + 60 * i,
            "label": rng.binomial(1, p), "score": p + rng.normal(0, 0.01, n).clip(-p + 1e-4, 1 - p - 1e-4),
            "baseline_score": np.round(habit / 3, 0) / 3}))
    return pd.concat(rows, ignore_index=True)


def test_list_size_is_the_evaluation_agents_rule():
    assert [list_size(n, 0.2) for n in (1, 3, 10, 1076)] == [1, 1, 2, 215]


def test_plan_matches_the_agents_measure_and_is_consistent():
    d = fake_predictions()
    view = d[d["campaign"] == 9]
    p = plan(view, 0.2, phi=1.0)
    assert p["k"] == list_size(len(view), 0.2) == len(p["list"])
    assert list(p["list"]["rank"]) == list(range(1, p["k"] + 1))
    assert p["list"]["score"].is_monotonic_decreasing
    assert p["reached"] == p["list"]["label"].sum() and p["redeemers"] == view["label"].sum()
    # no ties at the cut-off here, so the list's share equals the Evaluation Agent's own measure
    assert p["capture"] == pytest.approx(within_campaign_capture(view["campaign"], view["label"], view["score"], 0.2))
    assert p["low"] <= p["forecast"] <= p["high"]
    assert p["forecast"] == pytest.approx(p["list"]["score"].sum())


def test_wider_range_when_earlier_campaigns_missed():
    d = fake_predictions()
    view = d[d["campaign"] == 10]
    narrow, wide = plan(view, 0.3, phi=1.0), plan(view, 0.3, phi=4.0)
    assert wide["high"] - wide["forecast"] == pytest.approx(2 * (narrow["high"] - narrow["forecast"]))   # sqrt(4) = 2
    assert wide["low"] <= narrow["low"]


def test_plan_needs_one_campaign():
    with pytest.raises(ValueError):
        plan(fake_predictions(), 0.2)


def test_mailing_everyone_reaches_every_redeemer():
    d = fake_predictions()
    p = plan(d[d["campaign"] == 8], 1.0)
    assert p["k"] == p["n"] and p["capture"] == pytest.approx(1.0) and p["reached"] == p["redeemers"]


def test_curve_rises_with_depth_and_beats_random():
    d = fake_predictions()
    curve = capture_curve(d)
    assert curve["Model in use"].is_monotonic_increasing
    assert (curve["Model in use"] >= curve["Random mailing"]).all()
    assert curve["Model in use"].iloc[-1] == pytest.approx(1.0)


def test_forecast_inflation_is_never_below_one():
    phis = forecast_inflation(fake_predictions(campaigns=(8, 9, 10, 11)))
    assert set(phis) == {8, 9, 10, 11} and all(v >= 1.0 for v in phis.values())
    assert phis[8] == 1.0      # nothing had finished before the first campaign


def test_all_campaigns_pools_like_the_evaluation_agent():
    d = fake_predictions()
    table, pooled = all_campaigns(d, 0.2)
    assert list(table["campaign"]) == [8, 9, 10]
    assert pooled["model"] == pytest.approx(within_campaign_capture(d["campaign"], d["label"], d["score"], 0.2))
    assert pooled["reached"] == table["reached"].sum()


def test_list_mix_keeps_households_with_no_record():
    view = pd.DataFrame({"age": ["25-34", "35-44", None, None, "25-34"]})
    mix = list_mix(view, view.head(2), "age").set_index("group")
    assert mix["Everyone mailed"].sum() == pytest.approx(1.0) and mix["On this list"].sum() == pytest.approx(1.0)
    assert mix.loc["No record", "Everyone mailed"] == pytest.approx(0.4)
    assert mix.loc["No record", "On this list"] == 0.0
