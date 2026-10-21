"""Tests for src/evaluation/metrics.py."""
import numpy as np
import pandas as pd
import pytest

from evaluation.metrics import balance, cluster_bootstrap, e_value, fit_propensity, forecast_coverage, mae, stratified_qini
from modeling.model3_uplift import qini_points


def _uplift_data(n=20000, campaigns=4, seed=0):
    """Uplift grows with x: P(buy) = 0.1 + 0.3 * x if mailed, 0.1 if not."""
    rng = np.random.default_rng(seed)
    x, t = rng.random(n), (rng.random(n) < 0.5).astype(int)
    y = (rng.random(n) < 0.1 + 0.3 * x * t).astype(int)
    return pd.DataFrame({"campaign": rng.integers(0, campaigns, n), "household_key": np.arange(n), "label": y,
                         "treatment": t, "score": x, "ipw_weight": 2.0, "in_overlap": True})


def test_qini_rewards_the_right_order_and_punishes_the_reverse():
    d = _uplift_data()
    good, bad = stratified_qini(d), stratified_qini(d.assign(score=-d["score"]))
    rand = stratified_qini(d.assign(score=np.random.default_rng(1).random(len(d))))
    assert good > 0.02 and bad < -0.02 and abs(rand) < 0.005


def test_equal_weights_give_the_unweighted_qini():
    d = _uplift_data()
    assert stratified_qini(d, weighted=True) == pytest.approx(stratified_qini(d))


def test_weighted_qini_only_uses_rows_in_overlap():
    d = _uplift_data()
    d.loc[d.index % 2 == 0, "in_overlap"] = False
    assert stratified_qini(d, weighted=True) == pytest.approx(stratified_qini(d[d["in_overlap"]]))


def test_single_campaign_matches_shreyas_pooled_qini_per_treated_household():
    d = _uplift_data(n=4000, campaigns=1).assign(campaign=0)
    ours = stratified_qini(d, grid_points=4001)
    *_, theirs = qini_points(d.rename(columns={"score": "predicted_uplift", "treatment": "mailed_flag",
                                               "label": "bought_campaign_product_excl_display_flyer"}))
    assert ours == pytest.approx(theirs / d["treatment"].sum(), rel=0.02)


def test_qini_is_blank_without_treated_rows():
    assert np.isnan(stratified_qini(_uplift_data().assign(treatment=0)))


def test_propensity_never_sees_the_outcome():
    rng = np.random.default_rng(2)
    n = 2000
    t = pd.DataFrame({"spend": rng.random(n), "campaign": rng.integers(0, 3, n), "household_key": np.arange(n) % 500,
                      "mailed": (rng.random(n) < 0.3).astype(int), "label": rng.integers(0, 2, n)})
    e1 = fit_propensity(t, ["spend"], "mailed")
    e2 = fit_propensity(t.assign(label=1 - t["label"]), ["spend"], "mailed")
    assert np.array_equal(e1, e2) and ((e1 >= 0.01) & (e1 <= 0.99)).all()


def test_propensity_learns_who_was_mailed():
    rng = np.random.default_rng(3)
    spend = rng.random(4000)
    t = pd.DataFrame({"spend": spend, "campaign": 1, "household_key": np.arange(4000),
                      "mailed": (rng.random(4000) < spend).astype(int)})
    e = fit_propensity(t, ["spend"], "mailed")
    assert np.corrcoef(e, spend)[0, 1] > 0.9


def test_cluster_bootstrap_moves_a_households_rows_together():
    d = pd.DataFrame({"household_key": [1, 1, 1, 2], "v": [1.0, 1.0, 1.0, 0.0]})
    # each resample holds households 1 and/or 2; household 1 always brings all 3 rows
    draws = cluster_bootstrap(d, lambda s: len(s[s["household_key"] == 1]) % 3, reps=50, seed=0)
    assert (draws == 0).all()


def test_e_value_matches_the_known_worked_example():
    # VanderWeele & Ding 2017's own worked example: RR = 3.9 -> E-value ~= 7.27
    assert e_value(0.039, 0.01) == pytest.approx(7.264, abs=1e-2)


def test_e_value_is_direction_free():
    # Same gap, control higher than treated this time: same E-value either way round.
    assert e_value(0.01, 0.039) == pytest.approx(e_value(0.039, 0.01))


def test_e_value_grows_with_the_effect_size():
    small = e_value(0.15, 0.10)
    large = e_value(0.30, 0.10)
    assert large > small > 1.0   # E-value is always >= 1; a bigger gap needs a stronger confounder to explain


def test_e_value_is_none_when_there_is_no_gap():
    assert e_value(0.10, 0.10) is None


def test_mae():
    assert mae([10, 20], [12, 17]) == pytest.approx(2.5)


def test_forecast_coverage_widens_ranges_with_past_misses():
    # campaign 1 misses by 4 chance-sds (phi = 16 for later ones); campaign 2's miss of 3 sds is then inside
    pc = pd.DataFrame({"start_day": [100, 200], "end_day": [130, 230], "expected": [10.0, 10.0],
                       "chance_var": [1.0, 1.0], "actual": [14.0, 13.0]})
    cov = forecast_coverage(pc)
    assert cov["inside"] == [False, True] and cov["phi"] == [1.0, 16.0] and cov["coverage"] == 0.5


def test_balance_is_zero_for_identical_groups_and_large_for_different_ones():
    d = pd.DataFrame({"campaign": 1, "treatment": [1] * 30 + [0] * 30, "w": 1.0,
                      "same": list(range(30)) * 2, "diff": list(range(30)) + list(range(100, 130))})
    gaps = balance(d, ["same", "diff"], "w")
    assert gaps["same"] == 0 and gaps["diff"] > 5


def test_balance_uses_the_weights():
    # mailed spend 10; unmailed 0 and 10 -- weighting the unmailed 10s up closes the gap
    d = pd.DataFrame({"campaign": 1, "treatment": [1] * 20 + [0] * 40, "x": [10.0] * 20 + [0.0] * 20 + [10.0] * 20,
                      "even": 1.0})
    d["tilted"] = np.where((d["treatment"] == 0) & (d["x"] == 10), 1000.0, 1.0)
    assert balance(d, ["x"], "tilted")["x"] < 0.1 < balance(d, ["x"], "even")["x"]


def test_balance_skips_campaigns_with_too_few_mailed_rows():
    d = pd.DataFrame({"campaign": 1, "treatment": [1] * 5 + [0] * 5, "w": 1.0, "x": range(10)})
    assert np.isnan(balance(d, ["x"], "w")["x"])
