import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from modeling.model2_value import (  # noqa: E402
    LABEL, USUAL, CalibratedUsualSpend, UsualSpend, bootstrap_vs_benchmark, errors,
    time_split, usual_spend, walk_forward_splits,
)


def test_usual_spend_is_weekly_spend_times_campaign_weeks():
    table = pd.DataFrame({"campaign_product_spend_per_week_26w": [2.0, 0.0, 3.5],
                          "duration_days": [35, 33, 14]})
    assert usual_spend(table).tolist() == pytest.approx([10.0, 0.0, 7.0])


def test_calibrated_usual_spend_recovers_an_exact_line():
    usual = np.arange(0, 40, dtype=float)
    train = pd.DataFrame({USUAL: usual, LABEL: 0.8 * usual + 5.0})
    model = CalibratedUsualSpend().fit(train)
    assert model.slope == pytest.approx(0.8, abs=1e-6)
    assert model.intercept == pytest.approx(5.0, abs=1e-6)


def test_calibrated_usual_spend_is_robust_to_one_huge_spender():
    # A median fit should barely move when one household spends far more than the line predicts.
    usual = np.arange(1, 21, dtype=float)
    spend = 0.8 * usual + 5.0
    spend[-1] = 500.0
    model = CalibratedUsualSpend().fit(pd.DataFrame({USUAL: usual, LABEL: spend}))
    assert model.slope == pytest.approx(0.8, abs=0.05)


def test_calibrated_usual_spend_never_predicts_negative_spend():
    train = pd.DataFrame({USUAL: [0.0, 10.0, 20.0], LABEL: [-20.0, 0.0, 20.0]})
    model = CalibratedUsualSpend().fit(train)
    assert (model.predict(pd.DataFrame({USUAL: [0.0, 1.0]})) >= 0).all()


def test_usual_spend_benchmark_returns_usual_spend_unchanged():
    table = pd.DataFrame({USUAL: [1.5, 0.0, 9.0]})
    assert UsualSpend().fit(table).predict(table).tolist() == [1.5, 0.0, 9.0]


def _campaign_table():
    # 4 campaigns, 30 rows each, starting on days 100 < 200 < 300 < 400
    rows = [{"campaign": c, "start_day": 100 * (i + 1), LABEL: float(i)}
            for i, c in enumerate([7, 3, 9, 5]) for _ in range(30)]
    return pd.DataFrame(rows)


def test_walk_forward_trains_only_on_earlier_campaigns_and_skips_short_history():
    splits = list(walk_forward_splits(_campaign_table(), min_train_rows=60))
    # campaign 7 has no history, campaign 3 only 30 rows -> both skipped
    assert [campaign for campaign, _, _ in splits] == [9, 5]
    for campaign, train, test in splits:
        assert set(test["campaign"]) == {campaign}
        assert campaign not in set(train["campaign"])
        assert train["start_day"].max() < test["start_day"].min()


def test_time_split_trains_on_earliest_campaigns_by_start_day():
    train, test = time_split(_campaign_table(), train_fraction=0.5)
    assert set(train["campaign"]) == {7, 3}
    assert set(test["campaign"]) == {9, 5}


def test_errors_match_hand_calculation():
    result = errors([10, 20, 30, 40], [12, 18, 35, 40])
    assert result["mae"] == pytest.approx(2.25)        # (2 + 2 + 5 + 0) / 4
    assert result["median_ae"] == pytest.approx(2.0)
    assert result["rank_rho"] == pytest.approx(1.0)


def test_bootstrap_of_identical_predictions_is_exactly_zero():
    actual = np.array([5.0, 10.0, 15.0, 20.0])
    pred = np.array([6.0, 9.0, 14.0, 25.0])
    result = bootstrap_vs_benchmark(actual, pred, pred, np.array([1, 1, 2, 2]), n=200)
    assert result["diff"] == 0.0
    assert result["ci_low"] == 0.0 and result["ci_high"] == 0.0


def test_bootstrap_detects_a_uniformly_better_model():
    actual = np.full(6, 10.0)
    result = bootstrap_vs_benchmark(actual, actual + 1, actual + 3, np.array([1, 1, 2, 2, 3, 3]), n=200)
    assert result["diff"] == pytest.approx(-2.0)
    assert result["share_better"] == 1.0
