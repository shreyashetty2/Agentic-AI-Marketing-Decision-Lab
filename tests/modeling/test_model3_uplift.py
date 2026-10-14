import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from modeling.model3_uplift import (  # noqa: E402
    LABEL, TREATMENT, WITHOUT_REDEMPTION, ate_test, redemption_segmentation,
    spend_tier_segmentation, time_split,
)


def test_ate_test_matches_hand_calculation():
    # treated: 6/10 = 0.6, control: 2/10 = 0.2 -- a textbook two-proportion z-test
    treated = pd.Series([1] * 6 + [0] * 4)
    control = pd.Series([1] * 2 + [0] * 8)
    result = ate_test(treated, control)
    assert result["n_treated"] == 10 and result["n_control"] == 10
    assert result["treated_rate"] == pytest.approx(0.6)
    assert result["control_rate"] == pytest.approx(0.2)
    assert result["diff"] == pytest.approx(0.4)
    # p_pool = 8/20 = 0.4, se = sqrt(0.4*0.6*(1/10+1/10)) = sqrt(0.048) = 0.21909
    # z = 0.4 / 0.21909 = 1.8257
    assert result["z"] == pytest.approx(1.8257, abs=1e-3)
    assert 0 < result["p_value"] < 1


def test_ate_test_identical_rates_gives_z_near_zero():
    treated = pd.Series([1, 0, 1, 0, 1, 0])
    control = pd.Series([1, 0, 1, 0, 1, 0])
    result = ate_test(treated, control)
    assert result["diff"] == pytest.approx(0.0)
    assert result["z"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)


def _spend_table():
    # 9 treated households (3 per tercile by construction) + 9 control, spend 1..9 for treated
    # so treated terciles are exactly {1,2,3}=Low, {4,5,6}=Medium, {7,8,9}=High.
    rows = []
    for i in range(1, 10):
        rows.append({"spend_per_week_26w": float(i), TREATMENT: 1, LABEL: 1 if i >= 7 else 0})
    # control households spread across the same spend range, mostly in the Low tier range
    for i in range(1, 10):
        rows.append({"spend_per_week_26w": float(i), TREATMENT: 0, LABEL: 1 if i >= 8 else 0})
    return pd.DataFrame(rows)


def test_spend_tier_segmentation_uses_treated_group_cutoffs():
    table = _spend_table()
    result = spend_tier_segmentation(table)
    assert list(result["spend_tier"]) == ["Low", "Medium", "High"]
    # Every tier has exactly 3 treated households by construction (cutoffs come from the
    # TREATED group's own terciles), regardless of how control households are distributed.
    assert list(result["n_treated"]) == [3, 3, 3]
    # Low tier (spend 1-3): treated outcomes [0,0,0] -> rate 0.0; High tier (7-9): [1,1,1] -> rate 1.0
    low = result[result["spend_tier"] == "Low"].iloc[0]
    high = result[result["spend_tier"] == "High"].iloc[0]
    assert low["treated_rate"] == pytest.approx(0.0)
    assert high["treated_rate"] == pytest.approx(1.0)


def test_redemption_segmentation():
    table = pd.DataFrame([
        {TREATMENT: 1, LABEL: 1, WITHOUT_REDEMPTION: 1},   # bought, no redemption
        {TREATMENT: 1, LABEL: 1, WITHOUT_REDEMPTION: 1},   # bought, no redemption
        {TREATMENT: 1, LABEL: 1, WITHOUT_REDEMPTION: 0},   # bought, via redemption
        {TREATMENT: 1, LABEL: 0, WITHOUT_REDEMPTION: 0},   # didn't buy -- excluded from the denominator
        {TREATMENT: 0, LABEL: 1, WITHOUT_REDEMPTION: 1},   # control purchase -- excluded (not treated)
    ])
    result = redemption_segmentation(table)
    assert result["n_treated_bought"] == 3
    assert result["n_without_redemption"] == 2
    assert result["pct_without_redemption"] == pytest.approx(200 / 3)


def test_time_split_is_chronological_and_covers_every_row():
    table = pd.DataFrame({
        "campaign": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "start_day": [10, 10, 20, 20, 30, 30, 40, 40, 50, 50],
    })
    train, test = time_split(table, train_fraction=0.6)
    assert len(train) + len(test) == len(table)
    assert set(train["campaign"]) == {1, 2, 3}      # earliest 3 of 5 campaigns (60%)
    assert set(test["campaign"]) == {4, 5}
    assert train["start_day"].max() < test["start_day"].min()
