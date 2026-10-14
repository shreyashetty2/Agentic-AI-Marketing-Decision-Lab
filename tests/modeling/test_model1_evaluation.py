"""Hand-checkable tests for the Model 1 evaluation helpers (no real data needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from modeling.model1_evaluation import (  # noqa: E402
    forward_chain_folds, random_expected_capture, within_campaign_capture,
)


def _calendar():
    # campaign, type, start, end   (cutoff = start - 7)
    rows = [(1, "A", 100, 140), (2, "B", 120, 150), (3, "B", 160, 190), (4, "A", 200, 240)]
    return pd.DataFrame(rows, columns=["campaign", "campaign_type", "start_day", "end_day"])


def test_folds_train_only_on_campaigns_ended_before_cutoff():
    folds = forward_chain_folds(_calendar(), first_test_day=160, gap=7)
    assert [f["test_campaign"] for f in folds] == [3, 4]
    # campaign 3: cutoff 153 -> 1 (ended 140) and 2 (ended 150) are allowed
    assert folds[0]["train_campaigns"] == [1, 2]
    # campaign 4: cutoff 193 -> 1, 2, 3 (ended 190)
    assert folds[1]["train_campaigns"] == [1, 2, 3]
    assert folds[1]["n_train_type_a"] == 1


def test_overlapping_campaign_is_excluded_from_training():
    cal = _calendar()
    cal.loc[cal["campaign"] == 2, "end_day"] = 170      # campaign 2 now runs past campaign 3's cutoff (153)
    folds = forward_chain_folds(cal, first_test_day=160, gap=7)
    assert folds[0]["train_campaigns"] == [1]


def test_within_campaign_capture_scores_each_list_separately():
    # Two campaigns of 10 rows; top 20% = 2 rows each.
    # Campaign 1: redeemers at scores 0.9 and 0.1 -> 1 of 2 caught. Campaign 2: redeemer at 0.5, top -> caught.
    campaign = [1] * 10 + [2] * 10
    y = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0] + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    score = [0.9, 0.1, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2] + [0.5, 0.4, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    assert within_campaign_capture(campaign, y, score, n_seeds=1) == pytest.approx(2 / 3)


def test_pooling_would_differ_from_within_campaign():
    # Campaign 2's scores are all higher (think: "Type A"), so a pooled top 20% would take only campaign 2 rows.
    campaign = [1] * 10 + [2] * 10
    y = [1] + [0] * 9 + [0] * 9 + [1]
    score = [0.3] + [0.1] * 9 + [0.9] * 9 + [0.95]
    assert within_campaign_capture(campaign, y, score, n_seeds=1) == pytest.approx(1.0)


def test_random_expectation_is_list_share():
    campaign = [1] * 10 + [2] * 5
    y = [1] * 5 + [0] * 5 + [1, 0, 0, 0, 0]
    # campaign 1: 2/10 of 5 redeemers = 1.0; campaign 2: 1/5 of 1 = 0.2 -> 1.2 / 6
    assert random_expected_capture(campaign, y) == pytest.approx(1.2 / 6)


def test_rule_ties_are_broken_randomly_not_by_row_order():
    campaign = [1] * 10
    y = [1] + [0] * 9          # redeemer sits first; a tie-blind sort would always catch it
    score = np.zeros(10)
    assert within_campaign_capture(campaign, y, score, n_seeds=500) == pytest.approx(0.2, abs=0.05)
