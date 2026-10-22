"""Hand-checkable tests for the Model 1 feature search (no real data needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from features.spec import column_roles, load_spec  # noqa: E402
from modeling.model1_feature_search import ADOPTED, EXTENDED_PLUS, paired_gap, same_habit_auc  # noqa: E402


def test_same_habit_auc_is_weighted_by_redeemers_within_cells():
    # cell 1: the column puts the one redeemer on top (AUC 1); cell 2: at the bottom (AUC 0); one redeemer each
    d = pd.DataFrame({"campaign": [1, 1, 1, 1, 1], "coupon_fifth": [0, 0, 1, 1, 1],
                      "redeemed_flag": [1, 0, 1, 0, 0], "x": [2, 1, 0, 1, 2]})
    assert same_habit_auc(d, "x") == pytest.approx(0.5)


def test_same_habit_auc_ignores_differences_between_cells():
    # x only differs BETWEEN coupon fifths, so once coupon habit is known it says nothing: no usable cell
    d = pd.DataFrame({"campaign": [1] * 4, "coupon_fifth": [0, 0, 1, 1], "redeemed_flag": [0, 1, 1, 0], "x": [1, 1, 5, 5]})
    assert np.isnan(same_habit_auc(d, "x"))


def test_paired_gap_of_identical_scores_is_zero():
    rng = np.random.default_rng(0)
    test = pd.DataFrame({"household_key": np.arange(200) % 50, "campaign": np.repeat([1, 2], 100),
                         "redeemed_flag": rng.integers(0, 2, 200)})
    s = rng.random(200)
    g = paired_gap(test, s, s, reps=20)
    assert g["auc_gap"] == 0 and g["auc_low"] == 0 and g["auc_high"] == 0 and g["campaigns_better"] == 0


def test_adopted_columns_are_2a_profile_columns_and_match_the_modeling_spec():
    roles = column_roles(load_spec(ROOT / "src/features/feature_2a_spec.yaml"))
    for col in (c for cols in ADOPTED.values() for c in cols):
        assert roles.get(col) == "profile", col
    spec = yaml.safe_load((ROOT / "src/modeling/modeling_spec.yaml").read_text())
    assert spec["models"][1]["feature_sets"]["extended_plus"] == EXTENDED_PLUS
