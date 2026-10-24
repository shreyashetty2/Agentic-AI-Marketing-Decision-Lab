"""Tests for the Model 1 wrap-up analysis (no real data needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from modeling.model1_feature_search import ADOPTED, EXTENDED  # noqa: E402
from modeling.model1_importance import LEAN, LEAN_ADDED, Runner, permutation_importance  # noqa: E402


def test_lean_set_matches_the_modeling_spec_and_uses_only_new_coupon_columns():
    spec = yaml.safe_load((ROOT / "src/modeling/modeling_spec.yaml").read_text())
    assert spec["models"][1]["feature_sets"]["lean"] == LEAN
    assert LEAN[:len(EXTENDED)] == EXTENDED
    assert set(LEAN_ADDED) <= set(ADOPTED["coupon habit, refined"])


def _toy():
    rng = np.random.default_rng(0)
    rows = []
    for c, start in ((1, 100), (2, 200), (3, 300)):
        for h in range(60):
            x = rng.random()
            rows.append({"household_key": h, "campaign": c, "campaign_type": "A", "start_day": start, "end_day": start + 30,
                         "signal": x, "noise": rng.random(), "redeemed_flag": int(x > 0.7)})
    return pd.DataFrame(rows)


def test_permutation_finds_the_column_that_carries_the_signal():
    t = _toy()
    folds = [{"test_campaign": 3, "train_campaigns": [1, 2]}]
    r = Runner(t, folds)
    p = permutation_importance(r, ["signal", "noise"], "logistic_log", reps=3).set_index("column")
    assert p.loc["signal", "auc_drop"] > 0.2
    assert abs(p.loc["noise", "auc_drop"]) < 0.05


def test_runner_trains_only_on_the_fold_training_campaigns():
    t = _toy()
    t.loc[t["campaign"] == 3, "redeemed_flag"] = 1 - t.loc[t["campaign"] == 3, "redeemed_flag"]   # flip the test labels
    r = Runner(t, [{"test_campaign": 3, "train_campaigns": [1, 2]}])
    # trained on campaigns 1-2 (signal -> redeem), so on the flipped campaign 3 it ranks exactly backwards
    assert r.score(r.predict(["signal"], "logistic_log"))["auc"] < 0.1
