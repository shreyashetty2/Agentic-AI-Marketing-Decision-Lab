"""Tests for the Lead Agent surprise checks (src/lead/surprises.py)."""
import numpy as np
import pandas as pd

from lead.surprises import never_seen, unusual_moves

CFG = {"columns": ["spend", "coupons"], "move_sd": 4, "min_relative": 0.10, "min_history": 5}


def write_data(p, extra_lines=(), campaigns=None):
    p.mkdir(parents=True, exist_ok=True)
    lines = [(d, 1 + d % 2) for d in range(1, 200)] + list(extra_lines)          # products 1 and 2 bought every day
    pd.DataFrame(lines, columns=["DAY", "PRODUCT_ID"]).to_parquet(p / "transaction_data.parquet")
    pd.DataFrame({"PRODUCT_ID": [1, 2, 3], "DEPARTMENT": ["GROCERY", "PRODUCE", "PHARMACY"]}).to_parquet(p / "product.parquet")
    cal = campaigns or [(1, "A", 10), (2, "B", 50), (3, "B", 150)]
    pd.DataFrame(cal, columns=["campaign", "campaign_type", "start_day"]).to_parquet(p / "spine_household_campaign.parquet")


def test_a_department_first_bought_in_the_new_data_is_a_question(tmp_path):
    write_data(tmp_path, extra_lines=[(150, 3), (160, 3)])
    qs = never_seen(tmp_path, 100, 199, set())
    assert [q["key"] for q in qs] == ["department:PHARMACY"]
    assert "2 receipt line(s)" in qs[0]["text"]


def test_a_department_seen_before_the_new_data_is_not(tmp_path):
    write_data(tmp_path, extra_lines=[(50, 3), (150, 3)])
    assert never_seen(tmp_path, 100, 199, set()) == []


def test_new_data_ends_at_upto(tmp_path):
    write_data(tmp_path, extra_lines=[(190, 3)])
    assert never_seen(tmp_path, 100, 180, set()) == []


def test_an_answered_department_is_not_asked_again(tmp_path):
    write_data(tmp_path, extra_lines=[(150, 3)])
    assert never_seen(tmp_path, 100, 199, {"department:PHARMACY"}) == []


def test_a_new_campaign_type_is_a_question(tmp_path):
    write_data(tmp_path, campaigns=[(1, "A", 10), (2, "B", 50), (3, "C", 150)])
    qs = never_seen(tmp_path, 100, 199, set())
    assert [q["key"] for q in qs] == ["campaign_type:C"]


def test_no_earlier_run_means_nothing_to_compare_with(tmp_path):
    write_data(tmp_path, extra_lines=[(150, 3)])
    assert never_seen(tmp_path, None, 199, set()) == []
    assert unusual_moves(tmp_path, None, 199, CFG, set()) == []


def test_a_planted_department_is_flagged_and_labelled_as_planted(tmp_path):
    write_data(tmp_path)
    qs = never_seen(tmp_path, 100, 199, set(), plant="TEST DEPARTMENT")
    assert [q["key"] for q in qs] == ["department:TEST DEPARTMENT"]
    assert "planted to test this check" in qs[0]["text"]


def features(tmp_path, coupons_last: float):
    """Ten campaigns, 100 households each; spend and coupons drift a little, then the last campaign's coupons."""
    rng = np.random.default_rng(0)
    rows = []
    for c in range(10):
        level = 1.0 if c < 9 else coupons_last
        rows += [{"campaign": c, "start_day": 10 * c, "spend": 30 + rng.normal(0, 0.3),
                  "coupons": level * (0.15 + rng.normal(0, 0.002))} for _ in range(100)]
    pd.DataFrame(rows).to_parquet(tmp_path / "features_household_campaign.parquet")


def test_a_broken_feed_is_caught(tmp_path):
    features(tmp_path, coupons_last=0.2)                     # coupon use drops 80% at once
    qs = unusual_moves(tmp_path, 85, 95, CFG, set())
    assert [q["key"] for q in qs] == ["unusual_move:coupons:campaign9"]
    assert "times its usual move" in qs[0]["text"]


def test_ordinary_campaign_to_campaign_moves_are_not(tmp_path):
    features(tmp_path, coupons_last=1.0)
    assert unusual_moves(tmp_path, 15, 95, CFG, set()) == []


def test_too_little_history_flags_nothing(tmp_path):
    features(tmp_path, coupons_last=0.2)
    assert unusual_moves(tmp_path, 85, 95, {**CFG, "min_history": 9}, set()) == []


def test_an_answered_move_is_not_asked_again(tmp_path):
    features(tmp_path, coupons_last=0.2)
    assert unusual_moves(tmp_path, 85, 95, CFG, {"unusual_move:coupons:campaign9"}) == []
