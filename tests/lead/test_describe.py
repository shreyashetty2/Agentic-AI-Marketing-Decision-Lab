"""Tests for the lead agent's plain words (src/lead/describe.py)."""
from lead.describe import gap_words, model_words, score_words


def test_each_kind_of_model_in_plain_words():
    forest = {"family": "forest", "params": {"n_estimators": 300, "max_depth": 6}, "features": list("abcdefghijk")}
    assert model_words(forest, "classify") == "random forest (300 trees, 6 levels deep) on 11 columns"
    assert model_words({"family": "median_linear", "features": ["usual_spend"]}, "regress") == (
        "calibrated usual spend (a straight line on usual spend, fit through the middle of the answers) on 1 column")
    assert model_words({"family": "logistic", "features": ["a", "b"]}, "classify").startswith("logistic regression")
    assert model_words(forest, "uplift").startswith(
        "pair of random forests (300 trees, 6 levels deep), one for mailed and one for unmailed households")


def test_scores_in_plain_words():
    assert score_words("within_campaign_capture", 0.5629) == "56.3% (redeemers caught in the top 20% of each campaign's list)"
    assert score_words("mae", 8.32) == "$8.32 (average dollar error per household)"
    assert score_words("ipw_stratified_qini", 0.0065).startswith("+0.7 (extra buyers per 100 mailed")


def test_gaps_say_whether_they_are_real():
    assert gap_words("mae", {"gap": 0.9, "low": 0.32, "high": 1.59, "status": "pass"}) == (
        "+$0.90 (likely range +$0.32 to +$1.59: a real difference)")
    assert gap_words("within_campaign_capture", {"gap": -0.033, "low": -0.057, "high": -0.011, "status": "fail"}) == (
        "-3.3 points (likely range -5.7 points to -1.1 points: really worse)")
    assert gap_words("within_campaign_capture", {"gap": 0.011, "low": -0.02, "high": 0.027, "status": "inconclusive"}).endswith(
        "could be luck)")
