import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from modeling.check_expected_revenue import (  # noqa: E402
    SPEND, USUAL, attach_spend, locked_campaigns, open_folds, ranking, score_folds, totals, usual_spend,
)
from modeling.model1_baseline import BASE_FEATURES  # noqa: E402


def _tables(n_campaigns=14, households=120, seed=0):
    """Small Model 1 / Model 2 tables shaped like Feature Agent 2B's: campaigns every 40 days, 30 days long."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(1, n_campaigns + 1):
        ctype = "A" if c % 4 == 1 else ("B" if c % 2 == 0 else "C")
        start = 300 + 40 * c
        for h in range(households):
            row = {f: rng.random() for f in BASE_FEATURES}
            row.update(household_key=h, campaign=c, campaign_type=ctype, start_day=start, end_day=start + 30,
                       duration_days=30, campaign_product_spend_per_week_26w=rng.random() * 5)
            row["redeemed_flag"] = int(rng.random() < 0.15 + 0.4 * row["past_redemption_rate"])
            rows.append(row)
    m1 = pd.DataFrame(rows)
    m1 = m1.assign(is_type_a=(m1["campaign_type"] == "A").astype(int), is_type_b=(m1["campaign_type"] == "B").astype(int))
    m1 = m1.assign(**{USUAL: usual_spend(m1)})
    m2 = m1[(m1["redeemed_flag"] == 1) & (m1["campaign_type"] != "A")].copy()
    m2[SPEND] = 6 + 0.8 * m2[USUAL] + rng.random(len(m2)) * 4
    return m1, m2


def test_usual_spend_reads_a_blank_weekly_spend_as_zero():
    t = pd.DataFrame({"campaign_product_spend_per_week_26w": [2.0, np.nan], "duration_days": [35, 14]})
    assert usual_spend(t).tolist() == pytest.approx([10.0, 0.0])


def test_attach_spend_gives_spend_to_redeemed_typebc_rows_only():
    m1, m2 = _tables(n_campaigns=4, households=20)
    out = attach_spend(m1, m2)
    redeemed_bc = (out["redeemed_flag"] == 1) & (out["campaign_type"] != "A")
    assert out.loc[redeemed_bc, SPEND].notna().all() and out.loc[~redeemed_bc, SPEND].isna().all()


def test_attach_spend_refuses_tables_that_do_not_line_up():
    m1, m2 = _tables(n_campaigns=4, households=20)
    with pytest.raises(ValueError):
        attach_spend(m1, m2.iloc[1:])


def test_locked_campaigns_are_each_models_last_ones():
    m1, m2 = _tables()
    spec = {"models": {1: {"locked_test_campaigns": 3}, 2: {"locked_test_campaigns": 2}}}
    assert locked_campaigns(spec, m1, m2) == {12, 13, 14}


def test_open_folds_skip_typea_and_locked_campaigns():
    m1, _ = _tables()
    folds = open_folds(m1, first_test_day=500, locked={13, 14}, min_train_rows=100, min_train_positives=10)
    tested = {int(f["test_campaign"]) for f in folds}
    assert tested and not tested & {13, 14}
    assert all(f["campaign_type"] != "A" for f in folds)


def test_scoring_trains_only_on_campaigns_ended_before_the_cutoff_and_multiplies_the_two_models():
    m1, m2 = _tables()
    m1 = attach_spend(m1, m2)
    folds = open_folds(m1, first_test_day=500, locked=set(), min_train_rows=100, min_train_positives=10)
    preds, info = score_folds(m1, m2, folds, min_m2_rows=20)
    for f, i in zip(folds, info):
        assert all(c in f["train_campaigns"] for c in m2.loc[m2["campaign"].isin(f["train_campaigns"]), "campaign"])
        assert i["m2_train_rows"] == m2["campaign"].isin(f["train_campaigns"]).sum()
        ended = m1.loc[m1["campaign"].isin(f["train_campaigns"]), "end_day"]
        assert (ended < f["cutoff_day"]).all()
    assert preds["expected_revenue"].to_numpy() == pytest.approx((preds["p_redeem"] * preds["expected_spend"]).to_numpy())
    assert (preds.loc[preds["redeemed"] == 0, "actual_revenue"] == 0).all()
    assert (preds["campaign_type"] != "A").all()


def test_totals_split_revenue_into_redemptions_and_spend():
    preds = pd.DataFrame({"campaign": [1, 1, 1, 1], "p_redeem": [0.5, 0.5, 0.5, 0.5],
                          "expected_spend": [10.0, 10.0, 10.0, 10.0], "redeemed": [1, 0, 0, 0],
                          "actual_revenue": [20.0, 0, 0, 0]})
    preds["expected_revenue"] = preds["p_redeem"] * preds["expected_spend"]
    row = totals(preds).set_index("campaign").loc["All"]
    assert row["revenue_ratio"] == pytest.approx(20 / 20)       # 2 forecast redemptions x $10 vs. 1 x $20
    assert row["redemption_ratio"] == pytest.approx(2.0)
    assert row["spend_ratio"] == pytest.approx(0.5)


def test_ranking_by_revenue_reaches_more_dollars_when_spend_differs_and_redemption_odds_do_not():
    n = 200
    rng = np.random.default_rng(3)
    spend = rng.random(n) * 50
    redeemed = (rng.random(n) < 0.3).astype(int)
    preds = pd.DataFrame({"campaign": 1, "p_redeem": 0.3 + rng.random(n) * 1e-3, "expected_spend": spend,
                          "redeemed": redeemed, "actual_revenue": redeemed * spend})
    preds["expected_revenue"] = preds["p_redeem"] * preds["expected_spend"]
    r = ranking(preds, reps=20)
    assert r["revenue_20"] > r["p_only_20"] and r["revenue_avg"] > r["p_only_avg"]
