import numpy as np
import pandas as pd
import pytest

from features.profile import compute_profiles, features


def _row(tables, household, campaign):
    rows = tables.spine[["household_key", "campaign", "start_day"]].copy()
    rows["cutoff_day"] = rows["start_day"] - 7
    out = compute_profiles(tables, rows[["household_key", "campaign", "cutoff_day"]])
    return out[(out.household_key == household) & (out.campaign == campaign)].iloc[0]


def test_purchase_on_cutoff_day_is_excluded(tables):
    r = _row(tables, 1, 1)                        # cutoff 93; the $100 purchase is on day 93
    assert r.spend_per_week_8w == pytest.approx(20 / 8)          # days 60, 60, 92 in [37, 93)
    assert r.spend_per_week_26w == pytest.approx(25 / (83 / 7))  # all 4 lines; 83 days of history


def test_activity_columns(tables):
    r = _row(tables, 1, 1)
    weeks26 = 83 / 7
    assert r.history_weeks_before_cutoff == pytest.approx(83 / 7)
    assert r.trips_per_week_8w == pytest.approx(2 / 8)
    assert r.trips_per_week_26w == pytest.approx(3 / weeks26)
    assert r.avg_basket_value_26w == pytest.approx(25 / 3)
    assert r.days_since_last_trip == 1
    assert r.n_departments_26w == 2
    assert r.share_spend_on_discount_26w == pytest.approx(4 / 25)
    assert r.coupon_lines_per_week_26w == pytest.approx(1 / weeks26)
    assert r.campaign_product_spend_per_week_26w == pytest.approx(4 / weeks26)
    assert r.campaign_product_share_26w == pytest.approx(4 / 25)
    assert r.short_history_flag == 1 and r.no_history_flag == 0
    assert r.classification_1 == "Age Group4"


def test_short_history_uses_weeks_actually_available(tables):
    r = _row(tables, 2, 1)                        # first purchase day 80, cutoff 93 -> 13 days
    assert r.spend_per_week_8w == pytest.approx(8 / (13 / 7))
    assert r.spend_per_week_26w == pytest.approx(8 / (13 / 7))
    assert r.short_history_flag == 1


def test_no_history_gives_blanks_and_flag(tables):
    r = _row(tables, 3, 1)                        # first purchase day 150, cutoff 93
    assert r.history_weeks_before_cutoff == 0 and r.no_history_flag == 1
    for col in ["spend_per_week_8w", "spend_per_week_26w", "trips_per_week_26w", "avg_basket_value_26w",
                "days_since_last_trip", "n_departments_26w", "coupon_lines_per_week_26w"]:
        assert np.isnan(r[col]), col
    assert r.past_redemptions == 0
    assert pd.isna(r.classification_1)


def test_redemption_counted_by_date_and_running_campaign_not_received(tables):
    r = _row(tables, 1, 2)                        # cutoff 113: c1 redemption on day 105, c1 still running
    assert r.past_redemptions == 1
    assert r.past_campaigns_received == 0         # c1 ends day 130, after the cutoff
    assert np.isnan(r.past_redemption_rate)
    assert r.other_campaigns_live_at_cutoff == 1  # c1: started day 100, still running on 113
    assert r.spend_per_week_8w == pytest.approx((4 + 6 + 10 + 100 + 3) / 8)


def test_received_redeemed_and_rate(tables):
    r = _row(tables, 1, 3)                        # cutoff 193: c1 and c2 both ended
    assert r.past_campaigns_received == 2
    assert r.past_campaigns_redeemed == 1
    assert r.past_redemption_rate == pytest.approx(0.5)


def test_features_matches_batch_and_works_for_any_cutoff(tables):
    r = _row(tables, 1, 2)
    single = features(tables, household=1, campaign=2, cutoff_day=113)
    for col in single.index:
        a, b = single[col], r[col]
        if isinstance(a, str):
            assert a == b, col
        else:
            assert (pd.isna(a) and pd.isna(b)) or a == pytest.approx(b), col
    today = features(tables, household=1, campaign=3, cutoff_day=140)   # not any campaign's cutoff
    assert today.spend_per_week_8w == pytest.approx((10 + 100 + 3 + 2 + 1 + 50) / 8)   # days 84-139


def test_scrambling_the_future_changes_nothing(tables):
    rows = tables.spine[["household_key", "campaign", "start_day"]].copy()
    rows["cutoff_day"] = rows["start_day"] - 7
    rows = rows[["household_key", "campaign", "cutoff_day"]]
    before = compute_profiles(tables, rows)
    rng = np.random.default_rng(0)
    for cutoff in rows.cutoff_day.unique():
        t = tables.copy()
        future = t.transactions.day >= cutoff
        t.transactions.loc[future, "sales_value"] = rng.uniform(0, 500, future.sum())
        t.transactions.loc[future, "coupon_disc"] = -1.0
        extra = t.transactions[future].assign(day=cutoff, basket_id=999)       # new trip ON the cutoff day
        t.transactions = pd.concat([t.transactions, extra], ignore_index=True)
        t.redemptions = pd.concat([t.redemptions, pd.DataFrame(
            [(h, cutoff, "U1", 1) for h in (1, 2, 3)], columns=t.redemptions.columns)], ignore_index=True)
        sub = rows[rows.cutoff_day == cutoff]
        after = compute_profiles(t, sub)
        pd.testing.assert_frame_equal(before.loc[sub.index], after, check_exact=True)
