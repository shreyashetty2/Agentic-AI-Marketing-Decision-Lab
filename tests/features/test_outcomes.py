import numpy as np
import pandas as pd
import pytest

from features.outcomes import compute_outcomes, promoted_keys


def _get(out, household, campaign):
    return out[(out.household_key == household) & (out.campaign == campaign)].iloc[0]


def test_campaign_product_spend_window_inclusive_and_no_double_count(tables, causal):
    out = compute_outcomes(tables, causal)
    # days 110, 120, 130 (end day) count; day 93 (before) and 131 (after) do not; U1 listed twice
    assert _get(out, 1, 1).campaign_product_spend_during == pytest.approx(3 + 2 + 1)
    assert _get(out, 2, 1).campaign_product_spend_during == pytest.approx(7)
    assert _get(out, 3, 1).campaign_product_spend_during == 0


def test_typea_spend_uses_pool_and_model3_is_blank(tables, causal):
    r = _get(compute_outcomes(tables, causal), 3, 2)
    assert r.campaign_product_spend_during == pytest.approx(9)
    assert np.isnan(r.bought_campaign_product_excl_display_flyer)
    assert np.isnan(r.bought_campaign_product_excl_display)


def test_redeemed_coupon_spend(tables, causal):
    out = compute_outcomes(tables, causal)
    assert _get(out, 1, 1).redeemed_coupon_product_spend_during == pytest.approx(6)
    assert np.isnan(_get(out, 2, 1).redeemed_coupon_product_spend_during)


def test_display_and_flyer_variants(tables, causal):
    out = compute_outcomes(tables, causal)
    h2 = _get(out, 2, 1)                       # only purchase was in the flyer, not on display
    assert h2.bought_campaign_product_excl_display_flyer == 0
    assert h2.bought_campaign_product_excl_display == 1
    h1 = _get(out, 1, 1)                       # day 110 on display, but day 120 not promoted
    assert h1.bought_campaign_product_excl_display_flyer == 1
    assert _get(out, 3, 1).bought_campaign_product_excl_display_flyer == 0


def test_conflicting_duplicates_count_as_promoted(causal):
    p = promoted_keys(causal)
    row = p[(p.product_id == 101) & (p.store_id == 1) & (p.week_no == 16)].iloc[0]
    assert row.on_display == 1 and row.in_flyer == 0
    assert len(p) == 2


def test_mailed_overlapping_campaign(tables, causal):
    out = compute_outcomes(tables, causal)
    assert _get(out, 1, 1).mailed_overlapping_campaign == 1   # mailed c2, which overlaps c1
    assert _get(out, 2, 1).mailed_overlapping_campaign == 0   # mailed c1 only
    assert _get(out, 2, 2).mailed_overlapping_campaign == 1   # not mailed c2, but mailed c1 which overlaps it
    assert _get(out, 1, 3).mailed_overlapping_campaign == 0


def test_one_row_per_spine_row(tables, causal):
    out = compute_outcomes(tables, causal)
    assert len(out) == len(tables.spine)
    assert not out.duplicated(["household_key", "campaign"]).any()
