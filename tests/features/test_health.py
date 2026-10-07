import numpy as np
import pandas as pd

from features.health import column_health, unexplained_blanks
from features.spec import load_spec, column_roles


def test_spec_is_consistent():
    spec = load_spec()
    roles = column_roles(spec)
    listed = list(spec["passthrough"]) + [c for s in ("profile", "outcome", "filter", "flag") for c in spec[s]]
    assert len(listed) == len(roles), "a column is listed in more than one section"
    assert set(roles.values()) <= {"key", "context", "profile", "outcome", "filter", "flag"}
    known = {"no_history", "no_spend_26w", "never_received", "no_demographics", "type_a", "not_redeemed"}
    for section in ("profile", "outcome", "filter", "flag"):
        for col, body in spec[section].items():
            assert body.get("desc"), col
            assert set(body.get("blank_when", [])) <= known, col


def _table():
    return pd.DataFrame({
        "campaign_type": ["A", "B", "B", "C"],
        "redeemed_flag": [0, 1, 0, 1],
        "mailed_flag": [1, 1, 1, 1],
        "has_demographics": [1, 1, 0, 1],
        "no_history_flag": [0, 0, 1, 0],
        "past_campaigns_received": [1, 0, 2, 3],
        "spend_per_week_26w": [1.0, 2.0, np.nan, 3.0],                 # blank only where no history: OK
        "past_redemption_rate": [0.5, np.nan, 0.0, 1.0],               # blank only where never received: OK
        "bought_campaign_product_excl_display_flyer": [np.nan, 1, 0, 1],
        "redeemed_coupon_product_spend_during": [np.nan, 5.0, np.nan, 2.0],
        "campaign_product_spend_during": [1.0, 5.0, 0.0, 2.0],
        "spend_per_week_8w": [1.0, np.nan, np.nan, 3.0],              # row 1 blank has no reason
    })


def test_unexplained_blanks_found_only_where_no_reason():
    bad = unexplained_blanks(_table(), load_spec())
    assert bad == {"spend_per_week_8w": 1}


def test_health_flags_a_column_that_copies_the_answer():
    rng = np.random.default_rng(0)
    n = 400
    t = pd.DataFrame({
        "campaign_type": rng.choice(["A", "B", "C"], n),
        "mailed_flag": rng.integers(0, 2, n),
        "redeemed_flag": rng.integers(0, 2, n),
        "campaign_product_spend_during": rng.uniform(0, 100, n),
        "bought_campaign_product_excl_display_flyer": rng.integers(0, 2, n).astype(float),
        "spend_per_week_26w": rng.uniform(0, 100, n),
    })
    t.loc[:99, "spend_per_week_26w"] = np.nan
    t["planted_leak"] = t["redeemed_flag"].astype(float)
    h = column_health(t, ["spend_per_week_26w", "planted_leak"], leak_alarm=0.9)
    assert h.loc["planted_leak", "leak_alarm"]
    assert not h.loc["spend_per_week_26w", "leak_alarm"]
    assert h.loc["spend_per_week_26w", "share_blank"] == 0.25
