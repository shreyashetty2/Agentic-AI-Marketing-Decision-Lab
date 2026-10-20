"""The Evaluation Agent's check of the newest finished campaign against the model's usual range."""
import pandas as pd

from evaluation.agent import newest_campaign

CFG = {"size": "rows", "min_size": 3, "min_history": 5, "width_sd": 2}


def table(errors: list[float], sizes: list[int] | None = None) -> pd.DataFrame:
    """One campaign per error, in time order; every household's prediction misses by that campaign's error."""
    sizes = sizes or [4] * len(errors)
    rows = [{"campaign": i + 1, "end_day": 100 + 10 * i, "label": 10.0, "own": 10.0 + err}
            for i, (err, n) in enumerate(zip(errors, sizes)) for _ in range(n)]
    return pd.DataFrame(rows)


def test_newest_campaign_far_worse_than_usual_is_flagged():
    r = newest_campaign(table([1.0, 1.2, 0.8, 1.1, 0.9, 6.0]), "mae", "own", CFG)
    assert r["campaign"] == 6 and r["score"] == 6.0
    assert r["worse_than_usual"] and r["n_history"] == 5
    assert r["usual_low"] < 1.0 < r["usual_high"] < 6.0


def test_newest_campaign_inside_the_usual_range_is_not_flagged():
    r = newest_campaign(table([1.0, 1.2, 0.8, 1.1, 0.9, 1.05]), "mae", "own", CFG)
    assert not r["worse_than_usual"]


def test_better_than_usual_is_never_flagged():
    r = newest_campaign(table([1.0, 1.2, 0.8, 1.1, 0.9, 0.0]), "mae", "own", CFG)   # lower error = better
    assert not r["worse_than_usual"]


def test_small_campaigns_do_not_count():
    # the last campaign has 2 rows (below min_size 3): the newest that counts is campaign 6
    r = newest_campaign(table([1.0, 1.2, 0.8, 1.1, 0.9, 1.0, 50.0], [4, 4, 4, 4, 4, 4, 2]), "mae", "own", CFG)
    assert r["campaign"] == 6 and not r["worse_than_usual"]


def test_too_little_history_says_nothing():
    assert newest_campaign(table([1.0, 1.2, 0.8, 1.1, 6.0]), "mae", "own", CFG) is None


def capture_table(newest_reversed: bool, positives=(5, 6, 4, 5, 6, 5, 5)) -> pd.DataFrame:
    """Campaigns of 20 households; the model ranks redeemers first, except (optionally) on the newest one."""
    rows = []
    for i, k in enumerate(positives):
        last = i == len(positives) - 1
        for h in range(20):
            label = int(h < k)
            good = 100 - h + (h % 3) * (i % 2)            # slightly different ranking each campaign
            rows.append({"campaign": i + 1, "end_day": 100 + 10 * i, "label": label,
                         "own": -good if (last and newest_reversed) else good})
    return pd.DataFrame(rows)


def test_for_a_higher_is_better_score_a_drop_is_flagged_and_a_rise_is_not():
    cfg = {"size": "positives", "min_size": 3, "min_history": 5, "width_sd": 2}
    drop = newest_campaign(capture_table(newest_reversed=True), "within_campaign_capture", "own", cfg)
    assert drop["higher_is_better"] and drop["worse_than_usual"] and drop["score"] == 0.0
    rise = newest_campaign(capture_table(newest_reversed=False), "within_campaign_capture", "own", cfg)
    assert not rise["worse_than_usual"]


def test_size_counts_redeemers_when_asked():
    cfg = {"size": "positives", "min_size": 5, "min_history": 3, "width_sd": 2}
    r = newest_campaign(capture_table(False, positives=(5, 6, 5, 6, 5, 4)), "within_campaign_capture", "own", cfg)
    assert r["campaign"] == 5 and r["size"] == 5          # campaign 6 has only 4 redeemers


def test_size_counts_mailed_households_for_uplift():
    t = table([1.0, 1.2, 0.8, 1.1, 0.9, 1.0]).assign(treatment=1)
    t.loc[t["campaign"] == 6, "treatment"] = 0
    r = newest_campaign(t, "mae", "own", {"size": "treated", "min_size": 1, "min_history": 4, "width_sd": 2})
    assert r["campaign"] == 5                              # campaign 6 mailed nobody


def test_two_campaigns_ending_the_same_day_newest_is_the_higher_number():
    t = table([1.0, 1.2, 0.8, 1.1, 0.9, 1.0, 1.05])
    t.loc[t["campaign"] == 7, "end_day"] = int(t.loc[t["campaign"] == 6, "end_day"].iloc[0])
    r = newest_campaign(t, "mae", "own", CFG)
    assert r["campaign"] == 7 and r["n_history"] == 6
