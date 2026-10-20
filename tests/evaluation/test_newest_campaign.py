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
