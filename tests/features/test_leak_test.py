import features.leak_test as leak_test
from features.profile import compute_profiles


def _rows(tables):
    rows = tables.spine[["household_key", "campaign", "start_day"]].copy()
    rows["cutoff_day"] = rows["start_day"] - 7
    return rows[["household_key", "campaign", "cutoff_day"]]


def test_honest_columns_pass(tables):
    rows = _rows(tables)
    built = compute_profiles(tables, rows)
    assert leak_test.columns_that_leak(tables, rows, built, ["spend_per_week_8w", "past_redemptions"]) == {}


def test_a_column_that_sees_30_days_ahead_is_caught(tables, monkeypatch):
    def leaky(t, r):          # a planted bug: profiles built as if the cutoff were 30 days later
        return compute_profiles(t, r.assign(cutoff_day=r["cutoff_day"] + 30)).assign(cutoff_day=r["cutoff_day"])
    monkeypatch.setattr(leak_test, "compute_profiles", leaky)
    rows = _rows(tables)
    leaks = leak_test.columns_that_leak(tables, rows, leaky(tables, rows), ["spend_per_week_8w"])
    assert "spend_per_week_8w" in leaks
