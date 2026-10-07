import pandas as pd

from features.health import column_health, decide_columns
from features.outcomes import compute_outcomes
from features.profile import compute_profiles
from features.report import render_report
from features.spec import column_roles, load_spec

KEYS = ["household_key", "campaign"]


def _built(tables, causal):
    spec = load_spec()
    rows = tables.spine[KEYS].assign(cutoff_day=tables.spine["start_day"] - 7)
    table = (tables.spine.merge(compute_profiles(tables, rows).drop(columns="cutoff_day"), on=KEYS)
             .merge(compute_outcomes(tables, causal), on=KEYS))[list(column_roles(spec))]
    health = column_health(table, list(spec["profile"]), spec["settings"]["leak_alarm"])
    return spec, table, health


def test_every_added_column_gets_one_decision(tables, causal):
    spec, table, health = _built(tables, causal)
    decisions = decide_columns(table, health, spec, leak_cleared=[])
    added = [c for s in ("profile", "outcome", "filter", "flag") for c in spec[s]]
    assert list(decisions.index) == added
    assert set(decisions["decision"]) <= {"accept", "accept_with_note"}


def test_blank_columns_get_a_note_naming_their_reasons(tables, causal):
    spec, table, health = _built(tables, causal)
    decisions = decide_columns(table, health, spec, leak_cleared=[])
    rate = decisions.loc["past_redemption_rate"]          # blank for households never mailed before
    assert rate.decision == "accept_with_note"
    assert spec["blank_reasons"]["never_received"] in rate.note
    assert decisions.loc["past_redemptions"].decision == "accept"   # never blank


def test_leak_cleared_column_note_says_habit_not_leakage(tables, causal):
    spec, table, health = _built(tables, causal)
    health.loc["spend_per_week_8w", ["model2_spend", "max_signal"]] = 0.93
    decisions = decide_columns(table, health, spec, leak_cleared=["spend_per_week_8w"])
    note = decisions.loc["spend_per_week_8w"].note
    assert decisions.loc["spend_per_week_8w"].decision == "accept_with_note"
    assert "0.93" in note and "Model 2" in note and "not leakage" in note


def test_report_has_summary_checks_decisions_and_findings(tables, causal):
    spec, table, health = _built(tables, causal)
    decisions = decide_columns(table, health, spec, leak_cleared=[])
    text = render_report(table, health, decisions, ["inputs match", "no negative values"], spec)
    assert f"{len(table):,} rows x {table.shape[1]} columns" in text
    assert "- [OK] no negative values" in text
    assert "`past_redemption_rate`" in text
    for heading in ("## Summary", "## Checks", "## Column decisions", "## What each model leans on"):
        assert heading in text
    assert render_report(table, health, decisions, ["inputs match"], spec) == \
        render_report(table, health, decisions, ["inputs match"], spec)      # same run, same report
