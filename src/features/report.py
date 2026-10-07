"""The 2A run report: a plain-language page of what one run built, checked and decided."""
import pandas as pd

from features.health import OUTCOME_LABELS
from features.spec import column_roles

TOP_N = 3   # columns listed per model in "What each model leans on"


def render_report(table: pd.DataFrame, health: pd.DataFrame, decisions: pd.DataFrame,
                  checks: list[str], spec: dict) -> str:
    roles = column_roles(spec)
    added = {section: len(spec[section]) for section in ("profile", "outcome", "filter", "flag")}
    noted = decisions[decisions["decision"] == "accept_with_note"]
    accepted = decisions[decisions["decision"] == "accept"]

    lines = [
        "# Feature Agent 2A: run report",
        "",
        "## Summary",
        "",
        f"- Input: the Data Agent's table, {len(table):,} rows x {len(spec['passthrough'])} columns.",
        f"- Output: {len(table):,} rows x {table.shape[1]} columns. Same rows; {sum(added.values())} columns added: "
        f"{added['profile']} profile, {added['outcome']} outcome, {added['filter']} filter, {added['flag']} flags.",
        f"- Profile columns use only data from before each row's cutoff "
        f"(campaign start day minus {spec['settings']['cutoff_gap_days']}).",
        f"- All {len(checks)} checks passed. {len(noted)} columns accepted with a note, {len(accepted)} accepted.",
        "- What each column means and whether a model may use it: `src/features/feature_2a_spec.yaml`. "
        "Never use `outcome` or `filter` columns as model inputs.",
        "",
        "## Checks",
        "",
        *[f"- [OK] {c}" for c in checks],
        "",
        "## Column decisions",
        "",
        "### Accepted with a note",
        "",
        *[f"- {', '.join(f'`{c}`' for c in columns)} ({role}): {note}"
          for (role, note), columns in _group_by_note(noted, roles)],
        "",
        "### Accepted",
        "",
        ", ".join(f"`{column}`" for column in accepted.index),
        "",
        "## What each model leans on",
        "",
        "Profile columns that track each model's answer most strongly. For yes/no answers the score runs from "
        "0.5 (no better than a coin flip) to 1.0 (perfect); for dollar answers, from 0 (unrelated) to 1 "
        "(same ranking). A strong score means relevant, not caused by.",
        "",
    ]
    for outcome, label in OUTCOME_LABELS.items():
        top = health[outcome].dropna().sort_values(ascending=False).head(TOP_N)
        listed = ", ".join(f"`{column}` {score:.2f}" for column, score in top.items()) or "not enough rows to score"
        lines.append(f"- {label}: {listed}")
    return "\n".join(lines) + "\n"


def _group_by_note(noted: pd.DataFrame, roles: dict[str, str]) -> list[tuple[tuple[str, str], list[str]]]:
    """Columns sharing the same role and exact note, in spec order, so identical notes are written once."""
    groups: dict[tuple[str, str], list[str]] = {}
    for column, note in noted["note"].items():
        groups.setdefault((roles[column], note), []).append(column)
    return list(groups.items())
