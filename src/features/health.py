"""Column health: the numbers the 2A agent judges each column on."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features.spec import blank_reasons

MIN_ROWS = 30   # below this a signal score is too noisy to report

# outcome name -> (answer column, rows it is defined on, kind)
OUTCOMES = {
    "model1_redeemed": ("redeemed_flag", lambda t: t["mailed_flag"] == 1, "yes_no"),
    "model2_spend": ("campaign_product_spend_during", lambda t: t["redeemed_flag"] == 1, "dollars"),
    "model3_bought": ("bought_campaign_product_excl_display_flyer",
                      lambda t: t["bought_campaign_product_excl_display_flyer"].notna(), "yes_no"),
}
OUTCOME_LABELS = {
    "model1_redeemed": "Model 1's answer (redeemed, yes/no)",
    "model2_spend": "Model 2's answer (spend on campaign products during the campaign)",
    "model3_bought": "Model 3's answer (bought a campaign product, yes/no)",
}


def _reason_masks(t: pd.DataFrame) -> dict[str, pd.Series]:
    """Each named blank reason as a yes/no per row (all False if the table lacks the column it needs)."""
    def where(column, test):
        return test(t[column]) if column in t else pd.Series(False, index=t.index)
    return {
        "no_history": where("no_history_flag", lambda c: c == 1),
        "no_spend_26w": where("spend_per_week_26w", lambda c: c.fillna(0) == 0),
        "never_received": where("past_campaigns_received", lambda c: c == 0),
        "no_demographics": where("has_demographics", lambda c: c == 0),
        "type_a": where("campaign_type", lambda c: c == "A"),
        "not_redeemed": where("redeemed_flag", lambda c: c == 0),
    }


def unexplained_blanks(table: pd.DataFrame, spec: dict) -> dict[str, int]:
    """Column -> number of blank cells not covered by any of its blank_when reasons (only columns with some)."""
    masks = _reason_masks(table)
    found = {}
    for column, reasons in blank_reasons(spec).items():
        if column not in table:
            continue
        explained = pd.Series(False, index=table.index)
        for reason in reasons:
            explained |= masks[reason]
        n = int((table[column].isna() & ~explained).sum())
        if n:
            found[column] = n
    return found


def _signal(x: pd.Series, y: pd.Series, kind: str) -> float:
    """How strongly x tracks y: AUC (0.5 = none, 1 = perfect) for yes/no, |rank correlation| for dollars."""
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    if len(x) < MIN_ROWS or x.nunique() < 2 or (kind == "yes_no" and y.nunique() < 2):
        return np.nan
    if kind == "yes_no":
        auc = roc_auc_score(y, x)
        return max(auc, 1 - auc)
    return abs(x.corr(y, method="spearman"))


def column_health(table: pd.DataFrame, columns: list[str], leak_alarm: float) -> pd.DataFrame:
    rows = []
    for column in columns:
        values = table[column]
        numeric = pd.api.types.is_numeric_dtype(values)
        row = {"column": column,
               "share_blank": float(values.isna().mean()),
               "n_distinct": int(values.nunique()),
               "min": float(values.min()) if numeric else np.nan,
               "max": float(values.max()) if numeric else np.nan}
        for name, (answer, defined_on, kind) in OUTCOMES.items():
            if numeric and answer in table and answer != column:
                subset = table[defined_on(table)]
                row[name] = _signal(subset[column], subset[answer], kind)
            else:
                row[name] = np.nan
        rows.append(row)
    health = pd.DataFrame(rows).set_index("column")
    health["max_signal"] = health[list(OUTCOMES)].max(axis=1)
    health["leak_alarm"] = health["max_signal"] > leak_alarm
    return health


def decide_columns(table: pd.DataFrame, health: pd.DataFrame, spec: dict, leak_cleared: list[str]) -> pd.DataFrame:
    """One decision per column 2A adds: accept, or accept_with_note when there is something a reader should know.
    Called only after the hard checks passed, so every blank is already known to be explained."""
    alarm = spec["settings"]["leak_alarm"]
    rows = []
    for column, reasons in blank_reasons(spec).items():
        notes = []
        if column in leak_cleared:
            scores = health.loc[column, list(OUTCOMES)]
            top = scores.idxmax()
            notes.append(f"Scored {scores[top]:.2f} against {OUTCOME_LABELS[top]}, above the {alarm:.2f} leak alarm. "
                         "Unchanged when all data after the cutoff is scrambled, so this is genuine habit, not leakage.")
        share_blank = table[column].isna().mean()
        if share_blank > 0:
            why = ", or because ".join(spec["blank_reasons"][r] for r in reasons)
            notes.append(f"Blank on {share_blank:.1%} of rows, because {why}.")
        rows.append({"column": column, "decision": "accept_with_note" if notes else "accept", "note": " ".join(notes)})
    return pd.DataFrame(rows).set_index("column")
