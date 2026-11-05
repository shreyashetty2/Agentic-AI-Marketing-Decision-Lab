"""Shared chart helpers for the per-model pages."""
import re

import altair as alt
import pandas as pd
import streamlit as st

from lib.data import DEMOGRAPHIC_LABELS


def _natural_key(label) -> tuple:
    """Sorts 'Level2' before 'Level10', '3+' after '2'; anything with no digit sorts after, alphabetically."""
    m = re.search(r"(\d+)", str(label))
    return (0, int(m.group(1))) if m else (1, str(label))


def population_breakdown(df: pd.DataFrame, score_col: str, score_label: str) -> None:
    """Average score by each demographic column, side by side. Households with no demographic record
    (about two-thirds of the panel, per the Data Agent's own note) are left out of each column's own
    chart rather than silently folded into an 'unknown' average. Plain Altair, not st.bar_chart: the
    latter always re-sorts category labels alphabetically, which breaks 'Level2' before 'Level10'."""
    cols = st.columns(3)
    for i, (col, label) in enumerate(DEMOGRAPHIC_LABELS.items()):
        if col not in df.columns:
            continue
        with cols[i % 3]:
            st.caption(label)
            known = df[df[col].notna() & (df[col] != "None/Unknown")]
            if known.empty:
                st.caption("No records for this group.")
                continue
            grouped = known.groupby(col)[score_col].mean().reset_index()
            order = sorted(grouped[col].unique(), key=_natural_key)
            chart = alt.Chart(grouped).mark_bar().encode(
                x=alt.X(col, sort=order, title=None), y=alt.Y(score_col, title=None))
            st.altair_chart(chart, use_container_width=True)
    st.caption(f"Each bar: average {score_label} for households in that group, among the ~third of the "
               "panel with a demographic record on file.")


def score_histogram(df: pd.DataFrame, score_col: str, bins: int = 10) -> None:
    """A clean-labelled histogram in bin order -- plain `value_counts(bins=...)` both renders ugly
    Interval objects as labels and, once relabelled to strings, gets re-sorted alphabetically by
    st.bar_chart (so '-0.08 to 0.09' jumps ahead of '-0.24 to -0.08'); Altair with an explicit sort
    avoids both."""
    cut = pd.cut(df[score_col], bins=bins)
    counts = cut.value_counts().sort_index()
    labels = [f"{i.left:.2f} to {i.right:.2f}" for i in counts.index]
    chart_df = pd.DataFrame({"bin": labels, "count": counts.values})
    chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("bin", sort=labels, title=None), y=alt.Y("count", title=None))
    st.altair_chart(chart, use_container_width=True)
