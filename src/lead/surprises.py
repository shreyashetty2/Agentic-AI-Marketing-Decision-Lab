"""Lead Agent surprise checks: what no agent's rules cover, found by comparing the new data with what came before.

1. Never-seen values: a product department, or a campaign type, that appears for the first time in the data
   that arrived since the last run. Nobody has decided how to treat it yet (does it count as household spend?).
2. Unusual moves: an everyday-shopping column (spend, trips, coupon use, ...) whose average over all households
   moves between one campaign and the next by far more than it usually does. That usually means a data feed
   broke (for example an empty coupon file), not that shoppers changed overnight. "Usual" is learned from the
   campaigns before: move_sd times the standard deviation of their campaign-to-campaign moves, and at least
   min_relative of the column's level, so tiny moves are never flagged. On our two years no campaign trips it;
   an 80% drop in coupon use would be about 40 times the usual move.

   A column that goes blank for many more households than usual (a feed producing empty values, not zeros) is
   flagged the same way, on its share of blank rows.

The new data is everything after `since` up to `upto` (days). With no earlier run (since=None) there is
nothing to compare with: the checks only learn. Each finding is a question {key, who, text}; its key names the
value or column, so the engine can match it to a saved rule and file it as answered.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def never_seen(processed: Path, since: int | None, upto: int, plant: str | None = None) -> list[dict]:
    """plant: add a made-up department to the new data, to test that the check stops the run (said plainly in
    the question)."""
    if since is None or upto <= since:
        return []
    out = []
    tx = pd.read_parquet(processed / "transaction_data.parquet", columns=["DAY", "PRODUCT_ID"])
    departments = pd.read_parquet(processed / "product.parquet", columns=["PRODUCT_ID", "DEPARTMENT"])
    tx = tx[tx["DAY"] <= upto].merge(departments, on="PRODUCT_ID", how="left")
    before = set(tx.loc[tx["DAY"] <= since, "DEPARTMENT"].dropna())
    new_lines = tx[tx["DAY"] > since]
    counts = new_lines["DEPARTMENT"].value_counts()
    if plant:
        counts[plant] = 1
    for dept, n in counts.items():
        if dept in before:
            continue
        planted = " (planted to test this check; not in the real data)" if dept == plant else ""
        out.append({"key": f"department:{dept}", "who": "team",
                    "text": f"A product department never seen before appeared in the data since day {since}: "
                            f"'{dept}', on {n:,} receipt line(s){planted}. Should its purchases count as household "
                            "spend, like groceries, or be set apart, like gas and miscellaneous lines?"})
    cal = pd.read_parquet(processed / "spine_household_campaign.parquet", columns=["campaign", "campaign_type", "start_day"])
    cal = cal.drop_duplicates("campaign")
    seen_types = set(cal.loc[cal["start_day"] <= since, "campaign_type"])
    for ctype, group in cal[(cal["start_day"] > since) & (cal["start_day"] <= upto)].groupby("campaign_type"):
        if ctype not in seen_types:
            out.append({"key": f"campaign_type:{ctype}", "who": "team",
                        "text": f"A campaign type never seen before started since day {since}: type '{ctype}' "
                                f"(campaigns {sorted(int(c) for c in group['campaign'])}). The models were only "
                                "trained on the types before it. Should they score it, or wait until it has history?"})
    return out


def unusual_moves(processed: Path, since: int | None, upto: int, cfg: dict) -> list[dict]:
    """cfg: columns, move_sd, min_relative, min_history (moves needed before 'usual' means anything)."""
    if since is None or upto <= since:
        return []
    cols = cfg["columns"]
    table = pd.read_parquet(processed / "features_household_campaign.parquet", columns=["campaign", "start_day", *cols])
    by_campaign = table.groupby("campaign")
    starts = by_campaign["start_day"].first().sort_values()
    starts = starts[starts <= upto]
    means = by_campaign[cols].mean().loc[starts.index]
    blank = by_campaign[cols].agg(lambda c: c.isna().mean()).loc[starts.index]
    out = []
    for position, campaign in enumerate(starts.index):
        if starts[campaign] <= since or position < cfg["min_history"] + 1:
            continue
        previous = starts.index[position - 1]
        for col in cols:
            for kind, series, level in (("unusual_move", means[col], abs(means[col].iloc[position - 1])),
                                        ("unusual_blanks", blank[col], 1.0)):
                now, before = series.iloc[position], series.iloc[position - 1]
                usual = series.iloc[:position].diff().iloc[1:].std()
                move = now - before
                if pd.isna(now) and not pd.isna(before):          # every household blank: the mean is gone
                    move, now = np.inf, "blank for every household"
                if kind == "unusual_blanks" and move <= 0:         # fewer blanks is never a problem
                    continue
                if not (abs(move) > cfg["move_sd"] * usual and abs(move) > cfg["min_relative"] * level):
                    continue
                times = "" if usual == 0 or np.isinf(move) else f", {abs(move) / usual:.0f} times its usual move between campaigns ({usual:.3f})"
                what = "share of households with no value for" if kind == "unusual_blanks" else "average of"
                shown = (lambda v: v if isinstance(v, str) else (f"{v:.1%}" if kind == "unusual_blanks" else f"{v:.3f}"))
                out.append({"key": f"{kind}:{col}:campaign{int(campaign)}", "who": "team",
                            "text": f"The {what} {col} went from {shown(before)} (campaign {int(previous)}) to "
                                    f"{shown(now)} (campaign {int(campaign)}){times}. That usually means a data feed "
                                    "changed, not shoppers. Is the data for this campaign complete?"})
                break   # one question per column per campaign
    return out
