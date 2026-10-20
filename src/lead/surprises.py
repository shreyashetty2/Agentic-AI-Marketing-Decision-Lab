"""Lead Agent surprise checks: what no agent's rules cover, found by comparing the new data with what came before.

1. Never-seen values: a product department, or a campaign type, that appears for the first time in the data
   that arrived since the last run. Nobody has decided how to treat it yet (does it count as household spend?).
2. Unusual moves: an everyday-shopping column (spend, trips, coupon use, ...) whose average over all households
   moves between one campaign and the next by far more than it usually does. That usually means a data feed
   broke (for example an empty coupon file), not that shoppers changed overnight. "Usual" is learned from the
   campaigns before: move_sd times the standard deviation of their campaign-to-campaign moves, and at least
   min_relative of the column's level, so tiny moves are never flagged. On our two years no campaign trips it;
   an 80% drop in coupon use would be about 40 times the usual move.

The new data is everything after `since` up to `upto` (days). With no earlier run (since=None) there is
nothing to compare with: the checks only learn. Each finding is a question {key, who, text}; its key names the
value or column, so an answered question is never asked again.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def never_seen(processed: Path, since: int | None, upto: int, known: set[str], plant: str | None = None) -> list[dict]:
    """known: values a person already decided about (the lead agent's rules). plant: add a made-up department to
    the new data, to test that the check stops the run (said plainly in the question)."""
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
        if dept in before or f"department:{dept}" in known:
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
        if ctype not in seen_types and f"campaign_type:{ctype}" not in known:
            out.append({"key": f"campaign_type:{ctype}", "who": "team",
                        "text": f"A campaign type never seen before started since day {since}: type '{ctype}' "
                                f"(campaigns {sorted(int(c) for c in group['campaign'])}). The models were only "
                                "trained on the types before it. Should they score it, or wait until it has history?"})
    return out


def unusual_moves(processed: Path, since: int | None, upto: int, cfg: dict, known: set[str]) -> list[dict]:
    """cfg: columns, move_sd, min_relative, min_history (moves needed before 'usual' means anything)."""
    if since is None or upto <= since:
        return []
    cols = cfg["columns"]
    table = pd.read_parquet(processed / "features_household_campaign.parquet", columns=["campaign", "start_day", *cols])
    means = table.groupby("campaign")[[*cols, "start_day"]].mean().sort_values("start_day")
    means = means[means["start_day"] <= upto]
    moves = means[cols].diff()
    out = []
    for position, (campaign, row) in enumerate(means.iterrows()):
        if row["start_day"] <= since or position == 0:
            continue
        past = moves.iloc[1:position]
        if len(past) < cfg["min_history"]:
            continue
        usual = past.std()
        previous = means.iloc[position - 1]
        for col in cols:
            move = row[col] - previous[col]
            key = f"unusual_move:{col}:campaign{int(campaign)}"
            relative = abs(move) / abs(previous[col]) if previous[col] else np.inf
            if abs(move) > cfg["move_sd"] * usual[col] and relative > cfg["min_relative"] and key not in known:
                out.append({"key": key, "who": "team",
                            "text": f"The average of {col} moved from {previous[col]:.3f} (campaign "
                                    f"{int(means.index[position - 1])}) to {row[col]:.3f} (campaign {int(campaign)}), "
                                    f"{abs(move) / usual[col]:.0f} times its usual move between campaigns "
                                    f"({usual[col]:.3f}). That usually means a data feed changed, not shoppers. "
                                    "Is the data for this campaign complete?"})
    return out
