"""The definitive leak test: scramble everything on or after each cutoff and rebuild.

A profile column that changes cannot be trusted; one that does not change provably uses only the past.
"""
import numpy as np
import pandas as pd

from features.data import Tables
from features.profile import compute_profiles


def scramble_future(tables: Tables, cutoff: int, rng: np.random.Generator) -> Tables:
    """Same tables with every purchase and redemption on or after cutoff randomised."""
    tx = tables.transactions.copy()
    future = tx["day"] >= cutoff
    tx.loc[future, "sales_value"] = rng.uniform(0, 500, future.sum())
    tx.loc[future, ["retail_disc", "coupon_disc"]] = -1.0
    tx.loc[future, "product_id"] = rng.permutation(tx.loc[future, "product_id"].to_numpy())
    red = tables.redemptions.copy()
    late = red["day"] >= cutoff
    red.loc[late, "household_key"] = rng.permutation(red.loc[late, "household_key"].to_numpy())
    return Tables(tables.spine, tx, tables.coupons, red, tables.products, tables.demographics)


def columns_that_leak(tables: Tables, rows: pd.DataFrame, built: pd.DataFrame, columns: list[str],
                      seed: int = 0) -> dict[str, list[int]]:
    """rows: household_key, campaign, cutoff_day; built: the profile values already computed for those rows.
    Returns column -> cutoffs at which scrambling the future changed it (empty dict = no leak)."""
    rng = np.random.default_rng(seed)
    leaks: dict[str, list[int]] = {}
    for cutoff, sub in rows.groupby("cutoff_day"):
        after = compute_profiles(scramble_future(tables, int(cutoff), rng), sub)
        for column in columns:
            if not built.loc[sub.index, column].reset_index(drop=True).equals(after[column].reset_index(drop=True)):
                leaks.setdefault(column, []).append(int(cutoff))
    return leaks
