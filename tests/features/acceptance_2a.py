"""Real-data acceptance checks for step 2A (the done_when list in feature_2a_spec.yaml).

Slow (about a minute), so it is a script rather than part of the pytest suite:
    python tests/features/acceptance_2a.py --processed data/processed --table data/processed/features_household_campaign.parquet
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd  # noqa: E402

from features.data import Tables, load_tables  # noqa: E402
from features.leak_test import columns_that_leak  # noqa: E402
from features.outcomes import compute_outcomes  # noqa: E402
from features.profile import compute_profiles, features  # noqa: E402
from features.spec import load_spec  # noqa: E402

KEYS = ["household_key", "campaign"]
results = []


def check(ok: bool, message: str) -> None:
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {message}")


def with_transactions(tables: Tables, transactions: pd.DataFrame, redemptions: pd.DataFrame | None = None) -> Tables:
    """Same tables, different purchases (and optionally redemptions); everything else shared, not copied."""
    return Tables(tables.spine, transactions, tables.coupons,
                  tables.redemptions if redemptions is None else redemptions, tables.products, tables.demographics)


def future_scramble_changes_nothing(tables: Tables, table: pd.DataFrame, gap: int, profile_cols: list[str]) -> None:
    rows = table[KEYS].assign(cutoff_day=table["start_day"] - gap)
    leaks = columns_that_leak(tables, rows, table, profile_cols)
    check(not leaks, f"scrambling everything on/after each cutoff leaves all profile columns unchanged "
                     f"({rows.cutoff_day.nunique()} cutoffs tested){' LEAKS: ' + str(leaks) if leaks else ''}")


def window_scramble_changes_outcomes(tables: Tables, table: pd.DataFrame, causal: Path) -> None:
    tx = tables.transactions.copy()
    in_window = tx["day"] >= int(table["start_day"].min())
    tx.loc[in_window, "sales_value"] = tx.loc[in_window, "sales_value"] * 2 + 1
    after = compute_outcomes(with_transactions(tables, tx), causal).set_index(KEYS)
    before = table.set_index(KEYS)["campaign_product_spend_during"]
    nonzero = before[before > 0]
    moved = after["campaign_product_spend_during"].reindex(nonzero.index) != nonzero
    check(bool(moved.all()), f"changing in-window purchases changes all {len(nonzero):,} non-zero campaign spend outcomes")


def known_numbers(table: pd.DataFrame) -> None:
    mailed = table[table["mailed_flag"] == 1]
    check(int(mailed["redeemed_flag"].sum()) == 889, "889 redeemed mailed rows")
    both = int((table["bought_campaign_product_excl_display_flyer"] == 1).sum())
    display = int((table["bought_campaign_product_excl_display"] == 1).sum())
    check(both == 11838, f"Model 3 outcome (display + flyer excluded) = 1 on {both:,} rows (expected 11,838)")
    check(display == 13235, f"Model 3 outcome (display only excluded) = 1 on {display:,} rows (expected 13,235)")
    hh208 = table[(table["household_key"] == 208) & (table["campaign"] == 18)].iloc[0]
    check(round(hh208["campaign_product_spend_during"], 2) == 209.58,
          f"household 208, campaign 18 spend on campaign products = {hh208['campaign_product_spend_during']:.2f} (expected 209.58)")


def single_household_matches_table(tables: Tables, table: pd.DataFrame, gap: int, profile_cols: list[str]) -> None:
    sample = table.sample(25, random_state=0)
    mismatches = 0
    for _, row in sample.iterrows():
        one = features(tables, row["household_key"], row["campaign"], row["start_day"] - gap)
        mismatches += any(not (pd.isna(one[c]) and pd.isna(row[c])) and one[c] != row[c] for c in profile_cols)
    check(mismatches == 0, f"features() for one household matches the table on 25 random rows ({mismatches} mismatches)")
    today = features(tables, household=208, campaign=18, cutoff_day=700)
    check(today.notna()["spend_per_week_8w"], "features() works for a cutoff that is not a campaign's (day 700)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--table", type=Path, default=Path("data/processed/features_household_campaign.parquet"))
    parser.add_argument("--causal", type=Path, default=Path("data/raw/causal_data.csv"))
    args = parser.parse_args()
    spec = load_spec()
    gap = spec["settings"]["cutoff_gap_days"]
    profile_cols = list(spec["profile"]) + list(spec["flag"])     # the before-cutoff columns 2A builds
    tables, table = load_tables(args.processed), pd.read_parquet(args.table)
    known_numbers(table)
    single_household_matches_table(tables, table, gap, profile_cols)
    future_scramble_changes_nothing(tables, table, gap, profile_cols)
    window_scramble_changes_outcomes(tables, table, args.causal)
    print(f"\n{sum(results)} of {len(results)} acceptance checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
