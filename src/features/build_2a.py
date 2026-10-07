"""Feature Agent step 2A: Data Agent tables in, one shared household x campaign table out.

Run from the repo folder:
    python src/features/build_2a.py --processed data/processed --causal data/raw/causal_data.csv

Writes to --processed: features_household_campaign.parquet and feature_2a_health.csv
Stops with [STOP] and the reason if any hard check fails.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # lets `features.*` import when run as a script

import pandas as pd  # noqa: E402

from features.data import Tables, load_tables  # noqa: E402
from features.health import column_health, unexplained_blanks  # noqa: E402
from features.leak_test import columns_that_leak  # noqa: E402
from features.outcomes import compute_outcomes  # noqa: E402
from features.profile import compute_profiles  # noqa: E402
from features.spec import column_roles, load_spec  # noqa: E402

KEYS = ["household_key", "campaign"]


def check(ok: bool, message: str) -> None:
    if not ok:
        raise SystemExit(f"[STOP] {message}")
    print(f"[OK]   {message}")


def check_inputs(tables: Tables, spec: dict) -> None:
    spine, expected = tables.spine, spec["input"]["expected"]
    mailed = spine[spine["mailed_flag"] == 1]
    check(len(spine) == expected["rows"], f"spine has {len(spine):,} rows (expected {expected['rows']:,})")
    check(len(mailed) == expected["mailed_rows"], f"{len(mailed):,} mailed rows (expected {expected['mailed_rows']:,})")
    redeemed = int(mailed["redeemed_flag"].sum())
    check(redeemed == expected["redeemed_mailed_rows"],
          f"{redeemed:,} redeemed mailed rows (expected {expected['redeemed_mailed_rows']:,})")
    tx = tables.transactions
    week_ok = bool(((tx["day"] + 8) // 7 == tx["week_no"]).all())     # (day + 8) // 7 = ceil((day + 2) / 7)
    check(week_ok, "week_no = ceil((day + 2) / 7) on every transaction")


def check_output(table: pd.DataFrame, spine: pd.DataFrame, spec: dict) -> None:
    roles = column_roles(spec)
    check(list(table.columns) == list(roles), f"output has exactly the {len(roles)} columns listed in the spec")
    check(len(table) == len(spine) and not table.duplicated(KEYS).any(),
          f"{len(table):,} rows, one per household x campaign")
    passthrough = list(spec["passthrough"])
    same = table[passthrough].reset_index(drop=True).equals(spine[passthrough].reset_index(drop=True))
    check(same, "Data Agent columns carried over unchanged")
    numeric = [c for c, r in roles.items() if r in ("profile", "outcome") and pd.api.types.is_numeric_dtype(table[c])]
    negative = [c for c in numeric if (table[c] < 0).any()]
    check(not negative, f"no negative values {negative or ''}".strip())
    shares = [c for c in numeric if "share" in c or "rate" in c]
    outside = [c for c in shares if ((table[c] < 0) | (table[c] > 1)).any()]
    check(not outside, f"every share and rate between 0 and 1 {outside or ''}".strip())
    unexplained = unexplained_blanks(table, spec)
    check(not unexplained, f"every blank has a known reason {unexplained or ''}".strip())


def build(tables: Tables, causal, spec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    check_inputs(tables, spec)
    spine = tables.spine.sort_values(["household_key", "chrono_rank"]).reset_index(drop=True)
    rows = spine[KEYS].assign(cutoff_day=spine["start_day"] - spec["settings"]["cutoff_gap_days"])
    print("Building profile columns...")
    profiles = compute_profiles(tables, rows).drop(columns="cutoff_day")
    print("Building outcome columns...")
    outcomes = compute_outcomes(tables, causal)
    table = spine.merge(profiles, on=KEYS, how="left").merge(outcomes, on=KEYS, how="left")
    table = table[list(column_roles(spec))]
    check_output(table, spine, spec)

    profile_columns = list(spec["profile"])
    health = column_health(table, profile_columns, spec["settings"]["leak_alarm"])
    health["decision"], health["note"] = "accept", ""
    alarms = health.index[health["leak_alarm"]].tolist()
    if alarms:
        # A high score alone is not a leak: honest habits can score high. The scramble test decides.
        print(f"High score on {alarms}; running the leak test on them...")
        profile_rows = table[KEYS].assign(cutoff_day=rows["cutoff_day"])
        leaks = columns_that_leak(tables, profile_rows, table, alarms)
        check(not leaks, f"leak test: scrambling the future does not change {alarms}"
                         f"{' (LEAKS: ' + str(leaks) + ')' if leaks else ''}")
        health.loc[alarms, "decision"] = "accept_with_note"
        health.loc[alarms, "note"] = ("High score from genuine habit, not leakage: "
                                      "unchanged when all data after the cutoff is scrambled.")
    return table, health


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--causal", type=Path, default=Path("data/raw/causal_data.csv"))
    args = parser.parse_args()
    spec = load_spec()
    table, health = build(load_tables(args.processed), args.causal, spec)
    out = args.processed / spec["output"]["table"]
    table.to_parquet(out, index=False)
    health.to_csv(args.processed / "feature_2a_health.csv", float_format="%.4f")
    print(f"\nSaved {len(table):,} rows x {table.shape[1]} columns to {out}")


if __name__ == "__main__":
    main()
