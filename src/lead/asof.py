"""Lead Agent replay: the model tables as they would have looked on a given day.

Our dataset never changes, so the lead agent's daily runs are simulated by stepping through the two years
in order. Every feature column is computed only from data before its campaign started (Feature Agent 2A),
so keeping just the campaigns that had FINISHED by day X gives the same tables a rebuild on data up to
day X would give. Campaigns still running on day X are left out: their answers (who redeemed, what they
spent) are not known yet.

The tables depend only on which campaigns had finished, so each replay folder is named after the day the
latest of them ended: two replay days with no newly finished campaign share one folder, and the lead agent
skips retraining for the second (retrain only when a campaign finishes).

The one gap, said plainly: the Data Agent's outlier cut-offs were learned from all two years. They only
set flags on receipts; no model input depends on them.

Usage:  python src/lead/asof.py --day 660 --out <folder>
Writes model{1,2,3}_table.parquet and asof.json to --out.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data/processed"
MODELS = (1, 2, 3)
# The Modeling Agent's first test campaign starts on this day; read from its own instructions.
FIRST_TEST_DAY = yaml.safe_load((ROOT / "src/modeling/modeling_spec.yaml").read_text())["defaults"]["first_test_day"]
MIN_TEST_CAMPAIGNS = 3        # fewer finished test campaigns than this: nothing meaningful to test on


def calendar(processed: Path = PROCESSED) -> pd.DataFrame:
    """One row per campaign: type, start and end day (from the Data Agent's spine)."""
    spine = pd.read_parquet(processed / "spine_household_campaign.parquet",
                            columns=["campaign", "campaign_type", "start_day", "end_day"])
    return spine.drop_duplicates("campaign").sort_values(["start_day", "campaign"]).reset_index(drop=True)


def finished_by(cal: pd.DataFrame, day: int) -> list[int]:
    return sorted(int(c) for c in cal.loc[cal["end_day"] <= day, "campaign"])


def folder_name(cal: pd.DataFrame, day: int) -> str:
    """Named after the end day of the latest finished campaign, so days with the same campaigns share a folder."""
    ended = cal.loc[cal["end_day"] <= day, "end_day"]
    if ended.empty:
        raise SystemExit(f"[STOP] no campaign had finished by day {day}; the first one ends on day {int(cal['end_day'].min())}")
    return f"ended_by_day_{int(ended.max())}"


def describe(cal: pd.DataFrame, day: int) -> dict:
    done = cal[cal["end_day"] <= day]
    running = cal[(cal["start_day"] <= day) & (cal["end_day"] > day)]
    newest_end = int(done["end_day"].max())
    return {"day": day, "folder": folder_name(cal, day),
            "finished": [int(c) for c in done["campaign"]],
            "newest_finished": sorted(int(c) for c in done.loc[done["end_day"] == newest_end, "campaign"]),
            "newest_end_day": newest_end,
            "running": [int(c) for c in running["campaign"]],
            "not_started": int((cal["start_day"] > day).sum()),
            "test_campaigns": int((done["start_day"] >= FIRST_TEST_DAY).sum())}


def build(day: int, out: Path, processed: Path = PROCESSED) -> dict:
    cal = calendar(processed)
    info = describe(cal, day)
    if info["test_campaigns"] < MIN_TEST_CAMPAIGNS:
        raise SystemExit(f"[STOP] by day {day} only {info['test_campaigns']} campaign(s) that started on or after day "
                         f"{FIRST_TEST_DAY} had finished; the replay needs at least {MIN_TEST_CAMPAIGNS} to test on")
    out.mkdir(parents=True, exist_ok=True)
    keep = set(info["finished"])
    info["rows"] = {}
    for n in MODELS:
        table = pd.read_parquet(processed / f"model{n}_table.parquet")
        cut = table[table["campaign"].isin(keep)].reset_index(drop=True)
        cut.to_parquet(out / f"model{n}_table.parquet", index=False)
        info["rows"][f"model{n}"] = len(cut)
    (out / "asof.json").write_text(json.dumps(info, indent=2))
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Lead Agent replay: the model tables as of a given day")
    parser.add_argument("--day", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--processed", type=Path, default=PROCESSED)
    args = parser.parse_args()
    info = build(args.day, args.out, args.processed)
    print(f"As of day {info['day']}: {len(info['finished'])} campaigns finished (newest: {info['newest_finished']}, "
          f"day {info['newest_end_day']}); {len(info['running'])} running; rows {info['rows']}")


if __name__ == "__main__":
    main()
