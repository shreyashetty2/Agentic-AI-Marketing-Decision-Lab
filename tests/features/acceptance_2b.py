"""Real-data acceptance checks for step 2B (the done_when list in docs/agent_design/feature_agent_2b_design.md).

    python tests/features/acceptance_2b.py --processed data/processed
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd  # noqa: E402

from features.build_2b import feature_columns, label_columns, load_2b_spec  # noqa: E402
from features.spec import load_spec  # noqa: E402

KEYS = ["household_key", "campaign"]
results = []


def check(ok: bool, message: str) -> None:
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    spec_2b = load_2b_spec()
    features = feature_columns(load_spec())
    tables = {n: pd.read_parquet(args.processed / spec_2b["output"]["table_pattern"].format(n=n))
              for n in spec_2b["models"]}

    for model_id, model_spec in spec_2b["models"].items():
        out = tables[model_id]
        check(len(out) == model_spec["expected_rows"],
              f"model {model_id}: {len(out):,} rows (expected {model_spec['expected_rows']:,})")
        check(not out.duplicated(KEYS).any(), f"model {model_id}: unique on household_key x campaign")
        labels = label_columns(model_spec)
        check(all(c in out.columns for c in labels), f"model {model_id}: all label/treatment columns present {labels}")
        check(all(c in out.columns for c in features), f"model {model_id}: all {len(features)} candidate features present")
        # The rail that matters most: re-derive the model's own table columns and confirm no OTHER
        # model's label or filter column leaked in (e.g. Model 1's table must not carry Model 3's outcome).
        other_labels = {c for other_id, other_spec in spec_2b["models"].items() if other_id != model_id
                        for c in label_columns(other_spec)} - set(labels)
        leaked = other_labels & set(out.columns)
        check(not leaked, f"model {model_id}: no other model's label/filter columns present {leaked or ''}".strip())

    check(int(tables[1]["redeemed_flag"].sum()) == 889, "Model 1: 889 redeemed of 7,208 mailed rows")
    check(int(tables[3]["mailed_flag"].sum()) == 3229, "Model 3: 3,229 treated rows")
    check(len(tables[3]) - int(tables[3]["mailed_flag"].sum()) == 40167, "Model 3: 40,167 control rows")
    check(len(tables[2]) == 254, "Model 2: 254 redeemed TypeB/C rows")

    print(f"\n{sum(results)} of {len(results)} acceptance checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
