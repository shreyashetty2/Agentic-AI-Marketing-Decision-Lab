"""Feature Agent step 2B: 2A's shared table in, one table per model out.

Run from the repo folder:
    python src/features/build_2b.py --processed data/processed

Writes to --processed: model1_table.parquet, model2_table.parquet, model3_table.parquet,
feature_2b_report.md
Stops with [STOP] and the reason if any hard check fails.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # lets `features.*` import when run as a script

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from features.spec import column_roles, load_spec  # noqa: E402

KEYS = ["household_key", "campaign"]
SPEC_2B_PATH = Path(__file__).with_name("feature_2b_spec.yaml")


def load_2b_spec(path: Path = SPEC_2B_PATH) -> dict:
    return yaml.safe_load(path.read_text())


def check(ok: bool, message: str, passed: list[str]) -> None:
    """Stop the run on failure; otherwise print and record the check for the report."""
    if not ok:
        raise SystemExit(f"[STOP] {message}")
    print(f"[OK]   {message}")
    passed.append(message)


def feature_columns(spec_2a: dict) -> list[str]:
    """Every column safe as a model input: context (Data Agent) + profile (2A). Same candidate
    set for all three models -- which of them to actually use is the Modeling Agent's job (step 3).
    """
    roles = column_roles(spec_2a)
    return [c for c, r in roles.items() if r in ("context", "profile")]


def label_columns(model_spec: dict) -> list[str]:
    cols = [model_spec["label"]]
    # sensitivity_label may be a single column (most models) or a list (Model 3 now has two:
    # the display/flyer-only variant, and the redemption-segmentation refinement)
    sensitivity = model_spec.get("sensitivity_label")
    if sensitivity:
        cols.extend(sensitivity if isinstance(sensitivity, list) else [sensitivity])
    if model_spec.get("treatment_column"):
        cols.append(model_spec["treatment_column"])
    return cols


def build_model_table(table: pd.DataFrame, model_id, model_spec: dict,
                       features: list[str], passed: list[str]) -> pd.DataFrame:
    rows = table.query(model_spec["rows"]).reset_index(drop=True)
    labels = label_columns(model_spec)
    out = rows[KEYS + features + labels].copy()

    check(len(out) == model_spec["expected_rows"],
          f"model {model_id}: {len(out):,} rows (expected {model_spec['expected_rows']:,})", passed)
    check(not out.duplicated(KEYS).any(), f"model {model_id}: unique on household_key x campaign", passed)

    # The one leakage rail 2B is responsible for: a model's answer (or its sensitivity/treatment
    # column) can never also appear in the list of columns a model is allowed to train on.
    leaked = [c for c in features if c in labels]
    check(not leaked, f"model {model_id}: label/treatment columns are not duplicated into the feature "
                       f"list {leaked or ''}".strip(), passed)

    if "expected_label_positive" in model_spec:
        positive = int(out[model_spec["label"]].sum())
        check(positive == model_spec["expected_label_positive"],
              f"model {model_id}: {positive:,} positive labels (expected {model_spec['expected_label_positive']:,})",
              passed)

    treatment_col = model_spec.get("treatment_column")
    if treatment_col:
        treated = int(out[treatment_col].sum())
        control = len(out) - treated
        check(treated == model_spec["expected_treated_rows"],
              f"model {model_id}: {treated:,} treated rows (expected {model_spec['expected_treated_rows']:,})", passed)
        check(control == model_spec["expected_control_rows"],
              f"model {model_id}: {control:,} control rows (expected {model_spec['expected_control_rows']:,})", passed)

    return out


def render_report(spec_2b: dict, tables: dict, features: list[str], passed: list[str]) -> str:
    lines = [
        "# Feature Agent 2B: run report",
        "",
        "## Summary",
        "",
        "- Input: Feature Agent 2A's shared table, 75,000 rows.",
        f"- {len(features)} shared context/profile columns are available as model inputs for all "
        "three models; which of them to use is the Modeling Agent's choice (step 3).",
        f"- All {len(passed)} checks passed.",
        "- Never use a model's label, sensitivity label, or treatment column as a model input -- "
        "`src/features/feature_2b_spec.yaml` marks every one.",
        "",
        "## Per-model tables",
        "",
    ]
    for model_id, model_spec in spec_2b["models"].items():
        out = tables[model_id]
        lines += [
            f"### Model {model_id}: {model_spec['name']}",
            "",
            f"- Rows kept: `{model_spec['rows']}` -> {len(out):,} rows, {out.shape[1]} columns.",
            f"- {model_spec['rows_desc'].strip()}",
            f"- Label: `{model_spec['label']}`"
            + (f"; sensitivity: `{model_spec['sensitivity_label']}`" if "sensitivity_label" in model_spec else ""),
        ]
        if model_spec.get("treatment_column"):
            lines.append(f"- Treatment/split column (NOT a feature): `{model_spec['treatment_column']}` "
                          f"({model_spec['expected_treated_rows']:,} treated, "
                          f"{model_spec['expected_control_rows']:,} control)")
        lines.append("")
    lines += ["## Checks", "", *[f"- [OK] {c}" for c in passed], ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    spec_2a = load_spec()
    spec_2b = load_2b_spec()
    features = feature_columns(spec_2a)

    table = pd.read_parquet(args.processed / spec_2b["input"]["table"])
    passed: list[str] = []
    check(len(features) > 0, f"{len(features)} shared context/profile columns available as model inputs", passed)

    tables = {}
    for model_id, model_spec in spec_2b["models"].items():
        print(f"\nModel {model_id}: {model_spec['name']}")
        out = build_model_table(table, model_id, model_spec, features, passed)
        out_path = args.processed / spec_2b["output"]["table_pattern"].format(n=model_id)
        out.to_parquet(out_path, index=False)
        tables[model_id] = out
        print(f"  Saved {len(out):,} rows x {out.shape[1]} columns to {out_path}")

    report_path = args.processed / spec_2b["output"]["report"]
    report_path.write_text(render_report(spec_2b, tables, features, passed))
    print(f"\nSaved {len(passed)} checks. Report: {report_path}")


if __name__ == "__main__":
    main()
