"""
Ingestion agent — step 1 of the pipeline.

Loads the 8 raw dunnhumby "Complete Journey" CSVs, validates each
against the expected schema, casts dtypes, runs sanity checks, and
writes clean parquet tables to data/processed/.

Run:
    python agents/ingestion/ingest.py

Input : data/raw/*.csv   (place the raw dunnhumby files here)
Output: data/processed/*.parquet
        data/processed/ingestion_report.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TableSpec:
    """Expected shape of one raw table."""
    filename: str
    columns: dict[str, str]          # column_name -> pandas dtype
    key_columns: list[str]           # columns that together should be unique (empty = not enforced)
    not_null_columns: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Table specs, taken from the dunnhumby "Complete Journey" user guide.
# NOTE: hh_demographic uses the column names found in our actual file
# (classification_1..5, homeowner_desc, kid_category_desc), verified on
# the real data. The guide's page-4 table is wrong (it lists the
# transaction_data columns instead).
# ---------------------------------------------------------------------------

TABLE_SPECS: dict[str, TableSpec] = {
    "transaction_data": TableSpec(
        filename="transaction_data.csv",
        columns={
            "household_key": "int64",
            "BASKET_ID": "int64",
            "DAY": "int64",
            "PRODUCT_ID": "int64",
            "QUANTITY": "int64",
            "SALES_VALUE": "float64",
            "STORE_ID": "int64",
            "RETAIL_DISC": "float64",
            "TRANS_TIME": "int64",
            "WEEK_NO": "int64",
            "COUPON_DISC": "float64",
            "COUPON_MATCH_DISC": "float64",
        },
        key_columns=[],  # a basket can have multiple line items; no natural single key
        not_null_columns=["household_key", "PRODUCT_ID", "BASKET_ID"],
    ),
    "hh_demographic": TableSpec(
        filename="hh_demographic.csv",
        columns={
            "classification_1": "object",
            "classification_2": "object",
            "classification_3": "object",
            "classification_4": "object",
            "classification_5": "object",
            "homeowner_desc": "object",
            "kid_category_desc": "object",
            "household_key": "int64",
        },
        key_columns=["household_key"],
        not_null_columns=["household_key"],
    ),
    "product": TableSpec(
        filename="product.csv",
        columns={
            "PRODUCT_ID": "int64",
            "MANUFACTURER": "object",
            "DEPARTMENT": "object",
            "BRAND": "object",
            "COMMODITY_DESC": "object",
            "SUB_COMMODITY_DESC": "object",
            "CURR_SIZE_OF_PRODUCT": "object",
        },
        key_columns=["PRODUCT_ID"],
        not_null_columns=["PRODUCT_ID"],
    ),
    "campaign_table": TableSpec(
        filename="campaign_table.csv",
        columns={
            "DESCRIPTION": "object",
            "household_key": "int64",
            "CAMPAIGN": "int64",
        },
        key_columns=["household_key", "CAMPAIGN"],
        not_null_columns=["household_key", "CAMPAIGN"],
    ),
    "campaign_desc": TableSpec(
        filename="campaign_desc.csv",
        columns={
            "DESCRIPTION": "object",
            "CAMPAIGN": "int64",
            "START_DAY": "int64",
            "END_DAY": "int64",
        },
        key_columns=["CAMPAIGN"],
        not_null_columns=["CAMPAIGN", "START_DAY", "END_DAY"],
    ),
    "coupon": TableSpec(
        filename="coupon.csv",
        columns={
            "COUPON_UPC": "object",
            "PRODUCT_ID": "int64",
            "CAMPAIGN": "int64",
        },
        key_columns=[],  # one coupon can apply to many products
        not_null_columns=["COUPON_UPC", "PRODUCT_ID", "CAMPAIGN"],
    ),
    "coupon_redempt": TableSpec(
        filename="coupon_redempt.csv",
        columns={
            "household_key": "int64",
            "DAY": "int64",
            "COUPON_UPC": "object",
            "CAMPAIGN": "int64",
        },
        key_columns=[],
        not_null_columns=["household_key", "COUPON_UPC", "CAMPAIGN"],
    ),
    "causal_data": TableSpec(
        filename="causal_data.csv",
        columns={
            "PRODUCT_ID": "int64",
            "STORE_ID": "int64",
            "WEEK_NO": "int64",
            "display": "object",
            "mailer": "object",
        },
        key_columns=["PRODUCT_ID", "STORE_ID", "WEEK_NO"],
        not_null_columns=["PRODUCT_ID", "STORE_ID", "WEEK_NO"],
    ),
}


def load_and_validate(name: str, spec: TableSpec) -> tuple[pd.DataFrame, dict]:
    """Load one raw table, validate it against its spec, cast dtypes.
    Returns the cleaned dataframe and a report dict for this table."""

    path = RAW_DIR / spec.filename
    report: dict = {"table": name, "file": str(path)}

    if not path.exists():
        report["status"] = "MISSING"
        return pd.DataFrame(), report

    header = pd.read_csv(path, nrows=0).columns
    str_cols = {c: str for c in header if c.lower() == "coupon_upc"}
    df = pd.read_csv(path, dtype=str_cols)
    report["raw_row_count"] = len(df)
    report["raw_columns"] = list(df.columns)

    # 1. Column presence check (case-insensitive match, since raw dunnhumby
    #    files mix upper/lower case across tables)
    lower_map = {c.lower(): c for c in df.columns}
    missing_cols = [c for c in spec.columns if c.lower() not in lower_map]
    unexpected_cols = [c for c in df.columns if c.lower() not in [e.lower() for e in spec.columns]]
    report["missing_columns"] = missing_cols
    report["unexpected_columns"] = unexpected_cols

    if missing_cols:
        report["status"] = "SCHEMA_MISMATCH"
        return df, report

    # normalize to the expected column names
    df = df.rename(columns={lower_map[c.lower()]: c for c in spec.columns})
    df = df[list(spec.columns.keys())]

    # 2. Dtype casting
    cast_errors = {}
    for col, dtype in spec.columns.items():
        try:
            df[col] = df[col].astype(dtype)
        except (ValueError, TypeError) as e:
            cast_errors[col] = str(e)
    report["cast_errors"] = cast_errors

    # 3. Null checks on required columns
    null_counts = {c: int(df[c].isna().sum()) for c in spec.not_null_columns}
    report["null_counts"] = {c: n for c, n in null_counts.items() if n > 0}

    # 4. Key uniqueness check
    if spec.key_columns:
        dup_count = int(df.duplicated(subset=spec.key_columns).sum())
        report["duplicate_key_rows"] = dup_count
    else:
        report["duplicate_key_rows"] = None

    # 5. Drop fully-null rows introduced by cast failures, then finalize
    df = df.dropna(subset=spec.not_null_columns) if spec.not_null_columns else df
    report["clean_row_count"] = len(df)
    report["rows_dropped"] = report["raw_row_count"] - report["clean_row_count"]

    report["status"] = "OK" if not cast_errors and not report["null_counts"] else "OK_WITH_WARNINGS"
    return df, report


def run_ingestion() -> dict:
    full_report = {"tables": []}
    ok = True

    for name, spec in TABLE_SPECS.items():
        df, report = load_and_validate(name, spec)
        full_report["tables"].append(report)

        if report["status"] in ("MISSING", "SCHEMA_MISMATCH"):
            print(f"[FAIL] {name}: {report['status']}")
            ok = False
            continue

        out_path = PROCESSED_DIR / f"{name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"[OK]   {name}: {report['clean_row_count']} rows -> {out_path}")

        if report["status"] == "OK_WITH_WARNINGS":
            print(f"       warnings: null_counts={report['null_counts']}, "
                  f"cast_errors={report['cast_errors']}")

    full_report["all_tables_loaded"] = ok
    report_path = PROCESSED_DIR / "ingestion_report.json"
    report_path.write_text(json.dumps(full_report, indent=2))
    print(f"\nReport written to {report_path}")

    return full_report


if __name__ == "__main__":
    result = run_ingestion()
    if not result["all_tables_loaded"]:
        sys.exit(1)
