import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "preprocessing"))
import ingest  # noqa: E402


@pytest.fixture
def raw(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "RAW_DIR", tmp_path)
    return tmp_path


def test_coupon_exact_duplicates_removed_and_counted(raw):
    pd.DataFrame({"COUPON_UPC": ["10000089073", "10000089073", "10000089073", "51800000050"],
                  "PRODUCT_ID": [1, 1, 2, 1],          # rows 1-2 identical; row 3 differs only in product
                  "CAMPAIGN": [8, 8, 8, 8]}).to_csv(raw / "coupon.csv", index=False)
    df, report = ingest.load_and_validate("coupon", ingest.TABLE_SPECS["coupon"])
    assert len(df) == 3
    assert report["exact_duplicates_removed"] == 1
    assert report["clean_row_count"] == 3 and report["raw_row_count"] == 4


@pytest.mark.parametrize("name", ["transaction_data", "coupon_redempt"])
def test_repeated_rows_kept_where_they_can_be_real(raw, name):
    spec = ingest.TABLE_SPECS[name]
    row = {c: ("1" if t == "object" else 1) for c, t in spec.columns.items()}
    pd.DataFrame([row, row]).to_csv(raw / spec.filename, index=False)
    df, report = ingest.load_and_validate(name, spec)
    assert len(df) == 2
    assert report["exact_duplicates_removed"] == 0
