"""Unit tests for Feature Agent 2B, on the same tiny hand-built dataset as 2A (conftest.py).

Expected row counts worked out by hand from conftest.py's spine:
  MAILED = {(1,1), (1,2), (2,1)}, REDEEMED = {(1,1)}, c1=TypeB, c2=TypeA (overlaps c1), c3=TypeC.

  Model 1 (mailed_flag==1): (1,1),(1,2),(2,1) -> 3 rows, 1 redeemed.
  Model 2 (mailed & redeemed & not TypeA): only (1,1) -> 1 row.
  Model 3 (TypeB/C, mailed OR clean control):
    c2 (TypeA) excluded entirely.
    c1: (1,1) treated [mailed], (2,1) treated [mailed], (3,1) control [never mailed anything, no overlap].
    c3: (1,3),(2,3),(3,3) all control [not mailed, c3 overlaps nothing].
    -> 6 rows: 2 treated [(1,1),(2,1)], 4 control [(3,1),(1,3),(2,3),(3,3)].
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from features.build_2b import build_model_table, feature_columns, label_columns, load_2b_spec  # noqa: E402
from features.outcomes import compute_outcomes  # noqa: E402
from features.profile import compute_profiles  # noqa: E402
from features.spec import column_roles, load_spec  # noqa: E402

KEYS = ["household_key", "campaign"]


@pytest.fixture
def table_2a(tables, causal):
    """A 2A-shaped table (profiles + outcomes merged onto the spine), built the same way
    build_2a.build() assembles it -- but skipping 2A's own hard-coded real-data shape checks
    (e.g. "spine has 75,000 rows"), which this tiny toy fixture will never satisfy and which
    are already exercised by 2A's own tests. 2B's tests only need a table shaped like 2A's output.
    """
    spec = load_spec()
    spine = tables.spine.sort_values(["household_key", "chrono_rank"]).reset_index(drop=True)
    rows = spine[KEYS].assign(cutoff_day=spine["start_day"] - spec["settings"]["cutoff_gap_days"])
    profiles = compute_profiles(tables, rows).drop(columns="cutoff_day")
    outcomes = compute_outcomes(tables, causal)
    table = spine.merge(profiles, on=KEYS, how="left").merge(outcomes, on=KEYS, how="left")
    return table[list(column_roles(spec))]


@pytest.fixture
def spec_2b():
    return load_2b_spec()


def _toy_spec(spec_2b, model_id, **expected_overrides):
    """A model's real spec (rows filter, label, treatment column -- genuine production logic),
    with its real-data `expected_*` counts replaced by this toy dataset's hand-computed ones.
    """
    model_spec = dict(spec_2b["models"][model_id])
    for key in ("expected_rows", "expected_label_positive", "expected_treated_rows", "expected_control_rows"):
        model_spec.pop(key, None)
    model_spec.update(expected_overrides)
    return model_spec


def test_feature_columns_excludes_outcome_filter_flag(spec_2b):
    features = feature_columns(load_spec())
    assert "household_key" not in features and "campaign" not in features     # keys, not features
    for model_spec in spec_2b["models"].values():
        assert model_spec["label"] not in features
        if "sensitivity_label" in model_spec:
            assert model_spec["sensitivity_label"] not in features
        if model_spec.get("treatment_column"):
            assert model_spec["treatment_column"] not in features


def test_model1_rows_and_label(table_2a, spec_2b):
    features = feature_columns(load_spec())
    toy = _toy_spec(spec_2b, 1, expected_rows=3, expected_label_positive=1)
    out = build_model_table(table_2a, 1, toy, features, [])
    assert set(zip(out.household_key, out.campaign)) == {(1, 1), (1, 2), (2, 1)}
    assert out["redeemed_flag"].sum() == 1


def test_model2_rows_and_label(table_2a, spec_2b):
    features = feature_columns(load_spec())
    toy = _toy_spec(spec_2b, 2, expected_rows=1)
    out = build_model_table(table_2a, 2, toy, features, [])
    assert (out.household_key.iloc[0], out.campaign.iloc[0]) == (1, 1)
    assert out["campaign_product_spend_during"].iloc[0] == pytest.approx(3 + 2 + 1)   # days 110,120,130


def test_model3_rows_treatment_and_control(table_2a, spec_2b):
    features = feature_columns(load_spec())
    toy = _toy_spec(spec_2b, 3, expected_rows=6, expected_treated_rows=2, expected_control_rows=4)
    out = build_model_table(table_2a, 3, toy, features, [])
    treated = set(zip(out[out.mailed_flag == 1].household_key, out[out.mailed_flag == 1].campaign))
    control = set(zip(out[out.mailed_flag == 0].household_key, out[out.mailed_flag == 0].campaign))
    assert treated == {(1, 1), (2, 1)}
    assert control == {(3, 1), (1, 3), (2, 3), (3, 3)}
    assert set(out["campaign_type"]) == {"B", "C"}   # TypeA rows never make it into Model 3's table


def test_label_columns_never_leak_into_features(table_2a, spec_2b):
    """If a label were ever accidentally included in the feature list, build_model_table must refuse."""
    features = feature_columns(load_spec())
    toy = _toy_spec(spec_2b, 1, expected_rows=3, expected_label_positive=1)
    broken_features = features + [toy["label"]]   # simulate the leak
    with pytest.raises(SystemExit, match="label/treatment columns are not duplicated"):
        build_model_table(table_2a, 1, toy, broken_features, [])


def test_label_columns_helper():
    assert label_columns({"label": "x"}) == ["x"]
    assert label_columns({"label": "x", "sensitivity_label": "y"}) == ["x", "y"]
    assert label_columns({"label": "x", "treatment_column": "t"}) == ["x", "t"]
