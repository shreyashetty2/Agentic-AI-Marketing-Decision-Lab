"""Tests for choosing the model in use (src/lead/in_use.py) and passing it to the Modeling Agent."""
import pytest
import yaml

from lead.engine import in_use_args, plan_steps
from lead.in_use import clear, load, set_in_use

IDEAS = [{"id": "r2_p1", "title": "Fit through the average", "reason": "totals", "family": "linear",
          "add_columns": ["spend_per_week_26w"], "approved": True, "status": "ready", "check": "ok"},
         {"id": "r2_p4", "title": "A new column", "new_column": {"name": "x", "definition": "y"}}]


def setup(tmp_path):
    (tmp_path / "model2").mkdir()
    (tmp_path / "model2/proposals.yaml").write_text(yaml.safe_dump(IDEAS))
    return tmp_path, tmp_path / "in_use.yaml"


def test_choosing_an_idea_saves_what_it_is_who_chose_it_and_why(tmp_path):
    ideas, path = setup(tmp_path)
    e = set_in_use(2, "r2_p1", "Pranav Jain", "fixes the low totals", ideas, path, today="2026-10-21")
    assert e == {"id": "r2_p1", "title": "Fit through the average", "reason": "totals", "add_columns": ["spend_per_week_26w"],
                 "family": "linear", "decided_by": "Pranav Jain", "decided_on": "2026-10-21", "why": "fixes the low totals"}
    assert load(path) == {2: e}                                   # the bookkeeping fields (status, check, approved) are not kept


def test_clear_goes_back_to_the_owners_model(tmp_path):
    ideas, path = setup(tmp_path)
    set_in_use(2, "r2_p1", "Pranav", "r", ideas, path)
    assert clear(2, path) and load(path) == {}
    assert not clear(2, path)


def test_an_idea_that_needs_a_new_column_or_does_not_exist_cannot_be_chosen(tmp_path):
    ideas, path = setup(tmp_path)
    with pytest.raises(SystemExit, match="needs a column that does not exist"):
        set_in_use(2, "r2_p4", "Pranav", "r", ideas, path)
    with pytest.raises(SystemExit, match="nothing called 'r9'"):
        set_in_use(2, "r9", "Pranav", "r", ideas, path)
    with pytest.raises(SystemExit, match="name of the person"):
        set_in_use(2, "r2_p1", " ", "r", ideas, path)


def test_the_engine_tells_the_modeling_agent_only_for_the_chosen_model(tmp_path):
    from lead.engine import load_spec
    ideas, path = setup(tmp_path)
    assert in_use_args(2, path) == ""
    set_in_use(2, "r2_p1", "Pranav", "r", ideas, path)
    assert in_use_args(2, path) == f"--in-use {path}" and in_use_args(1, path) == ""
    steps = plan_steps(load_spec(), [1, 2], in_use_path=path)
    modeling = {s.model: s.runs[0] for s in steps if s.name == "modeling"}
    assert modeling[2].endswith(f"--in-use {path}") and "--in-use" not in modeling[1]


def test_the_real_models_in_use_file_loads():
    from lead.in_use import IN_USE_PATH
    assert isinstance(load(IN_USE_PATH), dict)


def test_an_option_the_modeling_agent_always_tests_can_be_chosen_too(tmp_path):
    import json
    ideas, path = setup(tmp_path)
    (tmp_path / "model2/manifest.json").write_text(json.dumps({
        "final": {"task": "regress"},
        "candidates": {"usual_only_linear": {"family": "linear", "features": ["usual_spend"]}}}))
    e = set_in_use(2, "usual_only_linear", "Pranav", "totals pass", ideas, path, today="2026-10-21")
    assert e["option"] is True and e["id"] == "usual_only_linear"
    assert e["title"] == "linear regression (one weighted formula, giving dollars) on 1 column"
    with pytest.raises(SystemExit, match="options: \\['usual_only_linear'\\]"):
        set_in_use(2, "nope", "Pranav", "r", ideas, path)
