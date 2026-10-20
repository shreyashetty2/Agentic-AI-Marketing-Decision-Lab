"""Tests for the Lead Agent rules (src/lead/rules.py): nothing is saved until a person approves it."""
import pytest
import yaml

from lead.rules import answered, approve, load_rules, propose

Q = "data:-:department:PHARMACY"


def paths(tmp_path):
    return {"drafts_path": tmp_path / "drafts.json"}, tmp_path / "rules.yaml"


def test_a_draft_is_not_a_rule(tmp_path):
    d, rules_path = paths(tmp_path)
    propose(Q, "no, set it apart", "PHARMACY lines are not household spend.", **d)
    assert load_rules(rules_path) == []


def test_approving_saves_the_rule_with_who_and_when(tmp_path):
    d, rules_path = paths(tmp_path)
    draft = propose(Q, "no, set it apart", "PHARMACY lines are not household spend.", **d)
    saved = approve(draft, "Pranav", rules_path=rules_path, today="2026-10-21", **d)
    assert saved == {"question": Q, "rule": "PHARMACY lines are not household spend.", "answer": "no, set it apart",
                     "decided_by": "Pranav", "decided_on": "2026-10-21"}
    assert load_rules(rules_path) == [saved]
    assert yaml.safe_load(rules_path.read_text())["rules"][0]["decided_by"] == "Pranav"


def test_the_persons_edit_is_what_is_saved(tmp_path):
    d, rules_path = paths(tmp_path)
    draft = propose(Q, "no", "PHARMACY is not spend.", **d)
    saved = approve(draft, "Pranav", rule="PHARMACY lines are set apart, like gas.", rules_path=rules_path, **d)
    assert saved["rule"] == "PHARMACY lines are set apart, like gas."


def test_a_new_answer_to_the_same_question_replaces_the_old_rule(tmp_path):
    d, rules_path = paths(tmp_path)
    approve(propose(Q, "no", "old", **d), "Pranav", rules_path=rules_path, **d)
    approve(propose(Q, "yes", "new", **d), "Shira", rules_path=rules_path, **d)
    assert [r["rule"] for r in load_rules(rules_path)] == ["new"]


def test_a_rule_needs_a_person_and_an_existing_draft(tmp_path):
    d, rules_path = paths(tmp_path)
    draft = propose(Q, "no", "r", **d)
    with pytest.raises(SystemExit, match="name of the person"):
        approve(draft, " ", rules_path=rules_path, **d)
    with pytest.raises(SystemExit, match="no draft 'd9'"):
        approve("d9", "Pranav", rules_path=rules_path, **d)
    for bad in ("PHARMACY", "data:department:PHARMACY", "data:-:"):
        with pytest.raises(SystemExit, match="not a question id"):
            propose(bad, "no", "r", **d)


def test_lookups(tmp_path):
    d, rules_path = paths(tmp_path)
    approve(propose(Q, "no", "r", **d), "Pranav", rules_path=rules_path, **d)
    rules = load_rules(rules_path)
    assert set(answered(rules)) == {Q}


def test_the_real_rules_file_loads():
    from lead.rules import RULES_PATH
    assert isinstance(load_rules(RULES_PATH), list)


def test_a_new_draft_never_takes_the_id_of_one_still_waiting(tmp_path):
    d, rules_path = paths(tmp_path)
    first, second = propose(Q, "a", "r1", **d), propose("data:-:department:X", "b", "r2", **d)
    approve(first, "Pranav", rules_path=rules_path, **d)
    third = propose("data:-:department:Y", "c", "r3", **d)
    assert third not in (first, second)
    saved = approve(second, "Pranav", rules_path=rules_path, **d)
    assert saved["rule"] == "r2"


def test_department_names_with_colons_and_spaces_survive(tmp_path):
    d, rules_path = paths(tmp_path)
    question = "data:-:department:MISC: SALES TRAN"
    approve(propose(question, "no", "r", **d), "Pranav", rules_path=rules_path, **d)
    assert set(answered(load_rules(rules_path))) == {question}
