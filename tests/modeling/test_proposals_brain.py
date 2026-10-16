"""Tests for proposals (src/modeling/proposals.py), the brain (brain.py) and locked campaigns (agent.py)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from modeling import brain, model1_baseline, model2_baseline, model3_uplift
from modeling.agent import load_modeling_spec, resolve_candidates
from modeling.proposals import approved_candidates, changes, ledger_upsert, load_proposals, validate

SPEC = load_modeling_spec()
DEFAULT = {"family": "forest", "params": {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 20, "random_state": 42},
           "features": ["spend", "trips"]}


def _ctx(task="classify", existing=None):
    table = pd.DataFrame({"spend": [1.0, 2.0], "trips": [1.0, 1.0], "recency": [3.0, np.nan], "demo": ["a", None],
                          "odd_blank": [np.nan, 1.0], "start_day": [1, 2], "label": [0, 1], "mailed": [0, 1],
                          "outcome_col": [0.0, 1.0]})
    roles = {"spend": "profile", "trips": "profile", "recency": "profile", "demo": "profile", "odd_blank": "profile",
             "start_day": "context", "label": "outcome", "mailed": "filter", "outcome_col": "outcome"}
    return {"task": task, "default": DEFAULT, "existing": existing or {"own": DEFAULT},
            "dtypes": table.dtypes.to_dict(), "has_blanks": table.isna().any().to_dict(), "roles": roles,
            "blank_reasons": {"recency": ["no_history"], "demo": ["no_demographics"], "odd_blank": ["not_redeemed"]},
            "defaults": SPEC["defaults"] | {"never_inputs": ["start_day"], "derived_inputs": []},
            "label": "label", "treatment": "mailed" if task == "uplift" else None, "spec": SPEC}


# --------------------------------------------------------------- what a candidate changes
def test_changes_names_each_kind_of_change():
    assert changes({**DEFAULT, "features": ["spend", "trips", "x"]}, DEFAULT) == ["columns"]
    assert changes({**DEFAULT, "features": ["trips", "spend"]}, DEFAULT) == []          # order is not a change
    assert changes({**DEFAULT, "params": {**DEFAULT["params"], "max_depth": 10}}, DEFAULT) == ["settings"]
    assert changes({**DEFAULT, "family": "logistic", "params": {"C": 1}}, DEFAULT) == ["learner"]   # not also settings
    assert changes({**DEFAULT, "family": "lightgbm", "features": ["spend"]}, DEFAULT) == ["columns", "learner"]


def test_spec_candidates_change_what_their_names_say():
    c1 = resolve_candidates(1, SPEC["models"][1])
    assert c1["owner_forest"]["changes"] == []
    assert c1["extended_forest"]["changes"] == ["columns"]
    assert c1["owner_logistic"]["changes"] == ["learner"]
    assert c1["owner_cols_lightgbm"]["changes"] == ["learner"]
    assert c1["deeper_forest"]["changes"] == c1["shallower_forest"]["changes"] == ["settings"]
    assert c1["extended_lightgbm"]["changes"] == ["columns", "learner"]
    c3 = resolve_candidates(3, SPEC["models"][3])
    assert c3["shallower_t_forest"]["changes"] == ["settings"] and c3["extended_t_forest"]["changes"] == ["columns"]


def _synthetic(features, label, n=200, seed=0, extra=None):
    rng = np.random.default_rng(seed)
    t = pd.DataFrame({f: rng.random(n) for f in features})
    t[label] = (rng.random(n) < 0.3).astype(int)
    return t.assign(**(extra or {}))


@pytest.mark.parametrize("model_id,family", [(1, "forest"), (1, "logistic"), (2, "forest"), (3, "forest")])
def test_owner_settings_in_the_spec_match_the_owners_scripts(model_id, family):
    """`settings` attribution is only honest if the spec records the owner's real settings."""
    cand = next(c for c in SPEC["models"][model_id]["candidates"].values()
                if c["source"] == "owner" and c["family"] == family)
    if model_id == 3:
        t = _synthetic(model3_uplift.FEATURES, model3_uplift.LABEL, extra={model3_uplift.TREATMENT: [0, 1] * 100})
        est = model3_uplift.train_t_learner(t)[0]
    else:
        module = {1: model1_baseline, 2: model2_baseline}[model_id]
        est = module.train_models(_synthetic(module.FEATURES, module.LABEL))[
            {"forest": "Random forest", "logistic": "Logistic regression"}[family]]
    got = est.get_params()
    for k, v in cand["params"].items():
        assert got.get(k, got.get(f"logisticregression__{k}")) == v, k


# --------------------------------------------------------------- checking a proposal
def test_a_safe_column_proposal_is_ready_and_starts_from_the_owners_settings():
    status, why, cand = validate({"id": "p1", "add_columns": ["recency"]}, _ctx())
    assert status == "ready" and cand["features"] == ["spend", "trips", "recency"]
    assert cand["params"] == DEFAULT["params"] and cand["changes"] == ["columns"]


def test_settings_merge_over_the_owners_and_a_new_learner_starts_from_its_own_base():
    _, _, cand = validate({"id": "p", "params": {"max_depth": 9}}, _ctx())
    assert cand["params"] == {**DEFAULT["params"], "max_depth": 9} and cand["changes"] == ["settings"]
    _, _, cand = validate({"id": "p", "family": "lightgbm"}, _ctx())
    assert cand["params"] == SPEC["lightgbm_params"] and cand["changes"] == ["learner"]


@pytest.mark.parametrize("proposal,phrase", [
    ({"add_columns": ["start_day"]}, "never-input"),
    ({"add_columns": ["outcome_col"]}, "role"),
    ({"add_columns": ["label"]}, "answer"),
    ({"add_columns": ["demo"]}, "text"),
    ({"add_columns": ["odd_blank"]}, "blanks"),
    ({"add_columns": ["nope"]}, "not a column"),
    ({"add_columns": ["spend"]}, "already"),
    ({"drop_columns": ["recency"]}, "cannot drop"),
    ({"drop_columns": ["spend", "trips"]}, "nothing to learn"),
    ({"family": "linear"}, "cannot do this task"),
    ({"params": {"criterion": "gini"}}, "not one a proposal may change"),
    ({"params": {"max_depth": 500}}, "outside the allowed range"),
    ({"params": {"max_depth": True}}, "outside the allowed range"),
    ({"params": {"max_depth": 6}}, "unchanged"),
])
def test_unsafe_or_pointless_proposals_are_rejected_with_a_reason(proposal, phrase):
    status, why, cand = validate({"id": "p", **proposal}, _ctx())
    assert status == "rejected" and phrase in why and cand is None


def test_restating_unchanged_settings_or_the_owners_learner_is_not_a_change():
    """The brain often repeats the full settings, fixed seed included; only real changes count."""
    status, _, cand = validate({"id": "p", "family": "forest", "add_columns": ["recency"],
                                "params": dict(DEFAULT["params"])}, _ctx())
    assert status == "ready" and cand["changes"] == ["columns"] and cand["params"] == DEFAULT["params"]
    status, why, _ = validate({"id": "p", "params": {**DEFAULT["params"], "random_state": 7}}, _ctx())
    assert status == "rejected" and "random_state" in why        # actually changing a fixed setting is still refused
    assert validate({"id": "p", "new_column": {"name": "x", "definition": "y"}, "family": "forest",
                     "params": dict(DEFAULT["params"])}, _ctx())[0] == "needs_feature_agent"


def test_a_repeat_of_an_existing_candidate_is_rejected():
    existing = {"own": DEFAULT, "deep": {**DEFAULT, "params": {**DEFAULT["params"], "max_depth": 9}}}
    status, why, _ = validate({"id": "p", "params": {"max_depth": 9}}, _ctx(existing=existing))
    assert status == "rejected" and "`deep`" in why


def test_a_new_column_idea_goes_to_the_feature_agent_even_when_bundled_with_other_changes():
    status, why, cand = validate({"id": "p", "new_column": {"name": "x", "definition": "y"}}, _ctx())
    assert status == "needs_feature_agent" and cand is None and "wait" not in why
    status, why, _ = validate({"id": "p", "new_column": {"name": "x", "definition": "y"}, "family": "logistic"}, _ctx())
    assert status == "needs_feature_agent" and "wait until it exists" in why


def test_describe_states_exactly_what_a_candidate_changes():
    deeper = {**DEFAULT, "params": {**DEFAULT["params"], "max_depth": 10}}
    assert brain.describe(deeper, DEFAULT) == "settings {'max_depth': 10}"
    wider = {**DEFAULT, "features": ["spend", "recency"], "family": "logistic", "params": {"C": 1}}
    assert brain.describe(wider, DEFAULT) == ("adds columns ['recency']; drops columns ['trips']; "
                                              "learner logistic with settings {'C': 1}")
    assert brain.describe(DEFAULT, DEFAULT) == "nothing"


def test_uplift_uses_classifiers_and_never_the_treatment():
    assert validate({"id": "p", "family": "linear"}, _ctx("uplift"))[0] == "rejected"
    assert validate({"id": "p", "family": "logistic"}, _ctx("uplift"))[0] == "ready"
    assert "treatment" in validate({"id": "p", "add_columns": ["mailed"]}, _ctx("uplift"))[1]


def test_only_approved_proposals_run_and_a_bad_approved_one_stops():
    props = [{"id": "a", "add_columns": ["recency"], "approved": True},
             {"id": "b", "family": "logistic", "approved": False}]
    assert list(approved_candidates(props, _ctx())) == ["a"]
    with pytest.raises(SystemExit, match="STOP"):
        approved_candidates([{"id": "c", "add_columns": ["start_day"], "approved": True}], _ctx())


def test_two_approved_proposals_that_are_the_same_change_stop():
    props = [{"id": "a", "params": {"max_depth": 9}, "approved": True},
             {"id": "b", "params": {"max_depth": 9}, "approved": True}]
    with pytest.raises(SystemExit, match="already tested as `a`"):
        approved_candidates(props, _ctx())


# --------------------------------------------------------------- the ledger
def test_ledger_adds_new_rows_and_updates_old_ones_without_losing_fields(tmp_path):
    path = tmp_path / "ledger.csv"
    ledger_upsert(path, [{"model": 1, "id": "r2_p1", "title": "add recency", "status": "ready", "approved": False}])
    ledger_upsert(path, [{"model": 1, "id": "r2_p1", "status": "tested", "approved": True, "open_gap": 0.01}])
    ledger_upsert(path, [{"model": 2, "id": "r2_p1", "status": "rejected"}])        # same id, other model
    led = pd.read_csv(path)
    assert len(led) == 2
    row = led[led["model"] == 1].iloc[0]
    assert row["title"] == "add recency" and row["status"] == "tested" and row["open_gap"] == 0.01


# --------------------------------------------------------------- the brain
def _brain_ctx(tmp_path, monkeypatch):
    monkeypatch.setattr(brain, "ROOT", tmp_path)
    spec = {**SPEC, "output": {"dir": "out"}, "brain": {**SPEC["brain"], "max_proposals": 2}}
    return {"model_id": 1, "spec": spec, "ctx": _ctx(), "out_dir": tmp_path / "out" / "model1", "round": 2}


def test_record_caps_ids_checks_and_logs_every_proposal(tmp_path, monkeypatch):
    c = _brain_ctx(tmp_path, monkeypatch)
    raw = [{"title": "recency", "add_columns": ["recency"], "family": None},
           {"title": "timeline", "add_columns": ["start_day"]},
           {"title": "over the cap", "family": "logistic"}]
    props = brain.record(c, raw, "ai", "the forest wins because ...")
    assert [p["id"] for p in props] == ["r2_p1", "r2_p2"]
    assert [p["status"] for p in props] == ["ready", "rejected"] and not any(p["approved"] for p in props)
    assert "family" not in props[0]                                   # empty fields dropped
    saved = yaml.safe_load((c["out_dir"] / "proposals.yaml").read_text())
    assert [p["id"] for p in saved] == ["r2_p1", "r2_p2"]
    assert "forest wins" in (c["out_dir"] / "brain_diagnosis.md").read_text()
    led = pd.read_csv(tmp_path / "out" / "proposal_ledger.csv")
    assert led["status"].tolist() == ["ready", "rejected"] and led["from"].tolist() == ["ai", "ai"]


def test_a_second_round_keeps_earlier_proposals_and_never_reuses_an_id(tmp_path, monkeypatch):
    c = _brain_ctx(tmp_path, monkeypatch)
    brain.record(c, [{"title": "a", "add_columns": ["recency"]}], "ai", None)
    second = brain.record(c, [{"title": "b", "family": "logistic"}], "ai", None)
    assert second[0]["id"] == "r2_p2"
    assert [p["id"] for p in load_proposals(c["out_dir"])] == ["r2_p1", "r2_p2"]


def test_recheck_checks_hand_written_and_edited_proposals_and_keeps_ids(tmp_path, monkeypatch):
    c = _brain_ctx(tmp_path, monkeypatch)
    brain.record(c, [{"title": "a", "add_columns": ["start_day"]}], "ai", None)      # rejected: timeline column
    props = load_proposals(c["out_dir"])
    props[0]["add_columns"] = ["recency"]                                            # a person fixes it ...
    props.append({"title": "mine", "family": "logistic", "approved": True})          # ... and adds their own
    (c["out_dir"] / "proposals.yaml").write_text(yaml.safe_dump(props))
    out = brain.recheck(c)
    assert [(p["id"], p["from"], p["status"]) for p in out] == [("r2_p1", "ai", "ready"), ("r2_p2", "person", "ready")]
    assert out[1]["approved"] is True and out[1]["changes"] == ["learner"]
    led = pd.read_csv(tmp_path / "out" / "proposal_ledger.csv")
    assert led["status"].tolist() == ["ready", "ready"] and led["approved"].tolist() == [False, True]


def test_recheck_does_not_call_a_tested_proposal_a_repeat_of_itself(tmp_path, monkeypatch):
    c = _brain_ctx(tmp_path, monkeypatch)
    brain.record(c, [{"title": "a", "add_columns": ["recency"]}], "ai", None)
    _, _, cand = validate(load_proposals(c["out_dir"])[0], _ctx())
    c["ctx"] = {**c["ctx"], "existing": {**c["ctx"]["existing"], "r2_p1": cand}}       # it ran, so it is a candidate now
    assert brain.recheck(c)[0]["status"] == "ready"


class _FakeClient:
    def __init__(self, content):
        self.sent = None
        self.messages = SimpleNamespace(create=self._create)
        self._content = content

    def _create(self, **kw):
        self.sent = kw
        return SimpleNamespace(content=self._content, usage=SimpleNamespace(input_tokens=10, output_tokens=5))


def test_ask_claude_makes_one_tool_call_and_returns_its_answer():
    answer = {"diagnosis": "d", "proposals": []}
    client = _FakeClient([SimpleNamespace(type="tool_use", input=answer)])
    out = brain.ask_claude("brief", SPEC["brain"], client)
    assert out["input"] == answer and out["usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert client.sent["tools"][0]["name"] == "submit_proposals" and "submit_proposals" in client.sent["system"]
    assert client.sent["model"] == SPEC["brain"]["model"]


def test_ask_claude_stops_if_the_answer_is_not_structured():
    with pytest.raises(SystemExit, match="STOP"):
        brain.ask_claude("brief", SPEC["brain"], _FakeClient([SimpleNamespace(type="text", text="hi")]))


def test_api_key_comes_from_the_environment_or_the_local_env_file(tmp_path, monkeypatch):
    monkeypatch.setattr(brain, "ROOT", tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="--manual"):
        brain.api_key()
    (tmp_path / ".env").write_text("OTHER=1\nANTHROPIC_API_KEY='abc'\n")
    assert brain.api_key() == "abc"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "xyz")
    assert brain.api_key() == "xyz"
