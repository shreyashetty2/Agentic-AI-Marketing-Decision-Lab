"""Tests for the Modeling Agent (src/modeling/agent.py, learners.py, owners.py, decision_log.py)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modeling import model1_baseline
from modeling.agent import ModelingAgent, choose_candidate, train_set_id, validate_inputs
from modeling.decision_log import DecisionLog
from modeling.learners import build_estimator, fill_blanks, fit_agent_candidate
from modeling.owners import fit_owner

CANDS = {"own": {"complexity": 2, "features": ["a", "b"]}, "simple": {"complexity": 1, "features": ["a"]},
         "big": {"complexity": 3, "features": ["a", "b", "c"]}}


# --------------------------------------------------------------- the selection rule
def test_default_kept_when_no_alternative_beats_it_by_min_gain():
    pick, why = choose_candidate({"own": 0.80, "simple": 0.804, "big": 0.79}, CANDS, "own", 0.005, True)
    assert pick == "own" and "no alternative" in why


def test_alternative_wins_when_it_clears_min_gain():
    assert choose_candidate({"own": 0.80, "simple": 0.79, "big": 0.82}, CANDS, "own", 0.005, True)[0] == "big"


def test_near_tie_among_winners_goes_to_the_simpler_one():
    # big is best (0.820) but simple (0.818) is within min_gain of it and simpler
    assert choose_candidate({"own": 0.80, "simple": 0.818, "big": 0.820}, CANDS, "own", 0.005, True)[0] == "simple"


def test_lower_is_better_metrics_flip_the_direction():
    # mae: own 10, simple 9 (better by 1.0 >= 0.5)
    assert choose_candidate({"own": 10.0, "simple": 9.0, "big": 11.0}, CANDS, "own", 0.5, False)[0] == "simple"


def test_unscoreable_default_is_kept():
    assert choose_candidate({"own": np.nan, "simple": 0.9}, CANDS, "own", 0.005, True)[0] == "own"


def test_train_set_id_ignores_order():
    assert train_set_id([3, 1, 2]) == train_set_id([1, 2, 3]) != train_set_id([1, 2])


# --------------------------------------------------------------- input safety
DEFAULTS = {"never_inputs": ["start_day"], "derived_inputs": ["is_type_a"], "allowed_roles": ["profile", "context"]}
ROLES = {"spend": "profile", "start_day": "context", "redeemed_flag": "outcome", "mailed_flag": "filter"}


def test_validate_inputs_accepts_safe_columns():
    used = validate_inputs(1, {"c": {"features": ["spend", "is_type_a"]}}, ROLES, DEFAULTS, None, [])
    assert used == ["is_type_a", "spend"]


@pytest.mark.parametrize("cols,treatment", [(["spend", "start_day"], None),      # never-input
                                            (["spend", "redeemed_flag"], None),  # outcome role
                                            (["spend", "mailed_flag"], "mailed_flag")])   # treatment as input
def test_validate_inputs_stops_on_unsafe_columns(cols, treatment):
    with pytest.raises(SystemExit, match="STOP"):
        validate_inputs(1, {"c": {"features": cols}}, ROLES, DEFAULTS, treatment, [])


def test_fill_blanks_fills_only_when_zero_is_the_right_reading():
    t = pd.DataFrame({"x": [1.0, np.nan], "y": [np.nan, 2.0]})
    out = fill_blanks(t, ["x"], {"x": ["no_history"]}, ["no_history"])
    assert out["x"].tolist() == [1.0, 0.0]
    with pytest.raises(SystemExit, match="STOP"):
        fill_blanks(t, ["y"], {"y": ["no_demographics"]}, ["no_history"])
    with pytest.raises(SystemExit, match="STOP"):
        fill_blanks(t, ["y"], {}, ["no_history"])          # blank with no known reason


# --------------------------------------------------------------- learners and owner adapters
def _table(n_campaigns=8, rows=150, seed=0):
    """Campaigns every 40 days, each running 30 days; label driven by feature a."""
    rng = np.random.default_rng(seed)
    parts = []
    for c in range(1, n_campaigns + 1):
        a = rng.random(rows)
        parts.append(pd.DataFrame({
            "household_key": np.arange(rows), "campaign": c, "campaign_type": "B",
            "start_day": 40 * c, "end_day": 40 * c + 30, "a": a, "b": rng.random(rows),
            "label": (rng.random(rows) < 0.05 + 0.5 * a).astype(int), "base": a}))
    return pd.concat(parts, ignore_index=True)


def test_build_estimator_rejects_unknown_combinations():
    with pytest.raises(ValueError):
        build_estimator("linear", "classify", {})


def test_t_learner_uplift_is_the_difference_of_the_two_models():
    t = _table().assign(mailed=lambda d: (d["household_key"] % 2).astype(int))
    fitted = fit_agent_candidate(t, "tl", "uplift", "logistic", {}, ["a", "b"], "label", "mailed")
    out = fitted.predict(t.head(20))
    assert np.allclose(out["score"], out["p_treated"] - out["p_control"])


def test_owner_adapter_is_the_owners_own_model():
    rng = np.random.default_rng(5)
    n = 600
    t = pd.DataFrame({f: rng.random(n) for f in model1_baseline.FEATURES})
    t[model1_baseline.LABEL] = (rng.random(n) < t["coupon_lines_per_week_26w"]).astype(int)
    mine = fit_owner(1, "forest", t, "owner_forest").predict(t)["score"]
    theirs = model1_baseline.train_models(t)["Random forest"].predict_proba(t[model1_baseline.FEATURES])[:, 1]
    assert np.array_equal(mine, theirs)


# --------------------------------------------------------------- the agent's folds and caching
def _agent(table, inner=None, fc=None):
    mcfg = {"task": "classify", "label": "label", "default": "own",
            "baseline": {"column": "base", "sign": 1},
            "selection": {"metric": "within_campaign_auc", "min_gain": 0.005,
                          "inner": inner or {"min_train_rows": 100, "min_test_positives": 10}},
            "forward_chain": fc or {"min_train_rows": 100}}
    cands = {"own": {"source": "agent", "family": "logistic", "params": {}, "features": ["a"], "complexity": 1},
             "both": {"source": "agent", "family": "logistic", "params": {}, "features": ["a", "b"], "complexity": 1}}
    spec = {"defaults": {"inner_last_n_campaigns": 3, "min_inner_folds": 2, "first_test_day": 200}}
    return ModelingAgent(1, mcfg, spec, table, cands, DecisionLog("modeling", 1, "test"))


def test_inner_folds_stay_inside_the_training_window_and_respect_the_cutoff():
    table = _table()
    agent = _agent(table)
    train = [1, 2, 3, 4, 5]
    ends = table.groupby("campaign")["end_day"].first()
    starts = table.groupby("campaign")["start_day"].first()
    folds = agent.inner_folds(train)
    assert [f["test_campaign"] for f in folds] == [3, 4, 5]        # last 3 of the window
    for f in folds:
        assert set(f["train_campaigns"]) < set(train)
        assert all(ends[c] < starts[f["test_campaign"]] - 7 for c in f["train_campaigns"])


def test_too_little_inner_data_keeps_the_default_and_logs_why():
    agent = _agent(_table(), inner={"min_train_rows": 10**6})
    assert agent.select([1, 2, 3, 4, 5], fold=6) == ("own", {})
    assert agent.log.records[-1]["check_id"] == "MOD-03"


def test_fits_are_cached_by_training_set():
    agent = _agent(_table())
    assert agent.fit([1, 2], "own") is agent.fit([2, 1], "own")


def test_forward_run_gives_every_candidate_and_exactly_one_choice_per_row():
    agent = _agent(_table())
    preds, folds = agent.run_forward()
    per_row = preds.groupby(["household_key", "campaign"])
    assert (per_row.size() == 2).all() and (per_row["chosen"].sum() == 1).all()
    assert all(f["test_campaign"] not in f["train_campaigns"] for f in folds)


def test_forward_run_skips_and_logs_folds_with_too_little_training_data():
    # first test campaign (5, cutoff 193) has 4 finished campaigns = 600 rows; campaign 6 has 750
    agent = _agent(_table(), fc={"min_train_rows": 700})
    _, folds = agent.run_forward()
    skipped = [r for r in agent.log.records if r["check_id"] == "MOD-02"]
    assert [r["fold"] for r in skipped] == [5] and folds[0]["test_campaign"] == 6


def test_decision_log_requires_a_valid_decider_and_a_reason():
    log = DecisionLog("modeling", 1, "t")
    with pytest.raises(ValueError):
        log.record("X", "f", "a", "llm", "why")
    with pytest.raises(ValueError):
        log.record("X", "f", "a", "rule", "")
    assert log.record("X", "f", "a", "human_needed", "why")["llm_review"] is None
    assert len(log.human_needed()) == 1


# --------------------------------------------------------------- locked campaigns
def test_the_last_test_campaigns_are_locked_and_the_recommendation_never_uses_them():
    agent = _agent(_table(n_campaigns=10))
    agent.mcfg["locked_test_campaigns"] = 2
    preds, folds = agent.run_forward()
    tested = [f["test_campaign"] for f in folds]
    assert [f["test_campaign"] for f in folds if f["locked"]] == tested[-2:]
    assert set(preds.loc[preds["locked"], "campaign"]) == set(tested[-2:])
    assert any(r["check_id"] == "MOD-08" for r in agent.log.records)
    # wreck the locked predictions: the recommendation's scores must not move
    before = agent.fit_final(preds)["recommendation_scores"]
    wrecked = preds.assign(score=np.where(preds["locked"], -preds["score"], preds["score"]))
    assert agent.fit_final(wrecked)["recommendation_scores"] == before


def test_no_locked_campaigns_by_default():
    preds, folds = _agent(_table()).run_forward()
    assert not preds["locked"].any() and not any(f["locked"] for f in folds)


# --------------------------------------------------------------- what a model leans on
def test_importances_put_the_informative_column_first_and_sum_to_one():
    from modeling.learners import importances
    t = _table()                                   # label is driven by a; b is noise
    for family in ("forest", "logistic", "lightgbm"):
        params = {"n_estimators": 50, "random_state": 0, **({"verbose": -1} if family == "lightgbm" else {})}
        imp = importances(fit_agent_candidate(t, "c", "classify", family, params if family != "logistic" else {},
                                              ["a", "b"], "label"))
        assert list(imp)[0] == "a" and abs(sum(imp.values()) - 1) < 1e-3, family


def test_importances_of_an_uplift_pair_average_the_two_models():
    from modeling.learners import importances
    t = _table().assign(mailed=lambda d: (d["household_key"] % 2).astype(int))
    imp = importances(fit_agent_candidate(t, "tl", "uplift", "logistic", {}, ["a", "b"], "label", "mailed"))
    assert list(imp)[0] == "a" and abs(sum(imp.values()) - 1) < 1e-3


# --------------------------------------------------------------- Model 2's current model, and new owner scripts
def test_model2_current_model_is_yuxins_calibrated_usual_spend_unchanged():
    from modeling import model2_value
    rng = np.random.default_rng(3)
    t = pd.DataFrame({"usual_spend": rng.gamma(2, 5, 300)})
    t[model2_value.LABEL] = 0.8 * t["usual_spend"] + 6 + rng.normal(0, 3, 300)
    fitted = fit_owner(2, "calibrated", t, "owner_calibrated")
    assert fitted.features == ["usual_spend"]
    theirs = model2_value.CalibratedUsualSpend().fit(t).predict(t)
    assert np.array_equal(fitted.predict(t)["score"], theirs)


def test_a_new_model_script_in_the_repo_is_flagged_for_a_person(monkeypatch):
    from modeling import agent as agent_mod
    mcfg = {"scripts": {"current": "model2_value.py", "others": ["model2_baseline.py"]}}
    log = DecisionLog("modeling", 2, "t")
    monkeypatch.setattr(agent_mod, "model_scripts_on_git", lambda m: ["model2_baseline.py", "model2_value.py"])
    agent_mod.check_scripts(2, mcfg, log)
    assert log.records == []                                   # everything known: nothing to ask
    monkeypatch.setattr(agent_mod, "model_scripts_on_git",
                        lambda m: ["model2_baseline.py", "model2_better.py", "model2_value.py"])
    agent_mod.check_scripts(2, mcfg, log)
    rec = log.records[-1]
    assert rec["check_id"] == "MOD-09" and rec["decided_by"] == "human_needed" and "model2_better.py" in rec["finding"]
    monkeypatch.setattr(agent_mod, "model_scripts_on_git", lambda m: None)   # no git: say it could not check
    agent_mod.check_scripts(2, mcfg, log)
    assert log.records[-1]["decided_by"] == "human_needed" and "could not check" in log.records[-1]["finding"]


def test_model_scripts_on_git_lists_this_repos_committed_scripts():
    from modeling.agent import model_scripts_on_git
    assert "model2_value.py" in model_scripts_on_git(2)


def test_every_spec_script_list_names_its_current_script_and_the_owner_adapter_loads_from_it():
    from modeling.agent import load_modeling_spec
    from modeling.owners import OWNERS
    spec = load_modeling_spec()
    for m, mcfg in spec["models"].items():
        assert Path(OWNERS[m]["table"].__file__).name == mcfg["scripts"]["current"], m


def test_median_linear_is_a_learner_for_dollar_answers():
    est = build_estimator("median_linear", "regress", {"quantile": 0.5, "alpha": 0, "solver": "highs"})
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"u": rng.random(100)})
    est.fit(X, 2 * X["u"] + 1)
    assert np.allclose(est.predict(X), 2 * X["u"] + 1, atol=1e-6)
    with pytest.raises(ValueError):
        build_estimator("median_linear", "classify", {})


# --------------------------------------------------------------- tables that are not the full dataset (lead agent replay)
def _real_cut(tmp_path, keep_campaigns):
    from modeling.agent import ROOT
    real = ROOT / "data/processed/model1_table.parquet"
    if not real.exists():
        pytest.skip("the Feature Agent has not been run on this machine")
    table = pd.read_parquet(real)
    table[table["campaign"].isin(keep_campaigns)].to_parquet(tmp_path / "model1_table.parquet", index=False)
    return table


def _prepare(tmp_path, exact_counts):
    from modeling.agent import load_modeling_spec, prepare_table
    spec = load_modeling_spec()
    return prepare_table(1, spec["models"][1], spec, tmp_path, [], DecisionLog("modeling", 1, "t"),
                         exact_counts=exact_counts)


def test_a_cut_table_passes_without_exact_counts_and_stops_with_them(tmp_path):
    _real_cut(tmp_path, [1, 2, 3, 8, 13])
    table, _ = _prepare(tmp_path, exact_counts=False)
    assert 0 < len(table) < 7208
    with pytest.raises(SystemExit, match="expected 7,208"):
        _prepare(tmp_path, exact_counts=True)


def test_a_blank_label_stops_even_without_exact_counts(tmp_path):
    table = _real_cut(tmp_path, [1, 2, 3])
    cut = table[table["campaign"].isin([1, 2, 3])].copy()
    cut["redeemed_flag"] = cut["redeemed_flag"].astype(float)
    cut.iloc[0, cut.columns.get_loc("redeemed_flag")] = np.nan
    cut.to_parquet(tmp_path / "model1_table.parquet", index=False)
    with pytest.raises(SystemExit, match="every label filled"):
        _prepare(tmp_path, exact_counts=False)


def test_forward_runs_on_the_full_table_still_check_exact_counts():
    """Teammates running the agent as before keep the full dataset's count checks (the default)."""
    import inspect
    from modeling.agent import run_model
    assert inspect.signature(run_model).parameters["exact_counts"].default is True
