"""Tests for the Lead Agent engine (src/lead/engine.py).

The real agents are replaced by one small fake script (fake_agent.py, written into a temporary repo) that
writes the same files the real agents write. scenario.json controls what it writes, so each rule of the
engine can be checked with a known answer: skipping, stopping, known vs. new questions, failures.
"""
import json
import time
from pathlib import Path

import pandas as pd
import pytest

from lead.engine import (Step, classify, fingerprint, mark_current, needs_run, plan_steps, read_data, render,
                         run)

FAKE_AGENT = r'''
import json, sys
from pathlib import Path
import pandas as pd
step, model = sys.argv[1], sys.argv[2]
sc = json.loads(Path("scenario.json").read_text())
calls = Path("calls.txt"); calls.write_text(calls.read_text() + f"{step}:{model}\n" if calls.exists() else f"{step}:{model}\n")
if sc.get("fail") == f"{step}:{model}":
    print("[STOP] fake failure for testing"); sys.exit(1)
p = Path("data/processed"); p.mkdir(parents=True, exist_ok=True)
if step == "data":
    rows = [{"check_id": "Q1", "finding": "duplicate coupon rows", "action": "FIX", "decided_by": "rule"},
            {"check_id": "Q4", "finding": "odd discounts", "action": "FLAG", "decided_by": "rule"}]
    rows += [{"check_id": k, "finding": f"question {k}", "action": "ESCALATE", "decided_by": "human_needed"}
             for k in sc.get("escalate", [])]
    rows += [{"check_id": "X1", "finding": "tables disagree", "action": "HALT", "decided_by": "rule"}] if sc.get("halt") else []
    pd.DataFrame(rows).to_csv(p / "decision_log.csv", index=False)
    c = {k: {"actual": v} for k, v in {"households": 10, "campaigns": 3, "mailed pairs": 6, "mailed pairs with a redemption": 2}.items()}
    (p / "spine_check.json").write_text(json.dumps({"all_pass": True, "checks": c}))
elif step == "features":
    for n in (1, 2, 3):
        pd.DataFrame({"x": range(n + 4)}).to_parquet(p / f"model{n}_table.parquet")
elif step == "modeling":
    d = p / f"modeling_agent/model{model}"; d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({"run_id": str(__import__("time").time()), "default": "own", "candidates": {"own": {}, "alt": {}},
                                                 "folds": [1, 2], "final": {"recommended": "own"}}))
    (d / "predictions.parquet").write_bytes(b"preds")
elif step == "evaluation":
    d = p / f"evaluation_agent/model{model}"; d.mkdir(parents=True, exist_ok=True)
    qs = [{"key": k, "who": "team", "question": f"eval question {k}"} for k in sc.get("eval_questions", [])]
    (d / "verdict.json").write_text(json.dumps({"verdict": "approve", "decided_by": "rule", "reasons": ["EVA-05: fine"],
                                                "questions": qs, "scores": {"own": 0.5}, "primary_metric": "m",
                                                "owner_model": "own"}))
'''


def make_repo(tmp_path: Path, scenario: dict | None = None) -> tuple[Path, dict]:
    (tmp_path / "fake_agent.py").write_text(FAKE_AGENT)
    (tmp_path / "scenario.json").write_text(json.dumps(scenario or {}))
    (tmp_path / "raw.csv").write_text("a,b\n1,2\n")
    spec = {
        "output_dir": "data/processed/lead_agent",
        "steps": {
            "data": {"agent": "Data Agent", "runs": ["fake_agent.py data -"], "inputs": ["raw.csv"],
                     "outputs": ["data/processed/decision_log.csv"]},
            "features": {"agent": "Feature Agent", "runs": ["fake_agent.py features -"],
                         "inputs": ["data/processed/decision_log.csv"],
                         "outputs": ["data/processed/model1_table.parquet"]},
            "modeling": {"agent": "Modeling Agent", "runs": ["fake_agent.py modeling {n}"],
                         "inputs": ["data/processed/model{n}_table.parquet"],
                         "outputs": ["data/processed/modeling_agent/model{n}/manifest.json"]},
            "evaluation": {"agent": "Evaluation Agent", "runs": ["fake_agent.py evaluation {n}"],
                           "inputs": ["data/processed/modeling_agent/model{n}/manifest.json"],
                           "outputs": ["data/processed/evaluation_agent/model{n}/verdict.json"]},
        },
        "known_questions": [{"agent": "data", "key": "Q7b", "note": "gas"}],
    }
    return tmp_path, spec


def calls(root: Path) -> list[str]:
    path = root / "calls.txt"
    return path.read_text().split() if path.exists() else []


def set_scenario(root: Path, **scenario) -> None:
    (root / "scenario.json").write_text(json.dumps(scenario))


# ---- deciding what runs -----------------------------------------------------------------------------------

def test_plan_steps_runs_shared_steps_once_then_each_model_in_order(tmp_path):
    _, spec = make_repo(tmp_path)
    steps = plan_steps(spec, [2, 3])
    assert [s.key for s in steps] == ["data", "features", "modeling:model2", "evaluation:model2",
                                      "modeling:model3", "evaluation:model3"]
    assert steps[2].runs == ["fake_agent.py modeling 2"]
    assert steps[2].inputs == ["data/processed/model2_table.parquet"]


def test_fingerprint_follows_content_not_file_dates(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("same")
    first = fingerprint(tmp_path, ["a.txt"])
    time.sleep(0.01)
    f.write_text("same")                       # rewritten, newer date, same content
    assert fingerprint(tmp_path, ["a.txt"]) == first
    f.write_text("different")
    assert fingerprint(tmp_path, ["a.txt"]) != first


def test_fingerprint_marks_a_pattern_that_matches_nothing(tmp_path):
    assert fingerprint(tmp_path, ["nothing*.csv"]) != fingerprint(tmp_path, [])


def test_needs_run_gives_the_reason(tmp_path):
    root, _ = make_repo(tmp_path)
    step = Step("data", "Data Agent", None, [], ["raw.csv"], ["out.txt"])
    assert needs_run(step, root, {}) == (True, "an output is missing")
    (root / "out.txt").write_text("x")
    assert needs_run(step, root, {}) == (True, "no record of an earlier run")
    state = {"steps": {"data": {"inputs": fingerprint(root, ["raw.csv"])}}}
    assert needs_run(step, root, state) == (False, "up to date")
    (root / "raw.csv").write_text("a,b\n9,9\n")
    assert needs_run(step, root, state) == (True, "its inputs changed since it last ran")


# ---- whole runs --------------------------------------------------------------------------------------------

def test_first_run_runs_everything_and_second_run_skips_everything(tmp_path):
    root, spec = make_repo(tmp_path)
    s = run([1], spec, root)
    assert s["status"] == "finished"
    assert calls(root) == ["data:-", "features:-", "modeling:1", "evaluation:1"]
    s2 = run([1], spec, root)
    assert calls(root) == ["data:-", "features:-", "modeling:1", "evaluation:1"]      # nothing ran again
    assert [e["why"] for e in s2["steps"]] == ["up to date"] * 4
    assert s2["steps"][3]["verdict"] == "approve"                                    # still read and reported


def test_changed_input_reruns_that_step_and_only_what_depends_on_it(tmp_path):
    root, spec = make_repo(tmp_path)
    run([1, 2], spec, root)
    pd.DataFrame({"x": range(50)}).to_parquet(root / "data/processed/model2_table.parquet")   # new Model 2 rows
    run([1, 2], spec, root)
    assert calls(root)[6:] == ["modeling:2", "evaluation:2"]


def test_a_new_data_question_stops_the_run_before_any_model(tmp_path):
    root, spec = make_repo(tmp_path, {"escalate": ["Q99"]})
    s = run([1], spec, root)
    assert s["status"] == "stopped"
    assert "Data Agent raised 1 question" in s["stopped_because"]
    assert calls(root) == ["data:-"]
    assert [q["key"] for q in s["questions"]["new"]] == ["Q99"]


def test_an_unanswered_data_question_keeps_stopping_later_runs(tmp_path):
    root, spec = make_repo(tmp_path, {"escalate": ["Q99"]})
    run([1], spec, root)
    s = run([1], spec, root)
    assert s["status"] == "stopped"
    assert [q["key"] for q in s["questions"]["still_open"]] == ["Q99"]


def test_a_question_the_team_already_knows_does_not_stop_the_run(tmp_path):
    root, spec = make_repo(tmp_path, {"escalate": ["Q7b"]})
    s = run([1], spec, root)
    assert s["status"] == "finished"
    assert [q["key"] for q in s["questions"]["known"]] == ["Q7b"]
    assert s["questions"]["new"] == []


def test_a_data_agent_halt_stops_the_run(tmp_path):
    root, spec = make_repo(tmp_path, {"halt": True})
    s = run([1], spec, root)
    assert s["status"] == "stopped"
    assert s["stopped_because"] == "Data Agent: HALT: tables disagree"
    assert calls(root) == ["data:-"]


def test_a_failed_script_stops_the_run_with_its_own_reason(tmp_path):
    root, spec = make_repo(tmp_path, {"fail": "modeling:1"})
    s = run([1, 2], spec, root)
    assert s["status"] == "stopped"
    assert s["stopped_because"] == "Modeling Agent: `fake_agent.py` stopped: [STOP] fake failure for testing"
    assert calls(root) == ["data:-", "features:-", "modeling:1"]
    assert "modeling:1" not in json.loads((root / "data/processed/lead_agent/state.json").read_text())["steps"]


def test_a_failed_step_runs_again_next_time(tmp_path):
    root, spec = make_repo(tmp_path, {"fail": "modeling:1"})
    run([1], spec, root)
    set_scenario(root)
    s = run([1], spec, root)
    assert s["status"] == "finished"
    assert calls(root)[3:] == ["modeling:1", "evaluation:1"]


def test_evaluation_questions_never_stop_a_run_and_are_new_only_once(tmp_path):
    root, spec = make_repo(tmp_path, {"eval_questions": ["calibration"]})
    first = run([1, 2], spec, root)
    assert first["status"] == "finished"
    assert [q["id"] for q in first["questions"]["new"]] == ["evaluation:1:calibration", "evaluation:2:calibration"]
    second = run([1, 2], spec, root)
    assert second["questions"]["new"] == []
    assert len(second["questions"]["still_open"]) == 2
    assert second["questions"]["still_open"][0]["first_seen"] == first["run_at"]


def test_plan_only_runs_nothing_and_writes_nothing(tmp_path):
    root, spec = make_repo(tmp_path)
    s = run([1], spec, root, plan_only=True)
    assert calls(root) == []
    assert [e["why"] for e in s["steps"]] == ["an output is missing"] * 4
    assert not (root / "data/processed/lead_agent").exists()


def test_mark_current_adopts_existing_outputs_so_nothing_reruns(tmp_path):
    root, spec = make_repo(tmp_path)
    run([1, 2, 3], spec, root)
    (root / "data/processed/lead_agent/state.json").unlink()
    assert mark_current(spec, root) == ["data", "features", "modeling:model1", "evaluation:model1",
                                        "modeling:model2", "evaluation:model2", "modeling:model3", "evaluation:model3"]
    n = len(calls(root))
    run([1, 2, 3], spec, root)
    assert len(calls(root)) == n


def test_summary_files_are_written_and_readable(tmp_path):
    root, spec = make_repo(tmp_path, {"escalate": ["Q7b"], "eval_questions": ["measure_note"]})
    run([1], spec, root)
    out = root / "data/processed/lead_agent"
    md = (out / "run_summary.md").read_text()
    assert "Status: **finished**" in md
    assert "## Known to the team, being worked on" in md and "question Q7b" in md
    assert "## New questions" in md and "eval question measure_note" in md
    assert json.loads((out / "run_summary.json").read_text())["status"] == "finished"


# ---- reading the Data Agent's log ---------------------------------------------------------------------------

def test_read_data_counts_rule_decisions_and_lists_escalations(tmp_path):
    p = tmp_path / "data/processed"
    p.mkdir(parents=True)
    pd.DataFrame([
        {"check_id": "Q1", "finding": "dups", "action": "FIX", "decided_by": "rule"},
        {"check_id": "Q2", "finding": "a", "action": "FLAG", "decided_by": "rule"},
        {"check_id": "Q3", "finding": "b", "action": "FLAG", "decided_by": "rule"},
        {"check_id": "Q7b", "finding": "gas lines", "action": "ESCALATE", "decided_by": "human_needed"},
    ]).to_csv(p / "decision_log.csv", index=False)
    c = {k: {"actual": v} for k, v in {"households": 2500, "campaigns": 30, "mailed pairs": 7208, "mailed pairs with a redemption": 889}.items()}
    (p / "spine_check.json").write_text(json.dumps({"checks": c}))
    r = read_data(tmp_path, None)
    assert r["handled"] == "by its rules: 1 fixed, 2 flagged"
    assert r["questions"] == [{"key": "Q7b", "who": "team", "text": "gas lines"}]
    assert r["produced"] == ["2,500 households x 30 campaigns; 7,208 mailings, 889 redeemed"]
    assert r["stopped"] is None


def test_read_data_reads_the_real_data_agent_files():
    """Guards against the fixtures above drifting from the Data Agent's real file formats."""
    from lead.engine import ROOT
    if not (ROOT / "data/processed/decision_log.csv").exists():
        pytest.skip("the Data Agent has not been run on this machine")
    r = read_data(ROOT, None)
    assert r["produced"][0].startswith("2,500 households x 30 campaigns")
    assert r["stopped"] is None


def test_classify(tmp_path):
    _, spec = make_repo(tmp_path)
    data = Step("data", "Data Agent", None, [], [], [])
    assert classify(data, {"key": "Q7b"}, spec, {}) == "known"
    assert classify(data, {"key": "Q99"}, spec, {}) == "new"
    assert classify(data, {"key": "Q99"}, spec, {"questions": {"data:-:Q99": {}}}) == "still_open"
    evaluation = Step("evaluation", "Evaluation Agent", 1, [], [], [])
    assert classify(evaluation, {"key": "Q7b"}, spec, {}) == "new"      # known for the Data Agent only


def test_render_says_none_when_there_are_no_questions(tmp_path):
    root, spec = make_repo(tmp_path)
    md = render(run([1], spec, root))
    assert md.count("- none") == 3


# ---- the real instruction file ------------------------------------------------------------------------------

def test_the_real_spec_loads_and_names_files_that_exist():
    from lead.engine import ROOT, load_spec
    spec = load_spec()
    steps = plan_steps(spec, [1, 2, 3])
    assert [s.key for s in steps][:2] == ["data", "features"]
    for step in steps:
        assert step.runs and step.inputs and step.outputs
        for command in step.runs:
            assert (ROOT / command.split()[0]).exists(), f"{step.key}: {command} does not exist"
        assert all("{n}" not in p for p in step.inputs + step.outputs + step.runs)
    assert {(k["agent"], k["key"]) for k in spec["known_questions"]} == {("data", "Q7b"), ("data", "Q5-2")}
