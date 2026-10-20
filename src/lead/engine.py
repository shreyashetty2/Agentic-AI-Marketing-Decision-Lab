"""Lead Agent engine: runs the Data, Feature, Modeling and Evaluation Agents in order and writes one summary.

The engine is plain code with no AI. It decides what runs by the rules in lead_spec.yaml, runs each agent's
own scripts unchanged, reads what each agent wrote, and collects it into one run summary that the lead
agent's front door explains to a data scientist. Design: docs/agent_design/lead_agent_design.md

  - A step runs only if its inputs changed since it last ran (by content), or an output is missing.
  - A step that fails, a Data Agent HALT, or a NEW question from the Data or Feature Agent stops the run:
    nothing downstream can be trusted until a person answers it.
  - A question the team already knows about (known_questions in the spec) never stops a run.
  - Questions from the Modeling and Evaluation Agents come at the end of a run and never stop it.

Usage (from the repo root):
    python src/lead/engine.py --model 2          # bring Model 2 up to date, then summarise
    python src/lead/engine.py --model all --plan # only say what would run, and why
    python src/lead/engine.py --mark-current     # once, after running the agents by hand: record their
                                                 # current outputs as up to date

Writes to output_dir (lead_spec.yaml): run_summary.json, run_summary.md, state.json
"""
import argparse
import glob
import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path(__file__).with_name("lead_spec.yaml")
SHARED_STEPS = ("data", "features")
PER_MODEL_STEPS = ("modeling", "evaluation")
STOPPING_AGENTS = ("data", "features")   # a new question here means the data underneath is not understood


@dataclass
class Step:
    name: str
    agent: str
    model: int | None
    runs: list[str]
    inputs: list[str]
    outputs: list[str]

    @property
    def key(self) -> str:
        return self.name if self.model is None else f"{self.name}:model{self.model}"


def load_spec(path: Path = SPEC_PATH) -> dict:
    return yaml.safe_load(path.read_text())


def plan_steps(spec: dict, models: list[int]) -> list[Step]:
    """Data and features once, then modeling and evaluation for each model, in that order."""
    def make(name: str, model: int | None) -> Step:
        cfg = spec["steps"][name]
        fill = (lambda s: s.replace("{n}", str(model))) if model is not None else (lambda s: s)
        return Step(name, cfg["agent"], model, [fill(c) for c in cfg["runs"]],
                    [fill(p) for p in cfg["inputs"]], [fill(p) for p in cfg["outputs"]])
    return [make(n, None) for n in SHARED_STEPS] + [make(n, m) for m in models for n in PER_MODEL_STEPS]


def fingerprint(root: Path, patterns: list[str]) -> str:
    """One hash over the contents of every file the patterns match (and a marker for any that match nothing)."""
    digest = hashlib.sha256()
    for pattern in patterns:
        paths = sorted(glob.glob(str(root / pattern)))
        if not paths:
            digest.update(f"missing:{pattern}".encode())
        for p in paths:
            digest.update(Path(p).relative_to(root).as_posix().encode())
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def needs_run(step: Step, root: Path, state: dict) -> tuple[bool, str]:
    if any(not (root / o).exists() for o in step.outputs):
        return True, "an output is missing"
    last = state.get("steps", {}).get(step.key)
    if last is None:
        return True, "no record of an earlier run"
    if last["inputs"] != fingerprint(root, step.inputs):
        return True, "its inputs changed since it last ran"
    return False, "up to date"


def run_commands(step: Step, root: Path) -> tuple[bool, str]:
    """Run the agent's own scripts, unchanged, in order. Stops at the first failure and returns its reason."""
    for command in step.runs:
        done = subprocess.run([sys.executable, *shlex.split(command)], cwd=root, capture_output=True, text=True)
        if done.returncode != 0:
            lines = [l for l in (done.stdout + done.stderr).splitlines() if l.strip()]
            reason = next((l for l in reversed(lines) if "[STOP]" in l or "HALT" in l), lines[-1] if lines else "no output")
            return False, f"`{command.split()[0]}` stopped: {reason.strip()}"
    return True, "finished"


# ---- reading what each agent wrote -----------------------------------------------------------------------
# Each reader returns: produced (lines), handled (one line or None), questions [{key, who, text}],
# and stopped (a reason, or None) for an agent that finished its scripts but refused to continue.

def read_data(root: Path, _model) -> dict:
    log = pd.read_csv(root / "data/processed/decision_log.csv")
    checks = json.loads((root / "data/processed/spine_check.json").read_text())["checks"]
    halts = log[log["action"] == "HALT"]
    counts = log[log["decided_by"] == "rule"]["action"].value_counts()
    handled = ", ".join(f"{counts[a]} {word}" for a, word in
                        [("FIX", "fixed"), ("FLAG", "flagged"), ("NOTE", "noted")] if a in counts)
    return {
        "produced": [f"{checks['households']['actual']:,} households x {checks['campaigns']['actual']} campaigns; "
                     f"{checks['mailed pairs']['actual']:,} mailings, {checks['mailed pairs with a redemption']['actual']:,} redeemed"],
        "handled": f"by its rules: {handled}" if handled else None,
        "questions": [{"key": r.check_id, "who": "team", "text": r.finding}
                      for r in log[log["action"] == "ESCALATE"].itertuples()],
        "stopped": f"HALT: {'; '.join(halts['finding'])}" if len(halts) else None,
    }


def read_features(root: Path, _model) -> dict:
    rows = {n: pq.read_metadata(root / f"data/processed/model{n}_table.parquet").num_rows for n in (1, 2, 3)}
    return {"produced": [f"Model {n} table: {r:,} rows" for n, r in rows.items()],
            "handled": None, "questions": [], "stopped": None}


def read_modeling(root: Path, model: int) -> dict:
    m = json.loads((root / f"data/processed/modeling_agent/model{model}/manifest.json").read_text())
    rec, default = m["final"]["recommended"], m["default"]
    return {"produced": [f"tested {len(m['candidates'])} options on {len(m['folds'])} test campaigns"],
            "handled": "recommends keeping the current model" if rec == default
                       else f"recommends `{rec}` over the current model (the owner decides)",
            "questions": [], "stopped": None, "run_id": m["run_id"]}


def read_evaluation(root: Path, model: int) -> dict:
    v = json.loads((root / f"data/processed/evaluation_agent/model{model}/verdict.json").read_text())
    return {"produced": [f"verdict: {v['verdict']} (decided by {v['decided_by']})"],
            "handled": v["reasons"][0] if v["reasons"] else None,
            "questions": [{"key": q["key"], "who": q["who"], "text": q["question"]} for q in v["questions"]],
            "stopped": None, "verdict": v["verdict"], "scores": v["scores"], "primary_metric": v["primary_metric"],
            "owner_model": v["owner_model"]}


READERS = {"data": read_data, "features": read_features, "modeling": read_modeling, "evaluation": read_evaluation}


# ---- the run ---------------------------------------------------------------------------------------------

def question_id(step: Step, key: str) -> str:
    return f"{step.name}:{'-' if step.model is None else step.model}:{key}"


def classify(step: Step, question: dict, spec: dict, state: dict) -> str:
    """'known' (the team is working on it), 'still_open' (asked in an earlier run), or 'new'."""
    if any(k["agent"] == step.name and k["key"] == question["key"] for k in spec.get("known_questions", [])):
        return "known"
    return "still_open" if question_id(step, question["key"]) in state.get("questions", {}) else "new"


def run(models: list[int], spec: dict, root: Path = ROOT, plan_only: bool = False) -> dict:
    out_dir = root / spec["output_dir"]
    state_path = out_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"steps": {}, "questions": {}}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = {"run_at": now, "models": models, "status": "finished", "stopped_because": None,
               "steps": [], "questions": {"new": [], "still_open": [], "known": []}}

    for step in plan_steps(spec, models):
        go, why = needs_run(step, root, state)
        entry = {"step": step.key, "agent": step.agent, "model": step.model, "ran": go and not plan_only, "why": why}
        summary["steps"].append(entry)
        if plan_only:
            continue
        if go:
            ok, message = run_commands(step, root)
            if not ok:
                entry["status"] = "stopped"
                summary.update(status="stopped", stopped_because=f"{step.agent}: {message}")
                break
            state["steps"][step.key] = {"inputs": fingerprint(root, step.inputs), "finished_at": now}
        report = READERS[step.name](root, step.model)
        entry.update({k: v for k, v in report.items() if k != "questions"}, status="stopped" if report["stopped"] else "finished")
        new_here = []
        for q in report["questions"]:
            kind = classify(step, q, spec, state)
            item = {"id": question_id(step, q["key"]), "agent": step.agent, "model": step.model, **q}
            if kind != "known":
                item["first_seen"] = state["questions"].setdefault(item["id"], {"first_seen": now})["first_seen"]
            summary["questions"][kind].append(item)
            if kind != "known" and step.name in STOPPING_AGENTS:
                new_here.append(item)
        if report["stopped"]:
            summary.update(status="stopped", stopped_because=f"{step.agent}: {report['stopped']}")
            break
        if new_here:
            summary.update(status="stopped", stopped_because=f"{step.agent} raised {len(new_here)} question(s) its "
                                                             "rules do not cover; a person needs to answer first")
            break

    if not plan_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2))
        (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=float))
        (out_dir / "run_summary.md").write_text(render(summary))
    return summary


def mark_current(spec: dict, root: Path = ROOT) -> list[str]:
    """Record every step whose outputs exist as up to date with today's inputs. For adopting outputs that
    were produced by running the agents by hand; the engine cannot check how they were made."""
    out_dir = root / spec["output_dir"]
    state_path = out_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"steps": {}, "questions": {}}
    marked = []
    for step in plan_steps(spec, [1, 2, 3]):
        if all((root / o).exists() for o in step.outputs):
            state["steps"][step.key] = {"inputs": fingerprint(root, step.inputs), "finished_at": "marked current"}
            marked.append(step.key)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2))
    return marked


def render(s: dict) -> str:
    """Plain record of the run. The front door turns this into the conversation; this file is the evidence."""
    lines = [f"# Lead Agent run: {s['run_at']}", "",
             f"Models: {', '.join(map(str, s['models']))}. Status: **{s['status']}**"
             + (f" -- {s['stopped_because']}" if s["stopped_because"] else ""), "", "## Steps", ""]
    for e in s["steps"]:
        did = "ran" if e["ran"] else "skipped"
        lines.append(f"- **{e['step']}** ({e['agent']}): {did} ({e['why']}), {e.get('status', 'not reached')}")
        lines += [f"  - {p}" for p in e.get("produced", [])]
        if e.get("handled"):
            lines.append(f"  - {e['handled']}")
    for kind, title in [("new", "New questions"), ("still_open", "Still open from earlier runs"),
                        ("known", "Known to the team, being worked on")]:
        items = s["questions"][kind]
        lines += ["", f"## {title}", ""] + ([f"- ({q['agent']}{'' if q['model'] is None else ', Model ' + str(q['model'])}, "
                                              f"for {q['who']}) {q['text']}" for q in items] or ["- none"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="all", choices=["1", "2", "3", "all"])
    parser.add_argument("--plan", action="store_true", help="only say what would run, and why")
    parser.add_argument("--mark-current", action="store_true", help="record current outputs as up to date")
    args = parser.parse_args()
    spec = load_spec()
    if args.mark_current:
        print("Marked up to date:", ", ".join(mark_current(spec)) or "nothing")
        return
    models = [1, 2, 3] if args.model == "all" else [int(args.model)]
    summary = run(models, spec, plan_only=args.plan)
    print(render(summary))


if __name__ == "__main__":
    main()
