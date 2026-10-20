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
    python src/lead/engine.py --model 2 --as-of 660   # replay: run as if today were day 660 (src/lead/asof.py)
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
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # lets `lead.*` import when run as a script
from lead.asof import calendar, describe, folder_name  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path(__file__).with_name("lead_spec.yaml")
SHARED_STEPS = ("data", "features")
REPLAY_STEPS = ("asof",)
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
    folders: dict = field(default_factory=dict)   # where model steps read and write: tables, models, evals
    replay: str | None = None
    day: int | None = None

    @property
    def key(self) -> str:
        key = self.name if self.model is None else f"{self.name}:model{self.model}"
        return f"{key}@{self.replay}" if self.replay and self.name not in SHARED_STEPS else key


def load_spec(path: Path = SPEC_PATH) -> dict:
    return yaml.safe_load(path.read_text())


def plan_steps(spec: dict, models: list[int], day: int | None = None, folder: str | None = None) -> list[Step]:
    """Data and features once, the replay step if replaying, then modeling and evaluation for each model."""
    folders = {k: v.replace("{folder}", folder or "")
               for k, v in spec.get("folders", {}).get("normal" if day is None else "replay", {}).items()}

    def make(name: str, model: int | None) -> Step:
        cfg = spec["steps"][name]
        values = {**folders, "n": "" if model is None else str(model), "day": "" if day is None else str(day)}

        def fill(text: str) -> str:
            for k, v in values.items():
                text = text.replace("{" + k + "}", v)
            return " ".join(text.split())
        return Step(name, cfg["agent"], model, [fill(c) for c in cfg["runs"]], [fill(p) for p in cfg["inputs"]],
                    [fill(p) for p in cfg["outputs"]], folders, folder, day)
    names = SHARED_STEPS + (REPLAY_STEPS if day is not None else ())
    return [make(n, None) for n in names] + [make(n, m) for m in models for n in PER_MODEL_STEPS]


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

def read_data(root: Path, _step=None) -> dict:
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


def read_features(root: Path, _step=None) -> dict:
    rows = {n: pq.read_metadata(root / f"data/processed/model{n}_table.parquet").num_rows for n in (1, 2, 3)}
    return {"produced": [f"Model {n} table: {r:,} rows" for n, r in rows.items()],
            "handled": None, "questions": [], "stopped": None}


def read_asof(root: Path, step: Step) -> dict:
    """The tables are shared by every day with the same finished campaigns, so the day itself (what is running,
    what has not started) is described afresh rather than read from the day the tables were built."""
    built = json.loads((root / step.folders["tables"] / "asof.json").read_text())
    a = {**describe(calendar(root / "data/processed"), step.day), "rows": built["rows"]}
    return {"produced": [f"as of day {a['day']}: {len(a['finished'])} campaigns finished, the newest "
                         f"{a['newest_finished']} on day {a['newest_end_day']}; {len(a['running'])} running, "
                         f"{a['not_started']} not started yet",
                         "rows: " + ", ".join(f"Model {k[-1]} {v:,}" for k, v in a["rows"].items())],
            "handled": None, "questions": [], "stopped": None, "as_of": a}


def read_modeling(root: Path, step: Step) -> dict:
    m = json.loads((root / step.folders["models"] / f"model{step.model}/manifest.json").read_text())
    rec, default = m["final"]["recommended"], m["default"]
    # The lead agent's cross-check: the Modeling Agent used exactly the table the step before it produced.
    rows = pq.read_metadata(root / step.folders["tables"] / f"model{step.model}_table.parquet").num_rows
    used = m.get("table_rows")
    return {"produced": [f"tested {len(m['candidates'])} options on {len(m['folds'])} test campaigns"],
            "handled": "recommends keeping the current model" if rec == default
                       else f"recommends `{rec}` over the current model (the owner decides)",
            "questions": [], "run_id": m["run_id"],
            "stopped": None if used in (None, rows) else f"it used {used:,} rows but the table it was given has {rows:,}"}


def read_evaluation(root: Path, step: Step) -> dict:
    v = json.loads((root / step.folders["evals"] / f"model{step.model}/verdict.json").read_text())
    return {"produced": [f"verdict: {v['verdict']} (decided by {v['decided_by']})"],
            "handled": v["reasons"][0] if v["reasons"] else None,
            "questions": [{"key": q["key"], "who": q["who"], "text": q["question"]} for q in v["questions"]],
            "stopped": None, "verdict": v["verdict"], "scores": v["scores"], "primary_metric": v["primary_metric"],
            "owner_model": v["owner_model"], "newest_campaign": v.get("newest_campaign"), "run_id": v["run_id"]}


def update_approved(root: Path, spec: dict, step: Step, report: dict, when: str) -> dict:
    """Which model scoring should use. A retrained model replaces the approved one only if the Evaluation Agent
    approved it and its newest campaign was not worse than usual; otherwise the last approved model stays.
    A replay keeps its own record, so a simulation never replaces the real approved model."""
    folder = root / spec["output_dir"] / ("replay/approved" if step.replay else "approved")
    record_path = folder / f"model{step.model}.json"
    previous = json.loads(record_path.read_text()) if record_path.exists() else None
    newest = report.get("newest_campaign") or {}
    if report["verdict"] != "approve" or newest.get("worse_than_usual"):
        why = (f"the newest campaign ({newest['campaign']}) scored worse than usual" if report["verdict"] == "approve"
               else f"the Evaluation Agent's verdict is {report['verdict']}: {report['handled']}")
        return {"action": "kept" if previous else "none_approved", "reason": why, "approved": previous}
    if previous and previous["run_id"] == report["run_id"]:
        return {"action": "unchanged", "reason": "this version is already the approved one", "approved": previous}
    record = {"run_id": report["run_id"], "as_of_day": step.day, "replay_folder": step.replay, "approved_at": when,
              "verdict": report["verdict"], "primary_metric": report["primary_metric"],
              "score": report["scores"][report["owner_model"]], "owner_model": report["owner_model"]}
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / step.folders["models"] / f"model{step.model}/final_model.joblib",
                    folder / f"model{step.model}_final_model.joblib")
    record_path.write_text(json.dumps(record, indent=2, default=float))
    return {"action": "replaced" if previous else "first_approved", "reason": "passed every check",
            "approved": record, "previous": previous}


READERS = {"data": read_data, "features": read_features, "asof": read_asof, "modeling": read_modeling,
           "evaluation": read_evaluation}


# ---- the run ---------------------------------------------------------------------------------------------

def question_id(step: Step, key: str) -> str:
    return f"{step.name}:{'-' if step.model is None else step.model}:{key}"


def classify(step: Step, question: dict, spec: dict, state: dict) -> str:
    """'known' (the team is working on it), 'still_open' (asked in an earlier run), or 'new'."""
    if any(k["agent"] == step.name and k["key"] == question["key"] for k in spec.get("known_questions", [])):
        return "known"
    return "still_open" if question_id(step, question["key"]) in state.get("questions", {}) else "new"


def run(models: list[int], spec: dict, root: Path = ROOT, plan_only: bool = False, day: int | None = None) -> dict:
    """day: replay as if today were that day (None = the full data, as it is)."""
    out_dir = root / spec["output_dir"]
    state_path = out_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"steps": {}, "questions": {}}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = {"run_at": now, "models": models, "as_of_day": day, "status": "finished", "stopped_because": None,
               "since_last_replay": None, "steps": [], "questions": {"new": [], "still_open": [], "known": []}}

    def stop(reason: str) -> bool:
        summary.update(status="stopped", stopped_because=reason)
        return False

    def do(step: Step) -> bool:
        """Run (or skip) one step and read its report. False = the run stops here."""
        go, why = needs_run(step, root, state)
        entry = {"step": step.key, "agent": step.agent, "model": step.model, "ran": go and not plan_only, "why": why}
        summary["steps"].append(entry)
        if plan_only:
            return True
        if go:
            ok, message = run_commands(step, root)
            if not ok:
                entry["status"] = "stopped"
                return stop(f"{step.agent}: {message}")
            state["steps"][step.key] = {"inputs": fingerprint(root, step.inputs), "finished_at": now}
        report = READERS[step.name](root, step)
        if step.name == "evaluation" and not report["stopped"]:
            report["model_update"] = update_approved(root, spec, step, report, now)
        if "as_of" in report:
            summary["since_last_replay"] = since_last(state.get("last_replay"), report["as_of"])
            state["last_replay"] = {"day": day, "finished": report["as_of"]["finished"]}
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
            return stop(f"{step.agent}: {report['stopped']}")
        if new_here:
            return stop(f"{step.agent} raised {len(new_here)} question(s) its rules do not cover; "
                        "a person needs to answer first")
        return True

    # The shared steps come first: the replay folder depends on the campaign calendar the Data Agent writes.
    n_shared = len(SHARED_STEPS)
    if all(do(step) for step in plan_steps(spec, models, day)[:n_shared]):
        folder = None
        if day is not None:
            try:
                folder = folder_name(calendar(root / "data/processed"), day)
            except (SystemExit, FileNotFoundError) as e:
                folder = "unknown_until_the_data_agent_runs" if plan_only else None
                if not plan_only:
                    stop(f"Replay: {str(e).removeprefix('[STOP] ')}")
        if folder is not None or day is None:
            for step in plan_steps(spec, models, day, folder)[n_shared:]:
                if not do(step):
                    break

    if not plan_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2))
        (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=float))
        (out_dir / "run_summary.md").write_text(render(summary))
    return summary


def since_last(previous: dict | None, now: dict) -> dict | None:
    """What changed since the previous replay day: which campaigns finished in between."""
    if previous is None:
        return None
    return {"previous_day": previous["day"],
            "newly_finished": sorted(set(now["finished"]) - set(previous["finished"])),
            "went_back_in_time": now["day"] < previous["day"]}


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
             + (f" -- {s['stopped_because']}" if s["stopped_because"] else ""), ""]
    if s.get("as_of_day") is not None:
        lines.append(f"Replay: as if today were day {s['as_of_day']}.")
        if s.get("since_last_replay"):
            sl = s["since_last_replay"]
            if sl["went_back_in_time"]:
                lines.append(f"This day is earlier than the last replay (day {sl['previous_day']}).")
            else:
                lines.append(f"Since the last replay (day {sl['previous_day']}): "
                             + (f"campaigns {sl['newly_finished']} finished." if sl["newly_finished"] else "no campaign finished."))
        lines.append("")
    lines += ["## Steps", ""]
    for e in s["steps"]:
        did = "ran" if e["ran"] else "skipped"
        lines.append(f"- **{e['step']}** ({e['agent']}): {did} ({e['why']}), {e.get('status', 'not reached')}")
        lines += [f"  - {p}" for p in e.get("produced", [])]
        if e.get("handled"):
            lines.append(f"  - {e['handled']}")
        if e.get("model_update"):
            u = e["model_update"]
            lines.append(f"  - model used for scoring: {u['action'].replace('_', ' ')} ({u['reason']})")
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
    parser.add_argument("--as-of", type=int, default=None, help="replay: run as if today were this day")
    parser.add_argument("--mark-current", action="store_true", help="record current outputs as up to date")
    args = parser.parse_args()
    spec = load_spec()
    if args.mark_current:
        print("Marked up to date:", ", ".join(mark_current(spec)) or "nothing")
        return
    models = [1, 2, 3] if args.model == "all" else [int(args.model)]
    summary = run(models, spec, plan_only=args.plan, day=args.as_of)
    print(render(summary))


if __name__ == "__main__":
    main()
