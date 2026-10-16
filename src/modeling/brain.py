"""The brain: reads the latest Modeling + Evaluation results for one model and proposes what to try next.

It does two jobs a fixed menu cannot: (1) explains WHY the results came out as they did (which option
won, and what that says about the data), and (2) proposes changes -- columns to add or drop, a different
learner, different settings, or a column that does not exist yet -- each with a reason and the result
that would show it worked.

It proposes; it never decides. Every proposal is checked by written rules (proposals.validate), saved to
proposals.yaml with `approved: false`, and tested only after a person approves it. It never sees the
locked campaigns: its brief holds only open-campaign results, and column signals are measured on open
campaigns only.

Two ways to run it:
    python src/modeling/brain.py --model 1             # asks Claude (needs ANTHROPIC_API_KEY in the
                                                        # environment or the local, never-committed .env)
    python src/modeling/brain.py --model 1 --manual    # writes the brief only; a person writes proposals
    python src/modeling/brain.py --model 1 --check     # re-checks proposals.yaml after a person edits it

Output: data/processed/modeling_agent/model{n}/ brief.md, proposals.yaml, brain_response.json
        data/processed/modeling_agent/proposal_ledger.csv (every proposal ever made)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # lets `features.*` / `modeling.*` import as a script

from features.spec import blank_reasons, column_roles, load_spec  # noqa: E402
from modeling import model2_baseline  # noqa: E402
from modeling.agent import ROOT, load_modeling_spec, proposal_context  # noqa: E402
from modeling.owners import load_owner_table  # noqa: E402
from modeling.proposals import (column_problem, ledger_upsert, load_proposals, proposals_path,  # noqa: E402
                                save_proposals, validate)

EVAL_DIR = ROOT / "data/processed/evaluation_agent"

TOOL = {
    "name": "submit_proposals",
    "description": "Submit your reading of the results and your proposals for what to test next.",
    "input_schema": {
        "type": "object",
        "required": ["diagnosis", "proposals"],
        "properties": {
            "diagnosis": {"type": "string", "description": (
                "Plain-language reading of the results, 4-8 sentences: which option did best and why you think "
                "so, what the one-change-at-a-time comparison says (columns vs. learner vs. settings), and what "
                "that suggests about the data. Say what is uncertain.")},
            "proposals": {"type": "array", "items": {
                "type": "object",
                "required": ["title", "reason", "expected"],
                "properties": {
                    "title": {"type": "string", "description": "A few words."},
                    "reason": {"type": "string", "description": "Why this could help, grounded in the results."},
                    "expected": {"type": "string", "description": "What result would show it worked, or not."},
                    "add_columns": {"type": "array", "items": {"type": "string"}},
                    "drop_columns": {"type": "array", "items": {"type": "string"}},
                    "family": {"type": "string", "enum": ["forest", "lightgbm", "logistic", "linear"]},
                    "params": {"type": "object"},
                    "new_column": {"type": "object", "required": ["name", "definition"], "properties": {
                        "name": {"type": "string"},
                        "definition": {"type": "string", "description": (
                            "Exactly how to compute it from data dated before the campaign's cutoff day.")}}},
                }}},
        },
    },
}


# --------------------------------------------------------------------------- what the brain is told
def load_context(model_id: int, spec: dict, processed: Path) -> dict:
    """The latest hand-off, its verdict, the table, and the rules -- open campaigns only."""
    mcfg = spec["models"][model_id]
    mod_dir = ROOT / spec["output"]["dir"] / f"model{model_id}"
    manifest = json.loads((mod_dir / "manifest.json").read_text())
    verdict = json.loads((EVAL_DIR / f"model{model_id}" / "verdict.json").read_text())
    if verdict["run_id"] != manifest["run_id"]:
        raise SystemExit(f"[STOP] model {model_id}: the verdict is for run {verdict['run_id']} but the latest hand-off "
                         f"is {manifest['run_id']}; run the Evaluation Agent first")
    table = load_owner_table(model_id, processed)
    if model_id == 2:
        table = table.assign(usual_spend=model2_baseline.usual_spend(table))
    spec_2a = load_spec(ROOT / spec["input"]["feature_spec"])
    descs = {c: b.get("desc", "") for sec in ("profile", "outcome", "filter", "flag") for c, b in spec_2a[sec].items()}
    ctx = proposal_context(mcfg, spec, table, manifest["candidates"], column_roles(spec_2a), blank_reasons(spec_2a))
    return {"model_id": model_id, "mcfg": mcfg, "spec": spec, "manifest": manifest, "verdict": verdict,
            "table": table[~table["campaign"].isin(manifest["locked_campaigns"])], "descs": descs, "ctx": ctx,
            "out_dir": mod_dir, "round": int(manifest["iteration"]) + 1}


def column_signal(table: pd.DataFrame, cols: list[str], label: str) -> dict:
    """Rank correlation of each column with the answer, on open campaigns only (a rough hint, not a test)."""
    y = table[label].astype(float)
    return {c: float(table[c].astype(float).corr(y, method="spearman")) for c in cols}


def fmt_gap(r: dict) -> str:
    return f"{r['gap']:+.4f} (95% range {r['low']:+.4f} to {r['high']:+.4f}, {r['status']})"


def describe(cand: dict, default: dict) -> str:
    """A candidate's exact difference from the owner's model, so the brain never has to guess it."""
    parts = []
    added = [f for f in cand["features"] if f not in default["features"]]
    dropped = [f for f in default["features"] if f not in cand["features"]]
    if added:
        parts.append(f"adds columns {added}")
    if dropped:
        parts.append(f"drops columns {dropped}")
    if cand["family"] != default["family"]:
        parts.append(f"learner {cand['family']} with settings {cand['params']}")
    else:
        diff = {k: v for k, v in cand["params"].items() if default["params"].get(k) != v}
        if diff:
            parts.append(f"settings {diff}")
    return "; ".join(parts) or "nothing"


def build_brief(c: dict) -> str:
    """Everything the brain (or a person playing it) needs, in plain text. No locked-campaign result appears."""
    mcfg, v, m, ctx = c["mcfg"], c["verdict"], c["manifest"], c["ctx"]
    default = ctx["default"]
    owner_cols = default["features"]
    candidates_free = [col for col in c["table"].columns if col not in owner_cols]
    usable = [col for col in candidates_free if not column_problem(col, ctx)]
    unusable = {col: column_problem(col, ctx) for col in candidates_free
                if ctx["roles"].get(col) in ("profile", "context") and column_problem(col, ctx)}
    signal = column_signal(c["table"], owner_cols + usable, mcfg["label"])
    lines = [
        f"# Brief: Model {c['model_id']}, {mcfg['name']} (round {c['round']})", "",
        "## Your job",
        "You are advising a data scientist. Read these results, explain why they came out this way, and propose "
        f"at most {c['spec']['brain']['max_proposals']} changes worth testing next. Your proposals are checked by "
        "rules and tested only if a person approves them, so be specific and honest about uncertainty. Prefer "
        "proposals that change ONE thing (columns, learner, or settings), so a result can be traced to its cause. "
        "You may also suggest columns that do not exist yet (new_column), with an exact definition using only data "
        "from before the campaign's cutoff day.", "",
        "## The model",
        f"- Task: {mcfg['task']}. Answer column: `{mcfg['label']}`: {mcfg['label_desc']}",
        f"- Owner: {mcfg['owner']}. Owner's model: `{v['owner_model']}` ({default['family']}, settings "
        f"{default['params']}) on {len(owner_cols)} columns.",
        f"- Rows outside the locked campaigns: {len(c['table']):,}, across {c['table']['campaign'].nunique()} campaigns.",
        "",
        "## How it is tested",
        "Each test campaign is predicted by models trained only on campaigns that had ENDED before it was planned "
        f"(forward chaining). Measure: `{v['primary_metric']}`. Gaps compare an option with the owner's model, with a "
        "95% range from resampling whole households; 'inconclusive' means the range includes 0. Some later campaigns "
        "are locked away for a final check and are not described here.", "",
        "## Results on the open test campaigns",
        f"- Verdict: **{v['verdict']}**. " + " ".join(v["reasons"]),
        f"- Owner's model vs. baseline: {fmt_gap(v['vs_baseline'])}",
        f"- Calibration: {v['calibration']['summary']}",
        *[f"- `{cid}` vs. owner's model: {fmt_gap(r)}. Exactly what it changes: {describe(v['candidates'][cid], default)}"
          for cid, r in v["alternatives"].items()],
        "", "### Questions the Evaluation Agent raised", *[f"- ({q['who']}) {q['question']}" for q in v["questions"]],
        "", "## Owner's columns (rank correlation with the answer, open campaigns"
        + ("; for uplift this is NOT response to the mailer, only a hint" if mcfg["task"] == "uplift" else "") + ")",
        *[f"- `{col}` ({signal[col]:+.3f}): {c['descs'].get(col) or 'no written description'}" for col in owner_cols],
        "", "## Columns you may add (they pass every safety rule)",
        *[f"- `{col}` ({signal[col]:+.3f}): {c['descs'].get(col) or 'no written description'}" for col in usable],
        "", "## Columns that exist but cannot be added as they are (a new_column idea could fix some)",
        *[f"- `{col}`: {why}" for col, why in unusable.items()],
        "", "## What you may change",
        f"- Learners: forest, lightgbm, logistic{', linear' if mcfg['task'] == 'regress' else ''}"
        + (" (uplift: each is used as a pair, one for mailed and one for unmailed households)" if mcfg["task"] == "uplift" else "")
        + (" (linear is for dollar answers only)" if mcfg["task"] != "regress" else ""),
        "- Settings (others are fixed; allowed ranges): " + "; ".join(
            f"{fam}: " + (", ".join(f"{k} {lo}-{hi}" for k, (lo, hi) in lim.items()) or "none")
            for fam, lim in c["spec"]["brain"]["allowed_params"].items()),
        f"- Never as inputs: {', '.join(c['spec']['defaults']['never_inputs'])} (timeline positions or counts that "
        "include future campaigns), the answer, or the treatment column.",
        "", "## Proposals already made for this model", *previous_lines(c), "",
    ]
    return "\n".join(lines)


def previous_lines(c: dict) -> list[str]:
    path = ROOT / c["spec"]["output"]["dir"] / "proposal_ledger.csv"
    if not path.exists():
        return ["- none"]
    led = pd.read_csv(path)
    led = led[led["model"] == c["model_id"]]
    if led.empty:
        return ["- none"]
    return [f"- {r.id} '{r.title}': {r.status}" + (f", open-campaign gap {r.open_gap:+.4f} ({r.open_status})"
                                                    if pd.notna(r.open_gap) else "") for r in led.itertuples()]


# --------------------------------------------------------------------------- asking Claude
def api_key() -> str:
    """From the environment, else from the local .env (gitignored; never committed)."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise SystemExit("[STOP] no ANTHROPIC_API_KEY (environment or .env); use --manual to write proposals by hand")
    return key


def ask_claude(brief: str, cfg: dict, client=None) -> dict:
    """ONE call. The answer must come through the submit_proposals tool, so it is structured; the model
    cannot be forced to use a tool, so it is told to, and anything else stops the run."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key())
    msg = client.messages.create(
        model=cfg["model"], max_tokens=cfg["max_tokens"], tools=[TOOL],
        system="Answer only by calling the submit_proposals tool, once.",
        messages=[{"role": "user", "content": brief}])
    block = next((b for b in msg.content if getattr(b, "type", None) == "tool_use"), None)
    if block is None:
        raise SystemExit("[STOP] the brain did not answer through submit_proposals")
    usage = getattr(msg, "usage", None)
    return {"input": dict(block.input), "model": cfg["model"],
            "usage": {"input_tokens": getattr(usage, "input_tokens", None),
                      "output_tokens": getattr(usage, "output_tokens", None)}}


# --------------------------------------------------------------------------- recording proposals
def stamp(c: dict, proposals: list[dict]) -> list[dict]:
    """Check each proposal against the rules (and against earlier ready ones in the list, so two proposals
    for the same change are caught), write the result onto it, and log it in the ledger. A proposal is
    never compared with itself, so re-checking one that has already been tested is fine."""
    ready = {}
    for p in proposals:
        others = {k: v for k, v in {**c["ctx"]["existing"], **ready}.items() if k != p["id"]}
        status, why, cand = validate(p, {**c["ctx"], "existing": others})
        p.update(status=status, check=why, changes=(cand or {}).get("changes", []), approved=bool(p.get("approved")))
        if cand:
            ready[p["id"]] = cand
    ledger_upsert(ROOT / c["spec"]["output"]["dir"] / "proposal_ledger.csv",
                  [{"model": c["model_id"], "id": p["id"], "round": p["round"], "from": p["from"], "title": p.get("title", ""),
                    "changes": "+".join(p["changes"]), "status": p["status"], "approved": p["approved"]} for p in proposals])
    return proposals


def with_ids(c: dict, raw: list[dict], source: str, taken: set) -> list[dict]:
    """New proposals get the next free id (r<round>_p<k>, never reused), their round and author; empty fields
    are dropped."""
    out, k = [], 0
    for p in raw:
        k += 1
        while f"r{c['round']}_p{k}" in taken:
            k += 1
        pid = f"r{c['round']}_p{k}"
        taken.add(pid)
        out.append({"id": pid, "round": c["round"], "from": source,
                    **{key: val for key, val in p.items() if key not in ("id", "round", "from") and val not in (None, [], {})}})
    return out


def record(c: dict, raw: list[dict], source: str, diagnosis: str | None) -> list[dict]:
    """Add the brain's (or a person's) new proposals, at most max_proposals, to proposals.yaml, checked and logged."""
    existing = load_proposals(c["out_dir"])
    new = with_ids(c, raw[:c["spec"]["brain"]["max_proposals"]], source, {p["id"] for p in existing})
    stamp(c, existing + new)
    save_proposals(c["out_dir"], existing + new)
    if diagnosis is not None:
        (c["out_dir"] / "brain_diagnosis.md").write_text(f"# Brain's reading of the results, round {c['round']}\n\n{diagnosis}\n")
    return new


def recheck(c: dict) -> list[dict]:
    """Re-check every proposal in proposals.yaml: ones a person typed in (no id yet) get an id and from: person;
    edited ones are checked again as edited. Nothing is approved or tested here."""
    props = load_proposals(c["out_dir"])
    taken = {p["id"] for p in props if "id" in p}
    props = [p if "id" in p else with_ids(c, [p], "person", taken)[0] for p in props]
    stamp(c, props)
    save_proposals(c["out_dir"], props)
    return props


def main() -> None:
    parser = argparse.ArgumentParser(description="The brain: propose what to try next for one model")
    parser.add_argument("--model", required=True, choices=["1", "2", "3"])
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--manual", action="store_true", help="write the brief only; a person writes the proposals")
    mode.add_argument("--check", action="store_true", help="re-check every proposal in proposals.yaml (e.g. after editing)")
    args = parser.parse_args()
    spec = load_modeling_spec()
    c = load_context(int(args.model), spec, args.processed)
    brief = build_brief(c)
    (c["out_dir"] / "brief.md").write_text(brief)
    if args.manual:
        print(f"Brief written to {c['out_dir'] / 'brief.md'}. Add proposals to {proposals_path(c['out_dir'])} "
              f"(fields: title, reason, expected, add_columns, drop_columns, family, params, new_column), then run "
              f"with --check.")
        return
    if args.check:
        props = recheck(c)
    else:
        reply = ask_claude(brief, spec["brain"])
        (c["out_dir"] / "brain_response.json").write_text(json.dumps(reply, indent=2))
        props = record(c, reply["input"]["proposals"], "ai", reply["input"]["diagnosis"])
        print(f"Brain ({reply['model']}): {reply['usage']['input_tokens']} tokens in, "
              f"{reply['usage']['output_tokens']} out")
    for p in props:
        print(f"  {p['id']} [{p['status']}] {p.get('title', '')} -- {p['check']}")
    print(f"Review {proposals_path(c['out_dir'])}, set approved: true on the ones to test, then run:\n"
          f"  python src/modeling/agent.py --model {args.model} --mode forward --proposals --iteration {c['round']}\n"
          f"  python src/evaluation/agent.py --model {args.model}")


if __name__ == "__main__":
    main()
