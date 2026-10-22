"""Proposals: ideas for what to try next, from the brain (an AI, see brain.py) or from a person.

A proposal changes the owner's model in up to three ways -- columns (add / drop), learner (family) and
settings (params) -- or suggests a column that does not exist yet (new_column, which only the Feature
Agent can build). Every proposal goes through the same path, whoever wrote it:

  1. validate()   written rules check it is safe and testable (no answer-leaking or timeline columns,
                  blanks that can be filled, a learner the task supports, settings inside allowed ranges,
                  not a repeat of something already tested). Rejections keep their reason.
  2. a person sets `approved: true` in proposals.yaml -- nothing is tested without that.
  3. the Modeling Agent (--proposals) tests approved ones exactly like its own candidates.
  4. the Evaluation Agent judges them and writes the result to the ledger.

The ledger (proposal_ledger.csv) keeps every proposal ever made, approved or not, with its result on the
open test campaigns and, after --unlock, on the locked ones: how many ideas the brain had, how many people
tried, and how many held up.
"""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from modeling.learners import FAMILIES

COMPLEXITY = {"linear": 1, "median_linear": 1, "logistic": 1, "logistic_log": 1, "forest": 2, "lightgbm": 3}
AXES = ("columns", "learner", "settings")
LEDGER_COLUMNS = ["model", "id", "round", "from", "title", "changes", "status", "approved",
                  "open_gap", "open_low", "open_high", "open_status",
                  "locked_gap", "locked_low", "locked_high", "locked_status", "updated"]


def changes(cand: dict, default: dict) -> list[str]:
    """What a candidate changes compared with the owner's default: columns, learner and/or settings.
    Settings only count when the learner is the same (a different learner has different settings)."""
    out = []
    if set(cand["features"]) != set(default["features"]):
        out.append("columns")
    if cand["family"] != default["family"]:
        out.append("learner")
    elif dict(cand["params"]) != dict(default["params"]):
        out.append("settings")
    return out


def base_params(spec: dict, family: str) -> dict:
    """Starting settings for a learner when a proposal names it without settings."""
    return {"forest": spec["forest_params"], "lightgbm": spec["lightgbm_params"],
            "logistic": {"max_iter": 2000}, "logistic_log": {"max_iter": 2000}, "linear": {},
            "median_linear": spec["median_linear_params"]}[family]


def column_problem(col: str, ctx: dict) -> str | None:
    """Why a column cannot be added as an input (None = it can)."""
    d = ctx["defaults"]
    if col not in ctx["dtypes"]:
        return f"{col} is not a column of this model's table"
    if col in d["never_inputs"]:
        return f"{col} is on the never-input list (a timeline position or a count that includes future campaigns)"
    if col in (ctx["label"], ctx["treatment"]):
        return f"{col} is the answer or the treatment column"
    if col not in d["derived_inputs"] and ctx["roles"].get(col) not in d["allowed_roles"]:
        return f"{col} has role {ctx['roles'].get(col)!r}; only profile/context columns are inputs"
    if not pd.api.types.is_numeric_dtype(ctx["dtypes"][col]):
        return f"{col} is text; it needs turning into numbers first (a Feature Agent question)"
    reasons = ctx["blank_reasons"].get(col, [])
    if ctx["has_blanks"].get(col) and (not reasons or any(r not in d["fill_zero_when"] for r in reasons)):
        return f"{col} has blanks that 0 does not describe (reasons {reasons}); deciding how to fill them is a Feature Agent question"
    return None


def params_problem(family: str, params: dict, allowed: dict) -> str | None:
    limits = allowed.get(family, {})
    for k, v in params.items():
        if k not in limits:
            return f"setting {k!r} is not one a proposal may change for {family} (allowed: {sorted(limits) or 'none'})"
        lo, hi = limits[k]
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not lo <= v <= hi:
            return f"setting {k}={v!r} is outside the allowed range [{lo}, {hi}]"
    return None


def validate(p: dict, ctx: dict) -> tuple[str, str, dict | None]:
    """One proposal -> (status, plain reason, candidate or None). status: ready / rejected / needs_feature_agent.
    ctx: task, default (resolved default candidate), existing (resolved candidates by id), dtypes, has_blanks,
    roles, blank_reasons, defaults, label, treatment, spec (the modeling spec).
    Restating something unchanged (the owner's learner, a setting at its current value) is not a change."""
    default, spec = ctx["default"], ctx["spec"]
    family = p.get("family") or default["family"]
    start = default["params"] if family == default["family"] else base_params(spec, family)
    given = {k: v for k, v in (p.get("params") or {}).items() if start.get(k, object()) != v}
    if p.get("new_column"):
        bundled = bool(p.get("add_columns") or p.get("drop_columns") or family != default["family"] or given)
        return "needs_feature_agent", ("the column does not exist yet; the Feature Agent would have to build it"
                                       + ("; the other changes in this proposal wait until it exists" if bundled else "")), None
    add, drop = list(p.get("add_columns") or []), list(p.get("drop_columns") or [])
    for col in add:
        if col in default["features"]:
            return "rejected", f"{col} is already one of the owner's columns", None
        problem = column_problem(col, ctx)
        if problem:
            return "rejected", problem, None
    missing = [c for c in drop if c not in default["features"]]
    if missing:
        return "rejected", f"cannot drop {missing}: not among the owner's columns", None
    features = [c for c in default["features"] if c not in drop] + add
    if not features:
        return "rejected", "dropping every column leaves nothing to learn from", None

    task = "classify" if ctx["task"] == "uplift" else ctx["task"]
    if (family, task) not in FAMILIES:
        return "rejected", f"learner {family!r} cannot do this task ({ctx['task']})", None
    problem = params_problem(family, given, spec["brain"]["allowed_params"])
    if problem:
        return "rejected", problem, None
    params = {**start, **given}

    cand = {"source": "proposal", "family": family, "params": params, "features": features,
            "complexity": COMPLEXITY[family], "proposal_id": p["id"], "title": p.get("title")}
    cand["changes"] = changes(cand, default)
    if not cand["changes"]:
        return "rejected", "this is the owner's model unchanged", None
    for cid, other in ctx["existing"].items():
        if (other.get("family") == family and set(other["features"]) == set(features)
                and dict(other.get("params") or {}) == params):
            return "rejected", f"already tested as `{cid}`", None
    return "ready", "safe and testable: " + " + ".join(cand["changes"]) + " changed", cand


def proposals_path(out_dir: Path) -> Path:
    return out_dir / "proposals.yaml"


def load_proposals(out_dir: Path) -> list[dict]:
    path = proposals_path(out_dir)
    return (yaml.safe_load(path.read_text()) or []) if path.exists() else []


def save_proposals(out_dir: Path, proposals: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    header = ("# Proposals for what to try next. Set `approved: true` on the ones you want tested, then run\n"
              "#   python src/modeling/agent.py --model N --mode forward --proposals --iteration R\n"
              "# You can add your own (from: person); every proposal is re-checked before it runs.\n")
    proposals_path(out_dir).write_text(header + yaml.safe_dump(proposals, sort_keys=False, width=110))


def approved_candidates(proposals: list[dict], ctx: dict) -> dict:
    """The approved proposals, re-checked (a person may have edited them), as candidates for the agent.
    An approved proposal that fails the checks stops the run: a person asked for it, so a person fixes it."""
    out = {}
    for p in proposals:
        if not p.get("approved"):
            continue
        status, reason, cand = validate(p, {**ctx, "existing": {**ctx["existing"], **out}})
        if status != "ready":
            raise SystemExit(f"[STOP] approved proposal {p['id']} cannot be tested: {reason}")
        out[p["id"]] = cand
    return out


def ledger_upsert(path: Path, rows: list[dict]) -> pd.DataFrame:
    """Add or update ledger rows, keyed by (model, id); fields not given keep their old value."""
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=LEDGER_COLUMNS)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for r in rows:
        r = {**r, "updated": now}
        hit = (old["model"].astype(int) == int(r["model"])) & (old["id"].astype(str) == str(r["id"]))
        if hit.any():
            for k, v in r.items():
                old[k] = old[k].astype(object)
                old.loc[hit, k] = v
        else:
            old = pd.concat([old, pd.DataFrame([r])], ignore_index=True)
    old = old.reindex(columns=LEDGER_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    old.to_csv(path, index=False)
    return old
