"""Evaluation Agent: judges each model from the Modeling Agent's hand-off and decides what happens next.

For each model it reads data/processed/modeling_agent/model{n}/ (predictions on forward-chaining test
campaigns, never seen in training) and evaluation_spec.yaml, then:

  EVA-01  checks the hand-off: one prediction per household x test campaign per candidate, no blanks,
          every training campaign ended before its test campaign's cutoff, instructions unchanged since
          the run. Any failure stops.
  EVA-02  (uplift) checks that enough mailed households overlap with unmailed ones AND that weighting really
          made the two groups alike (balance); otherwise uplift numbers are not trusted.
  EVA-03  measures the OWNER's model against its baseline, with a 95% interval from resampling whole
          households: pass / fail / inconclusive.
          It also measures every alternative against the owner's model, and the agent's per-fold choices.
  It also shows where any gain comes from -- columns, learner or settings, one change at a time -- and
  writes each tested proposal's result to the proposal ledger.
  then applies the rules in rules.py: approve / send back to modelling / send back to features /
  human_needed. The agent exists to make the process better, not to save work: every genuine question
  (inconclusive results, disagreements with an owner, unchecked choices, anything the Modeling Agent
  flagged) is listed under "Questions for a person", never settled silently.

Everything above uses the OPEN test campaigns only. --unlock reads the locked campaigns once, for the
final check of the owner's model and every candidate; each unlock is recorded, and reading them again
after more changes is flagged, because a second look is no longer a clean test.

Run from the repo folder (after src/modeling/agent.py --mode forward):
    python src/evaluation/agent.py --model all
    python src/evaluation/agent.py --model 1 --unlock

Output: data/processed/evaluation_agent/model{n}/ verdict.json, decisions.json/.csv, report.md
        (--unlock: unlock.json, unlock_report.md, unlock_history.json)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # lets `evaluation.*` / `modeling.*` import as a script

from evaluation.metrics import balance, cluster_bootstrap, forecast_coverage, metric, within_campaign_capture  # noqa: E402
from evaluation.rules import gap_status, route  # noqa: E402
from modeling.agent import SPEC_PATH as MODELING_SPEC_PATH, file_hash, load_modeling_spec  # noqa: E402
from modeling.decision_log import DecisionLog, check  # noqa: E402
from modeling.model1_evaluation import CUTOFF_GAP  # noqa: E402
from modeling.proposals import AXES, ledger_upsert  # noqa: E402

SPEC_PATH = Path(__file__).with_name("evaluation_spec.yaml")
ROOT = Path(__file__).resolve().parents[2]
# In the bootstrap, capture uses one random tie-break per resample (as John's bootstrap_capture does)
FAST = {"within_campaign_capture": lambda d: within_campaign_capture(d["campaign"], d["label"], d["score"], n_seeds=1)}


def load_evaluation_spec(path: Path = SPEC_PATH) -> dict:
    return yaml.safe_load(path.read_text())


def scored_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return df.assign(score=df[column])


def compare(df: pd.DataFrame, name: str, a: str, b: str | None, reps: int, seed: int) -> dict:
    """How much better ranking `a` is than `b` on `name` (positive = a better; b=None means vs. 0, i.e.
    random for Qini), with a 95% household-bootstrap interval."""
    fn, higher = metric(name)
    fast = FAST.get(name, fn)
    sign = 1 if higher else -1

    def stat(d, f):
        return sign * (f(scored_by(d, a)) - (f(scored_by(d, b)) if b else 0.0))
    point = stat(df, fn)
    draws = cluster_bootstrap(df, lambda d: stat(d, fast), reps, seed)
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return {"gap": float(point), "low": float(low), "high": float(high), "status": gap_status(low, high),
            "sd": float(np.nanstd(draws))}


def wide(preds: pd.DataFrame) -> pd.DataFrame:
    """One row per household x test campaign, with each candidate's score in its own column."""
    keys = ["household_key", "campaign"]
    base_cols = [c for c in preds.columns if c not in ("score", "candidate_id", "is_default", "chosen", "source",
                                                       "p_treated", "p_control", "train_set_id")]
    base = preds[preds["is_default"]][base_cols].set_index(keys)
    scores = preds.pivot_table(index=keys, columns="candidate_id", values="score", aggfunc="first")
    chosen = preds[preds["chosen"]].set_index(keys)["score"].rename("agent_choice")
    return base.join(scores).join(chosen).reset_index()


def newest_campaign(d: pd.DataFrame, primary: str, default: str, cfg: dict) -> dict | None:
    """The owner's model on the newest finished test campaign, against its usual range on the campaigns before it.
    Only campaigns big enough to mean something count (cfg: size = positives | rows | treated, min_size); the
    usual range is the earlier campaigns' mean +/- width_sd standard deviations, from at least min_history of them.
    None when there is too little history to say what is usual."""
    fn, higher = metric(primary)
    size_of = {"positives": lambda g: g["label"].sum(), "rows": len, "treated": lambda g: g["treatment"].sum()}[cfg["size"]]
    rows = [{"campaign": int(c), "end_day": int(g["end_day"].iloc[0]), "size": int(size_of(g)),
             "score": float(fn(scored_by(g, default)))}
            for c, g in d.groupby("campaign") if size_of(g) >= cfg["min_size"]]
    if len(rows) <= cfg["min_history"]:
        return None
    rows.sort(key=lambda r: (r["end_day"], r["campaign"]))
    *history, newest = rows
    past = np.array([r["score"] for r in history])
    low, high = past.mean() - cfg["width_sd"] * past.std(ddof=1), past.mean() + cfg["width_sd"] * past.std(ddof=1)
    worse = newest["score"] < low if higher else newest["score"] > high
    return {**newest, "usual_low": float(low), "usual_high": float(high), "n_history": len(history),
            "higher_is_better": higher, "worse_than_usual": bool(worse)}


def contract(model_id: int, preds: pd.DataFrame, manifest: dict, mspec: dict, passed: list, processed: Path) -> None:
    keys = ["household_key", "campaign", "candidate_id"]
    check(not preds.duplicated(keys).any(), f"model {model_id}: one prediction per household x test campaign x candidate", passed)
    check(not preds["score"].isna().any(), f"model {model_id}: no blank scores", passed)
    per_row = preds[preds["chosen"]].groupby(["household_key", "campaign"]).size()
    check((per_row == 1).all() and len(per_row) == int(preds["is_default"].sum()),
          f"model {model_id}: exactly one agent choice per household x test campaign", passed)
    table = pd.read_parquet(processed / mspec["models"][model_id]["table"],
                            columns=["campaign", "end_day"]).drop_duplicates().set_index("campaign")["end_day"]
    late = [f["test_campaign"] for f in manifest["folds"]
            if any(table[c] >= f["cutoff_day"] for c in f["train_campaigns"])]
    check(not late, f"model {model_id}: every training campaign ended before its test campaign's cutoff", passed)
    check(manifest["spec_hash"] == file_hash(MODELING_SPEC_PATH),
          f"model {model_id}: modeling instructions unchanged since the hand-off was written", passed)
    check(not preds["locked"].any() and not set(manifest["locked_campaigns"]) & set(preds["campaign"]),
          f"model {model_id}: no locked campaign in the open hand-off", passed)


def attribution(cands: dict, alternatives: dict) -> dict:
    """Where a gain would come from: for each kind of change (columns / learner / settings), the best option
    that changes ONLY that, compared with the owner's model. Options changing several things are listed
    separately, since their gain cannot be split."""
    out = {}
    for axis in AXES:
        single = {c: r for c, r in alternatives.items() if cands[c]["changes"] == [axis]}
        best = max(single, key=lambda c: single[c]["gap"]) if single else None
        out[axis] = {"tried": sorted(single), "best": best, **(single[best] if best else {})}
    out["combined"] = sorted(c for c in alternatives if len(cands[c]["changes"]) > 1)
    return out


def ledger_rows(model_id: int, cands: dict, results: dict, prefix: str) -> list[dict]:
    """One ledger update per tested proposal: its gap over the owner's model on open or locked campaigns."""
    return [{"model": model_id, "id": c, "approved": True,
             **{f"{prefix}_{k}": results[c][k] for k in ("gap", "low", "high", "status")}}
            for c in results if cands[c]["source"] == "proposal"]


def calibration(kind: str, cfg: dict, d: pd.DataFrame, default: str) -> dict:
    p = d[default]
    if kind == "forecast_range":
        pc = d.assign(var=p * (1 - p)).groupby("campaign").agg(
            expected=(default, "sum"), chance_var=("var", "sum"), actual=("label", "sum"),
            start_day=("start_day", "first"), end_day=("end_day", "first")).reset_index()
        cov = forecast_coverage(pc)
        return {"ok": cov["coverage"] >= cfg["min_coverage"], "coverage": cov["coverage"], "n_campaigns": cov["n"],
                "expected_total": float(pc["expected"].sum()), "actual_total": float(pc["actual"].sum()),
                "summary": f"actual inside the honest forecast range for {cov['coverage']:.0%} of {cov['n']} campaigns "
                           f"(need {cfg['min_coverage']:.0%})"}
    if kind == "total_ratio":
        ratio = float(p.sum() / d["label"].sum())
        return {"ok": abs(ratio - 1) <= cfg["max_abs_rel_error"], "ratio": ratio,
                "summary": f"predicted total / actual total = {ratio:.2f} (need within +/-{cfg['max_abs_rel_error']:.0%})"}
    # att: mean predicted uplift on comparable mailed households vs. the weighted estimate of the real effect
    o = d[d["in_overlap"]]
    t, c = o[o["treatment"] == 1], o[o["treatment"] == 0]
    w = c["propensity"] / (1 - c["propensity"])
    att = float(t["label"].mean() - (w * c["label"]).sum() / w.sum())
    predicted = float(t[default].mean())
    return {"ok": abs(predicted - att) <= cfg["max_abs_error"], "predicted_uplift": predicted, "weighted_att": att,
            "summary": f"mean predicted uplift on mailed households {predicted:.3f} vs. weighted estimate {att:.3f} "
                       f"(need within {cfg['max_abs_error']})"}


def evaluate_model(model_id: int, spec: dict, mspec: dict, iteration: int | None, modeling_dir: Path | None = None,
                   processed: Path | None = None, out_root: Path | None = None) -> dict:
    """modeling_dir / processed / out_root default to the folders in the instruction files; the lead agent's
    replay points them at its own dated folders."""
    t0, passed = time.time(), []
    ecfg, mcfg = spec["models"][model_id], mspec["models"][model_id]
    modeling_dir = modeling_dir or ROOT / spec["input"]["dir"]
    in_dir = modeling_dir / f"model{model_id}"
    preds = pd.read_parquet(in_dir / "predictions.parquet")
    manifest = json.loads((in_dir / "manifest.json").read_text())
    log = DecisionLog("evaluation", model_id, manifest["run_id"])
    contract(model_id, preds, manifest, mspec, passed, processed or ROOT / mspec["input"]["dir"])
    log.record("EVA-01", f"hand-off from run {manifest['run_id']}: {len(manifest['folds'])} test campaigns",
               "accept hand-off", "rule", "All contract checks passed (see Checks).")

    default, reps, seed = mcfg["default"], spec["bootstrap"]["reps"], spec["bootstrap"]["seed"]
    d = wide(preds)
    primary = ecfg["primary"]
    facts = {"iteration": manifest["iteration"] if iteration is None else iteration}

    # every genuine question goes to a person: (key, who should answer, the question). The key names the kind of
    # question and stays the same from run to run, so the lead agent can tell a question it already asked from a new one.
    questions = []
    if mcfg["task"] == "uplift":
        pcfg = mcfg["propensity"]
        share = float(d.loc[d["treatment"] == 1, "in_overlap"].mean())
        gaps = balance(d[d["in_overlap"]], pcfg["balance_columns"], "ipw_weight")
        worst = max(gaps, key=gaps.get)
        facts["comparison_ok"] = share >= ecfg["min_overlap_share"] and gaps[worst] <= pcfg["unusable_above"]
        log.record("EVA-02", f"{share:.0%} of mailed test rows overlap with unmailed ones; largest gap after weighting: "
                   f"{worst} {gaps[worst]:.2f}", "continue" if facts["comparison_ok"] else "escalate",
                   "rule" if facts["comparison_ok"] else "human_needed",
                   f"Need overlap >= {ecfg['min_overlap_share']:.0%} and every gap <= {pcfg['unusable_above']} for weighted "
                   "uplift numbers to be usable.", evidence={"overlap_share": share, "balance": gaps})
        if facts["comparison_ok"] and gaps[worst] > pcfg["balanced_below"]:
            questions.append(("uplift_balance", "team", f"After weighting, mailed and unmailed households still differ ({worst} gap "
                              f"{gaps[worst]:.2f}; balanced is below {pcfg['balanced_below']}). Uplift numbers below are "
                              "approximate -- is that good enough to act on?"))

    vs_base = compare(d, primary, default, None if mcfg["task"] == "uplift" else "baseline_score", reps, seed)
    facts["gap_status"] = vs_base["status"]
    log.record("EVA-03", f"owner's {default} vs. {ecfg['baseline_name']} on {primary}: {vs_base['gap']:+.4f} "
               f"(95% interval {vs_base['low']:+.4f} to {vs_base['high']:+.4f})", vs_base["status"], "rule",
               "Interval from resampling whole households; above 0 = reliably better.", evidence=vs_base)
    vs_rule = compare(d, primary, default, "baseline_score", reps, seed) if mcfg["task"] == "uplift" else None

    cands = manifest["candidates"]
    alternatives = {c: compare(d, primary, c, default, reps, seed) for c in cands if c != default}
    agent_choice = compare(d, primary, "agent_choice", default, reps, seed)
    better = [c for c, r in alternatives.items() if r["status"] == "pass"]
    facts["better_alternative"] = max(better, key=lambda c: alternatives[c]["gap"]) if better else None
    facts["all_within_noise"] = all(r["status"] == "inconclusive" for r in alternatives.values())
    picks = pd.Series([f["chosen"] for f in manifest["folds"]])
    facts["stability"] = float(picks.value_counts(normalize=True).iloc[0])

    cal = calibration(ecfg["calibration"]["kind"], ecfg["calibration"], d, default)
    facts["calibration_ok"] = cal["ok"]
    log.record("EVA-CAL", cal["summary"], "ok" if cal["ok"] else "not ok", "rule",
               "The Decision Lab adds these numbers up, so their level must match reality.", evidence=cal)

    verdict, decided_by, reasons = route(facts, {"max_iterations": spec["max_iterations"],
                                                 "stability_min_share": spec["stability_min_share"],
                                                 "approve_requires_human": ecfg["approve_requires_human"],
                                                 "calibration_hint": ecfg.get("calibration_hint")})
    log.record("EVA-VERDICT", f"verdict for the owner's model: {verdict}", verdict, decided_by, " ".join(reasons),
               evidence={k: v for k, v in facts.items()})

    newest = newest_campaign(d, primary, default, ecfg["newest_campaign"]) if "newest_campaign" in ecfg else None
    if newest and newest["worse_than_usual"]:
        log.record("EVA-NEW", f"newest campaign {newest['campaign']}: {primary} {newest['score']:.4f}, usual range "
                   f"{newest['usual_low']:.4f} to {newest['usual_high']:.4f}", "ask the owner", "human_needed",
                   "The model did worse than usual on the newest campaign; something may have changed.", evidence=newest)
        questions.append(("worse_than_usual", mcfg["owner"].split(" (")[0],
                          f"On the newest campaign ({newest['campaign']}) the model scored {newest['score']:.4f} on "
                          f"{primary}, outside its usual range of {newest['usual_low']:.4f} to {newest['usual_high']:.4f} "
                          f"(from {newest['n_history']} earlier campaigns). Has something changed?"))
    questions += open_questions(model_id, mcfg, ecfg, d, default, vs_base, vs_rule, alternatives, cal, in_dir, cands)
    gains = attribution(cands, alternatives)
    ledger_upsert(modeling_dir / "proposal_ledger.csv", ledger_rows(model_id, cands, alternatives, "open"))
    for _, who, q in questions:
        log.record("EVA-Q", q, f"ask the {who}", "human_needed", "A question the rules should not settle on their own.")
    extra = {name: metric(name)[0](scored_by(d, default)) for name in [primary] + ecfg["also_report"]}
    # every option's own score on the primary measure, and the simple rule's (random targeting is 0 by definition)
    scores = {c: float(metric(primary)[0](scored_by(d, c))) for c in cands}
    scores["baseline_score"] = (0.0 if mcfg["task"] == "uplift" else float(metric(primary)[0](scored_by(d, "baseline_score"))))
    result = {"model_id": model_id, "verdict": verdict, "decided_by": decided_by, "reasons": reasons,
              "owner_model": default, "primary_metric": primary, "vs_baseline": vs_base, "vs_rule": vs_rule,
              "alternatives": alternatives, "attribution": gains, "candidates": cands,
              "locked_campaigns": manifest["locked_campaigns"], "agent_choice": agent_choice, "calibration": cal, "facts": facts,
              "metrics_owner_model": extra, "scores": scores, "newest_campaign": newest, "questions": [{"key": k, "who": w, "question": q} for k, w, q in questions], "checks": passed, "run_id": manifest["run_id"]}
    out = (out_root or ROOT / spec["output"]["dir"]) / f"model{model_id}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.json").write_text(json.dumps(result, indent=2, default=float))
    log.save(out)
    (out / "report.md").write_text(render_report(model_id, mcfg, ecfg, result, log, time.time() - t0))
    print(f"Model {model_id}: {verdict} ({decided_by}) -- {reasons[0]}")
    return result


def open_questions(model_id, mcfg, ecfg, d, default, vs_base, vs_rule, alternatives, cal, in_dir, cands) -> list[tuple[str, str, str]]:
    """Everything a person should look at: anything inconclusive, anything that disagrees with the owner,
    any choice the agents made that the team has not signed off, and whatever the Modeling Agent flagged."""
    out = []
    owner = mcfg["owner"].split(" (")[0]
    if vs_base["status"] == "inconclusive":
        out.append(("baseline_inconclusive", owner, f"Can't tell whether the model beats {ecfg['baseline_name']}: gap {vs_base['gap']:+.4f}, "
                    f"range {vs_base['low']:+.4f} to {vs_base['high']:+.4f} includes zero."))
    for c, r in alternatives.items():
        if r["status"] == "pass":
            out.append((f"alternative_beats_owner:{c}", owner, f"Alternative `{c}` ({' + '.join(cands[c]['changes'])} changed) beats the owner's "
                        f"`{default}` by {r['gap']:+.4f} on {ecfg['primary']} (range {r['low']:+.4f} to "
                        f"{r['high']:+.4f}). Adopt it? Confirm on the locked campaigns first (--unlock)."))
    if not cal["ok"]:
        out.append(("calibration", owner, f"The model's numbers don't match reality: {cal['summary']}."))
    if mcfg["task"] == "regress" and vs_base["status"] != "pass":
        out.append(("baseline_as_good", owner, f"{ecfg['baseline_name']} is as good as the model on unseen campaigns; should it be the "
                    "forecast to beat?"))
    if mcfg["task"] == "uplift":
        raw = metric("raw_stratified_qini")[0](scored_by(d, default))
        out.append(("uplift_unweighted_qini", owner, f"Unweighted Qini {raw:+.4f} vs. weighted {vs_base['gap']:+.4f}: the unweighted score mostly "
                    f"reflects who got mailed. Against '{ecfg['rule_name']}' the model scores {vs_rule['gap']:+.4f} "
                    f"(range {vs_rule['low']:+.4f} to {vs_rule['high']:+.4f})."))
    if ecfg.get("measure_note"):
        out.append(("measure_note", "team", ecfg["measure_note"]))
    # The Modeling Agent's own flags. Its recommendation (MOD-06) is already asked above, with an interval,
    # whenever the evaluation confirms it; otherwise it is asked here with the reason it was not confirmed.
    flagged = json.loads((in_dir / "decisions.json").read_text())
    for r in flagged:
        if r["decided_by"] != "human_needed":
            continue
        if r["check_id"] == "MOD-06":
            pick = r["action"].removeprefix("recommend ").removesuffix(" to the owner")
            if alternatives.get(pick, {}).get("status") == "pass":
                continue
            out.append((f"recommendation_unproven:{pick}", "team", f"Modeling Agent recommends `{pick}`, but its edge over the owner's model is within "
                        "noise on the evaluation's interval. Treat it as unproven."))
            continue
        out.append((r["check_id"], "team", f"Modeling Agent ({r['check_id']}): {r['finding']}. {r['rationale']}"))
    return out


def render_report(model_id, mcfg, ecfg, r, log, runtime) -> str:
    def row(name, x):
        return f"| {name} | {x['gap']:+.4f} | {x['low']:+.4f} to {x['high']:+.4f} | {x['status']} |"
    lines = [f"# Evaluation Agent report: Model {model_id}, {mcfg['name']}", "",
             f"**Verdict: {r['verdict']}** (decided by: {r['decided_by']})", "", *[f"- {x}" for x in r["reasons"]], "",
             f"Judged: the owner's model (`{r['owner_model']}`, {mcfg['owner']}) on forward-chaining test campaigns "
             f"(locked, not used here: {r['locked_campaigns'] or 'none'}). "
             f"Primary measure: `{r['primary_metric']}`. Intervals: 95%, resampling whole households.", "",
             "## How the owner's model compares", "",
             "| comparison | gap (positive = first is better) | 95% interval | status |", "|---|---|---|---|",
             row(f"owner's model vs. {ecfg['baseline_name']}", r["vs_baseline"])]
    if r["vs_rule"]:
        lines.append(row(f"owner's model vs. {ecfg['rule_name']}", r["vs_rule"]))
    lines += [row(f"{c} vs. owner's model", x) for c, x in r["alternatives"].items()]
    lines += [row("agent's per-fold choices vs. owner's model", r["agent_choice"]), "",
              "## Where a gain would come from (one change at a time)", "",
              *attribution_lines(r["attribution"]), "",
              f"## Every option's own score (`{r['primary_metric']}`)", "",
              *[f"- {'the simple rule (' + ecfg['baseline_name'] + ')' if c == 'baseline_score' else '`' + c + '`'}: {v:.4f}"
                for c, v in r["scores"].items()], "",
              "## Numbers", "", *[f"- {k}: {v:.4f}" for k, v in r["metrics_owner_model"].items()],
              f"- calibration: {r['calibration']['summary']}",
              f"- agent's choice stability: most common pick in {r['facts']['stability']:.0%} of folds", "",
              "## Questions for a person", "",
              *([f"{i}. **{q['who']}:** {q['question']}" for i, q in enumerate(r["questions"], 1)] or ["- none"]), "",
              "## Checks", "", *[f"- [OK] {p}" for p in r["checks"]], "",
              f"Decisions logged: {len(log.records)} ({len(log.human_needed())} need a person). Runtime {runtime:.0f}s."]
    return "\n".join(lines)


def attribution_lines(gains: dict) -> list[str]:
    plain = {"pass": "reliably better", "fail": "reliably worse", "inconclusive": "can't tell (range includes 0)"}
    lines = []
    for axis in AXES:
        g = gains[axis]
        if not g["best"]:
            lines.append(f"- **{axis.capitalize()} only:** no option changed only this.")
            continue
        lines.append(f"- **{axis.capitalize()} only:** best of {len(g['tried'])} is `{g['best']}`, {g['gap']:+.4f} vs. the "
                     f"owner's model (range {g['low']:+.4f} to {g['high']:+.4f}): {plain[g['status']]}.")
    if gains["combined"]:
        lines.append(f"- Changed several things at once (gain cannot be split): {', '.join(gains['combined'])}.")
    return lines


def unlock_model(model_id: int, spec: dict, mspec: dict) -> dict:
    """The final check: the owner's model and every candidate on the locked campaigns, read once. Each unlock is
    recorded; an earlier unlock makes this one a second look, which is flagged rather than trusted."""
    ecfg, mcfg = spec["models"][model_id], mspec["models"][model_id]
    in_dir = ROOT / spec["input"]["dir"] / f"model{model_id}"
    out = ROOT / spec["output"]["dir"] / f"model{model_id}"
    manifest = json.loads((in_dir / "manifest.json").read_text())
    if not manifest["locked_campaigns"]:
        raise SystemExit(f"[STOP] model {model_id}: no locked campaigns in this hand-off")
    preds = pd.read_parquet(in_dir / "locked_predictions.parquet")
    d, default, cands = wide(preds), mcfg["default"], manifest["candidates"]
    reps, seed, primary = spec["bootstrap"]["reps"], spec["bootstrap"]["seed"], ecfg["primary"]
    history_path = out / "unlock_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    vs_base = compare(d, primary, default, None if mcfg["task"] == "uplift" else "baseline_score", reps, seed)
    alternatives = {c: compare(d, primary, c, default, reps, seed) for c in cands if c != default}
    result = {"model_id": model_id, "run_id": manifest["run_id"], "locked_campaigns": manifest["locked_campaigns"],
              "n_rows": int(len(d)), "previous_unlocks": len(history), "vs_baseline": vs_base,
              "alternatives": alternatives, "recommended": manifest["final"]["recommended"]}
    history.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "run_id": manifest["run_id"],
                    "candidates": sorted(cands)})
    out.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2))
    (out / "unlock.json").write_text(json.dumps(result, indent=2, default=float))
    ledger_upsert(ROOT / mspec["output"]["dir"] / "proposal_ledger.csv", ledger_rows(model_id, cands, alternatives, "locked"))

    def row(name, x):
        return f"- {name}: {x['gap']:+.4f} (range {x['low']:+.4f} to {x['high']:+.4f}), {x['status']}"
    warning = ([f"**Second look:** these campaigns were already read {len(history) - 1} time(s). Anything chosen after "
                "the first look has been partly fitted to them, so this is no longer a clean test. A person decides "
                "how much to trust it."] if len(history) > 1 else [])
    lines = [f"# Locked-campaign check: Model {model_id}, {mcfg['name']}", "", *warning, *([""] if warning else []),
             f"Campaigns {manifest['locked_campaigns']} ({len(d):,} rows), never used for any choice. Measure: `{primary}`; "
             "positive = first is better; 95% intervals from resampling whole households.", "",
             row(f"owner's model vs. {ecfg['baseline_name']}", vs_base),
             *[row(f"`{c}` ({' + '.join(cands[c]['changes']) or 'no change'}) vs. owner's model", x)
               for c, x in alternatives.items()], "",
             f"Modeling Agent's recommendation from the open campaigns: `{result['recommended']}`."]
    if result["recommended"] != default:
        held = alternatives[result["recommended"]]["status"]
        lines.append(f"On the locked campaigns it is: {held}. "
                     + ("It held up." if held == "pass" else "It did not clearly hold up; keep the owner's model unless "
                        "the owner has another reason to switch."))
    (out / "unlock_report.md").write_text("\n".join(lines))
    print(f"Model {model_id}: locked check written ({len(history)} unlock(s) so far) -> {out / 'unlock_report.md'}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation Agent: judge each model and decide what happens next")
    parser.add_argument("--model", default="all", choices=["1", "2", "3", "all"])
    parser.add_argument("--iteration", type=int, default=None, help="override the hand-off's iteration count")
    parser.add_argument("--unlock", action="store_true", help="the final check on the locked campaigns (recorded)")
    parser.add_argument("--modeling", type=Path, default=None, help="folder with the Modeling Agent's model{n} hand-offs")
    parser.add_argument("--processed", type=Path, default=None, help="folder with the model tables")
    parser.add_argument("--out", type=Path, default=None, help="folder for the model{n} verdicts")
    args = parser.parse_args()
    spec, mspec = load_evaluation_spec(), load_modeling_spec()
    for model_id in ([1, 2, 3] if args.model == "all" else [int(args.model)]):
        if args.unlock:
            unlock_model(model_id, spec, mspec)
        else:
            evaluate_model(model_id, spec, mspec, args.iteration, args.modeling, args.processed, args.out)


if __name__ == "__main__":
    main()
