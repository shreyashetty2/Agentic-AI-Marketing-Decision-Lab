"""Modeling Agent: one shared, instruction-driven modelling step for all three models.

It orchestrates the owners' models; it does not replace them. For each model it reads its section of
modeling_spec.yaml and:

  1. Loads the 2B table with the OWNER's own load_table and checks it (expected counts, column roles,
     never-input columns, treatment column not an input, blank filling). Any failure stops the run.
  2. Builds forward-chaining test folds (John's forward_chain_folds): each test campaign is predicted by
     models trained only on campaigns that had ENDED before its cutoff.
  3. Chooses per fold, using only that fold's training campaigns: every candidate is scored on the last
     few of those campaigns, and a written rule picks one (an alternative must beat the owner's default
     by min_gain; near-ties go to the simpler candidate). Every choice is logged with its scores.
  4. Predicts each test campaign with EVERY candidate (rows marked default / chosen), so the Evaluation
     Agent can judge the owner's model and see whether the agent's choices would have done better.
  5. Refits the owner's default on all campaigns for scoring (final_model.joblib). If the agent prefers
     an alternative, that is logged as a recommendation for the owner (decided_by: human_needed).

The last few test campaigns (locked_test_campaigns) are predicted the same way but written to
locked_predictions.parquet, which the Evaluation Agent reads only on --unlock. --proposals adds the
proposals a person approved in proposals.yaml (from the brain or a person, see proposals.py) as extra
candidates, tested exactly like the agent's own.

--mode reproduce re-runs each owner's own evaluation (70/30 split; John's forward chain for Model 1)
through the agent and stops unless every number matches the owner's report.

Run from the repo folder (after src/features/build_2b.py):
    python src/modeling/agent.py --model all --mode both
    python src/modeling/agent.py --model 1 --mode forward --proposals --iteration 2

Output: data/processed/modeling_agent/model{n}/ predictions.parquet, locked_predictions.parquet,
        manifest.json, decisions.json/.csv, final_model.joblib, report.md
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # lets `features.*` / `modeling.*` import as a script

from evaluation.metrics import balance, fit_propensity, metric  # noqa: E402
from features.spec import blank_reasons, column_roles, load_spec  # noqa: E402
from modeling import model1_baseline, model2_baseline, model3_uplift  # noqa: E402
from modeling.decision_log import DecisionLog, check  # noqa: E402
from modeling.learners import fill_blanks, fit_agent_candidate, importances  # noqa: E402
from modeling.model1_evaluation import (CUTOFF_GAP, bootstrap_capture, campaign_calendar,  # noqa: E402
                                        forward_chain_folds, within_campaign_auc, within_campaign_capture)
from modeling.owners import fit_owner, load_owner_table, owner_features, owner_learner_features  # noqa: E402
from modeling.proposals import approved_candidates, changes, load_proposals  # noqa: E402

SPEC_PATH = Path(__file__).with_name("modeling_spec.yaml")
ROOT = Path(__file__).resolve().parents[2]


def load_modeling_spec(path: Path = SPEC_PATH) -> dict:
    return yaml.safe_load(path.read_text())


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def train_set_id(campaigns) -> str:
    return hashlib.sha1(",".join(map(str, sorted(int(c) for c in campaigns))).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- candidates + table
def resolve_candidates(model_id: int, mcfg: dict) -> dict:
    """candidate id -> its settings plus the concrete column list it uses and what it changes
    compared with the owner's default (columns / learner / settings)."""
    out = {}
    for cid, c in mcfg["candidates"].items():
        if c["source"] == "owner":
            features = owner_learner_features(model_id, c["owner_learner"])
        elif c.get("features") == "owner":
            features = owner_features(model_id)
        else:
            features = mcfg["feature_sets"][c["features"]]
        out[cid] = {**c, "features": list(features)}
    default = out[mcfg["default"]]
    return {cid: {**c, "changes": changes(c, default)} for cid, c in out.items()}


def proposal_context(mcfg: dict, spec: dict, table: pd.DataFrame, cands: dict, roles: dict, reasons: dict) -> dict:
    """Everything proposals.validate needs to check a proposal against this model's table and rules."""
    return {"task": mcfg["task"], "default": cands[mcfg["default"]], "existing": cands,
            "dtypes": table.dtypes.to_dict(), "has_blanks": table.isna().any().to_dict(), "roles": roles,
            "blank_reasons": reasons, "defaults": spec["defaults"], "label": mcfg["label"],
            "treatment": mcfg.get("treatment"), "spec": spec}


def choose_candidate(scores: dict, cands: dict, default: str, min_gain: float, higher_is_better: bool):
    """The selection rule. An alternative replaces the default only if it is better by at least
    min_gain; among those, any within min_gain of the best count as tied, and the simplest wins
    (lower complexity, then fewer columns). Returns (candidate id, plain-language reason)."""
    sign = 1 if higher_is_better else -1
    if default not in scores or np.isnan(scores[default]):
        return default, f"{default} could not be scored; kept as the owner's default"
    base = sign * scores[default]
    winners = [c for c, s in scores.items() if c != default and not np.isnan(s) and sign * s - base >= min_gain]
    if not winners:
        return default, f"no alternative beat the owner's default ({default}) by {min_gain}"
    best = max(sign * scores[c] for c in winners)
    tied = [c for c in winners if best - sign * scores[c] < min_gain]
    pick = min(tied, key=lambda c: (cands[c]["complexity"], len(cands[c]["features"]), c))
    return pick, (f"{pick} beat the owner's default ({default}) by {sign * scores[pick] - base:.4f} "
                  f"(>= {min_gain}); simplest of {len(tied)} within {min_gain} of the best")


def validate_inputs(model_id: int, cands: dict, roles: dict, defaults: dict, treatment, passed: list) -> list[str]:
    """Every column any candidate uses must be safe: not on the never-input list, a profile/context column in
    2A (or one the owner's load_table derives), and never the treatment column. Stops otherwise."""
    used = sorted({f for c in cands.values() for f in c["features"]})
    never = [f for f in used if f in defaults["never_inputs"]]
    check(not never, f"model {model_id}: no candidate uses a never-input column" + (f" {never}" if never else ""), passed)
    bad_role = [f for f in used if f not in defaults["derived_inputs"] and roles.get(f) not in defaults["allowed_roles"]]
    check(not bad_role, f"model {model_id}: every input is a profile/context or owner-derived column"
          + (f" {bad_role}" if bad_role else ""), passed)
    check(treatment not in used, f"model {model_id}: treatment column is not an input", passed)
    return used


def add_propensity(model_id: int, mcfg: dict, table: pd.DataFrame, log: DecisionLog) -> pd.DataFrame:
    """Each row's chance of being mailed, inverse-propensity weights, and whether the row is in the region
    where mailed and unmailed households overlap. Then CHECK that the weighting worked: if mailed and
    unmailed households still differ after weighting, uplift numbers are flagged for a person."""
    pcfg, treatment = mcfg["propensity"], mcfg["treatment"]
    e = fit_propensity(table, mcfg["feature_sets"][pcfg["features"]], treatment, folds=pcfg["folds"],
                       clip=tuple(pcfg["clip"]))
    t = table[treatment].to_numpy() == 1
    lo, hi = pcfg["overlap"]
    table = table.assign(propensity=e, ipw_weight=np.where(t, 1 / e, 1 / (1 - e)), in_overlap=(e >= lo) & (e <= hi))
    share = float(table.loc[t, "in_overlap"].mean())
    gaps = balance(table[table["in_overlap"]], pcfg["balance_columns"], "ipw_weight", treatment)
    worst = max(gaps, key=gaps.get)
    evidence = {"overlap_share_treated": share, "balance_after_weighting": gaps}
    finding = (f"{share:.0%} of mailed rows overlap with unmailed ones; after weighting the largest remaining gap is "
               f"{worst} at {gaps[worst]:.2f} (balanced below {pcfg['balanced_below']})")
    if gaps[worst] <= pcfg["balanced_below"]:
        log.record("MOD-01", finding, "use the weights", "rule",
                   "Weighting made mailed and unmailed households comparable on every checked column.", evidence=evidence)
    else:
        log.record("MOD-01", finding, "use the weights, flag the remaining gap", "human_needed",
                   "Mailed and unmailed households still differ after weighting, so every uplift number built on "
                   "these weights is approximate"
                   + (" and may be badly confounded" if gaps[worst] > pcfg["unusable_above"] else "")
                   + ". A person should judge whether the comparison is good enough.", evidence=evidence)
    return table


def model_scripts_on_git(model_id: int) -> list[str] | None:
    """The model's scripts as committed to the repo (model{n}_*.py), or None if git is not available.
    Committed only, so someone's local, unshared experiments do not count."""
    try:
        out = subprocess.run(["git", "ls-files", f"src/modeling/model{model_id}_*.py"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return None
    return sorted(Path(f).name for f in out)


def check_scripts(model_id: int, mcfg: dict, log: DecisionLog) -> None:
    """The agent calls the owner's code through a fixed list (owners.py). If a teammate adds a new script for
    this model -- possibly a newer, better model -- the agent would silently keep using the old one, so every
    run compares the repo with the instruction file's list and flags anything new for a person."""
    known = {mcfg["scripts"]["current"], *mcfg["scripts"].get("others", [])}
    found = model_scripts_on_git(model_id)
    if found is None:
        log.record("MOD-09", "git not available; could not check for new model scripts", "continue", "human_needed",
                   "A teammate may have added a newer model the agent does not know about; check src/modeling by hand.")
        return
    new = [f for f in found if f not in known]
    if new:
        log.record("MOD-09", f"new script(s) for Model {model_id} in the repo: {new}; the agent's current model "
                   f"comes from {mcfg['scripts']['current']}", "keep the current model, ask the owner", "human_needed",
                   "If one of these is the owner's new current model, add it to modeling_spec.yaml (scripts) and "
                   "owners.py; until then the agent judges an older model.", evidence={"new": new, "known": sorted(known)})


def prepare_table(model_id: int, mcfg: dict, spec: dict, processed: Path, passed: list, log: DecisionLog,
                  proposals: list[dict] | None = None, exact_counts: bool = True):
    """The owner's table, checked against the instructions, with blanks filled and (Model 3) propensity weights.
    proposals: the model's proposals.yaml entries; the approved ones become extra candidates.
    exact_counts: check the row counts recorded in the instructions. Those are the counts of the full dataset,
    so they belong with reproducing the owners' numbers; on any other table (new data, the lead agent's replay)
    the counts are recorded in the manifest instead, and the lead agent checks them against the Feature Agent."""
    check_scripts(model_id, mcfg, log)
    table = load_owner_table(model_id, processed)
    label, exp = mcfg["label"], mcfg["expected"]
    treatment = mcfg.get("treatment")
    check(len(table) > 0 and table[label].notna().all(), f"model {model_id}: {len(table):,} rows, every label filled", passed)
    if exact_counts:
        check(len(table) == exp["rows"], f"model {model_id}: {len(table):,} rows (expected {exp['rows']:,})", passed)
        if "positives" in exp:
            check(int(table[label].sum()) == exp["positives"], f"model {model_id}: {exp['positives']} positive labels", passed)
        if treatment:
            n_t = int(table[treatment].sum())
            check(n_t == exp["treated"] and len(table) - n_t == exp["control"],
                  f"model {model_id}: {n_t:,} treated / {len(table) - n_t:,} control", passed)

    spec_2a = load_spec(ROOT / spec["input"]["feature_spec"])
    roles, reasons, d = column_roles(spec_2a), blank_reasons(spec_2a), spec["defaults"]
    cands = resolve_candidates(model_id, mcfg)
    if proposals:
        extra = approved_candidates(proposals, proposal_context(mcfg, spec, table, cands, roles, reasons))
        check(bool(extra), f"model {model_id}: {len(extra)} approved proposal(s) to test", passed)
        cands = {**cands, **extra}
    used = validate_inputs(model_id, cands, roles, d, treatment, passed)
    agent_cols = sorted({f for c in cands.values() if c["source"] != "owner" for f in c["features"]} - set(d["derived_inputs"]))
    table = fill_blanks(table, agent_cols, reasons, d["fill_zero_when"])
    blanks = [f for f in used if table[f].isna().any()]
    check(not blanks, f"model {model_id}: no blanks left in any input" + (f" {blanks}" if blanks else ""), passed)

    if treatment:
        table = add_propensity(model_id, mcfg, table, log)
    return table, cands


# --------------------------------------------------------------------------- the agent
class ModelingAgent:
    def __init__(self, model_id: int, mcfg: dict, spec: dict, table: pd.DataFrame, cands: dict, log: DecisionLog):
        self.model_id, self.mcfg, self.spec, self.table, self.cands, self.log = model_id, mcfg, spec, table, cands, log
        self.task, self.label, self.treatment = mcfg["task"], mcfg["label"], mcfg.get("treatment")
        self.calendar = campaign_calendar(table)
        self._fits, self._choices = {}, {}
        self.metric_fn, self.higher = metric(mcfg["selection"]["metric"])

    # ---- fitting and predicting (cached by training set: many folds share one)
    def fit(self, train_campaigns, cid: str):
        key = (train_set_id(train_campaigns), cid)
        if key not in self._fits:
            rows, c = self.table[self.table["campaign"].isin(train_campaigns)], self.cands[cid]
            if c["source"] == "owner":
                self._fits[key] = fit_owner(self.model_id, c["owner_learner"], rows, cid)
            else:
                self._fits[key] = fit_agent_candidate(rows, cid, self.task, c["family"], c["params"],
                                                      c["features"], self.label, self.treatment)
        return self._fits[key]

    def predict(self, fitted, rows: pd.DataFrame) -> pd.DataFrame:
        b = self.mcfg["baseline"]
        out = rows[["household_key", "campaign", "campaign_type", "start_day", "end_day"]].copy()
        out["label"] = rows[self.label].to_numpy()
        out["baseline_score"] = b["sign"] * rows[b["column"]].to_numpy(dtype=float)
        if self.treatment:
            out["treatment"] = rows[self.treatment].to_numpy()
            for col in ("propensity", "ipw_weight", "in_overlap", *self.mcfg["propensity"]["balance_columns"]):
                out[col] = rows[col].to_numpy()
        for name, values in fitted.predict(rows).items():
            out[name] = values
        return out.assign(candidate_id=fitted.candidate_id)

    def enough(self, rows: pd.DataFrame, cfg: dict, prefix: str) -> bool:
        """Do these rows meet the instruction file's minimums (rows / positives / treated)?"""
        need = {k.removeprefix(f"min_{prefix}_"): v for k, v in cfg.items() if k.startswith(f"min_{prefix}_")}
        have = {"rows": len(rows), "positives": int(rows[self.label].sum()) if self.task != "regress" else len(rows),
                "treated": int(rows[self.treatment].sum()) if self.treatment else len(rows)}
        return all(have[k] >= v for k, v in need.items())

    # ---- choosing (nested inside each fold's training campaigns)
    def inner_folds(self, train_campaigns) -> list[dict]:
        cal = self.calendar[self.calendar["campaign"].isin(train_campaigns)]
        icfg, folds = self.mcfg["selection"]["inner"], []
        for row in cal.tail(self.spec["defaults"]["inner_last_n_campaigns"]).itertuples():
            inner_train = cal.loc[cal["end_day"] < row.start_day - CUTOFF_GAP, "campaign"].tolist()
            tr = self.table[self.table["campaign"].isin(inner_train)]
            if inner_train and self.enough(tr, icfg, "train"):
                folds.append({"test_campaign": row.campaign, "train_campaigns": inner_train})
        return folds

    def select(self, train_campaigns, fold) -> tuple[str, dict]:
        key = train_set_id(train_campaigns)
        if key in self._choices:
            return self._choices[key]
        default, scfg = self.mcfg["default"], self.mcfg["selection"]
        folds = self.inner_folds(train_campaigns)
        inner_test = self.table[self.table["campaign"].isin([f["test_campaign"] for f in folds])]
        if len(folds) < self.spec["defaults"]["min_inner_folds"] or not self.enough(inner_test, scfg["inner"], "test"):
            self.log.record("MOD-03", f"only {len(folds)} usable inner campaigns ({len(inner_test)} test rows combined)",
                            f"keep {default}", "rule",
                            "Too little finished history inside this training window to compare candidates fairly; "
                            "the owner's default is kept.", fold=fold)
            self._choices[key] = (default, {})
            return self._choices[key]
        scores = {}
        for cid in self.cands:
            preds = pd.concat([self.predict(self.fit(f["train_campaigns"], cid),
                                            self.table[self.table["campaign"] == f["test_campaign"]]) for f in folds])
            scores[cid] = float(self.metric_fn(preds))
        pick, reason = choose_candidate(scores, self.cands, default, scfg["min_gain"], self.higher)
        self.log.record("MOD-04", f"inner {scfg['metric']}: " + ", ".join(f"{c} {s:.4f}" for c, s in scores.items()),
                        f"choose {pick}", "rule", reason, fold=fold,
                        evidence={"scores": scores, "inner_test_campaigns": [f["test_campaign"] for f in folds]})
        self._choices[key] = (pick, scores)
        return self._choices[key]

    # ---- forward chaining
    def outer_folds(self) -> list[dict]:
        folds, fcfg = [], self.mcfg["forward_chain"]
        for f in forward_chain_folds(self.table, self.spec["defaults"]["first_test_day"]):
            train = self.table[self.table["campaign"].isin(f["train_campaigns"])]
            if self.enough(train, fcfg, "train"):
                folds.append(f)
            else:
                self.log.record("MOD-02", f"campaign {f['test_campaign']}: training window too small ({len(train)} rows)",
                                "skip this test campaign", "rule",
                                "Below the instruction file's minimum training size; a model this small would be noise.",
                                fold=f["test_campaign"])
        return folds

    def run_forward(self) -> tuple[pd.DataFrame, list[dict]]:
        """Predictions for every test campaign (column `locked` marks the last locked_test_campaigns, in
        time order) and one record per fold."""
        parts, fold_info, default = [], [], self.mcfg["default"]
        folds = self.outer_folds()
        n_locked = self.mcfg.get("locked_test_campaigns", 0)
        locked = {f["test_campaign"] for f in folds[len(folds) - n_locked:]} if n_locked else set()
        for f in folds:
            chosen, scores = self.select(f["train_campaigns"], fold=f["test_campaign"])
            test_rows = self.table[self.table["campaign"] == f["test_campaign"]]
            for cid in self.cands:
                parts.append(self.predict(self.fit(f["train_campaigns"], cid), test_rows)
                             .assign(is_default=cid == default, chosen=cid == chosen, source=self.cands[cid]["source"],
                                     train_set_id=train_set_id(f["train_campaigns"]),
                                     locked=f["test_campaign"] in locked))
            fold_info.append({"test_campaign": int(f["test_campaign"]), "cutoff_day": int(f["cutoff_day"]),
                              "train_campaigns": [int(c) for c in f["train_campaigns"]],
                              "train_set_id": train_set_id(f["train_campaigns"]), "chosen": chosen,
                              "inner_scores": scores, "locked": f["test_campaign"] in locked})
        if locked:
            self.log.record("MOD-08", f"{len(locked)} latest test campaigns locked: {sorted(int(c) for c in locked)}",
                            "write them to locked_predictions.parquet", "rule",
                            "Judged once, on --unlock, so ideas tried round after round on the open campaigns cannot "
                            "pass by luck on campaigns they were tuned against.", evidence={"locked": sorted(int(c) for c in locked)})
        kept = [f["test_campaign"] for f in fold_info if not f["inner_scores"]]
        if kept:
            self.log.record("MOD-07", f"{len(kept)} of {len(fold_info)} test campaigns had too little finished history to "
                            f"compare options (campaigns {kept})", "owner's default used for them", "human_needed",
                            "The agent could not tell whether an alternative would have done better there; a person "
                            "should know the comparison covers fewer campaigns than it appears.",
                            evidence={"campaigns": kept})
        return pd.concat(parts, ignore_index=True), fold_info

    def fit_final(self, preds: pd.DataFrame) -> dict:
        """The owner's default, refit on every campaign, is what scoring uses. The agent's own preference is
        only a recommendation, chosen by the same rule on its forward-chaining predictions -- every test
        campaign, each predicted out of time -- rather than on a last few campaigns."""
        everything = self.table["campaign"].unique()
        default, scfg = self.mcfg["default"], self.mcfg["selection"]
        preds = preds[~preds["locked"]]   # the recommendation never peeks at the locked campaigns
        scores = {cid: float(self.metric_fn(preds[preds["candidate_id"] == cid])) for cid in self.cands}
        preferred, reason = choose_candidate(scores, self.cands, default, scfg["min_gain"], self.higher)
        n_test = preds["campaign"].nunique()
        finding = f"over {n_test} test campaigns, {scfg['metric']}: " + ", ".join(f"{c} {v:.4f}" for c, v in scores.items())
        if preferred != default:
            self.log.record("MOD-06", finding, f"recommend {preferred} to the owner", "human_needed",
                            reason + ". Changing an owner's model is the owner's decision; the default stays in "
                            "final_model.joblib until they accept.", fold="final", evidence={"scores": scores})
        else:
            self.log.record("MOD-06", finding, "keep the owner's default", "rule", reason, fold="final",
                            evidence={"scores": scores})
        fitted = self.fit(everything, default)
        return {"model_id": self.model_id, "candidate_id": default, "fitted": fitted, "importances": importances(fitted),
                "recommended": preferred, "recommendation_scores": scores,
                "trained_on_campaigns": sorted(int(c) for c in everything), "task": self.task, "label": self.label}


# --------------------------------------------------------------------------- reproduce
def _compare(model_id, name, got, expected, decimals, passed, results):
    results[name] = {"got": got, "expected": expected}
    check(round(float(got), decimals) == round(float(expected), decimals),
          f"model {model_id} reproduce: {name} = {got:.{decimals}f} (owner's report: {expected})", passed)


def reproduce(agent: ModelingAgent, passed: list) -> dict:
    """Re-run the owner's own evaluation through the agent; stop unless the numbers match their report."""
    m, cfg = agent.model_id, agent.mcfg["reproduce"]
    dec, t, res = cfg["decimals"], agent.table, {}
    split = cfg["split_70_30"]
    module = {1: model1_baseline, 2: model2_baseline, 3: model3_uplift}[m]
    train, test = module.time_split(t)
    check(len(train) == split["rows"]["train"] and len(test) == split["rows"]["test"],
          f"model {m} reproduce: 70/30 split {len(train):,} / {len(test):,} rows", passed)
    camps = sorted(train["campaign"].unique())
    if m == 1:
        y = test[agent.label].to_numpy()
        scores = {cid: agent.fit(camps, cid).predict(test)["score"] for cid in ("owner_forest", "owner_logistic")}
        scores["rule"] = test["past_redemption_rate"].to_numpy()
        for name, s in scores.items():
            for metric_name, value in (("pooled_capture", model1_baseline.top_share_capture(s, y)),
                                       ("roc_auc", roc_auc_score(y, s)), ("pr_auc", average_precision_score(y, s))):
                _compare(m, f"70/30 {name} {metric_name}", value, split[name][metric_name], dec, passed, res)
        fc = cfg["forward_chain"]
        folds = forward_chain_folds(t)
        check(len(folds) == fc["folds"], f"model 1 reproduce: {len(folds)} forward-chaining folds", passed)
        preds = pd.concat([agent.predict(agent.fit(f["train_campaigns"], "owner_forest"),
                                         t[t["campaign"] == f["test_campaign"]]) for f in folds])
        check(len(preds) == fc["test_rows"], f"model 1 reproduce: {len(preds):,} forward-chaining test rows", passed)
        c, y, s, r = preds["campaign"], preds["label"], preds["score"], preds["baseline_score"]
        got = {"within_capture": within_campaign_capture(c, y, s), "pooled_capture": model1_baseline.top_share_capture(s.to_numpy(), y.to_numpy()),
               "pooled_auc": roc_auc_score(y, s), "within_auc": within_campaign_auc(c, y, s)}
        for k, v in got.items():
            _compare(m, f"forward owner_forest {k}", v, fc["owner_forest"][k], dec, passed, res)
        _compare(m, "forward rule within_capture", within_campaign_capture(c, y, r), fc["rule"]["within_capture"], dec, passed, res)
        ci = bootstrap_capture(c, y, s, r, reps=300)["diff_ci"]
        for side, v, e in zip(("low", "high"), ci, fc["gap_interval"]):
            _compare(m, f"forward gap interval {side}", v, e, dec, passed, res)
    elif m == 2:
        forest, linear = (agent.fit(camps, cid).estimator for cid in ("owner_forest", "owner_linear"))
        out = model2_baseline.evaluate({"Random forest": forest, "Linear regression": linear}, test,
                                       train[agent.label].mean())["comparison"]
        named = {"owner_forest": out["Random forest"], "owner_linear": out["Linear regression"],
                 "usual_spend": next(v for k, v in out.items() if k.startswith("Usual spend")),
                 "train_mean": next(v for k, v in out.items() if k.startswith("Training-set mean"))}
        calibrated = agent.fit(camps, "owner_calibrated").predict(test)["score"]
        named["owner_calibrated"] = {"mae": float(np.mean(np.abs(test[agent.label].to_numpy() - calibrated)))}
        for name, metrics in split.items():
            if name == "rows":
                continue
            for metric_name, value in metrics.items():
                _compare(m, f"70/30 {name} {metric_name}", named[name][metric_name], value, dec, passed, res)
    else:
        fitted = agent.fit(camps, "owner_t_forest")
        out = model3_uplift.evaluate(fitted.treated_model, fitted.control_model, test)
        _compare(m, "70/30 pooled Qini", model3_uplift.qini_points(out["test"])[3], split["pooled_qini"], dec, passed, res)
        for q, (u, n) in enumerate(zip(split["quintile_uplift"], split["quintile_treated"])):
            row = out["quintiles"].iloc[q]
            _compare(m, f"70/30 quintile {q} uplift", row["actual_uplift"], u, dec, passed, res)
            _compare(m, f"70/30 quintile {q} treated", row["n_treated"], n, 0, passed, res)
    agent.log.record("MOD-05", f"{len(res)} numbers match the owner's own evaluation", "reproduction passed", "rule",
                     "The agent runs the owner's code unchanged, so it must give the owner's numbers exactly.",
                     evidence={k: v["got"] for k, v in res.items()})
    return res


# --------------------------------------------------------------------------- outputs
def render_report(model_id, mcfg, cands, preds, folds, final, passed, reproduced, log, runtime) -> str:
    chosen_counts = pd.Series([f["chosen"] for f in folds]).value_counts().to_dict() if folds else {}
    default = mcfg["default"]
    lines = [f"# Modeling Agent run report: Model {model_id}, {mcfg['name']}", "",
             f"Owner: {mcfg['owner']}. The owner's model is the default candidate (`{default}`), called unchanged; "
             "the agent only adds alternatives and recommends.", "",
             "## Candidates", ""]
    for cid, c in cands.items():
        what = (f"owner's {c['owner_learner']}" if c["source"] == "owner"
                else f"{c['family']} on {len(c['features'])} columns")
        lines.append(f"- `{cid}` ({c['source']}): {what}; changes vs. owner: {', '.join(c['changes']) or 'none'}")
    lines += ["", "## Checks", "", *[f"- [OK] {p}" for p in passed], ""]
    if reproduced:
        lines += ["## Reproduction of the owner's numbers", "",
                  *[f"- {k}: {v['got']:.4f} (owner: {v['expected']})" for k, v in reproduced.items()], ""]
    if folds:
        lines += ["## Forward chaining", "",
                  f"{len(folds)} test campaigns, {int((preds['candidate_id'] == default).sum()):,} test rows per candidate. "
                  f"Chosen per fold: {chosen_counts}. Locked (judged only on --unlock): "
                  f"{[f['test_campaign'] for f in folds if f['locked']] or 'none'}.", "",
                  "| test campaign | training campaigns | chosen | inner scores |", "|---|---|---|---|"]
        lines += [f"| {f['test_campaign']}{' (locked)' if f['locked'] else ''} | {len(f['train_campaigns'])} | {f['chosen']} | "
                  + (", ".join(f"{c} {s:.4f}" for c, s in f["inner_scores"].items()) or "n/a (default kept)") + " |"
                  for f in folds]
        lines.append("")
    if final:
        lines += ["## Final model", "", f"- Saved for scoring: owner's default `{final['candidate_id']}`, "
                  f"trained on all {len(final['trained_on_campaigns'])} campaigns.",
                  f"- Agent's recommendation: `{final['recommended']}`"
                  + (" (same as the default)" if final["recommended"] == default else " (needs the owner's decision)"), "",
                  "## Columns the owner's model leans on most (share of its reliance, all campaigns)", "",
                  "Reliance is not added value: a column can matter here yet add little if another column carries the same "
                  "information." + (" For uplift this is what predicts buying, not what predicts the mailer's effect."
                                    if mcfg["task"] == "uplift" else ""), "",
                  *[f"- `{c}`: {v:.0%}" for c, v in final["importances"].items()], ""]
    needed = log.human_needed()
    lines += ["## Needs a person", "", *([f"- {r['check_id']}: {r['finding']} -- {r['rationale']}" for r in needed] or ["- nothing"]),
              "", f"Runtime {runtime:.0f}s."]
    return "\n".join(lines)


def run_model(model_id: int, spec: dict, processed: Path, mode: str, iteration: int, use_proposals: bool = False,
              out_root: Path | None = None, locked: int | None = None) -> Path:
    """out_root: where the model{n} folder goes (default: the spec's output dir). locked: overrides
    locked_test_campaigns; the lead agent's replay uses 0, since there every newly finished campaign is fresh."""
    t0, passed = time.time(), []
    mcfg = spec["models"][model_id] if locked is None else {**spec["models"][model_id], "locked_test_campaigns": locked}
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log = DecisionLog("modeling", model_id, run_id)
    out_dir = (out_root or ROOT / spec["output"]["dir"]) / f"model{model_id}"
    proposals = load_proposals(out_dir) if use_proposals else None
    if use_proposals:
        check(bool(proposals), f"model {model_id}: proposals.yaml found in {out_dir}", passed)
    table, cands = prepare_table(model_id, mcfg, spec, processed, passed, log, proposals,
                                 exact_counts=mode in ("reproduce", "both"))
    agent = ModelingAgent(model_id, mcfg, spec, table, cands, log)
    reproduced = reproduce(agent, passed) if mode in ("reproduce", "both") else {}
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "reproduce":   # a check only: never touches the forward hand-off the Evaluation Agent reads
        (out_dir / "reproduce.json").write_text(json.dumps({"run_id": run_id, "checks": passed,
                                                            "reproduced": reproduced}, indent=2, default=float))
        (out_dir / "reproduce_report.md").write_text(render_report(model_id, mcfg, cands, pd.DataFrame(), [], None, passed,
                                                                   reproduced, log, time.time() - t0))
        print(f"Model {model_id}: reproduction passed in {time.time() - t0:.0f}s -> {out_dir}")
        return out_dir
    preds, folds = agent.run_forward()
    final = agent.fit_final(preds)
    preds[~preds["locked"]].to_parquet(out_dir / "predictions.parquet", index=False)
    preds[preds["locked"]].to_parquet(out_dir / "locked_predictions.parquet", index=False)
    joblib.dump(final, out_dir / "final_model.joblib")
    manifest = {
        "model_id": model_id, "run_id": run_id, "iteration": iteration, "mode": mode,
        "spec_hash": file_hash(SPEC_PATH), "table_hash": file_hash(processed / mcfg["table"]),
        "feature_spec_hash": file_hash(ROOT / spec["input"]["feature_spec"]),
        "versions": {"pandas": pd.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__},
        "default": mcfg["default"], "candidates": cands, "proposals_used": sorted(c for c in cands if cands[c]["source"] == "proposal"),
        "locked_campaigns": [f["test_campaign"] for f in folds if f["locked"]],
        "table_rows": len(table), "folds": folds, "final": {k: v for k, v in final.items() if k != "fitted"},
        "checks": passed, "reproduced": {k: v["got"] for k, v in reproduced.items()},
        "runtime_s": round(time.time() - t0, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log.save(out_dir)
    (out_dir / "report.md").write_text(render_report(model_id, mcfg, cands, preds, folds, final, passed, reproduced,
                                                     log, time.time() - t0))
    print(f"Model {model_id}: done in {time.time() - t0:.0f}s -> {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Modeling Agent: one shared modelling step for all three models")
    parser.add_argument("--model", default="all", choices=["1", "2", "3", "all"])
    parser.add_argument("--mode", default="both", choices=["reproduce", "forward", "both"])
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--iteration", type=int, default=1, help="how many times the Evaluation Agent has sent this back")
    parser.add_argument("--proposals", action="store_true", help="also test the approved proposals in proposals.yaml")
    parser.add_argument("--out", type=Path, default=None, help="folder for the model{n} hand-offs (default: the spec's)")
    parser.add_argument("--locked", type=int, default=None, help="override how many latest test campaigns are locked")
    args = parser.parse_args()
    spec = load_modeling_spec()
    for model_id in ([1, 2, 3] if args.model == "all" else [int(args.model)]):
        run_model(model_id, spec, args.processed, args.mode, args.iteration, args.proposals, args.out, args.locked)


if __name__ == "__main__":
    main()
