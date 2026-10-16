"""Real-data acceptance checks for the Modeling Agent and the Evaluation Agent
(the "How we know it is done" list in docs/agent_design/modeling_evaluation_agents_design.md).

Runs both agents on the real tables (into data/processed/acceptance/), then checks the hand-offs. Slow (~9 minutes).
    python tests/modeling/acceptance_agents.py --processed data/processed
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from evaluation.agent import evaluate_model, load_evaluation_spec  # noqa: E402
from modeling import model1_baseline, model2_baseline, model3_uplift  # noqa: E402
from modeling.agent import load_modeling_spec, run_model  # noqa: E402
from modeling.decision_log import DECIDED_BY  # noqa: E402

EXPECTED = {1: {"folds": 18, "rows": 5785, "locked": [21, 22, 23, 25, 24]}, 2: {"folds": 11, "rows": 167, "locked": [23, 25, 24]},
            3: {"folds": 15, "rows": 24397, "locked": [22, 23, 25, 24]}}
OWNER_FILES = ["src/modeling/model1_baseline.py", "src/modeling/model2_baseline.py", "src/modeling/model1_evaluation.py"]
results = []


def check(ok: bool, message: str) -> None:
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {message}")


def predictions_hash(model_id: int, out: Path) -> str:
    return hashlib.sha256((out / f"model{model_id}" / "predictions.parquet").read_bytes()).hexdigest()


def owner_scores(model_id: int, processed: Path, train_campaigns: list, test_campaign: int) -> np.ndarray:
    """The owner's own functions, called directly -- what the agent's default candidate must equal."""
    module = {1: model1_baseline, 2: model2_baseline, 3: model3_uplift}[model_id]
    table = module.load_table(processed)
    train, test = table[table["campaign"].isin(train_campaigns)], table[table["campaign"] == test_campaign]
    if model_id == 3:
        model_t, model_c, _, _ = module.train_t_learner(train)
        return model_t.predict_proba(test[module.FEATURES])[:, 1] - model_c.predict_proba(test[module.FEATURES])[:, 1]
    model = module.train_models(train)["Random forest"]
    if model_id == 1:
        return model.predict_proba(test[module.FEATURES])[:, 1]
    return model.predict(test[module.FEATURES])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed")
    args = parser.parse_args()
    mspec, espec = load_modeling_spec(), load_evaluation_spec()
    # its own folders, so a check never overwrites the real hand-offs, proposals or unlock history
    mspec["output"]["dir"] = espec["input"]["dir"] = "data/processed/acceptance/modeling_agent"
    espec["output"]["dir"] = "data/processed/acceptance/evaluation_agent"
    out = ROOT / mspec["output"]["dir"]

    for model_id in (1, 2, 3):
        # reproduce + forward in one run; reproduce stops the run (SystemExit) if any owner number is off
        try:
            run_model(model_id, mspec, args.processed, "both", 1)
            check(True, f"model {model_id}: agent reproduces every number in the owner's own report")
        except SystemExit as e:
            check(False, f"model {model_id}: reproduction failed -- {e}")
            continue
        first_hash = predictions_hash(model_id, out)
        manifest = json.loads((out / f"model{model_id}" / "manifest.json").read_text())
        open_preds = pd.read_parquet(out / f"model{model_id}" / "predictions.parquet")
        locked = pd.read_parquet(out / f"model{model_id}" / "locked_predictions.parquet")
        preds = pd.concat([open_preds, locked])
        default = preds[preds["is_default"]]
        check(manifest["locked_campaigns"] == EXPECTED[model_id]["locked"]
              and set(locked["campaign"]) == set(EXPECTED[model_id]["locked"])
              and not set(open_preds["campaign"]) & set(locked["campaign"]),
              f"model {model_id}: campaigns {manifest['locked_campaigns']} locked, and only in the locked file")

        check(len(manifest["folds"]) == EXPECTED[model_id]["folds"] and len(default) == EXPECTED[model_id]["rows"],
              f"model {model_id}: {len(manifest['folds'])} test campaigns, {len(default):,} test rows "
              f"(expected {EXPECTED[model_id]['folds']}, {EXPECTED[model_id]['rows']:,})")
        ends = pd.read_parquet(args.processed / mspec["models"][model_id]["table"],
                               columns=["campaign", "end_day"]).drop_duplicates().set_index("campaign")["end_day"]
        check(all(ends[c] < f["cutoff_day"] for f in manifest["folds"] for c in f["train_campaigns"]),
              f"model {model_id}: every training campaign ended before its test campaign's cutoff")

        fold = manifest["folds"][-1]
        mine = default[default["campaign"] == fold["test_campaign"]].sort_values("household_key")
        theirs_table = {1: model1_baseline, 2: model2_baseline, 3: model3_uplift}[model_id].load_table(args.processed)
        order = theirs_table[theirs_table["campaign"] == fold["test_campaign"]]["household_key"].to_numpy()
        theirs = pd.Series(owner_scores(model_id, args.processed, fold["train_campaigns"], fold["test_campaign"]),
                           index=order).sort_index()
        check(np.array_equal(mine["score"].to_numpy(), theirs.to_numpy()),
              f"model {model_id}: default candidate's predictions are identical to calling the owner's code directly")

        decisions = json.loads((out / f"model{model_id}" / "decisions.json").read_text())
        check(all(d["decided_by"] in DECIDED_BY and d["rationale"] and d["llm_review"] is None for d in decisions),
              f"model {model_id}: all {len(decisions)} modeling decisions have a decider, a reason, and no LLM review")

        run_model(model_id, mspec, args.processed, "forward", 1)
        check(predictions_hash(model_id, out) == first_hash, f"model {model_id}: two runs give identical predictions")

        verdict = evaluate_model(model_id, espec, mspec, None)
        check(verdict["decided_by"] in DECIDED_BY and verdict["reasons"],
              f"model {model_id}: Evaluation Agent verdict '{verdict['verdict']}' with a reason")
        if model_id == 3:
            check(not (verdict["verdict"] == "approve" and verdict["decided_by"] == "rule"),
                  "model 3: never approved by a rule alone (causal sign-off is a person's)")

    changed = subprocess.run(["git", "diff", "--name-only", "HEAD", "--", *OWNER_FILES],
                             cwd=ROOT, capture_output=True, text=True).stdout.split()
    check(not changed, f"owners' Model 1/2 scripts unchanged {changed or ''}".strip())
    diff = subprocess.run(["git", "diff", "--numstat", "HEAD", "--", "src/modeling/model3_uplift.py"],
                          cwd=ROOT, capture_output=True, text=True).stdout.split()
    check(not diff or (int(diff[0]) <= 15 and int(diff[1]) <= 15),
          f"Shreya's script: only the surgical fix (+{diff[0] if diff else 0} / -{diff[1] if diff else 0} lines)")

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
