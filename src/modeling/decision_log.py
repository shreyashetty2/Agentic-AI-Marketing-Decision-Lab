"""Decision log shared by the Modeling Agent and the Evaluation Agent.

Same idea as the Data Agent's log (src/preprocessing/apply_decisions.py): every decision is a record
saying what was found, what was done, and WHO decided -- "rule" (a written rule covers it) or
"human_needed" (no rule can settle it, or it changes an owner's model). Code writes every record,
so every number in it is exact.

LLM-REVIEW-HOOK (not built): REVIEW_HOOK may later be set to a function that reads a record and
returns a review. It may only fill record["llm_review"]; it can never change action or decided_by.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DECIDED_BY = ("rule", "human_needed")
REVIEW_HOOK = None   # LLM-REVIEW-HOOK: not built (see module docstring)


def check(ok: bool, message: str, passed: list[str]) -> None:
    """Hard check, same style as Feature Agent 2A/2B: stop the run on failure, otherwise record it."""
    if not ok:
        raise SystemExit(f"[STOP] {message}")
    print(f"[OK]   {message}")
    passed.append(message)


class DecisionLog:
    def __init__(self, agent: str, model_id: int, run_id: str):
        self.agent, self.model_id, self.run_id = agent, model_id, run_id
        self.records: list[dict] = []

    def record(self, check_id: str, finding: str, action: str, decided_by: str, rationale: str,
               fold=None, evidence: dict | None = None) -> dict:
        if decided_by not in DECIDED_BY:
            raise ValueError(f"decided_by must be one of {DECIDED_BY}, got {decided_by!r}")
        if not rationale:
            raise ValueError(f"{check_id}: every decision needs a rationale")
        rec = {"run_time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "agent": self.agent,
               "model_id": self.model_id, "run_id": self.run_id, "check_id": check_id, "fold": fold,
               "finding": finding, "action": action, "decided_by": decided_by, "rationale": rationale,
               "evidence": evidence or {}, "llm_review": None}
        if REVIEW_HOOK is not None:
            rec["llm_review"] = REVIEW_HOOK(dict(rec))
        self.records.append(rec)
        return rec

    def human_needed(self) -> list[dict]:
        return [r for r in self.records if r["decided_by"] == "human_needed"]

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "decisions.json").write_text(json.dumps(self.records, indent=2, default=str))
        flat = [{**r, "evidence": json.dumps(r["evidence"], default=str)} for r in self.records]
        pd.DataFrame(flat).to_csv(out_dir / "decisions.csv", index=False)
