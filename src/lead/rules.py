"""Lead Agent rules: a person's answer to a question, saved so the same question is never asked again.

A rule starts as a draft: the lead agent's front door writes the rule in plain words from the person's answer
(`propose`). Nothing is saved until the person approves it, as written or edited (`approve`). A saved rule
records who decided, when, and why, in src/lead/lead_rules.yaml (versioned with the code, so the team can
always see where a rule came from).

What a rule does: the question it answers is reported as answered instead of open, and a never-seen value it
names (a department, a campaign type) is no longer a surprise. Where the answer changes another agent's
behaviour (for example, whether a department counts as household spend), that agent's owner applies it to
their own rules; the rule says so, and the lead agent never edits another agent's files.

Usage:
    python src/lead/rules.py propose --question data:-:department:PHARMACY --answer "..." --rule "..."
    python src/lead/rules.py approve <draft id> --by Pranav [--rule "edited wording"]
    python src/lead/rules.py drafts      # list drafts waiting for a person
"""
import argparse
import json
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = Path(__file__).with_name("lead_rules.yaml")
DRAFTS_PATH = ROOT / "data/processed/lead_agent/rule_drafts.json"
HEADER = """# Lead Agent rules: answers a person gave to the lead agent's questions (src/lead/rules.py).
# Each one is reported as answered instead of asked again. Written by `rules.py approve`; edit with care.
"""


def load_rules(path: Path = RULES_PATH) -> list[dict]:
    return (yaml.safe_load(path.read_text()) or {}).get("rules", []) if path.exists() else []


def answered(rules: list[dict]) -> dict[str, dict]:
    """question id -> the rule that answers it."""
    return {r["question"]: r for r in rules}


def _drafts(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def propose(question: str, answer: str, rule: str, drafts_path: Path = DRAFTS_PATH) -> str:
    """Save a draft rule; returns its id. question: the id the lead agent gave it (agent:model:key)."""
    parts = question.split(":", 2)
    if len(parts) < 3 or not (parts[1] == "-" or parts[1].isdigit()) or not parts[2]:
        raise SystemExit(f"[STOP] '{question}' is not a question id (agent:model:key, as in the run summary)")
    drafts = _drafts(drafts_path)
    draft_id = f"d{max((int(k[1:]) for k in drafts), default=0) + 1}"   # never reuses the id of a waiting draft
    drafts[draft_id] = {"question": question, "answer": answer, "rule": rule}
    drafts_path.parent.mkdir(parents=True, exist_ok=True)
    drafts_path.write_text(json.dumps(drafts, indent=2))
    return draft_id


def approve(draft_id: str, by: str, rule: str | None = None, drafts_path: Path = DRAFTS_PATH,
            rules_path: Path = RULES_PATH, today: str | None = None) -> dict:
    """Move a draft into the rules, as written or with the person's edited wording."""
    drafts = _drafts(drafts_path)
    if draft_id not in drafts:
        raise SystemExit(f"[STOP] no draft '{draft_id}'; drafts waiting: {sorted(drafts) or 'none'}")
    if not by.strip():
        raise SystemExit("[STOP] a rule needs the name of the person who decided it")
    draft = drafts.pop(draft_id)
    saved = {"question": draft["question"], "rule": rule or draft["rule"], "answer": draft["answer"],
             "decided_by": by, "decided_on": today or date.today().isoformat()}
    rules = [r for r in load_rules(rules_path) if r["question"] != saved["question"]] + [saved]
    rules_path.write_text(HEADER + yaml.safe_dump({"rules": rules}, sort_keys=False, allow_unicode=True, width=110))
    drafts_path.write_text(json.dumps(drafts, indent=2))
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Lead Agent rules: save a person's answer as a rule")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("propose")
    p.add_argument("--question", required=True)
    p.add_argument("--answer", required=True, help="the person's answer, in their words")
    p.add_argument("--rule", required=True, help="the rule, in plain words")
    a = sub.add_parser("approve")
    a.add_argument("draft")
    a.add_argument("--by", required=True)
    a.add_argument("--rule", default=None, help="the person's edited wording, if they changed it")
    sub.add_parser("drafts")
    args = parser.parse_args()
    if args.command == "propose":
        print(f"Draft {propose(args.question, args.answer, args.rule)} saved; nothing applies until it is approved.")
    elif args.command == "approve":
        saved = approve(args.draft, args.by, args.rule)
        print(f"Saved rule for {saved['question']}: {saved['rule']} (decided by {saved['decided_by']}, {saved['decided_on']})")
    else:
        drafts = _drafts(DRAFTS_PATH)
        print("\n".join(f"{k}: {v['question']} -> {v['rule']}" for k, v in drafts.items()) or "No drafts waiting.")


if __name__ == "__main__":
    main()
