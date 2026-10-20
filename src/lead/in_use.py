"""Lead Agent: which model is in use, when a person chose an improvement idea over the team's own model.

By default each model in use is its owner's script. A data scientist can make any option that was tested the
model in use: an improvement idea (from that model's proposals.yaml) or one of the options the Modeling Agent
always tries (e.g. Model 2's line on usual spend alone). `set` copies it into
src/lead/models_in_use.yaml (versioned with the code, so the team sees what is in use, who chose it, when and
why). From the next run the Modeling Agent judges and refits that idea as the model in use, and the owner's
script stays in as an option to compare with. `clear` goes back to the owner's script. Nobody's script changes.

Usage:
    python src/lead/in_use.py set --model 2 --idea r2_p1 --by "Pranav Jain" --reason "fixes the low totals"
    python src/lead/in_use.py clear --model 2
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lead.describe import model_words  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
IN_USE_PATH = Path(__file__).with_name("models_in_use.yaml")
IDEAS = ROOT / "data/processed/modeling_agent"
HEADER = """# Lead Agent: models a person chose to use instead of the team's own (src/lead/in_use.py).
# Empty = every model in use is its owner's script. Written by `in_use.py`; edit with care.
"""
KEEP = ("id", "title", "option", "reason", "expected", "add_columns", "drop_columns", "family", "params")


def load(path: Path = IN_USE_PATH) -> dict:
    return (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}


def _save(chosen: dict, path: Path) -> None:
    path.write_text(HEADER + (yaml.safe_dump(chosen, sort_keys=False, width=110) if chosen else "{}\n"))


def set_in_use(model: int, idea: str, by: str, reason: str, ideas_dir: Path = IDEAS, path: Path = IN_USE_PATH,
               today: str | None = None) -> dict:
    ideas_file = ideas_dir / f"model{model}/proposals.yaml"
    proposals = (yaml.safe_load(ideas_file.read_text()) or []) if ideas_file.exists() else []
    found = next((p for p in proposals if p.get("id") == idea), None)
    manifest_file = ideas_dir / f"model{model}/manifest.json"
    options = json.loads(manifest_file.read_text())["candidates"] if manifest_file.exists() else {}
    if found is None and idea in options:      # one of the options the Modeling Agent always tests
        found = {"id": idea, "title": model_words(options[idea], json.loads(manifest_file.read_text())["final"]["task"]),
                 "option": True}
    if found is None:
        raise SystemExit(f"[STOP] nothing called '{idea}' for Model {model}; ideas: {[p.get('id') for p in proposals]}, "
                         f"options: {sorted(options)}")
    if found.get("new_column"):
        raise SystemExit(f"[STOP] idea '{idea}' needs a column that does not exist yet; it cannot be the model in use")
    if not by.strip():
        raise SystemExit("[STOP] the choice needs the name of the person who made it")
    entry = {**{k: found[k] for k in KEEP if k in found}, "decided_by": by, "decided_on": today or date.today().isoformat(),
             "why": reason}
    chosen = {**load(path), model: entry}
    _save(chosen, path)
    return entry


def clear(model: int, path: Path = IN_USE_PATH) -> bool:
    chosen = load(path)
    had = chosen.pop(model, None) is not None
    _save(chosen, path)
    return had


def main() -> None:
    parser = argparse.ArgumentParser(description="Lead Agent: choose which model is in use")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("set")
    s.add_argument("--model", type=int, required=True, choices=[1, 2, 3])
    s.add_argument("--idea", required=True)
    s.add_argument("--by", required=True)
    s.add_argument("--reason", required=True)
    c = sub.add_parser("clear")
    c.add_argument("--model", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()
    if args.command == "set":
        e = set_in_use(args.model, args.idea, args.by, args.reason)
        print(f"Model {args.model} now uses '{e.get('title')}' (chosen by {e['decided_by']}, {e['decided_on']}).")
    else:
        print(f"Model {args.model} uses its owner's script again." if clear(args.model) else
              f"Model {args.model} already uses its owner's script.")


if __name__ == "__main__":
    main()
