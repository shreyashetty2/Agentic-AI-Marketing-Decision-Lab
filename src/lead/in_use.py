"""Lead Agent: which model is in use, when a person chose an improvement idea over the team's own model.

By default each model in use is its owner's script. A data scientist who tested an idea through the lead agent
can make it the model in use: `set` copies the idea from that model's proposals.yaml into
src/lead/models_in_use.yaml (versioned with the code, so the team sees what is in use, who chose it, when and
why). From the next run the Modeling Agent judges and refits that idea as the model in use, and the owner's
script stays in as an option to compare with. `clear` goes back to the owner's script. Nobody's script changes.

Usage:
    python src/lead/in_use.py set --model 2 --idea r2_p1 --by "Pranav Jain" --reason "fixes the low totals"
    python src/lead/in_use.py clear --model 2
"""
import argparse
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
IN_USE_PATH = Path(__file__).with_name("models_in_use.yaml")
IDEAS = ROOT / "data/processed/modeling_agent"
HEADER = """# Lead Agent: models a person chose to use instead of the team's own (src/lead/in_use.py).
# Empty = every model in use is its owner's script. Written by `in_use.py`; edit with care.
"""
KEEP = ("id", "title", "reason", "expected", "add_columns", "drop_columns", "family", "params")


def load(path: Path = IN_USE_PATH) -> dict:
    return (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}


def _save(chosen: dict, path: Path) -> None:
    path.write_text(HEADER + (yaml.safe_dump(chosen, sort_keys=False, width=110) if chosen else "{}\n"))


def set_in_use(model: int, idea: str, by: str, reason: str, ideas_dir: Path = IDEAS, path: Path = IN_USE_PATH,
               today: str | None = None) -> dict:
    proposals = yaml.safe_load((ideas_dir / f"model{model}/proposals.yaml").read_text()) or []
    found = next((p for p in proposals if p.get("id") == idea), None)
    if found is None:
        raise SystemExit(f"[STOP] no idea '{idea}' for Model {model}; ideas: {[p.get('id') for p in proposals]}")
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
