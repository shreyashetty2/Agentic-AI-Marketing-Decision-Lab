"""Score the live Decision Lab chat against tests/decision/questions.yaml, and measure what each question costs.

Calls Gemini on Vertex AI (billed to the project's GCP credits), so it is not part of the pytest suite. Needs
`gcloud auth application-default login`. Each question is asked fresh, with no history.

Usage:
    python scripts/eval_decision_lab.py                  # both sets
    python scripts/eval_decision_lab.py --set held_back  # run the held-back set once per change
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))
from decision.assistant import run_turn  # noqa: E402

QUESTIONS = ROOT / "tests/decision/questions.yaml"


def judge(case: dict, turn) -> str | None:
    """None if the turn did what the case expects, else what went wrong. Looks at every call in the turn, not
    just the first: checking a code or looking up a category name before answering is good behaviour."""
    answers = [(form, r) for name, form, r in turn.calls if name == "answer"]
    said = turn.answer.lower()
    missing = [w for w in case.get("says", []) if w.lower() not in said]
    if missing:
        return f"answer doesn't say {missing}"
    claimed = [w for w in case.get("never_says", []) if w.lower() in said]
    if claimed:
        return f"answer says {claimed}"
    if ({"says", "never_says"} & set(case)) and not ({"form", "status", "no_answer"} & set(case)):
        return None
    if case.get("no_answer"):
        computed = [r for _, r in answers if r.status == "answer" and r.numbers]
        return "worked out numbers for a question it can't answer" if computed else None
    if not turn.calls:
        return "made no function call"
    if "status" in case:
        got = [r.status for _, _, r in turn.calls]
        return None if case["status"] in got else f"code returned {got}, expected {case['status']}"
    want, got = dict(case["form"]), case.get("result", {})
    tool = want.pop("tool")
    matching = [(form, r) for name, form, r in turn.calls if name == tool]
    if not matching:
        return f"never called {tool} (called {[name for name, _, _ in turn.calls]})"
    if any(all(form.get(k) == v for k, v in want.items()) and all(r.numbers.get(k) == v for k, v in got.items())
           for form, r in matching):
        return None
    return f"differs: {[({k: f.get(k) for k in want}, {k: r.numbers.get(k) for k in got}) for f, r in matching]}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["design", "held_back", "team", "all"], default="all")
    args = ap.parse_args()
    from google import genai

    client = genai.Client(vertexai=True, project=os.environ.get("GOOGLE_CLOUD_PROJECT", "capstone-decision-lab"),
                          location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
    from lib.chat import background_for, lab   # the app's own Lab and background, so this tests what it runs
    lab = lab()
    sets = yaml.safe_load(QUESTIONS.read_text())
    for name in (["design", "held_back", "team"] if args.set == "all" else [args.set]):
        passed, tokens = 0, {"input": 0, "output": 0}
        print(f"\n== {name}")
        for case in sets[name]:
            turn = run_turn(client, lab, [], case["q"], background_for(case["q"]))
            problem = judge(case, turn)
            passed += problem is None
            tokens = {k: tokens[k] + turn.tokens[k] for k in tokens}
            print(f"[{'PASS' if problem is None else 'FAIL'}] {case['q']}" + (f"\n       {problem}" if problem else ""))
        n = len(sets[name])
        print(f"{passed}/{n} passed. Tokens per question: {tokens['input'] / n:,.0f} in, {tokens['output'] / n:,.0f} out.")


if __name__ == "__main__":
    main()
