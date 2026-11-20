"""The question set is well formed, and the live-eval judge scores turns the way the set says."""
import importlib.util

import yaml

from decision_helpers import ROOT
from decision.assistant import TOOLS, Turn
from decision.menu import HOWS, MEASURES, Result

spec = importlib.util.spec_from_file_location("eval_decision_lab", ROOT / "scripts/eval_decision_lab.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def test_every_question_names_exactly_one_kind_of_expectation_using_real_menu_names():
    sets = yaml.safe_load(ev.QUESTIONS.read_text())
    assert set(sets) == {"design", "held_back", "team"}
    tools = {t["name"] for t in TOOLS}
    for case in sets["design"] + sets["held_back"] + sets["team"]:
        assert sum(k in case for k in ("form", "status", "no_answer")) <= 1, case["q"]
        assert any(k in case for k in ("form", "status", "no_answer", "says", "never_says")), case["q"]
        form = case.get("form", {})
        assert not form or form["tool"] in tools, case["q"]
        assert form.get("how", "summary") in HOWS, case["q"]
        assert form.get("measure", "uplift") in (*MEASURES, "past_redemptions"), case["q"]
        assert case.get("status", "ask") in ("ask", "cannot"), case["q"]


def turn(*calls):
    return Turn(answer="", calls=list(calls))


def test_judge_form_cases_look_at_every_call_not_just_the_first():
    case = {"form": {"tool": "answer", "how": "top", "campaign": 10}}
    right = ("answer", {"how": "top", "campaign": 10, "top_n": 5}, Result("answer"))
    lookup = ("find_category", {"word": "yogurt"}, Result("answer"))
    assert ev.judge(case, turn(right)) is None
    assert ev.judge(case, turn(lookup, right)) is None
    assert "differs" in ev.judge(case, turn(("answer", {"how": "top", "campaign": 11}, Result("answer"))))
    assert "never called answer" in ev.judge(case, turn(lookup))
    assert ev.judge(case, turn()) == "made no function call"


def test_judge_status_cases():
    assert ev.judge({"status": "ask"}, turn(("answer", {}, Result("ask", "Which campaign?")))) is None
    assert ev.judge({"status": "ask"}, turn(("describe_code", {}, Result("answer")),
                                            ("answer", {}, Result("ask", "Which measure?")))) is None
    assert "expected ask" in ev.judge({"status": "ask"}, turn(("answer", {}, Result("answer", numbers={"x": 1}))))


def test_judge_no_answer_cases_allow_describing_but_not_calculating():
    assert ev.judge({"no_answer": True}, turn()) is None
    assert ev.judge({"no_answer": True}, turn(("answer", {}, Result("cannot", "no")))) is None
    assert ev.judge({"no_answer": True}, turn(("describe_campaign", {}, Result("answer", numbers={"x": 1})))) is None
    assert "worked out numbers" in ev.judge({"no_answer": True}, turn(("answer", {}, Result("answer", numbers={"x": 1}))))


def test_judge_result_checks_what_the_code_resolved_not_the_words_used():
    case = {"form": {"tool": "describe_code"}, "result": {"trait": "classification_1"}}
    resolved = ("describe_code", {"trait": "Age Group1"}, Result("answer", numbers={"trait": "classification_1"}))
    wrong = ("describe_code", {"trait": "Level1"}, Result("answer", numbers={"trait": "classification_3"}))
    assert ev.judge(case, turn(resolved)) is None
    assert "differs" in ev.judge(case, turn(wrong))


def test_judge_says_checks_the_final_answer_text():
    t = Turn(answer="889 of 7,208 mailings were redeemed.")
    assert ev.judge({"says": ["889", "7,208"]}, t) is None
    assert "doesn't say" in ev.judge({"says": ["12.3%"]}, t)
    assert ev.judge({"no_answer": True, "says": ["889"]}, t) is None


def test_judge_never_says_catches_an_unsupported_claim():
    assert ev.judge({"never_says": ["2 people"]}, Turn(answer="Household size isn't in the data.")) is None
    assert "answer says" in ev.judge({"never_says": ["2 people"]}, Turn(answer="It has 2 people."))
