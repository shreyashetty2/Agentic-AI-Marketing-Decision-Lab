"""The Evaluation Agent's questions for a person carry a key that names the kind of question and does not
change from run to run, so the lead agent can tell a question it already asked from a new one."""
import json

import pytest

from evaluation.agent import open_questions

MCFG = {"owner": "Yuxin (model2_value.py)", "task": "regress"}
ECFG = {"baseline_name": "Usual spend", "primary": "mae", "measure_note": "Does the team accept this measure?"}
CANDS = {"own": {"changes": []}, "alt": {"changes": ["columns"]}, "other": {"changes": ["settings"]}}


def ask(tmp_path, vs_gap: float, flagged: list[dict]) -> list[tuple[str, str, str]]:
    (tmp_path / "decisions.json").write_text(json.dumps(flagged))
    vs_base = {"status": "inconclusive", "gap": vs_gap, "low": -0.5, "high": 0.7}
    alternatives = {"alt": {"status": "pass", "gap": vs_gap, "low": 0.1, "high": 0.9},
                    "other": {"status": "inconclusive", "gap": 0.0, "low": -0.1, "high": 0.1}}
    cal = {"ok": False, "summary": f"predicted total / actual total = {1 - vs_gap:.2f}"}
    return open_questions(2, MCFG, ECFG, None, "own", vs_base, None, alternatives, cal, tmp_path, CANDS)


def test_every_question_has_a_key_and_the_right_person(tmp_path):
    flagged = [{"check_id": "MOD-07", "decided_by": "human_needed", "finding": "thin history", "rationale": "r",
                "action": "flag"},
               {"check_id": "MOD-04", "decided_by": "rule", "finding": "chose own", "rationale": "r", "action": "x"}]
    qs = ask(tmp_path, 0.2, flagged)
    assert [(k, who) for k, who, _ in qs] == [
        ("baseline_inconclusive", "Yuxin"),
        ("alternative_beats_owner:alt", "Yuxin"),
        ("calibration", "Yuxin"),
        ("baseline_as_good", "Yuxin"),
        ("measure_note", "team"),
        ("MOD-07", "team"),
    ]


def test_keys_stay_the_same_when_the_numbers_change(tmp_path):
    first = ask(tmp_path, 0.2, [])
    second = ask(tmp_path, 0.35, [])
    assert [k for k, _, _ in first] == [k for k, _, _ in second]
    assert [q for _, _, q in first] != [q for _, _, q in second]      # the wording carries the new numbers


def test_the_no_model_score_is_what_anyone_could_do_without_a_model():
    import pandas as pd
    from evaluation.agent import no_skill_score
    d = pd.DataFrame({"campaign": [1] * 10 + [2] * 10, "label": [1, 1] + [0] * 8 + [1, 0] * 5})
    assert no_skill_score("classify", d) == pytest.approx(0.2)            # mail 20% at random, reach ~20% of redeemers
    dollars = pd.DataFrame({"label": [0.0, 10.0, 20.0]})
    assert no_skill_score("regress", dollars) == pytest.approx(20 / 3)    # guess the average ($10) for everyone
    assert no_skill_score("uplift", d) == 0.0
