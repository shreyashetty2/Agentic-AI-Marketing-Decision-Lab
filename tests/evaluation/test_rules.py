"""One test per Evaluation Agent verdict branch (src/evaluation/rules.py)."""
import pytest

from evaluation.rules import gap_status, route

CFG = {"max_iterations": 2, "stability_min_share": 0.6, "approve_requires_human": False}


def facts(**kw):
    base = {"gap_status": "pass", "calibration_ok": True, "comparison_ok": None, "better_alternative": None,
            "stability": 1.0, "all_within_noise": False, "iteration": 1}
    return {**base, **kw}


@pytest.mark.parametrize("low,high,status", [(0.01, 0.2, "pass"), (-0.2, -0.01, "fail"), (-0.1, 0.1, "inconclusive"),
                                             (0.0, 0.1, "inconclusive")])
def test_gap_status(low, high, status):
    assert gap_status(low, high) == status


def test_approve_when_gap_passes_and_calibrated():
    assert route(facts(), CFG)[:2] == ("approve", "rule")


def test_approve_needs_a_person_when_required():
    assert route(facts(), {**CFG, "approve_requires_human": True})[:2] == ("human_needed", "human_needed")


def test_incomparable_groups_go_to_a_person_first():
    assert route(facts(comparison_ok=False), CFG)[0] == "human_needed"


def test_iteration_cap_goes_to_a_person_even_if_it_would_pass():
    verdict, by, reasons = route(facts(iteration=2), CFG)
    assert (verdict, by) == ("human_needed", "human_needed") and "EVA-04" in reasons[0]


def test_good_ranking_bad_calibration_sends_back_to_modeling_with_ranges_hint():
    verdict, _, reasons = route(facts(calibration_ok=False), CFG)
    assert verdict == "send_back_modeling" and "do not recalibrate" in reasons[0]


def test_failing_model_with_a_better_alternative_sends_back_to_modeling():
    verdict, _, reasons = route(facts(gap_status="fail", better_alternative="alt"), CFG)
    assert verdict == "send_back_modeling" and "alt" in reasons[0]


def test_unstable_choice_sends_back_to_modeling():
    assert route(facts(gap_status="inconclusive", stability=0.5), CFG)[0] == "send_back_modeling"


def test_everything_within_noise_sends_back_to_features():
    assert route(facts(gap_status="inconclusive", all_within_noise=True), CFG)[0] == "send_back_features"


def test_uncovered_case_goes_to_a_person():
    assert route(facts(gap_status="fail"), CFG)[:2] == ("human_needed", "human_needed")
