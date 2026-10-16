"""Tests for the Evaluation Agent's one-change-at-a-time attribution and ledger rows."""
from evaluation.agent import attribution, attribution_lines, ledger_rows


def _r(gap, status="inconclusive"):
    return {"gap": gap, "low": gap - 0.01, "high": gap + 0.01, "status": status, "sd": 0.005}


CANDS = {"cols_a": {"changes": ["columns"], "source": "agent"}, "cols_b": {"changes": ["columns"], "source": "proposal"},
         "deeper": {"changes": ["settings"], "source": "agent"},
         "both": {"changes": ["columns", "learner"], "source": "agent"}}
ALTS = {"cols_a": _r(0.004), "cols_b": _r(0.02, "pass"), "deeper": _r(-0.03, "fail"), "both": _r(0.05, "pass")}


def test_attribution_takes_the_best_single_change_per_kind_and_lists_combined_ones_apart():
    g = attribution(CANDS, ALTS)
    assert g["columns"]["best"] == "cols_b" and g["columns"]["tried"] == ["cols_a", "cols_b"]
    assert g["settings"]["best"] == "deeper" and g["settings"]["status"] == "fail"
    assert g["learner"]["best"] is None
    assert g["combined"] == ["both"]          # its +0.05 is never credited to columns or learner


def test_attribution_lines_say_it_plainly():
    text = "\n".join(attribution_lines(attribution(CANDS, ALTS)))
    assert "**Columns only:** best of 2 is `cols_b`" in text and "reliably better" in text
    assert "**Learner only:** no option changed only this." in text
    assert "gain cannot be split): both" in text


def test_ledger_rows_cover_proposals_only():
    rows = ledger_rows(1, CANDS, ALTS, "open")
    assert rows == [{"model": 1, "id": "cols_b", "approved": True, "open_gap": 0.02,
                     "open_low": 0.01, "open_high": 0.03, "open_status": "pass"}]
