"""Smoke tests: every Decision Lab page renders against the Lead Agent's real outputs, and the Redemption
Likelihood plan agrees with the Evaluation Agent. Skipped when data/processed has no Lead Agent run yet."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "data/processed/lead_agent/run_summary.json"
pytestmark = pytest.mark.skipif(not SUMMARY.exists(), reason="no Lead Agent run in data/processed")
PAGES = ["app/Overview.py", "app/pages/1_Redemption_Likelihood.py", "app/pages/2_Spend_Forecast.py",
         "app/pages/3_Campaign_Uplift.py"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_errors(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / page), default_timeout=120).run()
    assert not at.exception, at.exception


def test_redemption_page_follows_the_controls():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / PAGES[1]), default_timeout=120).run()
    from lib.plan import list_size
    at.slider[0].set_value(50).run()
    assert not at.exception
    assert at.metric[0].label == "Households to mail"
    note = next(c.value for c in at.caption if c.value.endswith("mailed in this campaign"))
    mailed_total = int(note.split()[1].replace(",", ""))                         # "of 1,133 mailed in this campaign"
    assert at.metric[0].value == f"{list_size(mailed_total, 0.5):,}"
    first = at.selectbox[0].options[0]                                           # switch to the earliest campaign
    at.selectbox[0].select(int(first.split()[1])).run()
    assert not at.exception and at.selectbox[0].value == int(first.split()[1])


def test_plan_matches_the_evaluation_agent_on_real_data():
    """At the top 20% the page's pooled number is the Evaluation Agent's own within-campaign capture."""
    import json

    from decision.table import in_use_rows
    from lib.plan import all_campaigns
    from modeling.model1_evaluation import within_campaign_capture

    preds = in_use_rows(1, ROOT / "data/processed/modeling_agent")
    table, pooled = all_campaigns(preds, 0.2)
    assert pooled["model"] == pytest.approx(within_campaign_capture(preds["campaign"], preds["label"], preds["score"], 0.2))
    verdict = json.loads((ROOT / "data/processed/evaluation_agent/model1/verdict.json").read_text())
    owner = verdict.get("owner_model", "owner_forest")
    assert set(preds["candidate_id"]) == {owner}
