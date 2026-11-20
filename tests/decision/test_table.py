"""The clean table: only the model in use, never held-back campaigns, and every score matching what its own
model wrote. The real-data checks recount from the agents' files directly, sharing no code with table.py."""
import json
import re

import numpy as np
import pandas as pd
import pytest

from decision_helpers import PROCESSED, needs_data
from decision import table as T
from decision.menu import Lab

MODELING = PROCESSED / "modeling_agent"


def write_predictions(tmp_path, rows):
    d = tmp_path / "model1"
    d.mkdir()
    pd.DataFrame(rows).to_parquet(d / "predictions.parquet")
    return tmp_path


def pred(hh, campaign, cand, default, chosen=False, locked=False):
    return {"household_key": hh, "campaign": campaign, "candidate_id": cand, "is_default": default,
            "chosen": chosen, "locked": locked, "score": 0.1, "label": 0}


def test_in_use_rows_keeps_the_model_in_use_not_the_experimental_pick(tmp_path):
    d = write_predictions(tmp_path, [pred(1, 8, "owner", True), pred(1, 8, "agent_pick", False, chosen=True)])
    assert T.in_use_rows(1, d)["candidate_id"].tolist() == ["owner"]


@pytest.mark.parametrize("rows, problem", [
    ([pred(1, 8, "owner", True, locked=True)], "held-back"),
    ([pred(1, 8, "owner", False)], "no rows marked"),
    ([pred(1, 8, "a", True), pred(2, 8, "b", True)], "more than one model"),
    ([pred(1, 8, "owner", True), pred(1, 8, "owner", True)], "more than once"),
])
def test_in_use_rows_refuses_outputs_that_dont_fit_together(tmp_path, rows, problem):
    with pytest.raises(T.TableError, match=problem):
        T.in_use_rows(1, write_predictions(tmp_path, rows))


@pytest.fixture(scope="module")
def table():
    return T.build(PROCESSED)


def in_use_directly(model):
    """Independent read: the candidate the manifest names as default, straight from the parquet."""
    default = json.loads((MODELING / f"model{model}/manifest.json").read_text())["default"]
    df = pd.read_parquet(MODELING / f"model{model}/predictions.parquet")
    return df[df["candidate_id"] == default].set_index(["household_key", "campaign"])


@needs_data
def test_one_row_per_household_per_campaign(table):
    assert not table.duplicated(["household_key", "campaign"]).any()


@needs_data
def test_redemption_scores_are_exactly_the_model_in_use(table):
    direct = in_use_directly(1)
    got = table.dropna(subset=["p_redeem"]).set_index(["household_key", "campaign"])
    assert len(got) == len(direct)
    assert np.allclose(got["p_redeem"], direct.loc[got.index, "score"])
    assert (got["redeemed"] == direct.loc[got.index, "label"]).all()


@needs_data
def test_uplift_scores_are_exactly_the_model_in_use(table):
    direct = in_use_directly(3)
    got = table.dropna(subset=["uplift"]).set_index(["household_key", "campaign"])
    assert len(got) == len(direct)
    assert np.allclose(got["uplift"], direct.loc[got.index, "score"])
    assert (got["mailed"] == direct.loc[got.index, "treatment"].astype(bool)).all()


@needs_data
def test_no_held_back_campaign_reaches_the_table(table):
    for model, col in ((1, "p_redeem"), (3, "uplift")):
        locked = set(pd.read_parquet(MODELING / f"model{model}/locked_predictions.parquet")["campaign"])
        assert not set(table.dropna(subset=[col])["campaign"]) & locked


@needs_data
def test_spend_follows_the_published_line_and_is_blank_where_spend_forecast_isnt_built(table):
    # Published in the deck: spend = $9.68 + 0.70 x usual spend. Recompute usual spend from model1_table.
    inputs = pd.read_parquet(PROCESSED / "model1_table.parquet",
                             columns=["household_key", "campaign", "campaign_product_spend_per_week_26w", "duration_days"])
    rows = table.dropna(subset=["spend_if_redeem"])[["household_key", "campaign", "spend_if_redeem"]].merge(
        inputs, on=["household_key", "campaign"])
    usual = rows["campaign_product_spend_per_week_26w"] * rows["duration_days"] / 7
    # The deck rounds to cents and two decimals; allow that rounding (0.005 x usual) plus half a cent.
    assert np.allclose(rows["spend_if_redeem"], 9.68 + 0.70 * usual, rtol=0, atol=0.005 * usual.max() + 0.01)
    type_a = table[(table["campaign_type"] == "A") & table["p_redeem"].notna()]
    assert len(type_a) and type_a["spend_if_redeem"].isna().all()


@needs_data
def test_expected_revenue_is_redemption_chance_times_spend(table):
    rows = table.dropna(subset=["expected_revenue"])
    assert np.allclose(rows["expected_revenue"], rows["p_redeem"] * rows["spend_if_redeem"])


@needs_data
def test_redemption_scores_exist_only_for_mailed_households(table):
    assert table.loc[table["p_redeem"].notna(), "mailed"].all()


@needs_data
def test_mailing_the_top_half_matches_the_independently_counted_week_10_result(table):
    """Week 10 writeup: mailing the top half of each test campaign reaches 594 of 707 redeemers. Recount here
    through the menu, campaign by campaign, from the table."""
    lab = Lab(table, T.campaign_products(PROCESSED), T.category_redemptions(PROCESSED))
    reached = total = 0
    for c in lab.covered("redemption_chance"):
        r = lab.answer({"how": "top", "measure": "redemption_chance", "campaign": c, "top_share": 0.5})
        line = r.numbers["what actually happened in this campaign"]
        got, of = map(int, re.search(r"reached (\d+) of the (\d+) ", line).groups())
        reached, total = reached + got, total + of
    assert (reached, total) == (594, 707)


def test_cautions_cover_only_unapproved_models_in_plain_words(tmp_path):
    (tmp_path / "lead_agent").mkdir()
    (tmp_path / "lead_agent/run_summary.json").write_text(json.dumps({"steps": [
        {"step": "evaluation:model1", "verdict": "approve", "facts": {}},
        {"step": "evaluation:model2", "verdict": "human_needed", "handled": "no rule covers this combination (x)",
         "facts": {"vs_simple_rule": "+$0.50 (could be luck)"}},
        {"step": "evaluation:model3", "verdict": "human_needed", "handled": "EVA-02: groups not comparable",
         "facts": {}},
        {"step": "modeling:model3", "verdict": "whatever"}]}))
    got = T.cautions(tmp_path)
    assert set(got) == {2, 3}
    assert got[2] == ("Spend Forecast hasn't been signed off by the team's checks yet. Compared with a simple rule "
                      "it scores +$0.50 (could be luck). Treat its numbers as a rough guide.")
    assert got[3] == ("Campaign Uplift hasn't been signed off by the team's checks yet: groups not comparable. "
                      "Treat its numbers as a rough guide.")


def test_sure_things_are_blank_when_the_classifier_has_not_been_run(tmp_path):
    out = T.sure_things(tmp_path)
    assert out.empty and list(out.columns) == ["household_key", "campaign", "buy_anyway"]


def test_sure_things_come_from_the_classifiers_per_household_file(tmp_path):
    import pandas as pd
    pd.DataFrame({"household_key": [1, 2], "campaign": [17, 17], "p_sure_thing": [0.9, 0.2],
                  "actually_bought_without_redeeming": [1, 0]}).to_parquet(tmp_path / "model3_sure_things_households.parquet")
    out = T.sure_things(tmp_path)
    assert out["buy_anyway"].tolist() == [0.9, 0.2]
