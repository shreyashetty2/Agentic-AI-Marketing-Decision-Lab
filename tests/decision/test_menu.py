"""The menu's checks and calculations, against numbers worked out by hand on the toy table (conftest.py)."""
import pytest

from decision.menu import DEFAULT_TOP, _ranges


def homeowners():
    return [{"trait": "homeowner", "values": ["Homeowner"]}]


# ---- top N -------------------------------------------------------------------------------------------------
def test_top_n_takes_the_best_scored_households_in_order(lab):
    r = lab.answer({"how": "top", "measure": "redemption_chance", "campaign": 1, "top_n": 3})
    assert r.status == "answer"
    assert r.households["household_key"].tolist() == [1, 2, 3]
    assert r.numbers["expected redemptions"] == round(0.99 + 0.98 + 0.97, 1)


def test_top_counts_real_redeemers_reached_and_the_random_baseline(lab):
    # top 8 = households 1..8; redeemers are every 4th -> 4 and 8. 10 redeemers in all 40; random 8 -> 2.0.
    r = lab.answer({"how": "top", "measure": "redemption_chance", "campaign": 1, "top_n": 8})
    assert r.numbers["what actually happened in this campaign"] == (
        "this list would have reached 2 of the 10 households that really redeemed; "
        "mailing 8 at random would have reached about 2.0")


def test_top_share_rounds_up_to_whole_households(lab):
    r = lab.answer({"how": "top", "measure": "redemption_chance", "campaign": 1, "top_share": 0.26})
    assert len(r.households) == 11          # 0.26 x 40 = 10.4 -> 11


def test_top_without_a_number_uses_the_default_and_says_so(lab):
    r = lab.answer({"how": "top", "measure": "redemption_chance", "campaign": 1})
    assert len(r.households) == DEFAULT_TOP
    assert any("didn't say how many" in a for a in r.assumptions)


def test_top_asking_for_more_than_exist_returns_all_and_says_so(lab):
    r = lab.answer({"how": "top", "measure": "redemption_chance", "campaign": 1, "top_n": 500})
    assert len(r.households) == 40
    assert any("only 40" in a for a in r.assumptions)


def test_top_ties_go_in_household_order():
    from decision.menu import Lab
    from decision_helpers import toy_products, toy_redemptions, toy_table
    t = toy_table()
    t.loc[t["campaign"] == 1, "p_redeem"] = 0.5
    r = Lab(t, toy_products(), toy_redemptions()).answer(
        {"how": "top", "measure": "redemption_chance", "campaign": 1, "top_n": 3})
    assert r.households["household_key"].tolist() == [1, 2, 3]


def test_top_by_uplift_includes_unmailed_households(lab):
    r = lab.answer({"how": "top", "measure": "uplift", "campaign": 2, "top_n": 2})
    assert r.households["household_key"].tolist() == [40, 39]
    assert any("weren't mailed" in s for s in r.steps)


def test_top_with_a_filter_ranks_only_that_group(lab):
    r = lab.answer({"how": "top", "measure": "redemption_chance", "campaign": 1, "top_n": 2, "filters": homeowners()})
    assert r.households["household_key"].tolist() == [1, 2]
    assert any("Only 10 households" in w for w in r.warnings)


# ---- summary, segments, compare ------------------------------------------------------------------------------
def test_summary_averages_the_measure_and_reports_what_actually_happened(lab):
    r = lab.answer({"how": "summary", "measure": "redemption_chance", "campaign": 1, "filters": homeowners()})
    assert r.numbers["households"] == 10
    assert r.numbers["average chance the household uses the coupon"] == "94.5%"   # mean of 0.99..0.90
    assert r.numbers["share that actually redeemed, in these past campaigns"] == "20.0%"   # 4 and 8 of 1..10


def test_summary_without_a_campaign_covers_every_campaign_the_measure_scores_and_says_so(lab):
    r = lab.answer({"how": "summary", "measure": "uplift"})
    assert r.numbers["household_campaign_pairs"] == 80
    assert any("1-2" in a for a in r.assumptions)


def test_by_segment_orders_groups_highest_first_and_flags_small_ones(lab):
    r = lab.answer({"how": "by_segment", "measure": "redemption_chance", "campaign": 1, "segment_by": "homeowner"})
    groups = r.numbers["groups, highest first"]
    assert [g["homeowner"] for g in groups] == ["Homeowner", "Renter"]
    assert len(r.warnings) == 2
    assert any("20 of 40" in a for a in r.assumptions)


def test_compare_two_groups(lab):
    r = lab.answer({"how": "compare", "measure": "expected_spend", "campaign": 1,
                    "groups": [homeowners(), [{"trait": "homeowner", "values": ["Renter"]}]]})
    a, b = r.numbers["comparison"]
    assert a["households"] == b["households"] == 10
    assert a["average dollars spent on the campaign's products if the household redeems"] == "$15.50"   # 11..20
    assert b["average dollars spent on the campaign's products if the household redeems"] == "$25.50"   # 21..30


def test_compare_needs_exactly_two_groups(lab):
    for groups in ([], [homeowners()], [homeowners()] * 3):
        assert lab.answer({"how": "compare", "measure": "uplift", "groups": groups}).status == "ask"


# ---- the rules: clarifying questions and scope ---------------------------------------------------------------
@pytest.mark.parametrize("form", [
    {"how": "top", "measure": "redemption_chance", "top_n": 5},                 # no campaign
    {"how": "summary"},                                                         # no measure ("best")
    {"how": "summary", "measure": "best"},                                      # not a measure
    {"how": "summary", "measure": "uplift", "filters": [{"trait": "household_size", "values": ["3"]}]},
    {"how": "summary", "measure": "uplift", "filters": [{"trait": "kids", "values": ["4"]}]},
    {"how": "summary", "measure": "uplift", "filters": [{"trait": "kids", "values": []}]},
    {"how": "top", "measure": "uplift", "campaign": 1, "top_share": 1.5},
    {"how": "top", "measure": "uplift", "campaign": 1, "top_n": 0},
    {"how": "by_segment", "measure": "uplift", "segment_by": "age"},
    {"how": "household"},
])
def test_unclear_forms_get_a_clarifying_question_not_a_number(lab, form):
    r = lab.answer(form)
    assert r.status == "ask" and r.message and not r.numbers


def test_no_household_size_is_explained_when_asked_to_filter_on_it(lab):
    r = lab.answer({"how": "summary", "measure": "uplift", "filters": [{"trait": "household_size", "values": ["3"]}]})
    assert "no household size" in r.message


def test_a_campaign_that_does_not_exist_is_refused(lab):
    r = lab.answer({"how": "summary", "measure": "uplift", "campaign": 99})
    assert r.status == "cannot" and "no campaign 99" in r.message


def test_a_campaign_the_measure_does_not_cover_is_refused_with_what_it_does_cover(lab):
    r = lab.answer({"how": "summary", "measure": "redemption_chance", "campaign": 2})
    assert r.status == "cannot" and "covers campaigns 1" in r.message


def test_an_unknown_arrangement_is_refused(lab):
    assert lab.answer({"how": "forecast_next_april", "measure": "uplift"}).status == "cannot"


def test_a_filter_matching_nobody_is_refused(lab):
    r = lab.answer({"how": "summary", "measure": "uplift", "filters": [{"trait": "homeowner", "values": ["Probable Owner"]}]})
    assert r.status == "cannot"


# ---- households, products, describe --------------------------------------------------------------------------
def test_household_lookup_lists_every_campaign_and_only_scores_that_exist(lab):
    r = lab.answer({"how": "household", "household_key": 3})
    by = r.numbers["by campaign"]
    assert [b["campaign"] for b in by] == [1, 2]
    assert "chance the household uses the coupon" in by[0] and "chance the household uses the coupon" not in by[1]
    assert lab.answer({"how": "household", "household_key": 999}).status == "cannot"


def test_by_category_compares_a_group_with_comparable_households(lab):
    # homeowners (10): 5 redeemed YOGURT -> 50%. Comparable = the 20 with a record: 5 -> 25%. 2x as likely.
    r = lab.answer({"how": "by_category", "filters": homeowners()})
    top = r.numbers["top categories"][0]
    assert top["category"] == "YOGURT"
    assert top["share of this group that redeemed a coupon in it"] == "50.0%"
    assert top["share of comparable households"] == "25.0%"
    assert top["times as likely"] == 2.0


def test_placeholder_categories_are_never_reported(lab):
    r = lab.answer({"how": "by_category", "filters": [{"trait": "homeowner", "values": ["Renter"]}]})
    assert [c["category"] for c in r.numbers["top categories"]] == ["BEEF"]


def test_category_by_segment_matches_case_insensitively(lab):
    r = lab.answer({"how": "by_segment", "measure": "past_redemptions", "category": "yogurt", "segment_by": "homeowner"})
    groups = {g["homeowner"]: g["share that ever redeemed a YOGURT coupon"] for g in r.numbers["groups"]}
    assert groups == {"Homeowner": "50.0%", "Renter": "0.0%"}


def test_a_near_miss_category_asks_and_an_unknown_one_is_refused(lab):
    near = lab.answer({"how": "by_segment", "measure": "past_redemptions", "category": "yog", "segment_by": "kids"})
    assert near.status == "ask" and "YOGURT" in near.message
    none = lab.answer({"how": "by_segment", "measure": "past_redemptions", "category": "spaceships", "segment_by": "kids"})
    assert none.status == "cannot"


def test_describe_campaign_lists_its_categories_type_length_and_coverage(lab):
    r = lab.describe_campaign(1)
    assert r.numbers["categories it promoted, most products first"] == ["YOGURT", "BEEF"]
    assert r.numbers["length in days"] == 30
    assert "Redemption Likelihood" in r.numbers["covered by"]
    assert lab.describe_campaign(3).numbers["covered by"] == "no model"   # promoted products, but no model scores it


def test_describe_code_is_honest_about_hidden_meanings(lab):
    assert "hides" in lab.describe_code("classification_4").numbers["meaning"]
    assert lab.describe_code("income").status == "ask"


def test_find_category(lab):
    assert lab.find_category("yog").numbers["matching categories"] == ["YOGURT"]
    assert lab.find_category("zzz").status == "cannot"


def test_ranges():
    assert _ranges([8, 9, 10, 12, 14, 15]) == "8-10, 12, 14-15"
    assert _ranges([3]) == "3"


# ---- the team's sign-off travels with every answer built on an unapproved model --------------------------------
def test_answers_built_on_an_unapproved_model_carry_its_caution():
    from decision.menu import Lab
    from decision_helpers import toy_products, toy_redemptions, toy_table
    lab = Lab(toy_table(), toy_products(), toy_redemptions(), cautions={2: "SPEND NOT SIGNED OFF", 3: "UPLIFT NOT SIGNED OFF"})
    uplift = lab.answer({"how": "summary", "measure": "uplift", "campaign": 1})
    redeem = lab.answer({"how": "summary", "measure": "redemption_chance", "campaign": 1})
    revenue = lab.answer({"how": "top", "measure": "expected_revenue", "campaign": 1, "top_n": 3})
    refused = lab.answer({"how": "summary", "measure": "uplift", "campaign": 99})
    assert "UPLIFT NOT SIGNED OFF" in uplift.warnings
    assert not any("SIGNED OFF" in w for w in redeem.warnings)
    assert revenue.warnings == ["SPEND NOT SIGNED OFF"]   # chance x spend: only the spend half is unapproved
    assert refused.warnings == []


def test_a_top_list_that_is_everyone_says_so_and_makes_no_random_comparison(lab):
    r = lab.answer({"how": "top", "measure": "redemption_chance", "campaign": 1, "top_n": 40})
    assert "what actually happened in this campaign" not in r.numbers
    assert any("every matching household" in a for a in r.assumptions)


@pytest.mark.parametrize("said, trait", [("classification_1", "classification_1"), ("Age Group1", "classification_1"),
                                         ("age group 1", "classification_1"), ("age_group_1", "classification_1"),
                                         ("Level12", "classification_3"), ("probable renter", "homeowner")])
def test_describe_code_accepts_a_trait_or_one_of_its_values(lab, said, trait):
    assert lab.describe_code(said).numbers["trait"] == trait


@pytest.mark.parametrize("said", ["1", "income", "", None])
def test_describe_code_asks_when_the_word_is_ambiguous_or_unknown(lab, said):
    assert lab.describe_code(said).status == "ask"     # "1" is a value of both kids and classification_4
