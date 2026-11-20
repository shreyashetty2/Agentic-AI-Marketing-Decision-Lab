"""The Decision Lab's menu: the pieces every question is built from, the checks a question's form must pass,
and the calculations behind each piece (docs/agent_design/decision_lab_design.md).

The chat's LLM never does arithmetic. It fills in a form -- who (a campaign, household traits, a product
category), what (one measure), how (one way of arranging the result) -- and `answer()` checks the form and runs
it on the clean table (decision/table.py). The scope, complexity and clarifying-question rules live here, in code:
a form that names something not on the menu, misses a required field, or asks for more than one measure gets a
question or a refusal back, never a number.
"""
import difflib
import math
from dataclasses import dataclass, field

import pandas as pd

MEASURES = {
    "redemption_chance": {"column": "p_redeem", "models": (1,), "tool": "Redemption Likelihood", "unit": "share",
                          "plain": "chance the household uses the coupon", "actual": "redeemed",
                          "total_as": "expected redemptions"},
    "expected_spend": {"column": "spend_if_redeem", "models": (2,), "tool": "Spend Forecast", "unit": "dollars",
                       "plain": "dollars spent on the campaign's products if the household redeems",
                       "total_as": "expected dollars, if every one of them redeemed"},
    "expected_revenue": {"column": "expected_revenue", "models": (1, 2), "tool": "Redemption Likelihood x Spend Forecast",
                         "unit": "dollars", "total_as": "expected dollars from mailing them",
                         "plain": "expected dollars from mailing the household (chance it redeems x spend if it does)"},
    "uplift": {"column": "uplift", "models": (3,), "tool": "Campaign Uplift", "unit": "share", "total_as": "expected extra buyers",
               "plain": "extra chance the household buys the campaign's products because it was mailed"},
    "buy_anyway": {"column": "buy_anyway", "models": (3,), "tool": "Campaign Uplift", "unit": "share",
                   "total_as": "expected buyers without any mailer",
                   "plain": "chance the household buys the campaign's products without a mailer"},
}

# Only homeowner status and kids have readable values. The dataset's own guide says the five classification
# fields were deliberately given generic names and values, so the Lab never claims what they stand for.
TRAITS = {
    "homeowner": {"column": "homeowner_desc",
                  "values": ["Homeowner", "Probable Owner", "Probable Renter", "Renter", "Unknown"]},
    "kids": {"column": "kid_category_desc", "values": ["1", "2", "3+", "None/Unknown"]},
    "classification_1": {"column": "classification_1", "values": [f"Age Group{i}" for i in range(1, 7)]},
    "classification_2": {"column": "classification_2", "values": ["X", "Y", "Z"]},
    "classification_3": {"column": "classification_3", "values": [f"Level{i}" for i in range(1, 13)]},
    "classification_4": {"column": "classification_4", "values": ["1", "2", "3", "4", "5+"]},
    "classification_5": {"column": "classification_5", "values": [f"Group{i}" for i in range(1, 7)]},
}
CODE_NOTES = {
    "homeowner": "Whether the household owns its home: Homeowner, Probable Owner, Probable Renter, Renter, Unknown.",
    "kids": "Number of kids in the household: 1, 2, 3+, or None/Unknown (no kids recorded, or not known).",
    "classification_1": "Six ordered age groups (Age Group1 to Age Group6). The dataset hides the actual age ranges.",
    "classification_2": "Three groups, X, Y and Z, with no order. The dataset hides what they stand for.",
    "classification_3": "Twelve ordered levels (Level1 to Level12). The dataset hides what they measure.",
    "classification_4": "Five ordered values, 1 to 5+. The dataset hides what they count; it could be household size, "
                        "but nothing in the data confirms that.",
    "classification_5": "Six ordered groups (Group1 to Group6). The dataset hides what they stand for.",
}
# Product labels that aren't products a coupon could promote: placeholders, the coupon lines themselves, deposits.
NOT_CATEGORIES = {"NO COMMODITY DESCRIPTION", "UNKNOWN", "", "COUPON", "COUPON/MISC ITEMS", "COUPONS/STORE & MFG",
                  "BOTTLE DEPOSITS", "MISCELLANEOUS(CORP USE ONLY)"}
HOWS = ("summary", "top", "by_segment", "compare", "by_category", "household")
MIN_GROUP = 30         # fewer households than this: answer, but say it's too few to trust and offer to widen
DEFAULT_TOP = 10
SHOW_HOUSEHOLDS = 20   # households listed in the answer itself; the full list goes to the download


@dataclass
class Result:
    status: str                      # "answer", "ask" (a clarifying question), or "cannot" (out of scope)
    message: str = ""                # for "ask" / "cannot": what to tell the marketer
    numbers: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)
    assumptions: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    households: pd.DataFrame | None = None   # the full list behind a "top" answer, for download

    def for_llm(self) -> dict:
        return {k: v for k, v in {"status": self.status, "message": self.message, "numbers": self.numbers,
                                  "steps": self.steps, "assumptions": self.assumptions,
                                  "warnings": self.warnings}.items() if v}


def ask(message: str) -> Result:
    return Result("ask", message)


def cannot(message: str) -> Result:
    return Result("cannot", message)


class Lab:
    """The menu over one clean table. `campaign_products` / `category_redemptions`: decision/table.py's."""

    def __init__(self, table: pd.DataFrame, campaign_products: pd.DataFrame, category_redemptions: pd.DataFrame,
                 cautions: dict[int, str] | None = None):
        """cautions: model number -> why the team's Evaluation Agent hasn't signed that model off
        (decision/table.py's `cautions`). Every answer built on that model carries it as a warning."""
        self.table, self.products, self.redemptions = table, campaign_products, category_redemptions
        self.cautions = cautions or {}
        self.all_campaigns = sorted(set(campaign_products["campaign"]) | set(table["campaign"]))
        self.redemptions = category_redemptions[~category_redemptions["category"].str.strip().isin(NOT_CATEGORIES)]
        self.categories = sorted(self.redemptions["category"].unique())

    # ---- checks -------------------------------------------------------------------------------------------
    def covered(self, measure: str) -> list[int]:
        col = MEASURES[measure]["column"]
        return sorted(self.table.loc[self.table[col].notna(), "campaign"].unique().tolist())

    def _check_campaign(self, campaign, measure: str | None) -> Result | None:
        if campaign is None:
            return None
        if campaign not in self.all_campaigns:
            return cannot(f"There is no campaign {campaign}. The campaigns are numbered "
                          f"{min(self.all_campaigns)} to {max(self.all_campaigns)}.")
        if measure and campaign not in self.covered(measure):
            m = MEASURES[measure]
            return cannot(f"{m['tool']} doesn't cover campaign {campaign}. It covers campaigns "
                          f"{_ranges(self.covered(measure))}. (Campaigns can be missing because they are held back "
                          f"to test the model fairly, or because the model wasn't built for that campaign type.)")
        return None

    def _check_filters(self, filters) -> Result | None:
        for f in filters or []:
            trait, values = f.get("trait"), f.get("values") or []
            if trait not in TRAITS:
                return ask(f"I can filter households by: {', '.join(TRAITS)}. '{trait}' isn't one of them. "
                           f"(This dataset has no household size, age range or income in readable form.)")
            bad = [v for v in values if v not in TRAITS[trait]["values"]]
            if bad or not values:
                return ask(f"For {trait}, the values are: {', '.join(TRAITS[trait]['values'])}. "
                           f"Which do you mean?" + (f" ('{bad[0]}' isn't one of them.)" if bad else ""))
        return None

    def _match_category(self, category: str) -> tuple[str | None, Result | None]:
        exact = [c for c in self.categories if c.lower() == category.lower()]
        if exact:
            return exact[0], None
        close = [c for c in self.categories if category.lower() in c.lower()]
        close += [c for c in difflib.get_close_matches(category.upper(), self.categories, n=5, cutoff=0.6) if c not in close]
        if close:
            return None, ask(f"I don't have a category called '{category}'. Did you mean: {', '.join(close[:6])}?")
        return None, cannot(f"No product category matches '{category}'. Ask me to list categories if that helps.")

    # ---- the pieces ---------------------------------------------------------------------------------------
    def _rows(self, measure: str | None, campaign, filters) -> tuple[pd.DataFrame, list, list]:
        steps, assumptions = [], []
        rows = self.table
        if measure:
            rows = rows[rows[MEASURES[measure]["column"]].notna()]
        if campaign is None:
            if measure:
                assumptions.append(f"You didn't name a campaign, so this covers every campaign "
                                   f"{MEASURES[measure]['tool']} scores ({_ranges(self.covered(measure))}).")
        else:
            rows = rows[rows["campaign"] == campaign]
            steps.append(f"Looked at campaign {campaign}.")
        if measure and MEASURES[measure]["column"] in ("p_redeem", "spend_if_redeem", "expected_revenue"):
            steps.append("Used mailed households only (the only ones these scores exist for).")
        for f in filters or []:
            rows = rows[rows[TRAITS[f["trait"]]["column"]].isin(f["values"])]
            steps.append(f"Kept households where {f['trait']} is {' or '.join(f['values'])}.")
        if filters:
            assumptions.append("Only about a third of households (801 of 2,500) have demographic details on file, "
                               "so a filter on them leaves out everyone without a record.")
        return rows, steps, assumptions

    def _summary(self, rows: pd.DataFrame, measure: str) -> dict:
        m = MEASURES[measure]
        out = {"households": int(rows["household_key"].nunique()), "household_campaign_pairs": len(rows),
               f"average {m['plain']}": _fmt(rows[m["column"]].mean(), m["unit"])}
        if m.get("actual"):
            out["share that actually redeemed, in these past campaigns"] = _fmt(rows[m["actual"]].mean(), "share")
        return out

    def _too_small(self, n: int) -> list:
        return [f"Only {n} households match, too few to trust; offer to widen the group."] if n < MIN_GROUP else []

    def answer(self, form: dict) -> Result:
        result = self._answer(form)
        measure = MEASURES.get(form.get("measure"))
        if result.status == "answer" and measure:
            result.warnings += [self.cautions[m] for m in measure["models"] if m in self.cautions]
        return result

    def _answer(self, form: dict) -> Result:
        how, measure, campaign = form.get("how"), form.get("measure"), form.get("campaign")
        filters = form.get("filters") or []
        if how not in HOWS:
            return cannot(f"I can't arrange an answer as '{how}'.")
        if how == "household":
            return self._household(form.get("household_key"))
        if how == "by_category":
            return self._by_category(campaign, filters)
        if measure == "past_redemptions":
            return self._category_by_segment(form.get("category"), form.get("segment_by"), filters)
        if measure not in MEASURES:
            return ask("Which do you want to go by: chance they use the coupon, how much they'd spend, expected "
                       "revenue, how much the mailer changes their buying, or whether they'd buy anyway?")
        problem = self._check_campaign(campaign, measure) or self._check_filters(filters)
        if problem:
            return problem
        if how == "compare":
            return self._compare(measure, campaign, form.get("groups") or [])
        rows, steps, assumptions = self._rows(measure, campaign, filters)
        if rows.empty:
            return cannot("No households match that description.")
        if how == "summary":
            numbers = self._summary(rows, measure)
            return Result("answer", numbers=numbers, steps=steps, assumptions=assumptions,
                          warnings=self._too_small(numbers["households"]))
        if how == "top":
            return self._top(rows, measure, campaign, form, steps, assumptions)
        return self._by_segment(rows, measure, form.get("segment_by"), steps, assumptions)

    def _top(self, rows, measure, campaign, form, steps, assumptions) -> Result:
        if campaign is None:
            return ask("Which campaign? Who's best to mail differs a lot from one campaign to the next.")
        m, n_all = MEASURES[measure], len(rows)
        top_n, share = form.get("top_n"), form.get("top_share")
        if share is not None and not 0 < share <= 1:
            return ask("What share of the list do you want, between 1% and 100%?")
        if top_n is not None and top_n < 1:
            return ask("How many households do you want?")
        n = (math.ceil(round(n_all * share, 9)) if share is not None else min(top_n, n_all) if top_n else
             min(DEFAULT_TOP, n_all))
        if top_n is None and share is None:
            assumptions.append(f"You didn't say how many, so this is the top {n}. Ask for more if you need them.")
        if top_n and top_n > n_all:
            assumptions.append(f"You asked for {top_n}, but only {n_all} households match, so this is all of them.")
        ranked = rows.sort_values([m["column"], "household_key"], ascending=[False, True])
        top = ranked.head(n)
        steps.append(f"Ranked the {n_all:,} matching households by {m['plain']} and took the top {n:,}.")
        numbers = {"households in the list": n, "households to choose from": n_all,
                   m["total_as"]: _fmt(top[m["column"]].sum(), "dollars" if m["unit"] == "dollars" else "count")}
        if n == n_all:
            assumptions.append("This list is every matching household, so there's no ranking to compare with "
                               "mailing at random. Ask for fewer to see who's best.")
        elif measure in ("redemption_chance", "expected_revenue"):
            # One line, so the real count is only ever compared with the random pick's real count -- never with
            # the model's forecast (expected redemptions above).
            reached, total = int(top["redeemed"].sum()), int(rows["redeemed"].sum())
            numbers["what actually happened in this campaign"] = (
                f"this list would have reached {reached} of the {total} households that really redeemed; "
                f"mailing {n} at random would have reached about {total * n / n_all:.1f}")
        if measure in ("uplift", "buy_anyway"):
            steps.append("Included households that weren't mailed, since the question is who to mail.")
        list_cols = ["household_key", "campaign", m["column"]]
        numbers["first households in the list"] = [
            {"household": int(r.household_key), m["plain"]: _fmt(getattr(r, m["column"]), m["unit"])}
            for r in top.head(SHOW_HOUSEHOLDS).itertuples()]
        return Result("answer", numbers=numbers, steps=steps, assumptions=assumptions,
                      warnings=self._too_small(n_all), households=top[list_cols].reset_index(drop=True))

    def _by_segment(self, rows, measure, segment_by, steps, assumptions) -> Result:
        if segment_by not in TRAITS:
            return ask(f"Which trait should I break it down by? I have: {', '.join(TRAITS)}.")
        m, col = MEASURES[measure], TRAITS[segment_by]["column"]
        known = rows[rows[col].notna()]
        steps.append(f"Grouped households with a demographic record by {segment_by} and averaged {m['plain']}.")
        groups = []
        means = known.groupby(col)[m["column"]].mean().sort_values(ascending=False)
        for value, mean in means.items():
            g = known[known[col] == value]
            entry = {segment_by: value, "households": int(g["household_key"].nunique()), "average": _fmt(mean, m["unit"])}
            if m.get("actual"):
                entry["share that actually redeemed"] = _fmt(g[m["actual"]].mean(), "share")
            groups.append(entry)
        assumptions.append(f"{known['household_key'].nunique():,} of {rows['household_key'].nunique():,} households "
                           f"here have a demographic record; the rest can't be grouped.")
        warnings = [f"{g[segment_by]}: only {g['households']} households." for g in groups if g["households"] < MIN_GROUP]
        return Result("answer", numbers={"groups, highest first": groups, "measure": m["plain"],
                                         "trait meaning": CODE_NOTES[segment_by]},
                      steps=steps, assumptions=assumptions, warnings=warnings)

    def _compare(self, measure, campaign, groups) -> Result:
        if len(groups) != 2:
            return ask("Which two groups should I compare? I compare two at a time.")
        out, steps, assumptions, warnings = [], [], [], []
        for i, g in enumerate(groups, 1):
            problem = self._check_filters(g)
            if problem:
                return problem
            rows, s, a = self._rows(measure, campaign, g)
            label = "; ".join(f"{f['trait']} = {' or '.join(f['values'])}" for f in g) or "everyone"
            out.append({"group": label, **self._summary(rows, measure)})
            steps += [f"Group {i}: {x}" for x in s]
            assumptions += [x for x in a if x not in assumptions]
            warnings += [f"Group {i}: {w}" for w in self._too_small(out[-1]["households"])]
        return Result("answer", numbers={"comparison": out}, steps=steps, assumptions=assumptions, warnings=warnings)

    def _household(self, key) -> Result:
        if key is None:
            return ask("Which household? Give me its number.")
        rows = self.table[self.table["household_key"] == key]
        if rows.empty:
            return cannot(f"No model scores household {key} in any campaign it covers.")
        per = []
        for r in rows.itertuples():
            entry = {"campaign": int(r.campaign), "mailed": bool(r.mailed)}
            for name, m in MEASURES.items():
                v = getattr(r, m["column"])
                if pd.notna(v):
                    entry[m["plain"]] = _fmt(v, m["unit"])
            per.append(entry)
        traits = {t: rows.iloc[0][c["column"]] for t, c in TRAITS.items() if pd.notna(rows.iloc[0][c["column"]])}
        return Result("answer", numbers={"household": int(key), "traits": traits or "no demographic record",
                                         "by campaign": per},
                      steps=[f"Looked up every campaign any model scores household {key} in."])

    def _by_category(self, campaign, filters) -> Result:
        problem = self._check_campaign(campaign, None) or self._check_filters(filters)
        if problem:
            return problem
        homes = self.table if campaign is None else self.table[self.table["campaign"] == campaign]
        for f in filters:
            homes = homes[homes[TRAITS[f["trait"]]["column"]].isin(f["values"])]
        group = set(homes["household_key"])
        if not group:
            return cannot("No households match that description.")
        # Compare like with like: a demographic filter keeps only households with a record, so the baseline is
        # every household with a record too.
        base_homes = self.table[self.table["has_demographics"]] if filters else self.table
        everyone = set(base_homes["household_key"])
        red = self.redemptions
        in_group = red[red["household_key"].isin(group)].groupby("category")["household_key"].nunique()
        overall = red[red["household_key"].isin(everyone)].groupby("category")["household_key"].nunique()
        rows = []
        for cat, n in in_group.sort_values(ascending=False).head(10).items():
            share, base = n / len(group), overall[cat] / len(everyone)
            rows.append({"category": cat, "share of this group that redeemed a coupon in it": _fmt(share, "share"),
                         "share of comparable households": _fmt(base, "share"), "times as likely": round(share / base, 2)})
        steps = ["Counted, for each product category, how many of these households ever redeemed a coupon on it "
                 "(any campaign), and compared that with all households."]
        if filters:
            steps.insert(0, "Kept households where " + "; ".join(f"{f['trait']} is {' or '.join(f['values'])}"
                                                                 for f in filters) + ".")
        return Result("answer", numbers={"households in the group": len(group), "top categories": rows},
                      steps=steps, assumptions=["This is past coupon behaviour, not a model's prediction."],
                      warnings=self._too_small(len(group)))

    def _category_by_segment(self, category, segment_by, filters) -> Result:
        if not category:
            return ask("Which product category?")
        cat, problem = self._match_category(category)
        if problem:
            return problem
        if segment_by not in TRAITS:
            return ask(f"Which trait should I break {cat} down by? I have: {', '.join(TRAITS)}.")
        col = TRAITS[segment_by]["column"]
        homes = self.table.drop_duplicates("household_key")
        homes = homes[homes[col].notna()]
        redeemers = set(self.redemptions.loc[self.redemptions["category"] == cat, "household_key"])
        groups = []
        for value, g in homes.groupby(col):
            n = len(g)
            groups.append({segment_by: value, "households": n,
                           f"share that ever redeemed a {cat} coupon": _fmt(g["household_key"].isin(redeemers).mean(), "share")})
        return Result("answer", numbers={"category": cat, "groups": groups, "trait meaning": CODE_NOTES[segment_by]},
                      steps=[f"For each {segment_by} group, counted the households that ever redeemed a coupon on {cat}."],
                      assumptions=["This is past coupon behaviour, not a model's prediction.",
                                   "Only households with a demographic record can be grouped."])

    # ---- describe -----------------------------------------------------------------------------------------
    def describe_campaign(self, campaign) -> Result:
        if campaign is None:
            return ask("Which campaign?")
        problem = self._check_campaign(campaign, None)
        if problem:
            return problem
        c = self.table[self.table["campaign"] == campaign]
        info = self.products[self.products["campaign"] == campaign]
        rows = c.drop_duplicates("campaign")
        numbers = {"campaign": campaign,
                   "categories it promoted, most products first": info.head(8)["category"].tolist(),
                   "number of categories": int(len(info)),
                   "covered by": list(dict.fromkeys(m["tool"] for k, m in MEASURES.items() if campaign in self.covered(k)))
                   or "no model"}
        if not rows.empty:
            r = rows.iloc[0]
            numbers.update({"type": r["campaign_type"], "length in days": int(r["length_days"]),
                            "runs from day": int(r["start_day"]), "to day": int(r["end_day"])})
        return Result("answer", numbers=numbers, assumptions=[
            "The dataset numbers its days 1 to 711 and has no calendar dates, so I can't say which month this ran."])

    def describe_code(self, trait) -> Result:
        """`trait` may be a trait's name (classification_1) or one of its values (Age Group1, "age group 1"):
        a marketer names what they see, not the column it lives in."""
        trait = trait if trait in CODE_NOTES else _trait_of_value(trait)
        if trait not in CODE_NOTES:
            return ask(f"I can explain: {', '.join(CODE_NOTES)}. Household size, age ranges and income aren't "
                       f"in this dataset in readable form.")
        return Result("answer", numbers={"trait": trait, "meaning": CODE_NOTES[trait],
                                         "values": TRAITS[trait]["values"]})

    def find_category(self, word: str) -> Result:
        if not word.strip():
            return ask("Which product are you looking for?")
        hits =[c for c in self.categories if word.lower() in c.lower()]
        hits += [c for c in difflib.get_close_matches(word.upper(), self.categories, n=8, cutoff=0.6) if c not in hits]
        if not hits:
            return cannot(f"No product category matches '{word}'.")
        return Result("answer", numbers={"matching categories": hits[:15]})


def _trait_of_value(text) -> str | None:
    """The trait one of whose values matches `text`, ignoring case, spaces and underscores ("age_group_1" ->
    classification_1). None if no value matches, or if it matches more than one trait (e.g. "1")."""
    squash = lambda v: str(v).lower().replace(" ", "").replace("_", "")  # noqa: E731
    hits = {t for t, c in TRAITS.items() for v in c["values"] if squash(v) == squash(text or "")}
    return hits.pop() if len(hits) == 1 else None


def _fmt(value, unit: str):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "not available"
    if unit == "share":
        return f"{value:.1%}"
    if unit == "dollars":
        return f"${value:,.2f}"
    return round(float(value), 1)


def _ranges(nums: list[int]) -> str:
    """[8, 9, 10, 12] -> '8-10, 12'."""
    out, start = [], None
    for i, n in enumerate(nums):
        start = n if start is None else start
        if i == len(nums) - 1 or nums[i + 1] != n + 1:
            out.append(str(start) if start == n else f"{start}-{n}")
            start = None
    return ", ".join(out)
