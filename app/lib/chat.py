"""Ask the Decision Lab: the marketer's chat. Two sources, never mixed up:

- Background facts (this file): what the data is, what campaign Types mean, each tool's verdict and inputs, real
  redemption counts, product-level results. Gemini reads these to explain things.
- Calculations (src/decision/): who to mail, what a group would do, segment and product breakdowns. Gemini fills
  in a form, tested code computes the numbers. See docs/agent_design/decision_lab_design.md.

Runs on Vertex AI (Gemini), billed to the project's own GCP credits, per docs/agent_design/gcp_deployment.md.
Needs `gcloud auth application-default login` once (ADC), the same login used for the Cloud Run deployment.
"""
import os
import re

import streamlit as st

from lib.data import (MODEL_NAMES, PROCESSED, load_campaign_desc, load_campaign_products, load_dataset_facts,
                      load_model_features, load_predictions, load_product_redemption_summary, load_run_summary,
                      load_sure_things, load_type_a_product_example, step)
from decision import table as T
from decision.assistant import Turn, run_turn
from decision.menu import Lab

GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "capstone-decision-lab")
GCP_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")

TYPE_MEANING = {
    "A": "a broad/storewide campaign -- the dataset only records a pool of ~180-210 possible coupons for "
         "the campaign; each household got 16 chosen from that pool, but which 16 is not recorded, so no "
         "single household's exact mailing is known, only the broad pool of products it could have covered",
    "B": "a targeted campaign with a small coupon set (2-33 coupons); every household mailed in it got "
         "ALL of those coupons, so its product list is exactly what every one of its households received",
    "C": "a targeted campaign with a small coupon set (1-34 coupons), the same as Type B -- every household "
         "mailed in it got ALL of those coupons",
}

CAMPAIGN_RE = re.compile(r"campaign\D{0,6}(\d+)", re.IGNORECASE)

BACKGROUND_RULES = """About this data and these tools:
- This dataset has NO real calendar information -- no month, season or date, only an anonymised day-count and \
each campaign's Type (A, B or C). If asked what month, season or time of year is best, say the data has no real \
dates to answer from; never guess a season. You CAN answer from campaign length (days) and Type.
- Campaign Type has a real meaning (below). Always say what it means when it's relevant, never just "Type A".
- A new or upcoming campaign that isn't in the data can't be scored. Answer by analogy to past campaigns of the \
same Type -- who responded, and the department-level products typical of that Type -- and say it's an analogy, \
not a prediction.
- Explain terms plainly every time they come up, using the glossary below; assume no earlier explanation was \
read. Some demographic codes have no known meaning at all; if asked, say so and never guess.
- When asked about "the model" behind a tool, name the actual algorithm and, if asked, its literal input \
columns (below).
- Expected revenue per household (chance of redeeming x spend if they redeem) IS available through the \
`answer` function. Turning a coupon budget into a household count is not: no costs are recorded. Offer the \
ranked list instead -- the marketer picks how deep to go.
- "Human needed" means the automatic checks could not settle it and a person should judge; "approve" means it \
passed. A range that includes zero means the result could be luck.
- When asked what share of coupons get redeemed, give the real counts and rates by Type (below), never an \
average predicted score.
- Product-by-product results exist (below); Type A product results are only an illustration.
- Roughly 120 words unless the question asks for a list or a comparison."""


def credentials_ok() -> bool:
    """Whether Application Default Credentials are available at all -- cheap, makes no API call."""
    try:
        import google.auth
        google.auth.default()
        return True
    except Exception:
        return False


def _glossary_facts() -> str:
    """Plain-language definitions for someone with zero background. The dataset's own guide says the five
    classification fields were given generic names and values on purpose, so none is described as more than
    what the data shows -- in particular classification_4 (1 to 5+) is never called household size."""
    return "\n".join([
        "Glossary (plain-language terms a newcomer would need):",
        "- Campaign: a single coupon-mailing promotion, covering specific real products, run over some number "
        "of days. Each campaign has a Type -- A, B or C -- with a real meaning (see below).",
        "- Mailed vs not mailed: Campaign Uplift compares households that were mailed a campaign with households "
        "that were not. 'Causal' means asking whether the mailing CAUSED the buying, not just whether the two "
        "happened together.",
        "- Verdict: each tool is checked automatically. 'Approve' = passed; 'human needed' = the checks could not "
        "settle it, so a person should judge. 'Could be luck' = the likely range includes zero.",
        "- AUC: a 0-to-1 score of how well a tool ranks likely redeemers above unlikely ones (0.5 = guessing, "
        "1 = perfect).",
        "- Redemption: a household used the mailed coupon on a real purchase.",
        "- Uplift: how many EXTRA buyers a household contributes if mailed, compared to not mailing them.",
        "- Sure things: households who would likely have bought anyway, coupon or not -- a likely-wasted send.",
        "- Homeowner status and kids in the household: plain, not anonymised -- exactly what they say.",
        "- Age Group 1-6 (classification_1): an ordered age grouping; the actual age ranges, and which end is "
        "older, were removed from this anonymised dataset.",
        "- classification_2 (X/Y/Z), classification_3 (Level1-12), classification_4 (1 to 5+), classification_5 "
        "(Group1-6): anonymised codes with NO documented meaning. classification_4 may look like household size, "
        "but nothing in the data confirms it, so never call it household size; the data has no household size, "
        "income or education.",
    ])


def _wasted_products_facts() -> str:
    """Real products (the team's own product-level analysis, not a model score) most often mailed to a household
    that never redeemed, among products seen in at least 2 campaigns (1-campaign rows are too thin)."""
    df = load_product_redemption_summary()
    df = df[df["n_campaigns"] >= 2].nlargest(8, "pct_without_redemption")
    lines = ["Real products most often sent to a household that never redeemed the coupon (own analysis, "
             "not a model score -- useful for which products to be cautious mailing):"]
    lines += [f"- {r.commodity_desc} ({r.sub_commodity_desc}, {r.brand}): no redemption {r.pct_without_redemption:.0f}% "
              f"of the time, across {r.n_campaigns} campaigns" for r in df.itertuples()]
    return "\n".join(lines)


def _model_facts(summary: dict, model: int) -> str:
    name = MODEL_NAMES[model]
    modeling, evaluation = step(summary, "modeling", model), step(summary, "evaluation", model)
    if modeling is None or evaluation is None:
        return f"{name}: not in the latest run."
    facts = evaluation["facts"]
    lines = [
        f"{name} -- the model in use: {modeling['current_model']}",
        f"Trained on {modeling['trained_on']['campaigns']} finished campaigns ({modeling['trained_on']['rows']:,} rows).",
        "Built from these exact input columns: " + ", ".join(load_model_features(model)),
        "Relies most on: " + "; ".join(f"{c['meaning']} ({c['share']:.0%})" for c in modeling.get("leans_on", [])),
        f"Score: {facts['current_score']}",
        f"Versus the simple rule: {facts['vs_simple_rule']}",
        f"Forecast check: {facts['forecast_check']}",
        f"Evaluation verdict: {evaluation['verdict']}" + (f" -- {evaluation['handled']}" if evaluation.get("handled") else ""),
    ]
    in_use = modeling.get("in_use")
    if isinstance(in_use, dict):
        lines.append(f"In use since {in_use['decided_on']}, chosen by {in_use['decided_by']}: {in_use.get('why', '')}")
    return "\n".join(lines)


def _campaign_calendar_facts() -> str:
    """Every campaign's real type and length -- raw data, so all 30 campaigns whether or not modeled."""
    desc = load_campaign_desc().sort_values("campaign")
    lines = ["What campaign Type means (real, documented, not a guess):",
             *(f"- Type {t}: {meaning}" for t, meaning in TYPE_MEANING.items()),
             "", "Campaign calendar (day-numbers are anonymised -- NOT real dates):"]
    lines += [f"- campaign {r.campaign}: Type {r.campaign_type}, runs {r.duration_days} days" for r in desc.itertuples()]
    return "\n".join(lines)


def _type_product_facts() -> str:
    """Department-level products typical of each Type, across every past campaign of it -- the honest answer to
    "what would a new Type X campaign likely cover"."""
    merged = load_campaign_products().merge(load_campaign_desc()[["campaign", "campaign_type"]], on="campaign",
                                            how="left").dropna(subset=["campaign_type"])
    lines = ["Typical product departments by campaign Type (across ALL past campaigns of that Type):"]
    for campaign_type, group in merged.groupby("campaign_type"):
        lines.append(f"- Type {campaign_type}: mostly " + ", ".join(group["department"].value_counts().head(6).index))
    return "\n".join(lines)


def _dataset_facts() -> str:
    """What the data is and how often mailings were actually redeemed -- real counts, not model scores."""
    d = load_dataset_facts()
    parts = ", ".join(f"Type {t}: {r} of {n} ({r / n:.1%})" for t, (r, n) in sorted(d["by_type"].items()))
    return "\n".join([
        "Dataset facts (real counts, not model scores):",
        "- Source: dunnhumby's 'The Complete Journey', a real grocery chain's loyalty-card data, anonymised: "
        "about 2,500 households and 2.6 million purchase lines over 711 anonymised days (no real dates).",
        f"- Mailing data: {d['campaigns']} coupon campaigns, {d['mailings']:,} household-campaign mailings to "
        f"{d['households_mailed']:,} distinct households. Only about 800 households have demographic details on file.",
        f"- Actually redeemed: {d['redeemed']} of {d['mailings']:,} mailings ({d['redeemed'] / d['mailings']:.1%}, "
        f"about 1 in 8). By campaign Type: {parts}.",
        "- Of mailed households that bought a campaign's products, about 88% (1,407 of 1,596) never redeemed "
        "the coupon: they bought anyway. That is the core reason uplift matters.",
    ])


def _type_a_example_facts() -> str:
    df = load_type_a_product_example()
    lines = ["Product-level illustration from campaign 8 (Type A, so only an illustration: each household's exact "
             "16 coupons are unknown). Mailed households who bought each product, and how many did so WITHOUT "
             "redeeming:"]
    lines += [f"- {r.sub_commodity_desc.title()} ({r.commodity_desc.title()}): {r.n_without_redemption} of "
              f"{r.n_treated_bought} mailed buyers never redeemed ({r.pct_without_redemption:.0f}%); unmailed "
              f"households that bought it: {r.n_control_bought}" for r in df.itertuples()]
    return "\n".join(lines)


def _per_campaign_facts() -> str:
    """Average predicted score per campaign for each tool (model in use), for "which campaign" questions."""
    desc = load_campaign_desc().set_index("campaign")
    cols = {label: load_predictions(model).groupby("campaign")["score"].mean()
            for model, label in ((1, "redemption-likelihood"), (2, "forecast spend $"), (3, "uplift"))}
    lines = ["Average predicted score per modelled campaign (predicted, not actual results; a campaign appears "
             "only for the tools that scored it):"]
    for c in sorted(set().union(*[set(v.index) for v in cols.values()])):
        vals = ", ".join(f"{k} {v[c]:.3f}" for k, v in cols.items() if c in v.index)
        lines.append(f"- campaign {c} (Type {desc.loc[c, 'campaign_type']}, {desc.loc[c, 'duration_days']} days): {vals}")
    return "\n".join(lines)


def _campaign_type_facts(model: int, score_label: str, fmt: str = "{:.4f}") -> str:
    """Past performance by Type -- the analogy for "a new Type X campaign is coming"."""
    lines = [f"{MODEL_NAMES[model]} ({score_label}) by campaign Type, for comparing a new campaign to similar past ones:"]
    for campaign_type, group in load_predictions(model).groupby("campaign_type"):
        campaigns = sorted(group["campaign"].unique().tolist())
        lines.append(f"- Type {campaign_type} ({len(campaigns)} past campaigns: {campaigns}): "
                     f"average {score_label} = {fmt.format(group['score'].mean())}")
    return "\n".join(lines)


def build_background() -> str:
    summary = load_run_summary()
    sure = load_sure_things()
    return "\n".join([
        BACKGROUND_RULES, "", "--- Background facts you may quote ---", "",
        _glossary_facts(), "",
        _model_facts(summary, 1), _campaign_type_facts(1, "redemption-likelihood score"), "",
        _model_facts(summary, 2), _campaign_type_facts(2, "forecast spend", fmt="${:.2f}"), "",
        _model_facts(summary, 3), _campaign_type_facts(3, "uplift score"), "",
        f"The sure-things classifier -- a separate tool alongside {MODEL_NAMES[3]} that flags likely-wasted sends "
        "among already-mailed households:",
        f"AUC {sure['auc']:.3f}; its most confident fifth is right {sure['top_fifth_rate']:.1%} of the time, versus "
        f"a {sure['base_rate']:.1%} base rate; trained on {sure['n_train']:,}, tested on {sure['n_test']:,} households.",
        "What it relies on most: " + ", ".join(f"{k} ({v:.0%})" for k, v in sure["top_features"].items()), "",
        _dataset_facts(), "", _campaign_calendar_facts(), "", _per_campaign_facts(), "", _type_product_facts(), "",
        _wasted_products_facts(), "", _type_a_example_facts(),
    ])


_STOP = {"what", "which", "products", "product", "about", "does", "that", "with", "have", "this", "from",
         "buy", "bought", "without", "coupon", "coupons", "ever", "tell", "show", "campaign", "redeem",
         "redeemed", "redemption", "many", "most", "worst", "best", "send", "mail", "their", "there"}


def _product_lookup(question: str) -> str:
    """If the question names a product (e.g. "bananas"), the matching rows from the team's product analysis
    and the campaign 8 illustration -- only real rows from files already computed."""
    words = {w.rstrip("s") for w in re.findall(r"[a-z]{4,}", question.lower()) if w not in _STOP}
    if not words:
        return ""
    lines = []
    df = load_product_redemption_summary()
    text = (df["commodity_desc"].fillna("") + " " + df["sub_commodity_desc"].fillna("")).str.lower()
    hit = df[text.apply(lambda t: any(w in t for w in words))]
    if not hit.empty:
        lines.append("Product matches in the team's product analysis (targeted Type B/C campaigns):")
        lines += [f"- {r.commodity_desc} / {r.sub_commodity_desc} ({r.brand}): {r.n_without_redemption} of "
                  f"{r.n_treated_bought} mailed buyers did not redeem ({r.pct_without_redemption:.0f}%), "
                  f"across {r.n_campaigns} campaign(s)" for r in hit.nlargest(8, "n_treated_bought").itertuples()]
    ex = load_type_a_product_example()
    exhit = ex[(ex["commodity_desc"] + " " + ex["sub_commodity_desc"]).str.lower().apply(lambda t: any(w in t for w in words))]
    if not exhit.empty:
        lines.append("Matches in the campaign 8 (Type A) illustration: " + "; ".join(
            f"{r.sub_commodity_desc.title()}: {r.n_without_redemption} of {r.n_treated_bought} mailed buyers never "
            f"redeemed" for r in exhit.itertuples()))
    return "\n".join(lines)


def _campaign_product_lookup(question: str) -> str:
    """If the question names a campaign, the product categories its coupons covered -- summarised by department
    for a broad (Type A) campaign instead of listing 200+ categories."""
    match = CAMPAIGN_RE.search(question)
    if not match:
        return ""
    key = int(match.group(1))
    rows = load_campaign_products()
    rows = rows[rows["campaign"] == key]
    if rows.empty:
        return f"No product-coupon record found for campaign {key}."
    counts = rows["commodity"].value_counts()
    if len(counts) <= 25:
        return (f"Real product categories campaign {key}'s coupons covered ({len(counts)} categories): "
                + ", ".join(sorted(counts.index)))
    return (f"Campaign {key}'s coupons covered {len(counts)} product categories -- broad/storewide (the Type A "
            "pattern: only a pool of possible coupons is recorded, not which 16 any household got). By department: "
            + ", ".join(f"{d} ({n})" for d, n in rows["department"].value_counts().head(8).items())
            + ". Top individual categories by coupon volume: "
            + ", ".join(f"{name} ({n})" for name, n in counts.head(15).items()))


@st.cache_data(ttl=60)
def cached_background() -> str:
    return build_background()


@st.cache_resource(ttl=600)
def lab() -> Lab:
    return Lab(T.build(PROCESSED), T.campaign_products(PROCESSED), T.category_redemptions(PROCESSED),
               T.cautions(PROCESSED))


def background_for(question: str) -> str:
    """Everything Gemini may quote for this question: the standing background plus any campaign or product the
    question names. The live check (scripts/eval_decision_lab.py) uses this too, so it tests what the app runs."""
    return "\n\n".join(p for p in (cached_background(), _campaign_product_lookup(question), _product_lookup(question)) if p)


def ask(history: list[dict], question: str) -> Turn:
    from google import genai

    client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
    turn = run_turn(client, lab(), history, question, background_for(question))
    # One line per question in the Cloud Run logs, so cost per question is measured, not guessed.
    print(f"[decision-lab] tokens in={turn.tokens['input']} out={turn.tokens['output']} calls={len(turn.calls)}",
          flush=True)
    return turn
