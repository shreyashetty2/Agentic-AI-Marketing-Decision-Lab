"""Ask the Decision Lab: a stakeholder-facing Q&A over all three models, grounded only in numbers
this repo already computed. Same rule the lead-agent skill follows for a data scientist, applied here
for a marketer: the model paraphrases and reasons about given facts, it never computes a new one.

Runs on Vertex AI (Gemini), not a third-party API key -- billed to this project's own GCP credits, per
docs/agent_design/gcp_deployment.md's plan for the real front door. Needs `gcloud auth application-default
login` once (ADC), the same login already used for this project's Cloud Run deployment.
"""
import os
import re

import streamlit as st

from lib.data import (MODEL_NAMES, load_campaign_desc, load_campaign_products, load_model_features,
                       load_predictions, load_product_redemption_summary, load_run_summary,
                       load_sure_things, step, with_demographics)

HOUSEHOLD_RE = re.compile(r"household\D{0,6}(\d+)", re.IGNORECASE)
CAMPAIGN_RE = re.compile(r"campaign\D{0,6}(\d+)", re.IGNORECASE)

GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "capstone-decision-lab")
GCP_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
CHAT_MODEL = "gemini-3.8-flash"

SYSTEM_PROMPT = """You are the Decision Lab, answering a marketing stakeholder's questions about three tools \
built on grocery-chain loyalty data: Redemption Likelihood (who is likely to redeem a mailed coupon), Spend \
Forecast (how much a mailed household is forecast to spend on the campaign's products), and Campaign Uplift \
(who mailing actually persuades to buy more -- "uplift" -- plus a separate tool that flags likely-wasted \
sends among households already mailed, called "sure things").

Rules, no exceptions:
- Never say "Model 1", "Model 2", "Model 3", or any other internal model/tool number -- the stakeholder you're \
talking to doesn't know or care about that numbering. Always refer to the tools by name: Redemption Likelihood, \
Spend Forecast, Campaign Uplift, sure things.
- Answer ONLY from the facts given to you below. Never compute, estimate, or guess a number that is not \
already there. If the answer needs a number nobody has computed, say plainly that it is not available yet \
and name what's closest (e.g. "I don't have that broken out by campaign, but across all campaigns the uplift \
score is...").
- Plain language for a marketer, not a data scientist: define a term the first time you use it (e.g. "uplift \
score, meaning how many extra buyers this household contributes if mailed, versus not mailing them").
- Never say a model "works" or "is better" more strongly than the facts support. If a result is "could be \
luck" or the verdict is "human needed", say that plainly rather than rounding it up to a confident claim.
- This dataset has NO real calendar information anywhere -- no month, season, or actual date, only an \
anonymized sequential day-count and each campaign's Type (A, B, or C). If asked what month, season, or \
"time of year" is best to mail, say plainly that this data has no real calendar dates to answer that from \
-- never guess a season. You CAN answer from campaign duration (in days) and campaign Type.
- If asked about a new or upcoming campaign that isn't in the historical list (e.g. "we're planning a new \
Type A campaign, who should we target"), you cannot score it directly -- nobody has run the models on it. \
Answer by analogy: use the historical performance of past campaigns of the *same Type* as the closest \
available guide, and say explicitly that this is an analogy to similar past campaigns, not a new prediction.
- You're talking to someone who may have zero background in this project -- explain plainly what things \
mean using the glossary given to you below (age group, household size, campaign, uplift, etc.), including \
being upfront about what is genuinely NOT known: some demographic labels in this dataset (see the glossary) \
were anonymised at the source with no documented real-world meaning -- if asked what one of those actually \
represents (e.g. income, education), say plainly that this anonymised dataset never revealed that, and do \
not guess what it might be.
- When discussing "the model" behind any tool, be specific, never generic: name the actual algorithm (e.g. \
"random forest", "x_learner") AND, if asked what it's built from, list the literal input columns given to \
you below -- never just say "a model" or "some data" without naming what's actually there.
- If asked what products a specific campaign covers, use the real product categories given to you below for \
that campaign -- never guess at what a campaign might be selling.
- Short, direct answers. A sentence or two for a simple question; a short list only when the question asks \
for several things (e.g. "which segments").
"""


def credentials_ok() -> bool:
    """Whether Application Default Credentials are available at all -- cheap, makes no API call."""
    try:
        import google.auth
        google.auth.default()
        return True
    except Exception:
        return False


def _glossary_facts() -> str:
    """Plain-language definitions for someone with zero background, and an honest line on what this
    anonymised dataset genuinely does not reveal -- checked directly against src/features/feature_2a_spec.yaml
    and the raw hh_demographic file: classification_2 is literally "X"/"Y"/"Z", classification_3 is
    "Level1".."Level12", classification_5 is "Group1".."Group6" in the source data itself, with no
    documented real-world meaning anywhere in this project. Never guess what those represent (e.g. income,
    education) -- they are not used anywhere in this app precisely because nobody knows what they mean."""
    return "\n".join([
        "Glossary (plain-language terms a newcomer would need):",
        "- Campaign: a single coupon-mailing promotion, covering specific real products, run over some "
        "number of days (see the campaign calendar below). Each campaign has a Type (A, B, or C) -- the "
        "dataset's own grouping, no finer description than that exists.",
        "- Redemption: a household used the mailed coupon on a real purchase.",
        "- Uplift: how many EXTRA buyers a household contributes if mailed, compared to not mailing them -- "
        "not just whether they buy, but whether mailing is what caused it.",
        "- Sure things: households who would likely have bought anyway, coupon or not -- mailing them is a "
        "likely-wasted send.",
        "- Age Group (1 through 6): a real age grouping, ordered, but the exact age ranges and which end is "
        "older were stripped from this anonymised dataset -- not knowable from what's here.",
        "- Household size (1, 2, 3, 4, or 5+): the number of people in the household.",
        "- Homeowner status / kids in the household: plain, not anonymised -- exactly what they say.",
        "- Other demographic labels this dataset carries (not used anywhere in this app): some are "
        "anonymised with NO documented real-world meaning at all -- not income, not education, nothing "
        "knowable. If asked what one of those might represent, say plainly it's not known, never guess.",
    ])


def _wasted_products_facts() -> str:
    """Real products (not a model output, the team's own product-level analysis), for "what products
    are the worst sends" style questions -- the products most often mailed to a household that never
    redeemed, among products seen in at least 2 campaigns (1-campaign rows are too thin to generalise)."""
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
    facts = evaluation["facts"]
    lines = [
        f"{name} -- the model in use: {modeling['current_model']}",
        f"Trained on {modeling['trained_on']['campaigns']} finished campaigns ({modeling['trained_on']['rows']:,} rows).",
        "Built from these exact input columns (not a generic count): " + ", ".join(load_model_features(model)),
        "Relies most on: " + "; ".join(f"{c['meaning']} ({c['share']:.0%})" for c in modeling.get("leans_on", [])),
        f"Score: {facts['current_score']}",
        f"Versus the simple rule: {facts['vs_simple_rule']}",
        f"Forecast check: {facts['forecast_check']}",
        f"Evaluation verdict: {evaluation['verdict']}" + (f" -- {evaluation['handled']}" if evaluation.get("handled") else ""),
    ]
    in_use = modeling.get("in_use")
    if isinstance(in_use, dict):
        lines.append(f"In use since {in_use['decided_on']}, chosen by {in_use['decided_by']}: {in_use.get('why', '')}")
    open_qs = [q["text"] for q in summary["questions"]["still_open"] + summary["questions"]["new"]
               if q["model"] == model]
    if open_qs:
        lines.append("Open questions nobody has answered yet: " + " | ".join(open_qs))
    return "\n".join(lines)


def _population_facts(model: int, score_label: str, fmt: str = "{:.4f}") -> str:
    preds = with_demographics(load_predictions(model))
    name = MODEL_NAMES[model]
    lines = [f"Population breakdown for {name} ({score_label}), households with a demographic record on file:"]
    for col in ("classification_1", "classification_4", "homeowner_desc", "kid_category_desc"):
        if col not in preds.columns:
            continue
        known = preds[preds[col].notna() & (preds[col] != "None/Unknown")]
        if known.empty:
            continue
        means = known.groupby(col)["score"].mean().sort_values(ascending=False)
        lines.append(f"- by {col}: " + ", ".join(f"{k} = {fmt.format(v)}" for k, v in means.items()))
    top = preds.nlargest(10, "score")[["household_key", "campaign", "score"]]
    lines.append("Top 10 households by score: " + "; ".join(
        f"household {r.household_key} (campaign {r.campaign}): {fmt.format(r.score)}" for r in top.itertuples()))
    return "\n".join(lines)


def _campaign_calendar_facts() -> str:
    """Every campaign's real type (A/B/C) and real duration in days -- raw data, not a model output, so
    this covers all 30 campaigns whether or not they were ever modeled. No real calendar dates exist
    anywhere in this dataset (checked both campaign_desc and transaction_data), so this is the only
    honest answer to "when"/"how long" questions."""
    desc = load_campaign_desc().sort_values("campaign")
    lines = ["Campaign calendar (real data, every campaign, day-numbers are anonymised/sequential -- "
             "NOT real calendar dates, there is no month or season anywhere in this dataset):"]
    lines += [f"- campaign {r.campaign}: Type {r.campaign_type}, runs {r.duration_days} days"
              for r in desc.itertuples()]
    return "\n".join(lines)


def _campaign_type_facts(model: int, score_label: str, fmt: str = "{:.4f}") -> str:
    """Historical performance grouped by campaign Type (A/B/C) -- the closest honest answer to "a new
    campaign of type X is coming, who should we target": an analogy to past campaigns of that same
    type, built only from scores the models already computed, never a new prediction."""
    preds = load_predictions(model)
    name = MODEL_NAMES[model]
    lines = [f"{name} ({score_label}) by campaign Type, for comparing a new campaign to similar past ones:"]
    for campaign_type, group in preds.groupby("campaign_type"):
        campaigns = sorted(group["campaign"].unique().tolist())
        lines.append(f"- Type {campaign_type} ({len(campaigns)} past campaigns: {campaigns}): "
                      f"average {score_label} = {fmt.format(group['score'].mean())}")
    return "\n".join(lines)


def build_context() -> str:
    summary = load_run_summary()
    sure = load_sure_things()
    parts = [
        _glossary_facts(),
        "",
        _model_facts(summary, 1),
        _population_facts(1, "redemption-likelihood score"),
        _campaign_type_facts(1, "redemption-likelihood score"),
        "",
        _model_facts(summary, 2),
        _population_facts(2, "forecast spend", fmt="${:.2f}"),
        _campaign_type_facts(2, "forecast spend", fmt="${:.2f}"),
        "",
        _model_facts(summary, 3),
        _population_facts(3, "uplift score"),
        _campaign_type_facts(3, "uplift score"),
        "",
        f"The sure-things classifier -- a separate tool alongside {MODEL_NAMES[3]} that flags likely-wasted "
        "sends among already-mailed households:",
        f"AUC {sure['auc']:.3f}; its most confident fifth is right {sure['top_fifth_rate']:.1%} of the time, "
        f"versus a {sure['base_rate']:.1%} base rate; trained on {sure['n_train']:,}, tested on {sure['n_test']:,} households.",
        "What it relies on most: " + ", ".join(f"{k} ({v:.0%})" for k, v in sure["top_features"].items()),
        "",
        _campaign_calendar_facts(),
        "",
        _wasted_products_facts(),
    ]
    return "\n".join(parts)


@st.cache_data(ttl=60)
def cached_context() -> str:
    return build_context()


def _household_lookup(question: str) -> str:
    """If the question names a specific household (e.g. "household 323"), pull its actual rows from
    all three models' predictions -- the top-10 lists in build_context() won't cover most IDs, so
    without this the model would have to say "not available" for almost any specific household asked
    about. Still grounded: only real rows, nothing computed here."""
    match = HOUSEHOLD_RE.search(question)
    if not match:
        return ""
    key = int(match.group(1))
    lines = [f"Specific household requested in this message: household {key}."]
    found = False
    for model, label, fmt in ((1, "redemption-likelihood score", "{:.4f}"),
                              (2, "forecast spend", "${:.2f}"), (3, "uplift score", "{:.4f}")):
        rows = load_predictions(model)
        rows = rows[rows["household_key"] == key]
        if rows.empty:
            continue
        found = True
        lines.append(f"{MODEL_NAMES[model]} ({label}): " + "; ".join(
            f"campaign {r.campaign}: {fmt.format(r.score)}" for r in rows.itertuples()))
    if not found:
        lines.append("No record of this household in any model's mailed households -- say so plainly, "
                      "don't guess at a score.")
    return "\n".join(lines)


def _campaign_product_lookup(question: str) -> str:
    """If the question names a specific campaign (e.g. "campaign 17"), pull the real product categories
    its coupons actually covered -- the campaign calendar in build_context() has every campaign's type
    and duration, but listing every campaign's full product list there would be huge, so this is a
    lookup instead, same pattern as _household_lookup."""
    match = CAMPAIGN_RE.search(question)
    if not match:
        return ""
    key = int(match.group(1))
    products = load_campaign_products()
    rows = products[products["campaign"] == key]
    if rows.empty:
        return f"No product-coupon record found for campaign {key}."
    counts = rows["commodity"].value_counts()
    if len(counts) <= 25:
        return (f"Real product categories campaign {key}'s coupons actually covered "
                f"({len(counts)} categories): " + ", ".join(sorted(counts.index)))
    top = counts.head(15)
    return (f"Campaign {key}'s coupons covered {len(counts)} product categories -- broad/storewide, "
            f"too many to list individually. By far its top categories (by coupon volume): "
            + ", ".join(f"{name} ({n})" for name, n in top.items()))


def ask(history: list[dict], question: str) -> tuple[str, str]:
    """One call: the full grounding context as the system instruction, the chat history, the new
    question. No tool use, no retries -- if the model can't answer from the context, it says so, per
    the system prompt (verified directly: asked a fact-check question outside given context, it
    refused rather than guessing). Returns (answer, the exact context block used) so the page can show
    it for transparency.

    thinking_level="low": verified this still holds the grounding rule (refuses what it's not given)
    while cutting a ~3x latency difference from the model's default extended thinking."""
    from google import genai
    from google.genai import types

    client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
    context = cached_context()
    household = _household_lookup(question)
    if household:
        context = f"{context}\n\n{household}"
    products = _campaign_product_lookup(question)
    if products:
        context = f"{context}\n\n{products}"
    contents = [types.Content(role="model" if h["role"] == "assistant" else "user",
                              parts=[types.Part(text=h["content"])]) for h in history]
    contents.append(types.Content(role="user", parts=[types.Part(text=question)]))
    resp = client.models.generate_content(
        model=CHAT_MODEL, contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=f"{SYSTEM_PROMPT}\n\n--- Facts you may use ---\n{context}",
            thinking_config=types.ThinkingConfig(thinking_level="low")))
    return resp.text, context
