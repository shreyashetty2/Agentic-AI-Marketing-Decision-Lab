"""Ask the Decision Lab: a stakeholder-facing Q&A over Model 1 and Model 3, grounded only in numbers
this repo already computed. Same rule the lead-agent skill follows for a data scientist, applied here
for a marketer: the model paraphrases and reasons about given facts, it never computes a new one."""
import os
from pathlib import Path

import streamlit as st

from lib.data import ROOT, load_predictions, load_run_summary, load_sure_things, step, with_demographics

CHAT_MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """You are the Decision Lab, answering a marketing stakeholder's questions about two models \
built on grocery-chain loyalty data: Model 1 (who is likely to redeem a mailed coupon) and Model 3 \
(who mailing actually persuades to buy more -- "uplift" -- plus a separate tool that flags likely-wasted \
sends among households already mailed, called "sure things").

Rules, no exceptions:
- Answer ONLY from the facts given to you below. Never compute, estimate, or guess a number that is not \
already there. If the answer needs a number nobody has computed, say plainly that it is not available yet \
and name what's closest (e.g. "I don't have that broken out by campaign, but across all campaigns the uplift \
score is...").
- Plain language for a marketer, not a data scientist: define a term the first time you use it (e.g. "uplift \
score, meaning how many extra buyers this household contributes if mailed, versus not mailing them").
- Never say a model "works" or "is better" more strongly than the facts support. If a result is "could be \
luck" or the verdict is "human needed", say that plainly rather than rounding it up to a confident claim.
- Short, direct answers. A sentence or two for a simple question; a short list only when the question asks \
for several things (e.g. "which segments").
- Model 2 (spend forecasting) is not in your scope -- if asked, say so and point to the Decision Lab's \
Overview page instead of guessing at its numbers.
"""


def api_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key or None


def _model_facts(summary: dict, model: int) -> str:
    modeling, evaluation = step(summary, "modeling", model), step(summary, "evaluation", model)
    facts = evaluation["facts"]
    lines = [
        f"Model {model} -- the model in use: {modeling['current_model']}",
        f"Trained on {modeling['trained_on']['campaigns']} finished campaigns ({modeling['trained_on']['rows']:,} rows).",
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


def _population_facts(model: int, score_label: str) -> str:
    preds = with_demographics(load_predictions(model))
    lines = [f"Population breakdown for Model {model} ({score_label}), households with a demographic record on file:"]
    for col in ("classification_1", "classification_4", "homeowner_desc", "kid_category_desc"):
        if col not in preds.columns:
            continue
        known = preds[preds[col].notna() & (preds[col] != "None/Unknown")]
        if known.empty:
            continue
        means = known.groupby(col)["score"].mean().sort_values(ascending=False)
        lines.append(f"- by {col}: " + ", ".join(f"{k} = {v:.4f}" for k, v in means.items()))
    top = preds.nlargest(10, "score")[["household_key", "campaign", "score"]]
    lines.append("Top 10 households by score: " + "; ".join(
        f"household {r.household_key} (campaign {r.campaign}): {r.score:.4f}" for r in top.itertuples()))
    return "\n".join(lines)


def build_context() -> str:
    summary = load_run_summary()
    sure = load_sure_things()
    parts = [
        _model_facts(summary, 1),
        _population_facts(1, "redemption-likelihood score"),
        "",
        _model_facts(summary, 3),
        _population_facts(3, "uplift score"),
        "",
        "Model 3's sure-things classifier (flags likely-wasted sends among already-mailed households):",
        f"AUC {sure['auc']:.3f}; its most confident fifth is right {sure['top_fifth_rate']:.1%} of the time, "
        f"versus a {sure['base_rate']:.1%} base rate; trained on {sure['n_train']:,}, tested on {sure['n_test']:,} households.",
        "What it relies on most: " + ", ".join(f"{k} ({v:.0%})" for k, v in sure["top_features"].items()),
    ]
    return "\n".join(parts)


@st.cache_data(ttl=60)
def cached_context() -> str:
    return build_context()


def ask(history: list[dict], question: str) -> str:
    """One call: the full grounding context plus the chat history plus the new question. No tool use,
    no retries -- if the model can't answer from the context, it says so, per the system prompt."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key())
    context = cached_context()
    messages = [*history, {"role": "user", "content": question}]
    msg = client.messages.create(
        model=CHAT_MODEL, max_tokens=700,
        system=f"{SYSTEM_PROMPT}\n\n--- Facts you may use ---\n{context}",
        messages=messages)
    return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
