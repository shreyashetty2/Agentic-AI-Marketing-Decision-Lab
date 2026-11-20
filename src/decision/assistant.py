"""The Decision Lab's chat turn: Gemini fills in the menu's form (decision/menu.py) as a function call, our code
checks and runs it, and Gemini explains the result. Gemini never sees the data and never does arithmetic; every
number in an answer comes back from a function call.

`run_turn` takes the client as an argument, so tests drive it with a fake one and no API call is made.
"""
import json
from dataclasses import dataclass, field

from decision.menu import HOWS, MEASURES, TRAITS, Lab, Result

MODEL = "gemini-3.8-flash"
MAX_CALLS = 4          # function calls per question; more means it's trying to do too much in one go

SYSTEM_PROMPT = """You are the Decision Lab. Marketers at a grocery chain ask you who to target with coupon \
mailers. You answer from three tools built on the chain's loyalty data: Redemption Likelihood (chance a mailed \
household uses the coupon), Spend Forecast (what a mailed household spends on the campaign's products if it \
redeems), and Campaign Uplift (how much a mailer changes a household's buying, and whether it would buy anyway).

How you work:
- Every number you give must come from a function result in this conversation or from the background facts \
below. Never compute, estimate or round a number into a new one yourself. For anything about particular \
households, groups or campaigns that the background doesn't state outright, call `answer`. If neither has it, say \
it isn't available.
- Fill in the form only with what the marketer said. Leave a field empty rather than guess it -- the code \
decides whether to ask or use a default. If they don't say which measure ("best", "top", "worth targeting"), \
leave `measure` empty.
- When a function returns status "ask", ask the marketer exactly that question, in plain words. When it \
returns "cannot", explain why and offer the closest question you can answer.
- One measure per call, at most two groups compared. If a question needs more, answer the first part and offer \
the next part.
- The data has no household size, age ranges or income in readable form -- only homeowner status, kids, and \
five coded fields whose meanings the dataset hides. If asked about those, say so and offer what exists.
- Out of scope: if it has nothing to do with these campaigns, households or products (the weather, other \
companies), say the Decision Lab doesn't cover it. If it needs something no tool has (timing, how often to \
mail, discount size), say so and suggest the nearest question you can answer. If it's a data scientist's job \
(scoring new households, retraining a model), say so and offer a short write-up of the question to send them.

How you answer:
- Plain language for someone with no data background. Say "about 1 in 5 will use it", not "p = 0.21". Define \
a term the first time you use it.
- Lead with the answer. Then one short line "How I worked it out:" from the function's steps, and any \
assumptions and warnings it returned, in plain words.
- Lead with what's reliable. Never open with a group the function flagged as too small, or with an "Unknown" \
group -- mention them after, with their size. If the only difference is between tiny groups, say there's no \
reliable answer yet.
- Keep forecasts and real outcomes apart: "expected" numbers are the tools' forecasts; "what actually \
happened" is the real campaign. Never compare one with the other.
- Don't overstate a gap. If two groups are within a couple of points of each other, say they're about the same; \
call something higher only when the numbers clearly show it.
- Never say "Model 1", "Model 2" or "Model 3"; use the tools' names.
- Short: a few sentences, or a short list when the question asks for several things."""

_FILTER = {"type": "object", "properties": {
    "trait": {"type": "string", "enum": list(TRAITS)},
    "values": {"type": "array", "items": {"type": "string"}}}, "required": ["trait", "values"]}

TOOLS = [
    {"name": "answer",
     "description": "Answer a question about campaigns and households by filling in this form. "
                    "how: summary (one set of numbers for a group), top (best N or top share of households to "
                    "mail in one campaign; the list can be downloaded), by_segment (break a measure down by one "
                    "trait), compare (two groups side by side), by_category (which product categories a group "
                    "redeems coupons on), household (everything about one household). For 'which households "
                    "for this product', use how=by_segment with measure=past_redemptions and a category.",
     "parameters": {"type": "object", "properties": {
         "how": {"type": "string", "enum": list(HOWS)},
         "measure": {"type": "string", "enum": [*MEASURES, "past_redemptions"],
                     "description": "; ".join(f"{k}: {m['plain']}" for k, m in MEASURES.items())
                                    + "; past_redemptions: households that redeemed coupons on a product category"},
         "campaign": {"type": "integer"},
         "filters": {"type": "array", "items": _FILTER},
         "top_n": {"type": "integer"},
         "top_share": {"type": "number", "description": "0.25 = the top quarter"},
         "segment_by": {"type": "string", "enum": list(TRAITS)},
         "groups": {"type": "array", "maxItems": 2, "items": {"type": "array", "items": _FILTER}},
         "category": {"type": "string"},
         "household_key": {"type": "integer"}}, "required": ["how"]}},
    {"name": "describe_campaign", "description": "What a campaign promoted, its type and length, and which tools cover it.",
     "parameters": {"type": "object", "properties": {"campaign": {"type": "integer"}}, "required": ["campaign"]}},
    {"name": "describe_code",
     "description": "What a household trait or one of its values means. Pass the trait's name ("
                    + ", ".join(TRAITS) + ") or a value exactly as the marketer said it (e.g. 'Age Group1').",
     "parameters": {"type": "object", "properties": {"trait": {"type": "string"}}, "required": ["trait"]}},
    {"name": "find_category", "description": "Find product category names matching a word, e.g. 'diaper'.",
     "parameters": {"type": "object", "properties": {"word": {"type": "string"}}, "required": ["word"]}},
]


@dataclass
class Turn:
    answer: str
    calls: list = field(default_factory=list)     # (function name, the form, Result) per call
    tokens: dict = field(default_factory=lambda: {"input": 0, "output": 0})

    @property
    def households(self):
        """The last list a 'top' answer produced, for the download button."""
        lists = [r.households for _, _, r in self.calls if r.households is not None]
        return lists[-1] if lists else None


def call(lab: Lab, name: str, args: dict) -> Result:
    if name == "answer":
        return lab.answer(args)
    if name == "describe_campaign":
        return lab.describe_campaign(args.get("campaign"))
    if name == "describe_code":
        return lab.describe_code(args.get("trait"))
    if name == "find_category":
        return lab.find_category(args.get("word", ""))
    return Result("cannot", f"There is no function called {name}.")


def run_turn(client, lab: Lab, history: list[dict], question: str, background: str = "") -> Turn:
    """background: facts and domain rules Gemini may quote when explaining (app/lib/chat.py builds them)."""
    from google.genai import types

    contents = [types.Content(role="model" if h["role"] == "assistant" else "user",
                              parts=[types.Part(text=h["content"])]) for h in history]
    contents.append(types.Content(role="user", parts=[types.Part(text=question)]))
    config = types.GenerateContentConfig(
        system_instruction=f"{SYSTEM_PROMPT}\n\n{background}" if background else SYSTEM_PROMPT,
        tools=[types.Tool(function_declarations=[
            types.FunctionDeclaration(name=t["name"], description=t["description"], parameters_json_schema=t["parameters"])
            for t in TOOLS])],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        thinking_config=types.ThinkingConfig(thinking_level="low"))
    turn = Turn(answer="")
    for _ in range(MAX_CALLS + 1):
        resp = client.models.generate_content(model=MODEL, contents=contents, config=config)
        usage = getattr(resp, "usage_metadata", None)
        if usage:
            turn.tokens["input"] += usage.prompt_token_count or 0
            turn.tokens["output"] += (usage.candidates_token_count or 0) + (getattr(usage, "thoughts_token_count", 0) or 0)
        calls = resp.function_calls or []
        if not calls:
            turn.answer = resp.text or ""
            return turn
        if len(turn.calls) + len(calls) > MAX_CALLS:
            break
        contents.append(resp.candidates[0].content)
        parts = []
        for fc in calls:
            args = dict(fc.args or {})
            result = call(lab, fc.name, args)
            turn.calls.append((fc.name, args, result))
            parts.append(types.Part.from_function_response(name=fc.name, response=json.loads(
                json.dumps(result.for_llm(), default=str))))
        contents.append(types.Content(role="user", parts=parts))
    turn.answer = ("That's more than I can work out in one go. Could you ask it in smaller parts -- for example "
                   "one campaign or one measure at a time?")
    return turn
