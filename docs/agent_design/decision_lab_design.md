# Decision Lab: design

## Why this exists

A marketer today has to take every targeting question to a data scientist, wait for an answer, and then decode it from
engineer language. The Decision Lab should let them ask in their own words and get a correct, plain answer in about
a minute — and, when it can't answer, say so and hand the question to a data scientist already written up.

The first version (a page per model plus a chat page) proved two things. The chat stays honest: asked about a campaign
that doesn't exist, it refused instead of guessing. But it can't work anything out. It reads one fixed summary page
(each model's accuracy, average scores for four demographic columns, the top 10 households), so at the 11/13 mentor
meeting nearly every question failed — "best 1,000 households for campaign 5", "a file of who to mail", "which
products for households of 5 or more". It isn't missing polish; it's missing the ability to calculate.

## How a question is answered

Every question goes through the same four steps.

1. **Understand.** Gemini turns the sentence into a short form: *who* we're looking at, *what* we measure, *how* to
   arrange it, and the *output*. This is the only step where the LLM decides anything.
2. **Check.** Code checks the form: does the campaign exist, is the trait real, is anything required missing. If
   something is missing it asks; if nothing on the menu fits, it says what it can't answer and offers the nearest
   question it can.
3. **Calculate.** Tested code runs the pieces on the data. The LLM does no arithmetic.
4. **Explain.** Gemini turns the result into plain English, listing the steps it took and any assumption it made
   ("you didn't name a campaign, so I used 8").

Example: *"Best 200 households of 3 or more for campaign 8?"* becomes
`who: campaign 8, household size 3+ · what: redemption chance · how: top 200 · output: answer`. "Top 3 instead"
changes one field. A question nobody planned for works as long as the menu can describe it.

Questions that need explaining rather than calculating -- what a Type A campaign is, what share of coupons get
redeemed, why a tool's verdict is "human needed" -- are answered from background facts the team already computed
(`app/lib/chat.py`: the glossary, each tool's verdict and inputs, real redemption counts, product-level results).
Gemini may quote those; anything about particular households, groups or campaigns goes through the form.

Gemini calls these as functions (tool calling), which the `google-genai` library the chat already uses supports
(`types.Tool`, `types.FunctionDeclaration`); it is a change to `app/lib/chat.py`, not a new service.

## The menu

The questions from the mentor meeting are examples and tests, not the spec. The spec is this menu; anything that can
be described with it is answerable.

- **Who:** a campaign (or all); household traits (homeowner, kids, the five coded demographics); a product
  category; one specific household; two groups to compare.
- **What:** redemption chance (Model 1), expected spend (Model 2), expected revenue (redemption chance × expected
  spend), uplift and "would buy anyway" (Model 3), past redemptions.
- **How:** one number; top N or top %; group and compare; breakdown by segment.
- **Output:** a written answer, or a downloadable list.
- **Describe:** what a campaign promoted, its type and length; what a code means.

How the mentors' questions map onto it:

- "Mail 1,000: which segments give the best uplift?" → who: a campaign · what: uplift · how: top 1,000, then by segment.
- "Response rate if I target households of 3 or more?" → the data has no household size (see below), so it says
  so and offers what exists: who: homeowners, or kids 3+ · what: redemption chance · how: one number.
- "A file of who to mail for campaign 5" → who: campaign 5 · what: redemption chance · how: top N · output: list.
- "Households with 3 or more kids: which products?" → who: kids 3+ · what: past redemptions · how: group by category.
- "What is age group 1?" / "What did campaign 16 promote?" → describe.
- "Best time of year?" / "How often to mail?" → no piece fits: say so, offer the nearest answerable question.

## Rules the code enforces

**Scope.** In scope: these campaigns, households and products, and what the three models predict about them. Out of
scope comes in three kinds, each with its own reply:

- Off-topic (the weather): "That's outside what the Decision Lab covers."
- On-topic but not in our data (timing, frequency, discount size): say so, offer the closest answerable question.
- A data scientist's job (score new households, retrain): write the question up for a data scientist.

If the form can't be filled from the menu, no number is produced.

**Complexity.** One question may cover one campaign (or all), up to two groups, one measure and one "how". Bigger
questions are split: answer the first part, offer the next. Every answer stays checkable and costs two short LLM calls.

**Clarifying questions.** Ask only when the answer would really change; otherwise use a default and say so.

- Must ask: a required field is missing (top N with no campaign — scores differ a lot between campaigns); a word maps
  to more than one measure ("best" could mean likely to redeem, biggest spend, or persuaded by the coupon); the group
  is too small to trust (only a dozen households match).
- Default and say so: no number given → top 10; no period given → all test campaigns.

## The data underneath: one clean table

Every menu piece reads one table: one row per household per campaign, with all three scores from the model actually
in use, the household's traits, and the campaign's details. Building it is the first job, and the checks below
(11/19, on the local copy of `data/processed/`) show why.

- **The first version showed a mix of models, on every page and in the chat.** `load_predictions()` in
  `app/lib/data.py` kept each campaign's `chosen` candidate — the agent's per-campaign experimental pick — instead
  of the model in use. It now reads only the rows marked `is_default`, through the same function the table uses. On
  Model 1 that is seven different models across 13 campaigns (`extended_plus_logistic_log` on 1,897 rows,
  `extended_plus_forest` 1,077, `coupon_offer_fit_forest` 1,076, …); the model in use, `owner_forest`, covers 176.
  Models 2 and 3 mix two and three. The model in use is recorded in `src/lead/models_in_use.yaml` and each
  `manifest.json`: Model 1 `owner_forest` (team's own), Model 2 `usual_only_linear`, Model 3 `owner_x_learner`.
- **Model 2 only scores households that redeemed.** Its predictions hold 113 household-campaign rows (spend if they
  redeem). Expected revenue needs a spend forecast for every mailed household, so the table applies Model 2's
  saved model in use ($9.68 + 0.70 × usual spend) to all of them. Two limits: it is left blank in Type A campaigns
  (8, 13, 18), which Model 2 was never built for and where the line stretches to ~$750; and the saved model was fit
  on every Model 2 campaign, so in the test campaigns it has seen a little of what it forecasts. For a two-number
  line this barely moves the ranking, but the expected-revenue figures are a shade optimistic.
- **The three models cover different households and campaigns.** Model 1: mailed households, campaigns 8–20.
  Model 2: redeemers, campaigns 14–22. Model 3: all 2,500 households, mailed or not, campaigns 9–21 except 13 and 18.
  Each also holds back a different set of campaigns (Model 1: 21–25, Model 2: 23–25, Model 3: 22–25). The table has to
  say, per row, which scores exist, and the Lab has to say when a model doesn't cover a campaign.
- **Product questions are buildable but not built.** `coupon.parquet` links each campaign's coupons to products,
  `coupon_redempt.parquet` records who redeemed which coupon, and `product.parquet` gives department and category. No
  processed table yet joins them into "past redemptions by household and category".
- **The local data may be out of date.** The local copy was last synced 11/05, and its Model 3 predictions don't
  contain `owner_x_learner`, though it became the model in use on 10/21. The hosted app reads the cloud bucket; that
  copy needs checking before the table is built.
- **Most demographics are coded, on purpose.** The dataset's own guide says the five classification fields were
  given generic names and values. Only homeowner status and kids are readable; classification_4 (1 to 5+) may be
  household size, but nothing confirms it, so the Lab never claims it. Only 801 of 2,500 households have a record.
  Asked about household size, age ranges or income, the Lab says the data doesn't have them and offers what exists.

## How we know it works

A question set with known correct answers, in three kinds: answerable questions (correct number), out-of-scope
questions (correct refusal and suggestion), and unclear questions (the correct result is a clarifying question).

- **Design set:** the mentor-meeting questions, used to shape the menu.
- **Team set:** the team's own sample questions, each checked for the point its answer must make (and, where it
  matters, a claim it must never make — such as calling the coded 1-to-5+ field household size).
- **Held-back set:** written after the menu is fixed — ideally real questions from TD's marketers — and run once. If it
  passes at about the design set's rate, the design generalizes; if not, the failures name the missing piece.

Each question's cost (tokens used) is logged from day one, so the cost per question is measured, not guessed.

## Build order

1. Build the clean table, using the model in use for each model; fixes the mixed-model pages on the way.
2. Build the menu pieces and the check step, each verified against an independent recount.
3. Switch the chat from the summary page to form-filling, with the three rules.
4. Plain language: campaign descriptions, product names, readable numbers.
5. Downloadable lists and the data-scientist hand-off.
6. Run the question set and log cost.
7. Fold the pages into one interface with the chat at the centre.

Later, not now: timing and frequency (needs date features no model uses) and scoring brand-new households (a data
scientist's job, and the lead agent's).

## Where it lives

- `src/decision/table.py` — builds the clean table from the agents' outputs, and refuses outputs that don't fit
  together (held-back rows, two models marked in use, a household scored twice).
- `src/decision/menu.py` — the menu, the three rules and every calculation.
- `src/decision/assistant.py` — one chat turn: Gemini fills the form as a function call, the menu answers, Gemini
  explains. `app/lib/chat.py` and `app/pages/4_Ask_the_Decision_Lab.py` put it on the page, with the steps under
  "How this was worked out" and a download button for any list.
- `tests/decision/` — the calculations against hand-worked numbers, the table against the agents' own files (it
  reproduces the Week 10 count: mailing the top half reaches 594 of 707 redeemers), and the chat loop with a fake
  Gemini, so the suite costs nothing.
- `tests/decision/questions.yaml` and `scripts/eval_decision_lab.py` — the question set (the mentor meeting, a
  held-back set, and the team's sample questions from `docs/shreya/decision_lab_sample_questions.md`), scored
  against live Gemini with the background the app uses and the tokens each question used.

## Open decisions

Each has a default in place that is easy to reverse.

- Held-back campaigns stay hidden: the table never reads `locked_predictions.parquet`, so they remain a fair test.
- The table reads whichever model each agent marks in use; switching Model 1 or Model 3 needs no Lab change.
- The local mailing tool (`src/scoring/mailing.py`) isn't used; the "top" piece does the same count on the new table.
