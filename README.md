# Agentic-AI-Marketing-Decision-Lab
Agentic AI Marketing Decision Lab capstone project with TD Bank

Columbia University · ENGI E4800 Data Science Capstone · Fall 2026

> **Status:** Draft. Structure, methods, and tooling will evolve as the project develops.

---

## Overview

Marketing teams rely on predictive models to decide who should receive a campaign, but building those models takes many steps, and their outputs are often hard for business users to act on.

This project builds an **Agentic AI Marketing Decision Lab**, an end-to-end decision-support prototype with three parts:

1. **Predict campaign response** using historical retail transaction and marketing data.
2. **Use specialized AI agents** to support the modeling lifecycle (data ingestion → feature engineering → model development → model evaluation).
3. **Deploy the model in an interactive Decision Lab**, where business users run what-if scenarios. An LLM-powered **Decision Support Agent** explains predictions, compares scenarios, and recommends actions.

## Research Questions

1. Can specialized AI agents improve the efficiency and quality of predictive model development?
2. How accurately can campaign response be predicted from historical retail and marketing data?
3. Which customer and campaign characteristics are most predictive of campaign response?
4. Can a generative-AI Decision Support Agent improve interpretation of model outputs?
5. Does an AI-assisted decision-support platform improve business users' ability to evaluate alternative marketing strategies?

---

## Data

We use **dunnhumby — The Complete Journey**: about 2 years (102 weeks) of household-level transactions for about 2,500 frequent-shopper households, plus demographics, campaigns, coupons, and promotions.

**The raw data is not stored in this repo.** Each team member must request access from [dunnhumby Source Files](https://www.dunnhumby.com/source-files/) and place the CSVs locally (see [Setup](#setup)).

| Table | Grain (one row =) | Role |
|---|---|---|
| `transaction_data` | product line on a receipt | Core purchase behavior |
| `hh_demographic` | household (subset only) | Customer attributes |
| `campaign_table` | household × campaign received | Marketing exposure |
| `campaign_desc` | campaign (30 total) | Campaign type & timing |
| `coupon` | coupon × eligible product | Campaign offer content |
| `coupon_redempt` | coupon redemption | Campaign response |
| `product` | product | Product hierarchy |
| `causal_data` | product × store × week on promo | In-store display / mailer |

**Main join keys:** `household_key`, `campaign`, `product_id`, `coupon_upc`, `(product_id, store_id, week_no)`.

### Known data caveats
- **Demographics are partial.** Demographics exist for only a subset of households (801 of 2,500, 32%), so any demographic analysis is a subsample analysis.
- **The user guide's `hh_demographic` variable table is incorrect.** Use the actual CSV headers.
- **Discount columns are stored as negative values.** `sales_value` is what the retailer receives, not what the customer paid, and already has the retailer's coupon-match discount reflected in it — subtracting `coupon_match_disc` again double-counts it. (Caught in the proposal's original net-revenue formula; Model 2 now uses `sales_value` directly.)
- **TypeA exposure is unobservable.** Each TypeA household receives only 16 coupons, personally selected from a pool of ~181–209 products (about 8% of the pool), and which 16 is not recorded. Model 2 and Model 3 scope their campaign-related/eligible-product targets to TypeB/TypeC campaigns for this reason; TypeA remains an open methodological question for both.
- **Campaigns overlap heavily and aren't in date order.** 74 of 435 campaign pairs overlap in time. Train/test splits must be built by date (`START_DAY`), not by campaign ID.
- **Campaigns are targeted, not randomized.** Response estimates are predictive, not causal. (Confirmed directly in the data for the uplift model: treated households spend ~4.6x more, pre-campaign, than control households.)
- **`day` and `week_no` are relative indices, not calendar dates.**

---

## ML Questions

### Prediction Question 1: Campaign Response Probability
For a given household level and campaign, what is the probability that the household will redeem at least one coupon from that campaign?
- Unit: one household and one campaign.
- Output: a probability from 0% to 100%.
- Example: Household 1234, TypeB campaign: 34% chance of redeeming.
- Training data: 7,208 past household-campaign mailings, of which 889 (12.33%) had at least one redemption — matches the original proposal exactly.
- Population: 2,500 households, 2 years (711 days) of history, demographics available for 801 (32%).

**Redemption rate by campaign type** (confirms why campaign type is a key feature):

| | TypeA | TypeB | TypeC |
|---|---|---|---|
| How coupons are assigned | Personalized — 16 coupons picked from a larger pool | Everyone in the campaign gets **all** of the campaign's coupons | Everyone in the campaign gets **all** of the campaign's coupons |
| Coupons per campaign | 181–209 in the pool (household sees 16, ~8%) | 2–33 | 1–34 |
| Number of campaigns | 5 | 19 | 6 |
| Typical length (days) | 41–56 (median 48) | 33–62 (median 33) | 33–162 (median 65) |
| Mailings (household × campaign) | 3,979 (55% of all) | 2,655 (37%) | 574 (8%) |
| Mailings per campaign (avg) | ~800 | ~140 | ~95 |
| Redemption rate | **16.0%** (635 of 3,979) | **7.9%** (210 of 2,655) | **7.7%** (44 of 574) |
| Share of all redeemers | 71% | 24% | 5% |

**Files needed (all verified against real row counts):**

| File | Rows | Role | Needed? |
|---|---|---|---|
| `campaign_table` | 7,208 | Defines rows: who was mailed which campaign | Required |
| `campaign_desc` | 30 | Campaign type, start/end day (the time cutoff) | Required |
| `coupon_redempt` | 2,318 | Defines label: did the household redeem? | Required |
| `transaction_data` | 2,595,732 | Purchase history, main source of features | Required |
| `product` | 92,353 | Product categories for category/brand features | Required |
| `coupon` | 124,548 | Which products each campaign's coupons cover | Required |
| `hh_demographic` | 801 | Demographic features for the 32% who have them | Optional |
| `causal_data` | 36,786,524 | In-store displays and weekly mailer features | Not for v1 |

### Prediction Question 2: Expected Financial Value
For a given household and campaign, what is the expected net revenue the household will generate from campaign-related products if they engage with the campaign?
- Unit: one household and one campaign.
- Output: a continuous dollar amount ($).
- Example: Household 1234, TypeB campaign: Expected net value of $42.50.
- **Target (corrected from the original proposal):** sum of `SALES_VALUE` for campaign-related products during `START_DAY`–`END_DAY`. `SALES_VALUE` is used **directly**, with no further subtraction — the original proposal's formula subtracted `COUPON_MATCH_DISC` again, but that discount is already reflected in `SALES_VALUE`, so subtracting it double-counts it (see [Known data caveats](#known-data-caveats)).
- **Predictive model:** regression on redeemed household × campaign pairs.
- **Defining campaign-related value:**
  - **TypeB/TypeC only:** 254 redeemed pairs with known coupon exposure — a clean, well-defined campaign-related-product target.
  - **TypeA:** the exact 16 coupons each household received are unobserved, so household-level product exposure is uncertain. Open methodological question: investigate alternative ways to operationalize the Model 2 target for TypeA.

### Prediction Question 3: True Incremental Uplift
For a given household and campaign, how much does receiving the campaign actually increase the household's likelihood of purchasing the targeted products, compared to their baseline behavior if we sent them nothing?
- Unit: one household and one campaign.
- Output: an incremental probability percentage (which can be positive, zero, or negative).
- Example: Household 1234, TypeB campaign: +12% incremental lift (meaning the household already had a 40% organic chance to buy the product, but the campaign pushes their total probability to 52%).

**Modeling objective:** estimate the *incremental* effect of mailing a campaign — how much more likely a household is to buy eligible products *because of the mailer*, beyond their organic baseline habit (what they'd buy anyway) or their response to in-store promotions.

**Observation level:** one household × one campaign — **scoped to TypeB/TypeC campaigns** (25 of 30 campaigns, 3,229 mailings, 45% of all mailings).
- *Scope rationale:* TypeB/TypeC households receive every coupon in the campaign, so we know exactly what they could redeem. TypeA households receive only 16 untracked coupons from a pool of 17,000–35,000 products — testing the full pool as "eligible" produced a 96% false-positive conversion rate in trial testing, since that's effectively measuring "did this household shop at all." TypeA is out of scope for v1.

**Target/outcome:** binary (1/0) — did the household purchase an eligible product (`PRODUCT_ID`s from `coupon`) during the active campaign window, **excluding** purchases that occurred while the product was concurrently featured on an in-store display or in a weekly mailer/circular (the `causal_data` confounder filter)?

**Data feasibility — treatment, control, and the confounder:**
- **Treatment group:** households mailed this campaign (verified against `campaign_table`).
- **Control group:** households not mailed this campaign, and not mailed any other campaign whose window overlaps it (verified against `campaign_table` + `campaign_desc`, since campaigns overlap heavily — see [Known data caveats](#known-data-caveats)).
- **Confounder filter:** `causal_data` joined to `transaction_data` on shared `PRODUCT_ID`/`STORE_ID`/`WEEK_NO` keys flags purchases driven by in-store displays or circulars, isolating direct-mail influence from baseline organic habit and store-level promotions.
- **Limitation:** campaign targeting is non-random — treated households spend ~4.6x more, pre-campaign, than control households. This is framed as **uplift modeling that adjusts for observed confounders**, not a randomized-experiment causal estimate.

**Methodology:** T-Learner (two-learner) architecture — Model T (treated households) and Model C (control households) trained independently; `Uplift = P(buy | Treatment) − P(buy | Control)`. Preferred over a single (S-learner) model, which tends to ignore the treatment flag when it's dominated by strong baseline features like past spend.

**Evaluation:** Qini Curve (uplift cumulative gain). Standard ROC-AUC is invalid here because real-world counterfactuals can't be observed directly (a household can't be simultaneously mailed and not mailed); the Qini curve instead measures true incremental conversion.

**Key business finding (the value-add over Model 1):** current campaigns disproportionately target the heaviest, most loyal spenders, who tend to buy the product anyway — a propensity model (Model 1) would misallocate budget to them because their overall purchase probability is high. Untargeted, moderate-spending households show the largest actual response gap between "mailed" and "not mailed." Uplift modeling isolates this true incremental impact to drive real incremental revenue, which Model 1 alone cannot surface.

---

## System Flow & Architecture

One shared agent pipeline answers all three prediction questions — every agent does the same job, in the same way, for every model; only the instructions it receives (which campaigns, which rows, which target, which evaluation metric) change per model. Core computation stays in deterministic, tested Python functions; agents orchestrate those tools and produce reports, which keeps results reproducible and agent performance measurable.

```
Raw data (8 dunnhumby CSV files)
   │
   ▼
Step 1: Data Agent
   │
   ▼
Step 2: Feature Agent
   │
   ▼
Step 3: Modeling Agent
   │
   ▼
Step 4: Evaluation Agent  ──(feedback)──▶ back to Step 2 or Step 3
   │
   ▼
Step 5: Scoring
   │
   ▼
Step 6: Decision Lab (application)
   │
   ▼
Step 7: Decision Support Agent
```

**Step 1 — Data Agent.** Input: raw CSV files. Task: cleans and checks all 8 files against one shared rulebook (expected columns, checks, whether to fix/flag/stop) — for example, that every redemption matches a campaign the household was mailed, and that demographics exist for only 801 of the 2,500 households. Decides: which issues to fix automatically and which to flag for team review. Output: clean tables and a data quality report, sent to the Feature Agent. **Progress:** first-draft ingestion script built and run successfully on all 8 real files (2.6M transaction rows, 36.8M display/mailer rows).

**Step 2 — Feature Agent.** Input: clean tables. Task: builds one shared table of 75,000 rows (2,500 households × 30 campaigns), holding each household's pre-campaign behavior (spend, visits, past redemptions, category affinity) plus every model's target column. Decides: which household features to build. Output: training table, sent to the Modeling Agent. Each model's instructions pick which campaigns, rows, and target column to use:

| Model | Campaigns | Rows | Target column |
|---|---|---|---|
| 1. Response | All 30 | Mailed: 7,208 | Redeemed (yes/no) |
| 2. Expected value | All 30 (or TypeB/C) | Redeemed: 889 (TypeB/C: 254) | Spend on campaign products ($) |
| 3. Uplift | TypeB/C only (25) | Mailed and not mailed: 43,396 | Bought a campaign product (yes/no) |

**Step 3 — Modeling Agent.** Input: training table. Task: reshapes the table for the model type being trained (e.g. yes/no columns and similarly-scaled numbers for logistic regression; near-as-is for a boosting model — same information, different format), then trains and tunes on the earlier campaigns. Decides: which model(s) to try and how to tune them. Output: trained model(s), sent to the Evaluation Agent.

**Step 4 — Evaluation Agent.** Input: trained model(s). Task: tests on the later campaigns and compares against a baseline. Decides: approve, or send feedback back to Step 2 (features) or Step 3 (modeling). Output: feedback, or the final model and an evaluation report, sent to Step 5.

| Model | Modeling instructions | Evaluation instructions |
|---|---|---|
| 1. Response | Yes/no models (e.g. logistic regression, LightGBM) | Redeemers caught in the top 20%, vs. "redeemed before" |
| 2. Expected value | Dollar models (e.g. linear regression) | Average dollar error, vs. usual spend |
| 3. Uplift | Two models (mailed, not mailed); uplift = difference | Qini curve, vs. random targeting |

**Step 5 — Scoring.** Input: final model(s) and each household's latest data. Task: calculates each model's score for every household, for a new campaign of each type (TypeA, TypeB, TypeC). Output: household scores, used by the Decision Lab.

**Step 6 — Decision Lab (application).** Input: the marketer's choices (campaign type, number of households to target). Task: ranks households by score and selects the top households. Output: target list, expected number of redemptions, and expected cost of discounts.

**Step 7 — Decision Support Agent.** Input: the marketer's question in the Decision Lab, e.g. "How many more redemptions would we get by targeting 1,000 households instead of 500?" Task: retrieves the relevant results and explains them in plain language. Decides: which results answer the question. Output: answer shown in the Decision Lab.

**Why each agent is shared, not model-specific:**

| Agent | Shared because... |
|---|---|
| 1. Data | All three models use the same data files, so the cleaning rules are the same. |
| 2. Feature | One shared table of household features serves all three; each model keeps only its own campaigns, rows and target. |
| 3. Modeling | Same steps (reshape the data, train, tune); the instructions say which kind of model to build. |
| 4. Evaluation | Same steps (test on later campaigns, compare with a baseline); the instructions say how to measure success. |
| 5. Scoring | Runs any approved model the same way; each model adds one score column. |
| 6. Decision Lab | One application; each score is a column it can rank by or add up. |
| 7. Decision Support | Same tools to look up and explain results; it is told what each score means. |

Model 3 plugs its own method (two models: mailed and not mailed) and its own evaluation (Qini curve) into the same shared Modeling and Evaluation agents — no model-specific agent classes needed.

### Notes
- **Prediction timing:** if predictions are made immediately before a campaign starts, household features must use only data available before `START_DAY` (no leakage).
- **Evaluation benchmark:** compare each model's performance against random selection and a simple rule based on past redemptions.
- **Agent decisions:** which data issues can agents handle automatically, and which should be flagged for team review?
- **Potential ultimate business Qs answered (tentative):** once deployed, the Decision Lab should answer strategic questions in natural language, e.g.:
  - *Targeting strategy:* "Which demographic segment (e.g., Homeowners with Kids in Group 4) yields the highest response rate for TypeA campaigns?"
  - *Model explainability:* "Why did the model predict this customer has a 10% chance of redeeming the frozen pizza coupon?" (the agent explains the household has no purchases in that `COMMODITY_DESC` over the last year).
  - *Scenario trade-offs (what-if):* "If we target 5,000 extra households, how does that impact our overall campaign budget?" (the agent projects the increase in `COUPON_MATCH_DISC` payouts vs. the rise in `SALES_VALUE`).
  - *Omnichannel impact (tentative):* "Did featuring this product on a 'Front End Cap' in-store display increase coupon redemption compared to direct mail alone?"

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .env.example                      # API keys / config template (never commit .env)
├── data/                             # gitignored — local raw & processed data (each member downloads their own)
│   ├── raw/
│   └── processed/
├── docs/                             # design docs, meeting notes, progress reports, presentations
│   ├── data_agent_design.md          # Step 1 — Data Agent: why each cleaning/validation choice was made
│   ├── data_agent_team_summary.md    # Step 1 — Data Agent: 5-minute team summary
│   ├── feature_agent_2a_design.md    # Step 2A — Feature Agent: shared-table design
│   ├── feature_agent_2b_design.md    # Step 2B — Feature Agent: per-model table design
│   ├── meeting-notes/                # dated notes from mentor/team meetings
│   ├── progress-reports/             # weekly progress write-ups
│   └── presentations/                # slide decks shown to mentors
├── notebooks/                        # EDA and experiments (prefix with initials + number, e.g. jw_01_eda.ipynb)
├── reports/
│   └── figures/                      # generated charts (e.g. Model 3's Qini curve)
├── validation/                       # Data Agent: standalone assumption-verification script + report
├── src/
│   ├── preprocessing/                # Step 1 — Data Agent: ingest.py, build_spine.py
│   ├── features/                     # Steps 2A/2B — Feature Agent: shared table + per-model tables
│   ├── modeling/                     # Step 3 — Modeling Agent (e.g. model3_baseline.py)
│   ├── evaluation/                   # Step 4 — Evaluation Agent (not yet built)
│   ├── scoring/                      # Step 5 — Scoring (not yet built)
│   ├── api/                          # Model-serving REST API (not yet built)
│   ├── app/                          # Step 6 — Decision Lab UI (not yet built)
│   └── agents/                       # Step 7 — Decision Support Agent (not yet built)
└── tests/
    ├── preprocessing/                # Data Agent tests
    └── features/                     # Feature Agent tests (unit + real-data acceptance)
```

---

## Tech Stack (proposed, TBD)

| Layer | Candidate |
|---|---|
| Language | Python 3.11+ |
| Data | DuckDB, Parquet, pandas / polars |
| Modeling | scikit-learn, LightGBM / XGBoost, SHAP |
| Experiment tracking | MLflow (optional) |
| API | FastAPI |
| UI | Streamlit (or React) |
| LLM / Agents | TBD (LLM provider and agent framework to be decided with mentors) |

---

## Setup

```bash
# 1. Clone
git clone https://github.com/shreyashetty2/Agentic-AI-Marketing-Decision-Lab.git
cd Agentic-AI-Marketing-Decision-Lab

# 2. Create environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Add data
mkdir -p data/raw
#    Download The Complete Journey CSVs and place them in data/raw/

# 4. Configure secrets
cp .env.example .env              # then fill in API keys
```

### Run the pipeline

```bash
python src/preprocessing/ingest.py       # Step 1, Data Agent: clean tables
python src/preprocessing/build_spine.py  # Step 1, Data Agent: household x campaign table
python src/features/build_2a.py          # Step 2A, Feature Agent: shared feature table + run report
python src/features/build_2b.py          # Step 2B, Feature Agent: one table per model + run report
python src/modeling/model3_baseline.py   # Step 3, Modeling Agent: Model 3 (uplift) T-learner baseline
pytest tests                             # unit tests
python tests/features/acceptance_2a.py   # real-data acceptance checks (slow, ~1 min)
python tests/features/acceptance_2b.py   # real-data acceptance checks
```

Step 2A writes `data/processed/features_household_campaign.parquet` (75,000 rows x 47 columns) and `feature_2a_report.md`. What every column means, and which ones a model may use, is in `src/features/feature_2a_spec.yaml`.

Step 2B writes `data/processed/model{1,2,3}_table.parquet` and `feature_2b_report.md`. Which rows and which answer column each model gets is in `src/features/feature_2b_spec.yaml` (why, in `docs/feature_agent_2b_design.md`): Model 1 is 7,208 mailed rows (889 redeemed); Model 2 is 254 redeemed TypeB/C rows; Model 3 is 43,396 TypeB/C rows (3,229 treated, 40,167 clean control).

*(Commands for the later steps and the API/UI will be added as components are built.)*

---

## Evaluation

**Quantitative**
- **Model performance:** AUC, PR-AUC, precision, recall, F1, lift / gains by decile, calibration
- **Improvement over baselines:** e.g., logistic regression, simple RFM rules
- **Agent task success:** each agent's steps completed correctly
- **Scenario simulation:** what-if scenarios execute correctly
- **API/app:** responsiveness and reliability

**Qualitative**
- **Agent usefulness:** how much the agents help across the modeling workflow
- **Decision Lab:** usability
- **LLM explanations:** clarity, usefulness, and **factual consistency** (claims checked against model outputs)
- **Scenario comparison:** whether users can compare scenarios and understand the trade-offs

## Limitations & Responsible Use

- **Predictive, not causal.** Campaign targeting was non-random, so what-if outputs are *model-projected responses*, not measured causal effects.
- **Assumed costs.** The dataset has no cost or margin data, so budget and ROI scenarios rely on user-specified assumptions.
- **LLM grounding.** LLM outputs must be grounded in model and API results, and ungrounded claims are treated as errors.
- **Data license.** The dataset is used under dunnhumby's terms and is not redistributed in this repository.

---

## Team

| Name | UNI | Primary Focus |
|---|---|---|
| Yuxin Cai | yc4763 | TBD |
| Pranav Jain | pj2459 | TBD |
| Shreya Shetty | svs2148 | TBD |
| John Won | jjw2206 | TBD |
| Zhiyan Yang | zy2745 | TBD |

**Mentors (TD Bank):** Shira Appelbaum, Elif Ulusal
**Course:** ENGI E4800, Columbia Data Science Institute

## Contributing

**No direct commits or pushes to `main`, ever — PRs only.** This is a hard rule, not a suggestion, so two people working on the same table (e.g. Feature Agent 2B) don't silently overwrite each other or create conflicting commits on main.

1. Branch from `main` using `feature/<short-description>` (or `docs/...`, `fix/...`).
2. Commit and push your branch, then open a pull request into `main`.
3. Get at least one reviewer's sign-off before merging. Only merge your own PR after that.
4. Never commit raw data, `.env`, or API keys.
5. Clear notebook outputs before committing.

Branch protection enforcing rule 1 (blocking direct pushes to `main` server-side) is pending a GitHub Pro upgrade — this repo is private, and GitHub only allows branch protection rules on private repos on a paid plan. Columbia students qualify for the free [GitHub Student Developer Pack](https://education.github.com/pack), which includes GitHub Pro; once that's active, enable it under **Settings → Branches → Add branch protection rule** for `main`, checking "Require a pull request before merging." Until then, rule 1 is enforced by convention — please follow it manually.

## Timeline

| Milestone | Date |
|---|---|
| Kickoff with mentors | Week of 9/14 |
| Midterm report | 11/8 |
| Poster session | 12/14 |
| Final report | 12/20 |
