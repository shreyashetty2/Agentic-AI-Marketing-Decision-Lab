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
- **Demographics are partial.** Demographics exist for only a subset of households, so any demographic analysis is a subsample analysis.
- **The user guide's `hh_demographic` variable table is incorrect.** Use the actual CSV headers.
- **Discount columns are stored as negative values.** `sales_value` is what the retailer receives, not what the customer paid.
- **TypeA exposure is unobservable.** TypeA campaigns send each household a personalized 16-coupon subset, and which coupons each household received is not recorded.
- **Campaigns are targeted, not randomized.** Response estimates are predictive, not causal.
- **`day` and `week_no` are relative indices, not calendar dates.**



---

## ML Question and System Flow

### Prediction Question 1: Campaign Response Probability
For a given household level and campaign, what is the probability that the household will redeem at least one coupon from that campaign?
- Unit: one household and one campaign.
- Output: a probability from 0% to 100%.
- Example: Household 1234, TypeB campaign: 34% chance of redeeming.
- Training data: 7,208 past household-campaign mailings, of which 889 (12.3%) had at least one redemption.

### Prediction Question 2: Expected Financial Value
For a given household and campaign, what is the expected net revenue (total sales value minus the retailer's coupon match cost) the household will generate if they engage with the campaign?
- Unit: one household and one campaign.
- Output: a continuous dollar amount ($).
- Example: Household 1234, TypeB campaign: Expected net value of $42.50.
- Training data: Historical transactions (transaction_data) for the 889 household-campaign pairings that had a redemption, calculating the sum of SALES_VALUE minus COUPON_MATCH_DISC during the campaign's active days.

### Prediction Question 3: True Incremental Uplift
For a given household and campaign, how much does receiving the campaign actually increase the household's likelihood of purchasing the targeted products, compared to their baseline behavior if we sent them nothing?
- Unit: one household and one campaign.
Output: an incremental probability percentage (which can be positive, zero, or negative).
- Example: Household 1234, TypeB campaign: +12% incremental lift (meaning the household already had a 40% organic chance to buy the product, but the campaign pushes their total probability to 52%).
- Training data: A comparison of household purchase frequencies for targeted PRODUCT_IDs during active campaign windows (START_DAY to END_DAY) versus their historical purchase frequencies in the weeks prior to the campaign.



## System Flow
```
Raw data
- 8 dunnhumby CSV files (transactions, campaigns, coupons, redemptions, products, demographics, promotions)
   │
   ▼
Step 1: Data Agent
- Input: raw CSV files
- Task: checks and cleans the data, for example that every redemption matches a campaign the household was mailed, and that demographics exist for only 801 of the 2,500 households
- Decides: which issues to fix and which to flag
- Output: clean tables and a data quality report, sent to the Feature Agent
   │
   ▼
Step 2: Feature Agent
- Input: clean tables
- Task: builds one row for each household that was mailed a campaign (7,208 rows). Each row holds the household's behavior before the campaign started (weekly spend, visits, past redemptions, share of discounted purchases), the campaign type, and whether the household redeemed at least one coupon during the campaign
- Decides: which household features to build
- Output: training table, sent to the Modeling Agent
   │
   ▼
Step 3: Modeling Agent
- Input: training table
- Task: trains models on the earlier campaigns
- Decides: which models to try (for example logistic regression and LightGBM) and how to tune them
- Output: trained models, sent to the Evaluation Agent
   │
   ▼
Step 4: Evaluation Agent
- Input: trained models
- Task: tests the models on the later campaigns, measuring what share of actual redeemers fall in the top 20% of households by score
- Decides: whether the model is good enough, and if not, whether to go back to Step 2 (features) or Step 3 (models)
- Output: feedback to Step 2 or Step 3, or the final model and an evaluation report, sent to Step 5
   │
   ▼
Step 5: Scoring
- Input: final model and each household's latest data
- Task: calculates the redemption probability for every household for a new campaign of each type (TypeA, TypeB, TypeC)
- Output: household scores, used by the Decision Lab
   │
   ▼
Step 6: Decision Lab (application)
- Input: the marketer's choices (campaign type, number of households to target)
- Task: ranks households by score and selects the w was top households
- Output: target list and expected number of redemptions and expected cost of discounts
   │
   ▼
Step 7: Decision Support Agent
- Input: the marketer's question in the Decision Lab, for example "How many more redemptions would we get by targeting 1,000 households instead of 500?"
- Task: retrieves the relevant results and explains them in plain language
- Decides: which results answer the question
- Output: answer shown in the Decision Lab
```

### Note:
- Prediction timing: 
If we make predictions immediately before a campaign starts, household features should use only data available before START_DAY
- Evaluation benchmark:
Maybe compare the model’s top-20% targeting performance with random selection and a simple rule based on past redemptions
- Agent decisions:
Which data issues can agents handle automatically, and which should be flagged for team review? 
- Potential Ultimate Business Qs Answered (Tentative): When the models and agents are deployed into the AI Marketing Decision Lab, the integrated model will be able to answer the following strategic questions in natural language --  
   - Targeting Strategy: "Which demographic segment (e.g., Homeowners with Kids in Group 4) yields the highest response rate for Type A campaigns?"   
   - Model Explainability: "Why did the model predict this customer has a 10% chance of redeeming the frozen pizza coupon?" (The agent will explain that the household's transaction history shows no purchases in that COMMODITY_DESC over the last year).   
   - Scenario Trade-Offs (What-If Analysis): "If we target 5,000 extra households, how does that impact our overall campaign budget?" (The agent will calculate the projected increase in COUPON_MATCH_DISC payouts versus the anticipated rise in overall SALES_VALUE).   
   - Omnichannel Impact (Tentative - need to discuss how this will be included in our model): "Did featuring this product on a 'Front End Cap' in-store display increase the coupon redemption rate compared to relying on direct mail alone?"  

---

## System Architecture (proposed)

```
Raw CSVs
   │
   ▼
[1] Data Ingestion Agent ─ profiling, validation, cleaning → DuckDB / Parquet
   │
   ▼
[2] Feature Engineering Agent ─ household × campaign feature table, leakage checks
   │
   ▼
[3] Model Development Agent ─ baselines → advanced models, tuning, comparison
   │
   ▼
[4] Model Evaluation Agent ─ metrics, validation, model card, final recommendation
   │
   ▼
Model API (REST) ─ /score  /scenario  /explain
   │
   ▼
[5] Decision Lab UI ─ segments, campaign config, budgets, what-if, dashboards
   │
   ▼
[6] Decision Support Agent (LLM) ─ explanations, drivers, scenario comparison, recommendations
```

**Design principle:** core computation stays in deterministic, tested Python functions. Agents orchestrate those tools and produce reports, which keeps results reproducible and agent performance measurable.

---

## Repository Structure (tentative)

```
.
├── README.md
├── requirements.txt
├── .env.example              # API keys / config template (never commit .env)
├── data/                     # gitignored — local raw & processed data
│   ├── raw/
│   └── processed/
├── notebooks/                # EDA and experiments (prefix with initials + number, e.g. jw_01_eda.ipynb)
├── src/
│   ├── ingestion/            # Step 1 — Data Ingestion Agent + tools
│   ├── features/             # Step 2 — Feature Engineering Agent + tools
│   ├── modeling/             # Step 3 — Model Development Agent + tools
│   ├── evaluation/           # Step 4 — Model Evaluation Agent + tools
│   ├── api/                  # Model-serving REST API
│   ├── app/                  # Step 5 — Decision Lab UI
│   └── agents/               # Step 6 — Decision Support Agent + shared agent utilities
├── tests/
├── reports/                  # weekly write-ups, midterm/final reports, figures
└── docs/                     # design notes, data dictionary, meeting notes
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

*(Commands for loading the data, running the pipeline, and launching the API/UI will be added as components are built.)*

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

- Branch from `main` using `feature/<short-description>`.
- Open a pull request with at least one reviewer before merging.
- Never commit raw data, `.env`, or API keys.
- Clear notebook outputs before committing.

## Timeline

| Milestone | Date |
|---|---|
| Kickoff with mentors | Week of 9/14 |
| Midterm report | 11/8 |
| Poster session | 12/14 |
| Final report | 12/20 |
