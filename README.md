# Agentic-AI-Marketing-Decision-Lab
Agentic AI Marketing Decision Lab capstone project with TD Bank
Columbia University · ENGI E4800 Data Science Capstone · Fall 2026

> **Status:** Draft — work in progress. Structure, methods, and tooling will evolve as the project develops.

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
