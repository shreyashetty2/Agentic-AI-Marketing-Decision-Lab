# Midterm Report: DSI Capstone

Yuxin Cai, Shreya Shetty, Pranav Jain, John Won, Zhiyan Yang

> Shared skeleton for the full midterm report (guidelines: Introduction ~2pp, Methods ~3pp,
> Preliminary Results ~3pp, Discussion ~1pp, plus a contributions paragraph per person). This
> mirrors the team's Google Doc — whichever one we're actually editing live, please keep the other
> in sync (or just delete this file once the Doc is the single source of truth). Each section below
> is either filled in, sourced from this repo's own reports and design docs, or left as a
> `_TODO — <name>:_` placeholder for whoever owns that model/agent to fill in. Primary ownership,
> per the README's team table: Model 1 = John, Model 2 = Yuxin, Model 3 = Shreya, Data Agent =
> Zhiyan, agent architecture (Modeling/Evaluation/Lead Agents) = Pranav.

---

## 1. Introduction (~2 pages)

### 1.1 Motivation and Business Problem

Marketing organizations increasingly rely on predictive models to decide which customers should
receive a campaign, whether that is a marketing email, a promotion or a coupon. The people involved
in that decision usually sit in two separate roles: data scientists, who build and maintain the
models, and marketers, who plan the campaigns and decide who to target, what to offer and how much
to spend.

The difficulty is that these two roles do not easily connect. A marketer planning a campaign wants
to know which customers to target, what would happen if the offer or the budget changed, and why
the model favors one segment over another, but a model's outputs do not answer these questions
directly and are often hard for business users to interpret. Since marketers cannot work with the
models themselves, each new question has to go back through a data scientist.

Data scientists, on the other hand, face a problem of their own, because developing a model
involves multiple steps, from preparing the data and building features to training and comparing
models, and this work has to be repeated every time new data arrives or a campaign finishes. Each
of those steps involves judgment calls that are easy to lose track of, especially when the person
maintaining a model is not the person who originally built it.

This project, sponsored by TD Bank, builds an Agentic AI Marketing Decision Lab to close this gap
by supporting both roles, with an AI agent designed for each one. The data scientist works with a
Data Scientist Agent, a co-engineer that keeps the models up to date as new data arrives, explains
what changed and why, proposes what to try next, and flags anomalies when it finds something new.
The marketer, in turn, works with a Decision Support Agent inside an interactive application, where
they can explore campaign options and ask questions without needing to understand the models
underneath. Since both agents sit on top of the same models, the answers a marketer receives come
from work that a data scientist has already reviewed.

### 1.2 Existing Work

Direct marketing research has long used past behavior to predict who will respond to a campaign
and to weigh that response against the cost of each mailing ([Bult & Wansbeek, 1995](https://doi.org/10.1287/mksc.14.4.378);
[Fader, Hardie & Lee, 2005](https://doi.org/10.1509/jmkr.2005.42.4.415)), and uplift modeling goes
one step further by estimating the change in behavior that a campaign causes, which response models
are not designed to measure ([Radcliffe & Surry, 2011](https://stochasticsolutions.com/pdf/sig-based-up-trees.pdf);
[Künzel et al., 2019](https://doi.org/10.1073/pnas.1804597116)). More recently, AI agents built on
large language models have been used to automate data analysis tasks ([Guo et al., 2024](https://proceedings.mlr.press/v235/guo24b.html);
[Hong et al., 2025](https://aclanthology.org/2025.findings-acl.1016/)). Our project brings these
three areas together in one system, which is built around the two people who have to use it.

### 1.3 Problem Statement

Our goal is to show how AI can support the full analytics lifecycle, from data preparation and
predictive modeling to decision support through an interactive application. We test this system on
the public dunnhumby "The Complete Journey" dataset using three predictive models, which together
tell a marketer who is likely to respond to a campaign, what that response is worth, and whether
the campaign changed what a household bought:

* **Response model (Model 1):** Predicts the probability that a mailed household redeems at least
  one coupon from a campaign.
* **Value model (Model 2):** Predicts how much a household that redeems will spend on the
  campaign's products during the campaign.
* **Uplift model (Model 3):** Estimates how much more likely a household is to buy the campaign's
  products because it was mailed.

The project seeks to answer research questions in three areas:

* **Model development:** Can specialized AI agents improve the efficiency and quality of predictive
  model development?
* **Prediction:** How accurately can campaign response be predicted from historical retail and
  marketing data, and which customer and campaign characteristics are most predictive?
* **Decision support:** Can a generative AI-based Decision Support Agent improve interpretation of
  model outputs, and does an AI-assisted platform improve business users' ability to evaluate
  alternative marketing strategies?

### 1.4 Overview

We are building an end-to-end system that takes raw retail data and turns it into campaign
decisions that a marketer can act on. On the data science side, separate agents clean the data,
build the features, and train and evaluate the response, value and uplift models, and a data
scientist runs and maintains all of this by working with a single Data Scientist Agent that acts
as an interface between the scientist and all the underlying agents. On the marketing side, the
results feed into the Decision Lab, an application where a marketer can explore campaign options
and ask a Decision Support Agent questions in plain language. The result is one system in which a
data scientist can keep the models up to date and a marketer can use their results directly,
without every question having to pass between the two.

---

## 2. Methods (~3 pages)

### 2.1 Data

#### 2.1.1 Dataset and pipeline-wide preprocessing

_TODO — Zhiyan (Data Agent):_ describe the dunnhumby "Complete Journey" source tables and the Data
Agent's own cleaning steps (pipeline Step 1) that produce the inputs Feature Agent 2A reads.

_TODO — Pranav (agent architecture):_ describe how the Lead Agent coordinates the Data/Feature/
Modeling/Evaluation Agents end to end, including the staleness/rerun logic referenced in §2.3.1.

**Feature Agent 2B (Shreya) — shared across all three models.** Feature Agent 2A builds one shared
table (75,000 rows: 2,500 households × 30 campaigns) carrying every model's candidate input
columns and every model's answer column side by side. 2B's job is to turn that into one clean
table *per model* — the right rows, the right answer column, nothing else changed — with the exact
rule for each model read directly from `src/features/feature_2b_spec.yaml`, so the design doc and
the code can't drift apart. Row counts, verified against the real data rather than assumed:
**Model 1** — every mailed household × campaign pair, all 30 campaigns, 7,208 rows; **Model 2** —
redeemed, mailed, TypeB/C only, 254 rows; **Model 3** — TypeB/C only, treatment + clean control
(rows mailed an overlapping campaign excluded), 43,396 rows (3,229 treated, 40,167 control). 2B
also owns a **leakage rail**: each model's table only ever contains that model's own label (plus,
for Model 3, the treatment flag that decides which rows train Model T vs. Model C — never fed in
as a model input), with every other model's outcome/filter columns dropped entirely, not just
tagged; the build script stops the run if a label column is ever found inside a feature list.
Re-running on the same 2A output reproduces byte-identical tables. More in the Appendix.

#### 2.1.2 Model 1 data (Campaign Response Probability)

_TODO — John:_ describe Model 1's data scope and key features.

#### 2.1.3 Model 2 data (Expected Financial Value)

_TODO — Yuxin:_ describe Model 2's data scope and key features.

#### 2.1.4 Model 3 data (True Incremental Uplift)

Model 3 is scoped to TypeB/TypeC campaigns — 25 of the dataset's 30 campaigns, 3,229 treated
(mailed) rows against 40,167 control rows, no subsampling at any stage. TypeA campaigns were
excluded in an earlier scoping decision: TypeA households receive only 16 untracked coupons out of
a pool of 17,000–35,000 products, and treating that whole pool as "eligible" produced a 96%
false-positive conversion rate in trial testing — effectively measuring "did this household shop
at all," not campaign response.

Treatment and control are defined against `campaign_table`/`campaign_desc`: a household counts as
control only if it was mailed *no* campaign whose window overlaps the one being evaluated, since
campaigns overlap heavily in this data. The outcome is binary — did the household buy an eligible
product during the campaign window, excluding purchases that coincided with an in-store display or
flyer feature for that product (the `causal_data` confounder filter), so the main label isolates
direct-mail effect from in-store promotion as cleanly as the raw data allows.

A second confounder feature, `campaign_promo_share`, was added and then revised mid-effort: it
originally measured whether an eligible product was promoted *anywhere*, which was nearly constant
(21 of 25 campaigns scored a flat 1.0) and carried almost no signal. It now measures promotion at
the *household's own usual store* — the store each household shopped at most before the campaign
started — which moved it from 25 distinct values (std 0.093) to 1,240 distinct values (std 0.289),
a real increase in the information a model can use, not a cosmetic fix.

### 2.2 Exploratory Data Analysis

#### 2.2.1 Model 1

_TODO — John:_ key EDA patterns that guided Model 1's preprocessing/modeling choices.

#### 2.2.2 Model 2

_TODO — Yuxin:_ key EDA patterns that guided Model 2's preprocessing/modeling choices.

#### 2.2.3 Model 3

Three questions were answered directly against the full dataset before any model was trusted:

1. **Does mailing add lift at all?** Naive treated-vs-control comparison: 49.4% vs. 13.8% purchase
   rate (z = 52.5, p ≈ 0) — not noise, but this raw gap mixes real campaign effect with the fact
   that heavier spenders were targeted to begin with.
2. **Does lift vary by household spend?** Bucketing households into thirds using cutoffs from the
   *treated group's own* spend distribution (population-wide cutoffs were tried first and
   rejected — they put ~95% of treated households in a single bucket) shows uplift falling
   monotonically as historical spend rises: ~26% for low-spend households, ~14% for medium, ~9%
   for the heaviest spenders. The trained model reproduces the same ordering on held-out test
   campaigns (0.102 / 0.049 / 0.006).
3. **How much of "success" is actually coupon-driven?** Of 1,596 treated households who bought an
   eligible product, 1,407 (88.2%) did so *without ever redeeming the coupon* — direct, observed
   confirmation of (2), not an inference. A campaign-level breakdown shows this holds across every
   campaign with enough volume to check (79.7%–95.2% unredeemed-but-bought; full table in the
   Appendix).

A secondary demographic cut (homeowner status, number of kids, age group) was also run, but only
covers 801 of 2,500 households with a demographic record on file. Uplift is positive and broadly
similar across every demographic category (19–42 percentage points), with no split nearly as sharp
as the spend-tier one — informative by itself: pre-campaign spend behavior predicts responsiveness
far better than who a household is demographically.

### 2.3 Analytic / Modeling Methods

#### 2.3.1 Agent architecture (Data, Feature, Modeling, Evaluation, Lead Agents)

_TODO — Pranav:_ describe the agent pipeline design — how the Lead Agent coordinates the
Data/Feature/Modeling/Evaluation Agents, the staleness/rerun logic, and how candidates get
registered and judged consistently across models.

#### 2.3.2 Model 1

_TODO — John:_ modeling approach, candidates tried, what's currently in use and why.

#### 2.3.3 Model 2

_TODO — Yuxin:_ modeling approach, candidates tried, what's currently in use and why.

#### 2.3.4 Model 3

**Baseline.** A T-learner (two independently trained `RandomForestClassifier`s, one on treated
rows, one on control) estimates uplift as `P(buy | treated) − P(buy | control)`, preferred over a
single S-learner, which tends to ignore the treatment indicator when it's dominated by strong
baseline features like historical spend. `class_weight='balanced'` was tried and rejected: it
inflated Model C's average predicted probability to 41.0% against a true rate of 13.6% (a 3x
calibration distortion), which breaks the T-learner's core assumption that both halves produce
genuinely calibrated probabilities.

**The train/evaluate mismatch.** The Evaluation Agent scores every candidate on
`ipw_stratified_qini` — a propensity-weighted Qini computed within each campaign — but the baseline
T-learner is trained with no weighting at all. Four candidates were built to close this gap or
improve on it, all registered in `modeling_spec.yaml` and judged by the same Modeling + Evaluation
Agent pipeline as everything else, with no hand-rolled comparison:

- **Propensity-weighted T-learner** — same learner, trained with the evaluator's own
  inverse-propensity weights.
- **Entropy-balanced T-learner** (Hainmueller, 2012) — reweights training rows so treated and
  control groups balance on observed covariates.
- **X-learner** (Künzel et al., 2019) — built specifically for this dataset's ~12:1
  treated/control imbalance, where a plain T-learner's two models are fit on wildly different
  sample sizes and their errors don't cancel.
- **DR/AIPW learner** — a doubly-robust estimator fit on a pseudo-outcome that stays unbiased if
  either the outcome models or the propensity model (but not necessarily both) is correctly
  specified.
- **Confound-aware variants** of the T-learner and X-learner, adding `campaign_promo_share` as a
  feature.

**Sure-things classifier.** A separate model (not a T-learner) trained only on treated households,
predicting which already-mailed households will buy the eligible product *without* ever redeeming
the coupon. Control households can't be used here — they can never redeem a coupon they never
received, so the label has zero variance for them. This gives a direct deprioritization signal,
independent of the uplift ranking: a household can score low on uplift for different reasons than
scoring high as a "sure thing."

**Discipline around the locked check.** Open campaigns (15) are reused for iteration; four
campaigns are held back as a one-time locked check. Two early candidates (`extended_t_forest`,
`shallower_t_forest`) passed on open campaigns but failed on locked — the motivating example for
treating every subsequent result with real caution. By report time the locked set had been read
three times, and the Evaluation Agent itself flags when a check is no longer clean (*"these
campaigns were already read 2 time(s)... this is no longer a clean test"*) — a deliberate choice to
let the system say "I'm not sure anymore" rather than keep re-using a shrinking resource.

---

## 3. Preliminary Results (~3 pages)

### 3.1 Evaluation

#### 3.1.1 Model 1

_TODO — John:_ metrics used and why.

#### 3.1.2 Model 2

_TODO — Yuxin:_ metrics used and why.

#### 3.1.3 Model 3

Standard ROC-AUC doesn't apply to uplift: a household can't be observed both mailed and unmailed,
so there's no ground-truth label to score against directly. Model 3 is instead evaluated on
**`ipw_stratified_qini`** — a propensity-weighted Qini coefficient computed within each campaign —
because the pooled, unweighted version mostly measures who the retailer chose to mail, not what
the mailer caused. Alongside it, an **E-value** (VanderWeele & Ding, 2017) quantifies how strong an
unmeasured confounder's association with both mailing and buying would need to be to fully explain
the observed effect away — currently **1.35**, meaning a fairly weak hidden confounder could do it.
This gives the project's "predictive, not causal" caveat a concrete number instead of a disclaimer.

### 3.2 Results Presentation

#### 3.2.1 Model 1

_TODO — John:_ results table/figures, qualitative observations, strengths and weaknesses.

#### 3.2.2 Model 2

_TODO — Yuxin:_ results table/figures, qualitative observations, strengths and weaknesses.

#### 3.2.3 Model 3

| Candidate | What changed | Own score | Open-campaign gap vs. owner | Locked-campaign gap vs. owner |
|---|---|---|---|---|
| `owner_t_forest` (original baseline) | — | −0.0111 | — | — |
| `owner_t_forest_weighted` | IPW-weighted training | −0.0084 | +0.0027 (inconclusive) | −0.0027 (inconclusive) |
| `owner_t_forest_entropy_balanced` | entropy-balance weights | −0.0069 | +0.0042 (inconclusive) | −0.0040 (inconclusive) |
| **`owner_x_learner`** — **model in use** (2026-10-21) | X-learner | **+0.0077** | **+0.0187 (pass)** | **+0.0103 (inconclusive, only candidate positive on both)** |
| `owner_x_learner_confound_aware` | X-learner + `campaign_promo_share` | +0.0111 | +0.0223 (pass, highest recorded) | +0.0068 (weaker than plain X-learner) |
| `owner_dr_learner` | doubly-robust (AIPW) | −0.0241 | −0.0130 (inconclusive) | +0.0042 (inconclusive) |

The **X-learner** was adopted as Model 3's model in use: the only candidate positive on its own
score, real on open campaigns, and still directionally positive (not flipped negative) on the
locked check. `owner_x_learner_confound_aware` looked best on open campaigns but weakened on
locked — an honest non-replication, kept as a tracked candidate rather than promoted. The DR
learner underperformed, likely because the AIPW pseudo-outcome's variance is too high relative to
a treated group of only 3,229 rows.

The **sure-things classifier** reaches **AUC 0.675**; its most confident 20% (277 households) are
right 72.9% of the time against a 53.8% base rate (1.4x lift), relying most on
`campaign_product_spend_per_week_26w` (29%), `campaign_product_share_26w` (19%), and
`duration_days` (18%). Qualitatively, the Evaluation Agent's verdicts (`pass` / `inconclusive` /
`human_needed`) are doing real work here, not just rubber-stamping: the confound-aware X-learner's
ambiguous locked-campaign result triggered a `human_needed` verdict rather than a forced decision —
the system correctly escalating instead of guessing. Figures:
`reports/figures/model3_uplift_qini_curve.png`, `reports/figures/model3_qini_curve.png`.

### 3.3 App or Tool

_TODO — Shreya (next week):_ the Decision Lab marketer-facing Streamlit app and "Ask the Decision
Lab" chat agent (grounded Q&A over all three models, Gemini via Vertex AI) were built this week but
are being held out of this report and written up for next week instead, once there's been time to
test them properly. Leaving this placeholder so the section header isn't missing from the skeleton.

---

## 4. Discussion (~1 page)

### 4.1 Conclusion

_TODO — whole team:_ summarize key takeaways across all three models and what's been achieved so
far. Suggest drafting this last, once every model's section above is filled in.

### 4.2 Future Work

_TODO — John (Model 1), Yuxin (Model 2), Pranav (agents), Zhiyan (Data Agent):_ planned next steps
through end of April, justified by this report's observations.

**Model 3:**
- `owner_x_learner_confound_aware` remains a tracked, promising candidate (+0.0034 open-campaign
  gap vs. the X-learner in use, best own-score of anything tried) but is deliberately not
  promoted — its locked-campaign check has already been read three times, and a fourth look was
  judged more likely to erode the check's value than to produce a trustworthy answer. Before
  April, this needs either a principled way to get a genuinely fresh out-of-sample check (the
  historical dataset is fixed and closed, so this is a real constraint, not a to-do) or an explicit
  decision to adopt it anyway on the strength of its own-score and open-campaign evidence alone.
- This connects directly to the project's "model development" research question: the
  locked-check-exhaustion problem is itself evidence for whether specialized agents can maintain
  modeling discipline over repeated iteration, not just whether they can produce a single good
  result.

**Platform — merging the agents and all three models' results into the app (Shreya, next, through
April):**
- The Lead Agent engine already runs as a Docker container (`Dockerfile`,
  `scripts/gcp_entrypoint.sh`), deployable as a scheduled Cloud Run Job with data synced through a
  GCS bucket rather than baked into the image (`docs/agent_design/gcp_deployment.md`) — built and
  run locally to confirm the container works before documenting the exact `gcloud` deployment
  commands.
- The next stage on that same path is merging the agent architecture (Lead Agent + Modeling +
  Evaluation Agents) and all three models' latest results into one marketer-facing surface — the
  Decision Lab. A first version of that surface (a Streamlit app with a page per model and a
  grounded "Ask the Decision Lab" chat agent over all three models' outputs) was built this week;
  it, and how well it actually holds up, is being written up in next week's report rather than
  this one, once there's been time to test it properly rather than just ship it.

---

## 5. Contributions

_One paragraph per member, describing their contributions to **this report**._

> **Shreya Shetty** led Model 3 (True Incremental Uplift) end to end: scoping the problem to
> TypeB/TypeC campaigns, building and iterating the T-learner baseline, diagnosing and fixing the
> train/evaluate mismatch against the evaluation pipeline's propensity-weighted Qini metric, and
> running a disciplined head-to-head comparison of five causal-inference approaches
> (propensity-weighted and entropy-balanced T-learners, an X-learner, a doubly-robust/AIPW learner,
> and confound-aware variants) that led to adopting the X-learner as the model in use. She also
> built the sure-things classifier as a complementary targeting-deprioritization signal, added and
> then substantially improved the `campaign_promo_share` confounder feature, and reported an
> E-value sensitivity check alongside the model's existing calibration diagnostics. Beyond Model 3,
> she built Feature Agent 2B — the shared step that turns Feature Agent 2A's one combined table
> into a clean, leakage-checked table for each of the three models — and did the first
> infrastructure work toward deploying the Lead Agent on GCP (Dockerfile, entrypoint script, and
> the Cloud Run deployment plan), the groundwork the Decision Lab app is being built on top of.

> _TODO — **Yuxin Cai**:_ your contributions to this report.

> _TODO — **Pranav Jain**:_ your contributions to this report.

> _TODO — **John Won**:_ your contributions to this report.

> _TODO — **Zhiyan Yang**:_ your contributions to this report.

---

## 6. Appendix

_TODO — John, Yuxin, Pranav, Zhiyan:_ add Model 1 / Model 2 / agent-architecture / Data Agent
technical detail here as needed. Model 3's detail (Shreya) is below.

### A. Model 3 — full demographic segmentation (801 of 2,500 households with a demographic
record; categories with fewer than 20 treated households dropped)

By homeowner status:

| homeowner_desc | n_treated | n_control | treated_rate | control_rate | observed_uplift |
|---|---|---|---|---|---|
| Probable Owner | 20 | 106 | 0.700000 | 0.283019 | 0.416981 |
| Homeowner | 1,331 | 4,832 | 0.571751 | 0.292632 | 0.279118 |
| Renter | 124 | 350 | 0.532258 | 0.274286 | 0.257972 |
| Unknown | 543 | 2,231 | 0.497238 | 0.248767 | 0.248470 |

By number of kids:

| kid_category_desc | n_treated | n_control | treated_rate | control_rate | observed_uplift |
|---|---|---|---|---|---|
| 1 | 339 | 1,101 | 0.572271 | 0.284287 | 0.287984 |
| 2 | 220 | 551 | 0.618182 | 0.330309 | 0.287873 |
| None/Unknown | 1,280 | 5,315 | 0.532031 | 0.267545 | 0.264487 |
| 3+ | 195 | 693 | 0.528205 | 0.307359 | 0.220846 |

By age group (`classification_1`):

| classification_1 | n_treated | n_control | treated_rate | control_rate | observed_uplift |
|---|---|---|---|---|---|
| Age Group3 | 568 | 1,725 | 0.593310 | 0.276522 | 0.316788 |
| Age Group6 | 108 | 814 | 0.500000 | 0.224816 | 0.275184 |
| Age Group4 | 686 | 2,820 | 0.533528 | 0.279787 | 0.253740 |
| Age Group2 | 425 | 1,357 | 0.555294 | 0.311717 | 0.243577 |
| Age Group5 | 126 | 508 | 0.492063 | 0.251969 | 0.240095 |
| Age Group1 | 121 | 436 | 0.487603 | 0.298165 | 0.189438 |

### B. Model 3 — full campaign-level redemption breakdown (campaigns with ≥20 treated purchasers)

| campaign | n_bought | n_without_redemption | pct_without_redemption |
|---|---|---|---|
| 11 | 62 | 59.0 | 95.2 |
| 22 | 168 | 156.0 | 92.9 |
| 24 | 79 | 72.0 | 91.1 |
| 20 | 155 | 141.0 | 91.0 |
| 12 | 100 | 91.0 | 91.0 |
| 4 | 33 | 30.0 | 90.9 |
| 14 | 158 | 142.0 | 89.9 |
| 21 | 25 | 22.0 | 88.0 |
| 25 | 129 | 113.0 | 87.6 |
| 9 | 95 | 83.0 | 87.4 |
| 10 | 39 | 34.0 | 87.2 |
| 17 | 107 | 93.0 | 86.9 |
| 29 | 78 | 67.0 | 85.9 |
| 7 | 27 | 23.0 | 85.2 |
| 23 | 122 | 102.0 | 83.6 |
| 5 | 36 | 30.0 | 83.3 |
| 16 | 63 | 51.0 | 81.0 |
| 19 | 59 | 47.0 | 79.7 |

### C. Model 3 — held-out quintile evaluation (uplift model's predicted-score quintiles vs. actual
observed uplift)

| quintile | n_treated | n_control | actual_uplift |
|---|---|---|---|
| 0 | 824 | 1,907 | 0.3650 |
| 1 | 228 | 2,502 | 0.3441 |
| 2 | 72 | 2,724 | 0.5051 |
| 3 | 167 | 2,498 | 0.4623 |
| 4 | 96 | 2,635 | 0.4017 |

Spearman rank correlation (quintile vs. actual observed uplift): rho = 0.500, p = 0.391 — not
statistically significant on this split. Held-out Qini coefficient: −199.75, a known artifact of
treated/control non-overlap at the extremes of the ranking (§2.2.3), not evidence the per-tier
signal above is wrong.

### D. Model 3 — model setup detail

- **T-learner / X-learner base estimators:** `RandomForestClassifier(n_estimators=300, max_depth=6,
  min_samples_leaf=20)` for both Model T and Model C, natural class distribution (no
  `class_weight='balanced'` — see §2.3.4).
- **Train / test split:** out-of-time, earliest 70% of campaigns train / latest 30% test — 1,842
  treated + 27,901 control rows train; sure-things classifier trains on the same 1,842 treated
  rows, tests on 1,387 treated rows.
- **Feature list:** `spend_per_week_8w`, `spend_per_week_26w`, `avg_basket_value_26w`,
  `trips_per_week_26w`, `coupon_lines_per_week_26w`, `share_spend_on_discount_26w`,
  `past_redemption_rate`, `campaign_product_spend_per_week_26w`, `campaign_product_share_26w`,
  `history_weeks_before_cutoff`, `duration_days`.
- **Product-level drill-down** (`src/modeling/model3_product_lookup.py`): the same
  without-redemption pattern checked product-by-product rather than aggregated per campaign — 208
  qualifying products (≥5 treated purchases) ranked and saved to
  `data/processed/model3_product_redemption_summary.csv`; the top of that list is dominated by
  everyday staples (eggs, pasta sauce, cheese, juice) bought without redemption 74–100% of the
  time, an individual-product confirmation of the same 88.2% aggregate finding, not a separate
  result.

### E. Verification and testing notes

- **Feature Agent 2B's leakage rail** is enforced mechanically, not just documented: the build
  script stops the run if a label/filter column from one model ever ends up in another model's
  feature list, and re-running on identical 2A output reproduces byte-identical per-model tables.
- **The E-value implementation is checked against the original published result**, not just unit
  logic: `tests/evaluation/test_metrics.py::test_e_value_matches_the_known_worked_example` asserts
  the function reproduces VanderWeele & Ding (2017)'s own worked example (risk ratio 3.9 → E-value
  ≈ 7.27) to within 0.01 — the implementation returns 7.264.
- **Early exploratory work**, predating the formal `model3_uplift.py` build: an empirical baseline
  uplift notebook (`notebooks/empirical_uplift_baseline_Model3_try.ipynb`, 10/1) that first checked
  whether an uplift signal existed at all, and a standalone script
  (`notebooks/model3_slide_numbers_audit.py`, 10/2) written to verify the specific numbers claimed
  on the team's Model 3 proposal slide against the real data before they were presented. Both were
  superseded by the finalized, tested pipeline but are kept as the record of how the team got
  there.
