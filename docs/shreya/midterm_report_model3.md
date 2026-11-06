# Midterm Report Draft: Model 3 (True Incremental Uplift)

Author: Shreya Shetty (svs2148). Written to be dropped into the shared team midterm report under
the matching headers. Sourced from `reports/model3_candidate_comparison.md`,
`reports/model3_uplift_report.md`, `reports/model3_sure_things_report.md`,
`reports/model3_redemption_analysis.md`, and `docs/agent_design/modeling_agent_model3_design.md`.

## Methods → Data (Model 3: True Incremental Uplift)

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

## Methods → EDA (Model 3)

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
   campaign with enough volume to check (80–95% unredeemed-but-bought).

A secondary demographic cut (homeowner status, number of kids, age group) was also run, but only
covers 801 of 2,500 households with a demographic record on file. Uplift is positive and broadly
similar across every demographic category (19–42 percentage points), with no split nearly as sharp
as the spend-tier one — informative by itself: pre-campaign spend behavior predicts responsiveness
far better than who a household is demographically.

## Methods → Analytic / Modeling Methods (Model 3)

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

## Preliminary Results → Evaluation (Model 3)

Standard ROC-AUC doesn't apply to uplift: a household can't be observed both mailed and unmailed,
so there's no ground-truth label to score against directly. Model 3 is instead evaluated on
**`ipw_stratified_qini`** — a propensity-weighted Qini coefficient computed within each campaign —
because the pooled, unweighted version mostly measures who the retailer chose to mail, not what
the mailer caused. Alongside it, an **E-value** (VanderWeele & Ding, 2017) quantifies how strong an
unmeasured confounder's association with both mailing and buying would need to be to fully explain
the observed effect away — currently **1.35**, meaning a fairly weak hidden confounder could do it.
This gives the project's "predictive, not causal" caveat (noted in the Limitations section) a
concrete number instead of a disclaimer.

## Preliminary Results → Results Presentation (Model 3)

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

## Discussion → Future Work (Model 3)

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

## Contribution paragraph (for the "each member's contributions" section)

> Shreya Shetty led Model 3 (True Incremental Uplift) end to end: scoping the problem to
> TypeB/TypeC campaigns, building and iterating the T-learner baseline, diagnosing and fixing the
> train/evaluate mismatch against the evaluation pipeline's propensity-weighted Qini metric, and
> running a disciplined head-to-head comparison of five causal-inference approaches
> (propensity-weighted and entropy-balanced T-learners, an X-learner, a doubly-robust/AIPW learner,
> and confound-aware variants) that led to adopting the X-learner as the model in use. She also
> built the sure-things classifier as a complementary targeting-deprioritization signal, added and
> then substantially improved the `campaign_promo_share` confounder feature, and reported an
> E-value sensitivity check alongside the model's existing calibration diagnostics.
