# Model 1: final report

**Question:** will a mailed household redeem at least one of the campaign's coupons?
**Use:** rank each campaign's mailing list so a marketer can mail the likeliest redeemers first, and add up the probabilities to forecast redemptions for the Decision Lab.
**Status (10/24):** development wrapped up. The baseline stays the model in use, one ready alternative is documented, and the open questions are decided at the end of this report.

## Decision

- **The model in use stays the baseline:** a random forest on 11 pre-campaign features (`model1_baseline.py`).
  - On the 13 open test campaigns it catches **56.3%** of redeemers in each campaign's top 20%.
  - The "redeemed before" rule catches 47.6%, a gap of +8.7 points (95% range +5.2 to +12.4).
  - Its forecast ranges contain the actual count for 92% of campaigns.
- **Why we're not switching:** the three finalists beat the baseline on the open campaigns, but on the held-back campaigns their capture gap was −2.7 to −4.0 points, with ranges spanning zero. The rule set before the check was to switch only if a finalist held up there, and none did on the team's main measure.
- **What did hold up:** the logistic finalists also ordered households better on the held-back campaigns (AUC +2.2 to +2.5 points). The ready alternative is `lean_logistic_log`:
  - logistic regression on log-scaled values, 24 columns, no demographics
  - it orders households better on both test sets
  - it's simpler than the forest (one weighted formula)
  - it's the Modeling Agent's own pick from the open campaigns
- **When to switch:**
  - From 10/24, Model 1 is judged by redeemers caught, averaged over mailing depths 5–50% (decision 1 below).
  - The lead agent re-tests every option as each new campaign finishes.
  - Switch to `lean_logistic_log` once its averaged gain over the baseline, measured only on campaigns that finish after 10/24, has a 95% range above zero. The command is `python src/lead/in_use.py set --model 1 --idea lean_logistic_log --by "<name>" --reason "..."`.

## How we tested

- **Data:** one row per mailed household × campaign, 7,208 rows (889 redeemed). Every feature uses only data from before the campaign's cutoff (start day − 7), and the Feature Agent's leak test passes for every column.
- **Forward chaining:** each test campaign is predicted by models trained only on campaigns that ended before its cutoff, campaign by campaign, as it would run for real.
- **Two test sets:**
  - **Open:** campaigns 8–20 (13 campaigns, 4,974 rows, 707 redeemers), used for every comparison and choice.
  - **Held back:** campaigns 21–25 (5 campaigns, 811 rows, 75 redeemers, all Type B), read only for the final check.
- **Measures:**
  - Main: **redeemers caught** in each campaign's top 20% (random mailing = 20%).
  - Also: **within-campaign AUC**, how well each campaign's list is ordered. It's smoother, and the Modeling Agent chooses by it.
  - Every gap has a 95% range from resampling whole households.

## What we found

### 1. Coupon use drives the baseline

Ranking by coupon use alone (coupon items per week, last 26 weeks) catches 56.5%, the same as the baseline. Coupon use is the bar any new feature must beat (`model1_evaluation.py`, PR #11).

### 2. The 18 new features (PR #29, `reports/model1_feature_investigation.md`)

The Feature Agent now also builds:
- coupon habit in more detail
- coupon use on the campaign's products and categories
- redemption history in more detail
- demographics as numbers

Together they improve ordering. On the open campaigns, AUC goes up 1.6 points for the forest and 0.8 for the log-scaled logistic.

### 3. Which of the 18 matter (mentor question, `model1_importance.py`)

**Three coupon-habit columns carry most of the gain:**
- `coupon_commodities_26w`: kinds of product bought with coupons
- `coupon_trip_share_26w`: share of trips with a coupon
- `coupon_lines_per_week_all`: coupon use over all history

The evidence:
- **Logistic, added one at a time:** these three take AUC from 0.818 to 0.828. The other 15 add nothing, and the last few (mostly demographics) slightly lower it, ending at 0.826.
- **Dropping the coupon-habit group from all 39 columns** costs the logistic model 1.2 AUC points (range 0.6 to 2.1). Dropping any other group costs it nothing.
- **The forest** relies on the same coupon columns, spread across them, so no single column is essential. Demographics help it a little: dropping them costs 0.3 points (range 0.1 to 0.6). They don't help the logistic model.
- **Correction to how the 10/23 result was described:** for the logistic model, the 18 new columns improve ordering but not redeemers caught.

  | Logistic regression on log-scaled values | Redeemers caught |
  |---|---|
  | The baseline's 11 columns | 59.3% |
  | The 21 already-built columns | 60.0% |
  | All 39 | 60.1% |

  Most of its gain comes from log scaling plus the 10 already-built columns.

### 4. Other kinds of model (mentor question)

All on the same 39 columns, open campaigns, compared with the baseline:

| Model | AUC gap (pts) | Capture gap (pts) |
|---|---|---|
| Log-scaled logistic, more shrinkage (C=0.01) | +2.5 [+1.3, +3.7] | +3.4 [+0.6, +5.6] |
| Average of logistic and forest rankings | +2.3 [+1.4, +3.3] | +2.8 [+0.0, +4.7] |
| Log-scaled logistic with L1 (drops weak columns) | +2.2 [+1.1, +3.1] | +3.1 [+0.6, +4.7] |
| Smooth-curve logistic (GAM-like) | +2.1 [+1.0, +3.2] | +2.8 [+0.3, +5.0] |
| **Log-scaled logistic (C=0.05)** | **+2.0 [+0.9, +3.1]** | **+3.8 [+0.3, +5.5]** |
| Random forest, baseline settings | +1.9 [+1.1, +2.6] | +2.7 [+0.7, +4.7] |
| LightGBM, slow and regularized | +1.6 [+0.6, +2.5] | +2.4 [−0.1, +4.3] |
| HistGradientBoosting | −0.4 [−1.6, +0.9] | +0.4 [−2.1, +2.7] |
| Separate models for Type A and Type B/C | worse than one shared model | |

Every reasonable model lands at about 0.82–0.83 AUC and 59–60% capture. The limit is the information in the data, not the kind of model. Smooth curves add nothing beyond the log scale, so the simplest model does as well as any.

### 5. Robustness (open campaigns)

- **Campaign type:** the gains show up in both.

  | | AUC gap (pts) | Campaigns better |
  |---|---|---|
  | Type A (3 campaigns, 568 redeemers) | +1.8 to +1.9 | 3 of 3 |
  | Type B/C (10 campaigns, 139 redeemers) | +2.1 to +2.9 | 6 or 7 of 10 |

  B/C capture gains are larger (+6.5 to +8.6) but very uncertain.
- **Mailing depth:** at 10%, 30% and 50% the new options stay ahead. For example, the lean logistic catches 38 / 72 / 87% against the baseline's 35 / 70 / 84%.
- **Forecast totals:** predicted redemptions add up to 92–96% of actual (652–678 vs. 707).
- **Leaving out gas and miscellaneous lines** (Claudia's switch, open question Q7b): it changes spend, trips and basket size for 58–68% of rows, but every model moves by under half a point. Model 1 doesn't depend on that decision.

### 6. The held-back check

**Finalists, fixed before the check:**
- `extended_plus_logistic_log`: log-scaled logistic, 39 columns
- `lean_logistic_log`: log-scaled logistic, 24 columns
- `extended_plus_forest`: forest, 39 columns

All three are recorded in `modeling_spec.yaml` in the commit before this report. The Modeling Agent ran every candidate on campaigns 21–25, and this report reads the result once. These campaigns had been read once before (the agents' first test run in October), but none of the finalists existed then.

| | Capture, top 20% | Gap, top 20% (pts) | Gap, averaged over 5–50% (pts) | AUC | AUC gap (pts) | Campaigns with better AUC |
|---|---|---|---|---|---|---|
| Baseline forest | 64.0% | | | 0.778 | | |
| Log-scaled logistic, 39 | 60.0% | −4.0 [−10.3, +6.7] | +0.9 [−2.7, +4.8] | 0.803 | +2.5 [−0.04, +4.9] | 4 of 5 |
| Lean log-scaled logistic, 24 | 61.3% | −2.7 [−8.1, +7.2] | +1.1 [−2.5, +4.3] | 0.800 | +2.2 [+0.01, +4.4] | 4 of 5 |
| Forest, 39 | 60.0% | −4.0 [−9.5, +5.7] | +0.8 [−1.7, +4.3] | 0.787 | +0.9 [−1.3, +3.1] | 3 of 5 |
| "Redeemed before" rule | 54.2% | | | 0.716 | | |

The top-20% gaps come from the Evaluation Agent's locked check (`unlock_report.md`). The averaged and AUC gaps score the same frozen predictions (`model1_importance.py --open-questions`).

How to read it:
- **This check is small.** It has 75 redeemers, and one redeemer is worth about 1.3 capture points. So −4.0 means 3 fewer redeemers caught. Even the baseline's lead over the "redeemed before" rule (+9.8 points) has a range that includes zero here.
- **Redeemers caught: no finalist held up.** Each point estimate is a few redeemers lower, and every range spans zero.
- **Ordering: the logistic finalists mostly held up.** Their AUC gain (+2.2 to +2.5) is the same size as on the open campaigns (+2.0 to +2.2), and they're better in 4 of 5 campaigns. But their ranges sit right at zero (−0.04 and +0.01 at the low end). The forest's ordering gain did not hold up.
- **Forecast totals:** every model, the baseline included, over-forecasts these campaigns by 21–41% (91–106 predicted vs. 75 actual), mostly in campaigns 22 and 24. That's not a trend over time. Campaigns swing between 0.4 and 1.7 times their forecast throughout the test period. The honest forecast ranges still contained the actual count for all 5 campaigns.

## Hand-off to the Decision Lab

- **What Model 1 provides:** for each mailed household and campaign, a probability of redeeming, from the model in use (`data/processed/modeling_agent/model1/final_model.joblib`, refit on all finished campaigns).
- **Rank within a campaign, not across campaigns.** Type A campaigns redeem about twice as often, so pooling lists lets campaign type crowd the top.
- **Forecasts:**
  - Sum the probabilities for the expected count, and use the Evaluation Agent's forecast ranges, not a single number.
  - Campaigns, mostly Type B, swing well above and below forecast, so the ranges are wide on purpose. They contained the actual count for 17 of 18 test campaigns, including all 5 held-back ones.
  - If `lean_logistic_log` is adopted, its ranges need to be about 1.4 times wider. Weighting recent campaigns more brings its totals back in line (decision 3).
  - Recalibrating on past errors was tested earlier and made forecasts worse.
- **Explanations:** the baseline's top driver is coupon use. If the lean logistic is adopted, its weights are easy to show, but the coupon columns move together. Read the coupon group as a whole, not each weight alone.

## Decisions on the open questions (10/24)

Evidence: `python src/modeling/model1_importance.py --open-questions` (`open_questions.md`).

### 1. Main measure: redeemers caught, averaged over mailing depths 5–50%

- **It matches how Model 1 is used.** In the Decision Lab the marketer picks how deep to mail, so the whole top half of each list matters, not only the top 20%.
- **It's about twice as steady.** On the open campaigns, the finalists' gaps over the baseline are 4.8 to 6.0 times their spread with the averaged measure, against 2.7 to 3.3 with the top 20% alone. On the held-back campaigns, one redeemer is worth about 1.3 points at 20%.
- **AUC is just as steady but isn't in the business's units,** so the Modeling Agent keeps choosing by it and the Evaluation Agent judges by the averaged measure.
- **Applied:** in `evaluation_spec.yaml`, Model 1's main measure is now `within_campaign_capture_avg`. The top 20% and AUC are still reported, and the team sign-off question now asks about this choice.
- **On the open campaigns:**

  | | Averaged redeemers caught |
  |---|---|
  | Baseline | 61.3% |
  | Finalists | 63.8–63.9% (gaps +2.5 to +2.6, ranges +1.2 to +3.4) |
  | "Redeemed before" rule | 50.2% |
  | Random mailing | 27.5% |

- **It does not change today's decision.** It was chosen after the held-back check, and on those campaigns the averaged gain is also inconclusive (+0.8 to +1.1, ranges spanning zero). Switching now on a measure picked afterwards wouldn't be a fair test, so the baseline stays and new campaigns decide.

### 2. Demographics: the labels stay out of the model in use

- **They add nothing reliable.** They add nothing to the logistic model, and only +0.3 AUC points to the forest.
- **What matters is having a record, not what it says.** Households with a record use about four times as many coupons (median 0.15 vs. 0.04 a week) and still redeem more at the same baseline score: 10.0% vs. 4.6% in the middle fifth. The `has_demographics` flag says whether a record exists, not anyone's age or income, so it stays in.
- **The labels don't change who gets mailed.** Every model's top 20% has the same age mix within about one point, with or without demographic inputs.
- **Every model under-selects the youngest age group:** 2.6–3.0% of the top 20% against 6.4% of everyone mailed, because of their shopping behavior. The Decision Lab should show who is on each list.
- **Applied:** the ready alternative, `lean_logistic_log`, uses no demographic labels. 2A still builds them, for reporting and checking list mix. TD may have its own policy on age and income in targeting, which the mentors can confirm.

### 3. What else could help

- **Weighting recent campaigns more** (half-life 120 days) brings the lean model's open-campaign total from 652 to 701, against 707 actual, without changing its ranking (AUC 0.828 either way). Use it for the Decision Lab's forecasts if the lean model is adopted. It was found on the open campaigns, so check it on new ones first.
- **Tried, didn't help:** extra Type A interaction terms (AUC 0.824 vs. 0.828), and training only on the 6 latest campaigns (AUC 0.823).
- **Already shown not to help:** other kinds of model (section 4) and separate models by campaign type.
- **What would actually move Model 1 further is new data:** which of a Type A campaign's roughly 200 coupons a household sees, and each coupon's value. Neither is in the dataset.

### Still for the team

1. Sign off on the main measure. The Evaluation Agent asks every run.
2. Confirm TD's view on using demographic labels for targeting.

## Reproduce

```bash
python src/features/build_2a.py --causal data/raw/causal_data.csv   # 2A, incl. the 18 Model 1 columns
python src/features/build_2b.py
python src/modeling/model1_evaluation.py         # baseline validation (PR #11)
python src/modeling/model1_feature_search.py     # feature-group search (PR #29)
python src/modeling/model1_importance.py         # which columns matter, other models, robustness (this report)
python src/modeling/model1_importance.py --open-questions   # evidence for the decisions on the open questions
python src/modeling/agent.py --model 1 --mode forward
python src/evaluation/agent.py --model 1         # open campaigns
python src/evaluation/agent.py --model 1 --unlock   # held-back check (already done once; a repeat is flagged)
```

The non-merchandise comparison needs 2A rebuilt with `--exclude-non-merchandise` into a second folder, then `model1_importance.py --compare-processed <that folder>`.
