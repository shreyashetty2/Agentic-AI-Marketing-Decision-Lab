# Model 2: final report

**Question:** when a household redeems a campaign coupon, how much will it spend on that campaign's products during the campaign?
**Use:** the dollar side of the Decision Lab. Expected revenue per mailed household = P(redeem) from Model 1 × expected spend from Model 2.
**Status (11/5):** development wrapped up. The model in use is a straight line on usual spend, set as the agents' default. The held-back check is done and its results are below. Two questions remain open, listed at the end.

## Decision

- **The model in use is a least-squares line on one column, usual spend** (`LinearUsualSpend` in `src/modeling/model2_value.py`; `owner_usual_linear` in `modeling_spec.yaml`).
  - Fit on all 254 rows: **expected spend = $9.68 + 0.70 × usual spend**.
  - Usual spend = the household's weekly spend on the campaign's products over the 26 weeks before the cutoff × the number of campaign weeks.
- **Why a line on usual spend:**
  - The mentors agreed (10/23) that Model 2 should stay a simple model on usual spend, given 254 rows.
  - Every larger model we tried did no better and often worse (29 alternatives in October, 10 more candidates in PR #30, feature groups added one at a time).
- **Why least squares instead of the median line from 10/15:**
  - The median line (0.79 × usual spend + $6) has a slightly lower average error. But its predictions add up to only **0.78–0.80** of actual spend, because spend is right-skewed and a median line aims at the typical household, not the total.
  - The Decision Lab adds predictions across households, so totals matter. The least-squares line adds up to **0.92–0.95** of actual, inside the Evaluation Agent's ±15% check.
  - The error difference between the two lines is small and not reliable: about $0.39 per household in the median line's favor on the open campaigns, and $0.64 in the least-squares line's favor on the held-back ones. Neither range excludes zero.
- **What else is on file:** the median line stays in the code (`CalibratedUsualSpend`), along with a rescaled version of it (`TotalsCalibratedUsualSpend`) and a Tweedie model. None of them is the model in use.

## How we tested

- **Data:** 254 redeemed household × campaign rows (TypeB/C only), 161 households, 25 campaigns. Median spend $12.73, mean $18.42. Every column uses only data from before the campaign's cutoff (start day − 7).
- **Forward chaining:** each test campaign is predicted by a model trained only on earlier campaigns.
- **Three views:**
  - `model2_value.py` walk-forward: 194 rows from 14 campaigns.
  - The Evaluation Agent on the open campaigns: 113 rows.
  - The held-back check on campaigns 23, 24 and 25 (54 rows). These were read once, today, after the model was chosen.
- **Measures:**
  - average dollar error (MAE);
  - predicted ÷ actual total (should be within 0.85–1.15);
  - the gap against usual spend, with a 95% range from resampling whole households or campaigns.
- **Decision rules (set before the held-back check):**
  - totals within ±15%;
  - no worse than usual spend;
  - an alternative replaces the model in use only if it is reliably better by at least $0.50.

## What we found

### 1. The line beats usual spend, but not reliably

| View | Line on usual spend | Usual spend | Median line (10/15) |
|---|---|---|---|
| Walk-forward, MAE | $9.81 | $10.40 | $9.72 |
| Walk-forward, predicted ÷ actual total | 0.92 | 0.68 | 0.78 |
| 70/30 split, MAE | $10.86 | $11.68 | $11.62 |
| Evaluation Agent (open), predicted ÷ actual total | 0.95 | — | 0.80 |

- The walk-forward gap against usual spend is $0.59 per household (95% range −$1.41 to +$0.31). The line is better in 91% of resamples.
- On the Evaluation Agent's open campaigns, the gap is +$0.50 in the line's favor (range −$0.46 to +$1.62).
- With 113 test rows, gaps under about $1 can't be told apart from noise.

### 2. The agents' verdict

- With the line as the default, the Evaluation Agent's verdict moved from "send back to modeling" (the median line failed the totals check) to **"human needed"**.
- The totals check passes, and the gap against usual spend is positive but inconclusive. The agent has no rule for that combination, so a person decides.
- The decision is to keep the line, for the reasons above. The Model 2 override in `models_in_use.yaml` was removed, so the agents use the spec default.

### 3. Household-specific promo share: re-tested, still no help

- The promo-share column is now computed per household. Added to usual spend, it is **reliably worse** on the open campaigns: −$0.47 (range −$0.92 to −$0.09).
- On the held-back campaigns it changes nothing (−$0.0004).
- The reason: in Model 2's rows the column has only 13 distinct values, and 84% of them are 1.0. Almost every redeeming household buys these products on promotion, so the column barely varies.

### 4. More columns still don't help

- Coupon history, campaign details, offer fit, demographics and the feature groups that helped Model 1 were each added on top of usual spend.
- None was reliably better. Several were reliably worse on the open campaigns, for example the random forest on all columns (−$1.04).
- With 60 to 220 training rows per test, extra columns mostly fit noise.

### 5. The held-back check (campaigns 23–25, 54 rows, read once)

- **Line vs. usual spend:** +$1.24 in the line's favor (range −$0.59 to +$2.98). This is the same direction as on the open campaigns and larger, but still inconclusive.
- Approximate levels (fit on all open rows):

  | Model | MAE | Predicted ÷ actual total |
  |---|---|---|
  | Line on usual spend | $12.50 | 0.76 |
  | Median line | $12.90 | 0.70 |
  | Usual spend | $13.69 | 0.71 |

- **No alternative is reliably better.** Two lean better, but their ranges include zero: the rescaled median line (+$0.64, range −$0.83 to +$2.24) and campaign details (+$0.02). One alternative, the shallower forest, is reliably worse (−$3.94).
- The Modeling Agent's own pick from the open campaigns is also `owner_usual_linear`.
- **Totals fall short on the held-back campaigns (0.76).** These later campaigns spend more than the earlier ones the model learned from. The line holds up best of the three, but no pre-campaign column we tried explains the jump.

## Hand-off to the Decision Lab

- **Per household:** expected revenue = P(redeem) (Model 1) × ($9.68 + 0.70 × usual spend).
- **Where to get it:** `LinearUsualSpend().fit(table).predict(table)` in `src/modeling/model2_value.py`, or the Modeling Agent's default for Model 2.
- **How much to trust it:**
  - Use it for ranking households and for campaign totals, not for single-household dollar amounts. Typical error is about $10 per household on a median of $12.73.
  - For recent campaigns, totals may run about 5–25% low. Show a range rather than a point estimate.
- **Scope:** TypeB/C only. TypeA has no Model 2 estimate yet (see open question 1).

## Open questions

1. **A broader target that includes TypeA:** spend on the products of the coupons a household actually redeemed (254 → 889 rows). It is a narrower question, and TypeA spends about 4× more. This is a side experiment; it is not planned to replace the current target.
2. **Why later campaigns spend more:** the held-back campaigns under-predict by about a quarter. Product mix and pricing are the likeliest reasons, but neither is checked yet.

## Reproduce

```bash
python src/modeling/model2_value.py --processed data/processed        # the model in use + report
python src/evaluation/agent.py --model 2                               # agent verdict (open campaigns)
python -m pytest tests/modeling -q
```

The held-back check (`python src/evaluation/agent.py --model 2 --unlock`) was run once on 11/5. Its report is in `data/processed/evaluation_agent/model2/unlock_report.md`, and further runs are counted in `unlock_history.json`.
