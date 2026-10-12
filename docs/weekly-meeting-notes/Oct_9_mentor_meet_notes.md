Here's a quick recap of what we covered in the mentor meeting on October 9th and our plan for next week (by John):

## Data Agent
- Ran pipeline on full dataset and built the master table (one row per household and campaign)
- Integrity checks: 23 passed, 7 flagged, 4 informational, 0 errors. Notable flags were duplicate coupon rows (5,164 removed and logged), positive retail discounts may have a sign error, quantities above 100.
- Agent labels who should decide each finding, never deletes anything unsure, and logs every action. The decision layer is rule-based for now.

## Feature Agent
- Step 2A adds pre-campaign features to the master table, giving one shared table for all three models.
- Step 2B splits it into one table per model (Model 1: 7,208 rows, Model 2: 254, Model 3: 43,396), with automatic check that stops the pipeline if a model's answer column leaks into inputs.
- Type A campaigns are excluded from Models 2 and 3, since we can't see which coupons each Type A household received.

## Baseline models
- Model 1: the random forest finds 61% of redeemers in the top 20% of households (logistic regression 58%), compared with 52% for "mail past redeemers" and 20% at random.
- Model 2: ranks households in the right order but doesn't yet beat "usual spend" on dollar error, with only 254 rows to learn from.
- Model 3: uplift modeling is feasible on Type B/C campaigns. Next step is testing whether current targeting actually adds lift.

## Model 1 evaluation
- Checks before we rely on the 61% (RF baseline): compare each campaign separately against a rule that also knows campaign type, test campaign by campaign over time, confirm the probabilities are calibrated for forecasting.

## YOUR FEEDBACK AND OPEN ITEMS
- Outliers: set cutoffs per field from the data (e.g., IQR method) rather than a fixed number like 100, and look into suspicious patterns (e.g., repeated values, positive discounts).
- Agent autonomy: clarify which issues the Data Agent should resolve on its own and which it should escalate, since most current flags are zeros and outliers.
- Heavy spenders: for households that spend 4-5x more, do they redeem because of the coupon or would they spend anyway? Look into as part of Model 3.
- Features: look into full feature list. To answer today's question, yes, includes each household's historical spend (weekly spend over the 26 weeks before each campaign). Clarify whether Feature Agent selects variables or builds them.

## PLAN FOR NEXT WEEK
- Data Agent: field-specific outlier rules, investigate positive discounts, start LLM layer to review flagged items
- Feature Agent: share the feature list with definitions and decide where feature selection happens
- Model 1: evaluation checks, LightGBM, new features
- Model 2: tuning given small sample
- Model 3: test whether targeting adds lift and explore heavy-spender question
