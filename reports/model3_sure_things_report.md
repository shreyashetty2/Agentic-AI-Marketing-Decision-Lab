# Model 3: "sure things" classifier

Trained only on treated (mailed) households -- not a T-learner, since control households can never
redeem a coupon they never received, so bought_campaign_product_without_redemption never varies for
them. Predicts, among households already being mailed, which ones will buy the eligible product
without ever redeeming the coupon: a direct, actionable flag for households the mailer is wasted on,
separate from the uplift model's ranking.

## Setup

- Train: 1,842 treated rows (earliest 70% of campaigns). Test: 1,387
  treated rows (latest 30%) -- same out-of-time split as model3_uplift.py.
- RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20), natural class
  distribution, no class_weight rebalancing (same reasoning as the uplift model).

## Held-out performance

- AUC: 0.675
- Base rate (share of treated test households who buy without redeeming): 53.8%
- Among the top 20% most confidently predicted "sure things" (277 households):
  72.9% actually bought without redeeming -- 1.4x the base rate.

## What the model relies on most

- campaign_product_spend_per_week_26w: 0.285
- campaign_product_share_26w: 0.193
- duration_days: 0.178
- spend_per_week_8w: 0.084
- history_weeks_before_cutoff: 0.063

## How to use this

Feed the predicted probability into targeting as a deprioritization signal: a household with a high
score here is likely to buy regardless of the mailer, independent of how high their raw purchase
probability looks. This directly operationalizes the spend-tier and redemption findings in
model3_uplift.py and model3_redemption_analysis.py -- heavy spenders score high here precisely
because they're the "sure things" those findings already identified.
