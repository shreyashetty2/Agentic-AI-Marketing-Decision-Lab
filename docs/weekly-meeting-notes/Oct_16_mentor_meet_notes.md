Data agent (outliers and anomalies)
- IQR does not work for quantity because the 25th, 50th and 75th percentiles are all 1. The extreme quantity and low unit price values were driven by gas and miscellaneous lines. After separating non-merchandise rows, the max quantity drops to 10 and max unit price to $75.
- We set rules for discount anomalies: a half-cent tolerance, flags for free items and zero lines, and sign typo fixes stored in a new column rather than overwriting the original.
- Decision: quantity outlier cutoffs now use percentiles instead of IQR.
- Open questions: whether gas and misc should count as household spend, and how to handle the zero-quantity rows. We also plan to use a default value (1 or the median) for missing quantities and to look into the 4,451 rows that have quantity but no sales.

Model 1 (redemption prediction)
- A random forest baseline beats the "redeemed before" rule by about nine points across campaign types A, B and C.
- Nearly all of the signal comes from coupon use per week; ranking on that feature alone matches the full model. LightGBM moved results by 1 to 2 points, which is within noise.
- We will keep demographics in the feature set even though they are only about 32% populated, and test feature-group iterations.

Model 2 (spend prediction)
- Target is total dollars spent per household per campaign. Only a one-feature median regression beat the usual-spend baseline (0.79 x usual spend + $6), bringing average error from $10.40 to $9.72, consistent across resampling.
- Random forest, LightGBM and a log transform did not help, most likely because the dataset is small (about 254 examples).

Model 3 (uplift and redemption analysis)
- A T-learner on the full data shows about 26% uplift for low spenders versus about 9% for high spenders, who currently receive most of the budget.
- The Qini curve is negative so far; we still need to decide on weighting or outlier removal.
- Only about 12% of mailed households used the coupon, and staples like bananas, eggs and cream cheese are bought almost entirely without coupons. Demographics cover only 22% of rows for the B and C campaigns, so they did not produce any major findings.

Modeling agent and architecture
- The modeling agent now orchestrates all models for a given prediction question, reports results, and suggests improvements in natural language.
- The feature engineering agent stays mechanical; feature selection and new feature ideas live in the modeling agent.

Plan for next week
- Investigate the missing-quantity and quantity-no-sales rows by campaign and product.
- Decide on the Qini weighting or outlier strategy for Model 3.
- Test features beyond coupon use for Model 1, including demographic groups.
- Plan how the agents interact with each other, how they fit into Decision Lab, and what the API and deployment story looks like for a prototype (possibly on GCP credits).
- Test whether the modeling agent's improvement suggestions are actually useful.
