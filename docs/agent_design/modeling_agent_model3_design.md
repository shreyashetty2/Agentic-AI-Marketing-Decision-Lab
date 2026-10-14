# Modeling Agent, Model 3: Design

## What this is

`src/modeling/model3_uplift.py` is the primary Model 3 build, superseding
`model3_baseline.py`. It trains a T-learner on the full `model3_table.parquet` (Feature Agent 2B)
and directly answers three questions: does mailing a campaign add measurable lift, does that lift
vary by household spend level, and how much of the outcome Model 3 counts as a "success" is
actually coupon-driven.

## Full dataset, not a sample

Train and evaluate on all 43,396 rows (3,229 treated, 40,167 control) -- no subsampling at any
stage. The earlier baseline subsampled the control group to 5x the treated count when training
Model C, purely for training speed; at this dataset size that tradeoff isn't necessary, and
discarding ~30,000 real control rows was discarding real information.

## Why not `class_weight='balanced'`

The natural first instinct for training Model C on a 13.8%-positive-rate dataset is to rebalance
the classes. This was tried and rejected: it badly distorts `predict_proba`'s calibration.
Verified directly -- with `class_weight='balanced'`, Model C's average predicted probability on
held-out data was 41.0%, vs. a true positive rate of 13.6%, a 3x inflation. A T-learner's whole
design depends on `P(buy | treated) - P(buy | control)` being two genuinely calibrated
probabilities; rebalancing breaks that by construction, even though it can help pure
classification ranking. With ~5,500 positive examples in the control group alone, there's no need
for it -- the natural class distribution has plenty of signal to fit a well-calibrated model
directly. (`train_t_learner()` in the code carries this reasoning as well, so it doesn't get
silently reintroduced later.)

## Household historical spend

`spend_per_week_8w` and `spend_per_week_26w` (Feature Agent 2A) are both model inputs and the
basis of the spend-tier segmentation below -- central to this build, not incidental.

## Three questions, answered directly

**1. Does targeting add lift?** A two-proportion z-test on the full dataset (naive, unadjusted):
treated rate 49.4% vs. control rate 13.8%, z=52.5, p≈0. Not noise -- but this number mixes the
campaign's real effect with the fact that heavier spenders were targeted in the first place, which
is exactly what question 2 investigates.

**2. Does lift vary by household spend level?** Households are bucketed into thirds by
`spend_per_week_26w`, using cutoffs taken from the *treated group's own* distribution. Cutoffs
from the full population were tried first and rejected: because treated households are
overwhelmingly heavy spenders, population-wide terciles put ~95% of treated households in a
single "High" bucket, making the other two buckets almost empty of treated households (5 and 157
out of 3,229) and the comparison meaningless. Treated-group cutoffs guarantee ~1,076 treated
households per bucket, a fair three-way split. Result: observed uplift falls monotonically as
spend rises -- Low ~26%, Medium ~14%, High ~9% -- and the trained model reproduces the same
ordering on held-out test campaigns (Low 0.102, Medium 0.049, High 0.006).

**3. How much of the outcome is actually coupon-driven?** `bought_campaign_product_without_redemption`
(Feature Agent 2B) answers this directly: of 1,596 treated households who bought an eligible
product, 1,407 (88.2%) did so without ever redeeming the coupon. This is a direct, observed
confirmation of (2), not an inference from spend level this time.

## Evaluation: what's solid, what isn't yet

The per-tier signal above (both observed and model-predicted) is clean and directionally
consistent. The formal held-out Qini coefficient is still negative, and the quintile Spearman
correlation isn't statistically significant on this split (p=0.39). This is a known,
already-diagnosed issue, not a new one: treated households cluster tightly in the high-spend
region of feature space, so treated and control populations barely overlap at the extremes of a
uplift ranking, which distorts the cumulative Qini calculation specifically (restricting to the
region where both groups actually overlap was shown earlier to recover a clean monotonic Qini
signal). The per-tier segmentation above sidesteps this because it compares like-for-like spend
ranges directly, rather than ranking the full population end to end.

## Does lift vary by demographics?

Added as a fourth segmentation, alongside spend tier. Partial-coverage view by construction: only
801 of 2,500 households have a demographic record (22% of Model 3's rows after TypeB/C + clean
control scoping), so this is read as a subsample finding, not a full-population one -- categories
with fewer than 20 treated households are dropped as too small to read anything into. By
homeowner status, number of kids, and age group: uplift is positive and broadly similar across
every category (roughly 19-42 percentage points), with no demographic split anywhere near as
sharp as the spend-tier one. That itself is informative: pre-campaign spend behavior is a much
stronger predictor of how responsive a household is to the mailer than who they are demographically.

## Product-level view (`model3_product_lookup.py`)

Model 3's main outcome is aggregated across a whole campaign's eligible products ("did they buy
*any* eligible product"). `src/modeling/model3_product_lookup.py` answers the finer-grained
version -- for one specific product, in one specific campaign (or coupon), what happened,
including the real product name (joined from `product.csv`). Run with no `--campaign` to scan all
25 TypeB/C campaigns at once and rank every eligible product by how often it's bought without
ever redeeming the coupon -- saves the full list (208 qualifying products, >=5 treated purchases)
to `data/processed/model3_product_redemption_summary.csv`. The top of that list is dominated by
everyday staples (eggs, pasta sauce, cheese, juice) bought without redemption 74-100% of the time
-- individual-product confirmation of the same pattern as the aggregate 88.2% finding, not a
separate result.

## How to run it

```bash
python src/modeling/model3_uplift.py --processed data/processed
python src/modeling/model3_product_lookup.py --top 20   # all campaigns, ranked, saved to CSV
```

Runtime: ~6s for the uplift model, ~3s for the all-campaigns product scan. Writes
`data/processed/model3_uplift_report.md`, `reports/figures/model3_uplift_qini_curve.png`, and
`data/processed/model3_product_redemption_summary.csv`.
