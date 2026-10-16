# Modeling Agent, Model 2: Design

## What this is

`src/modeling/model2_value.py` is the primary Model 2 build, superseding `model2_baseline.py` (which
stays in the repo as the comparison it was built to be). For a household that redeems a TypeB/C
campaign, it predicts the household's spend on that campaign's products during the campaign window
(`campaign_product_spend_during`, Feature Agent 2A/2B).

The chosen model is deliberately small -- a one-feature median regression on "usual spend":

**spend = 0.79 x usual spend + $6.13** (fit on all 254 rows)

where usual spend = the household's average weekly spend on this campaign's products over the 26
weeks before the cutoff (`campaign_product_spend_per_week_26w`), times the campaign's length in
weeks. This is the README's own "usual spend" benchmark, calibrated rather than replaced.

This document records why: every larger model tried failed to beat usual spend reliably, and this
is the only candidate that does. The full list is reproducible with
`notebooks/yuxin_01_model2_tuning_experiments.py`.

## Scope: 254 rows, TypeB/C only

Unchanged from Feature Agent 2B: redeemed, mailed TypeB/C pairs only. In TypeA campaigns each
household receives 16 coupons out of a ~200-product pool and which 16 is not recorded, so "spend on
campaign products" can't be measured the same way (see `feature_agent_2b_design.md`). TypeA remains
an open question for Model 2.

## What the data looks like

Three properties drive every result below:

- **Small.** 254 rows across 25 campaigns; the 70/30 split trains on 126 rows.
- **Heavily skewed.** Median spend $12.74, mean $18.42, max $140.44; 14 redeemers spent $0 on
  campaign products in the window. A few large spenders dominate squared-error fits.
- **A third of the variation is between campaigns.** 32% of the variance in spend is explained by
  which campaign it is (e.g. campaign 24 averages $60, most campaigns $10-20), and later campaigns
  spend more than earlier ones. With only 25 campaigns, there is very little to learn that from.

## How it's evaluated

Metric: mean absolute dollar error (README evaluation table: "average dollar error, vs. usual
spend"), plus median absolute error and Spearman rank correlation.

1. **70/30 out-of-time split**, identical to every `model{n}_baseline.py`: earliest 17 campaigns
   train (126 rows), latest 8 test (128 rows). Kept for comparability, but 128 test rows from 8
   campaigns is too few to separate candidates -- the six best land within $0.80 of each other.
2. **Walk-forward (primary).** Each campaign is predicted by a model trained only on campaigns that
   started before it, once at least 60 rows of earlier history exist. This pools 194 predictions
   from 14 campaigns, every one of them strictly out-of-time. The difference against usual spend is
   tested with a campaign-level bootstrap (2,000 resamples of campaigns, not rows, because rows
   within a campaign share that campaign's spend level).

Both schemes order campaigns by `start_day`, the same convention as every `time_split()` in the
repo. Campaign windows overlap (74 of 435 pairs), so a training campaign can still be running when a
test campaign starts; neither scheme excludes that, consistent with the existing splits.

## Results

Walk-forward (194 rows, 14 campaigns):

| Model | Mean abs. error | Median abs. error | Rank corr. |
|---|---|---|---|
| **Calibrated usual spend (chosen)** | **$9.72** | $6.22 | 0.51 |
| Usual spend (benchmark) | $10.40 | $6.19 | 0.54 |
| Huber regression, baseline features + usual spend | $9.79 | $6.68 | 0.49 |
| Linear regression (`model2_baseline.py`) | $10.35 | $7.72 | 0.48 |
| Random forest (`model2_baseline.py`) | $10.49 | $7.35 | 0.48 |
| Training-set mean | $12.55 | $9.00 | -0.05 |

Calibrated usual spend vs. usual spend: **-$0.68** (95% campaign-bootstrap CI -$1.18 to -$0.19;
better in 100% of resamples). It is the only candidate whose improvement over usual spend holds up:
the next best, Huber regression (-$0.60), is better in only 88% of resamples (CI crosses zero), and
both baselines are indistinguishable from usual spend (51% and 45%).

On the 70/30 split the chosen model is effectively tied with usual spend ($11.62 vs. $11.68) and the
baseline linear regression ($11.64) -- consistent with that split being too small to separate them,
not evidence against the walk-forward result.

## Why the chosen model is shaped this way

- **One feature.** Two parameters are all that roughly 60-220 training rows support reliably; every
  multi-feature model learned noise that didn't carry to later campaigns.
- **Median, not least squares.** Spend is right-skewed and the metric is absolute error, which a
  median fit targets directly; a least-squares fit gets pulled toward the few $100+ spenders.
- **What the two numbers mean.** The +$6 intercept: redeeming a coupon means buying at least one
  campaign product, so even households that usually buy almost none of them spend something (the
  lowest walk-forward quintile usually spends $2 but actually spends $10). The 0.79 slope: households
  that usually buy a lot don't scale that up proportionally during the campaign (regression to the
  mean). Both are easy to explain to a business user, which matters for the Decision Lab.

## What was tried and why it didn't help

All numbers are walk-forward mean absolute error (usual spend = $10.40), from
`notebooks/yuxin_01_model2_tuning_experiments.py`.

1. **Less conservative trees.** `model2_baseline.py` copied Model 3's `min_samples_leaf=20`, built
   for 43,396 rows. Lowering it to 5 made the forest worse ($10.95), not better: smaller leaves on
   ~150 rows fit noise. Adding usual spend as an explicit input didn't change that ($11.01).
2. **Log target** (fit log(1 + spend), convert back). Meant to tame the skew. No gain in walk-forward
   (random forest $10.22-10.42, ridge $10.55, linear $11.58) and clearly worse on the 70/30 split
   ($14.57-15.88): converting back from logs predicts something closer to a median, which
   systematically under-predicts the later, higher-spending test campaigns.
3. **Absolute-error loss in trees** (random forest $9.91, gradient boosting $10.14, LightGBM L1
   $10.15). Helps the forest in walk-forward, but none beats calibrated usual spend, and all are
   worse on the 70/30 split ($13.12-13.62).
4. **Predict the ratio to usual spend** instead of spend itself (random forest $10.28, ridge
   $10.69). No better than usual spend.
5. **More household features** (8-week spend and trips, basket value, recency, departments, past
   redemption counts, overlapping campaigns). Worse: random forest $10.68, LightGBM $11.58.
6. **Campaign-level features**, since a third of the variation is between campaigns. Built only from
   what is known before mailing (cutoff = start_day - 7): number of products and coupons in the
   campaign, and how popular those products were across all 2,500 households in the 26 weeks before
   cutoff. Unstable: Huber with these features was the best model on the 70/30 split ($11.05) but
   worse in walk-forward ($10.18), and least-squares linear degraded badly ($13.68). With 25
   campaigns there are effectively 15-25 data points for any campaign-level effect.
7. **Robust linear models with usual spend** (Huber $9.79, least-squares $10.33). Huber comes close
   but its gain over usual spend isn't reliable (see Results), and it needs 11 features to do what
   the chosen model does with one.

## Known limitations

- **Later campaigns are under-predicted.** Every model, including the chosen one, under-predicts
  each walk-forward quintile by $2-9 (top quintile: $31 predicted vs. $40 actual), because later
  campaigns spend more and no pre-campaign feature tried here captures why.
- **The slope is not stable yet.** Fit on the 70/30 training split it is 0.53 (intercept $5.29);
  on all 254 rows, 0.79. Expected with this little data -- worth re-checking as data grows.
- **TypeA is still out of scope.** See Scope.
- **The sensitivity label** (`redeemed_coupon_product_spend_during`, spend on only the products
  actually redeemed) has not been modeled yet.

## How to run it

```bash
python src/modeling/model2_value.py --processed data/processed
python notebooks/yuxin_01_model2_tuning_experiments.py --processed data/processed   # every alternative, ~35s
```

Runtime: ~3s for the model. Writes `data/processed/model2_value_report.md`. The experiment script
also needs `transaction_data.parquet` and `coupon.parquet` from `ingest.py` (for the campaign-level
features).
