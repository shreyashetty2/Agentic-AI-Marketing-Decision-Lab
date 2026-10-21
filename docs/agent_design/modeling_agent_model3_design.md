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

## Confounder handling as a feature, not a filter

The main outcome excludes any purchase that happened during a display/flyer week, so a household
can buy the product and still be counted as "didn't buy" if every one of their purchases happened
to coincide with in-store promotion. That's not fully satisfying: it shrinks the usable sample, and
display timing isn't random, so it may be removing a specific kind of purchase rather than a random
slice.

`campaign_promo_share` (Feature Agent 2A, `src/features/outcomes.py`) is the alternative tried
here: share of a campaign's own weeks where at least one eligible product was on display or in the
flyer, at any store. It depends only on the campaign's product list (coupon.csv) and the promo
calendar, never on any household's purchases, so it's known before mailing and safe as a model
input -- registered under a new `context` role (`src/features/feature_2a_spec.yaml`,
`src/features/spec.py`), since it's computed by 2A but, unlike the existing `outcome` columns, is
not household-specific or after-the-fact.

Tested as `confound_aware_t_forest` (owner features + this one column) against the owner's model via
the Modeling + Evaluation Agents: +0.0039 ipw_stratified_qini, 95% range -0.0031 to +0.0101 --
inconclusive, a small nudge in the right direction but not yet a proven gain. Kept as a registered
candidate rather than adopted, so it's tracked and re-checked as more campaigns finish.

## The "sure things" classifier (`model3_product_lookup.py`'s sibling: `model3_sure_things.py`)

Not a second T-learner: control households can never redeem a coupon they never received, so
`bought_campaign_product_without_redemption` never varies for them, and a classifier can't learn
from a label with zero variance there. Instead, a single classifier trained only on treated
households, predicting the same label, directly flags likely wasted sends among people already
being mailed -- usable as a targeting deprioritization signal independent of the uplift ranking.
Held-out AUC 0.675; the top 20% most confident predictions are right 72.9% of the time, against a
53.8% base rate (1.4x).

## Closing the train/evaluate gap, and three causal-inference candidates

The Evaluation Agent judges this model on a propensity-weighted, overlap-restricted Qini
(`ipw_stratified_qini`), but `train_t_learner()` fits on the raw data with no weights at all -- a
train/evaluate mismatch. Four candidates close it or try to do better, all registered in
`modeling_spec.yaml` and judged by the exact same Modeling + Evaluation Agent machinery as every
other candidate (no separate comparison hand-rolled):

- **`owner_t_forest_weighted`** (`train_t_learner_weighted()`): the same T-learner, trained with the
  evaluator's own inverse-propensity weights, restricted to the overlap region. Result: +0.0027 vs.
  the owner's model on open test campaigns (range -0.0168 to +0.0231) -- inconclusive. Aligning
  training with evaluation didn't move the needle much by itself.
- **`owner_x_learner`** (`train_x_learner()`): X-learner (Kunzel et al. 2019), built for this
  dataset's ~12:1 treated/control imbalance, where a plain T-learner's two models are fit on wildly
  different sample sizes. Result: +0.0187 on open campaigns (range +0.0057 to +0.0345) -- the only
  candidate all along with a positive own-score (0.0077). On the locked campaigns: +0.0103 (range
  -0.0178 to +0.0422) -- still positive, but no longer statistically clean. Most promising result to
  date; not yet adopted.
- **`owner_dr_learner`** (`train_dr_learner()`): doubly-robust (AIPW) learner -- one regressor fit
  directly on a pseudo-outcome that stays unbiased if either the outcome models or the propensity
  model is right, not both. Result: -0.0130 on open campaigns, own score -0.0241 (worse than the
  owner's model) -- the AIPW pseudo-outcome's high variance with a treated group this small likely
  outweighs its robustness guarantee here. Not pursued further.
- **E-value** (`src/evaluation/metrics.py: e_value()`): VanderWeele & Ding (2017) -- how strong an
  unmeasured confounder's association with both mailing and buying would need to be, on the
  risk-ratio scale, to fully explain away the observed effect. Reported alongside Model 3's existing
  calibration check. Current value: **1.35** -- a fairly weak hidden confounder could fully explain
  the result away, a concrete number behind `approve_requires_human: true`'s existing "no rule can
  check for hidden confounders" caveat.

## How to run it

```bash
python src/modeling/model3_uplift.py --processed data/processed
python src/modeling/model3_product_lookup.py --top 20   # all campaigns, ranked, saved to CSV
python src/modeling/model3_sure_things.py --processed data/processed
```

Runtime: ~6s for the uplift model, ~3s for the all-campaigns product scan, ~2s for the sure-things
classifier. Writes `data/processed/model3_uplift_report.md`,
`reports/figures/model3_uplift_qini_curve.png`, `data/processed/model3_product_redemption_summary.csv`,
and `data/processed/model3_sure_things_report.md`.
