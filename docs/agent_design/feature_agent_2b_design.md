# Feature Agent, Step 2B: Design

## What 2B is

2A builds one shared table (75,000 rows: 2,500 households x 30 campaigns) with every model's
candidate input columns and every model's answer column side by side. 2B's only job is to turn
that into one clean table per model: the right **rows**, and the right **answer column** -- nothing
else changes. The exact rule for each model lives in `src/features/feature_2b_spec.yaml`; the code
reads that file, so the two cannot disagree.

2B does **not** pick which context/profile columns a model should actually use. All 33 are carried
into every model's table as candidates; narrowing that down is the Modeling Agent's job (step 3),
same as 2A's design doc says.

## Why each model's rows are what they are

- **Model 1 (Response):** every mailed household x campaign pair, all 30 campaigns, 7,208 rows.
  There's no ambiguity here -- this is the full, original population from the proposal.

- **Model 2 (Expected Value):** redeemed, mailed, **TypeB/C only** -- 254 rows. The proposal's
  population is "pairs that redeemed," but the outcome column (`campaign_product_spend_during`)
  is only a clean, comparable number for TypeB/C, where a household's eligible products are the
  entire campaign's coupon set. For TypeA, the eligible-product list is the full pool of ~200
  products the 16 coupons were drawn from, not the 16 a household actually got -- so the same
  column measures something less precise for TypeA. Rather than averaging a precise number with
  a noisier one, the default table keeps the 254 clean TypeB/C rows; the 2A table still carries a
  computed TypeA value for anyone who wants to score it as a separate, documented exercise.

- **Model 3 (Uplift):** TypeB/C only, Treatment (mailed) + clean Control (not mailed this campaign
  or any overlapping one), 43,396 rows -- 3,229 treated, 40,167 control. Two decisions here, both
  already validated against the real data before this table was built:
  - **TypeA is out of scope**, for the same reason as Model 2: without knowing which 16 of
    ~17,000-35,000 pool products a household actually got, "did they buy an eligible product"
    is close to "did they shop at all." An earlier trial run confirmed this concretely: using a
    TypeA campaign's full pool produced a 96%+ false-positive "conversion" rate.
  - **"Clean" control excludes overlap-contaminated rows.** A household not mailed *this* campaign
    but mailed a different campaign running at the same time isn't a fair untreated control --
    74 of 435 campaign pairs overlap in time, so this isn't a rare edge case. 19,104 of the 62,500
    TypeB/C rows are excluded for this reason (`mailed_overlapping_campaign == 1`), leaving 43,396.

## Redemption-segmentation refinement (mentor feedback, 10/9 meeting)

Model 3's main outcome (`bought_campaign_product_excl_display_flyer`) counts any qualifying
purchase as a campaign "success" -- including a household that bought the product without ever
using the coupon, who would plausibly have bought it anyway. The mentor asked us to check this
directly instead of only inferring it from spend levels. `bought_campaign_product_without_redemption`
(now Model 3's second sensitivity label, alongside the display-only variant) answers it: for
treated households, 1 if they bought the product but that purchase wasn't tied to a coupon they
actually redeemed for this campaign.

**Real result:** of the 1,596 treated TypeB/C households who bought an eligible product, only 189
(11.8%) did so via an actually-redeemed coupon -- the other 1,407 (88.2%) bought it without ever
touching the coupon. This is a sharper, more direct version of the "sure things vs. persuadables"
finding from the T-learner baseline: most of what the main outcome currently counts as treatment
success isn't coupon-driven at all. Always 0 for control households by construction (nothing to
redeem), so it only adds information for treated rows.

## The leakage rail 2B owns

2A tags every column's role (`context`, `profile`, `outcome`, `filter`, `flag`) so a reader knows
what's safe. 2B enforces it mechanically: each model's table only ever contains that model's own
label (and, for Model 3, the `mailed_flag` treatment/split column, which is never fed to either
Model T or Model C as an input -- it only decides which rows train which of the two). Every other
model's outcome and filter columns are dropped entirely, not just tagged -- a model can't
accidentally train on an answer column that isn't even in its table. `build_2b.py` checks this on
every run and stops if a label ever ends up inside the feature list.

## How we know it's done

- Row counts match the project's known numbers exactly: 7,208 / 889 (Model 1), 254 (Model 2),
  43,396 = 3,229 + 40,167 (Model 3) -- verified against the real data, not assumed.
- Every table is unique on `(household_key, campaign)`.
- No model's table contains another model's label or filter column.
- Re-running on the same 2A output gives byte-identical tables.
