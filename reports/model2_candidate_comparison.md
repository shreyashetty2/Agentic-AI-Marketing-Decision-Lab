# Model 2: totals candidate comparison

Why this exists: the Evaluation Agent sent Model 2 back to modeling (EVA-06) because the owner's model,
`owner_calibrated` (median line on usual spend, `model2_value.py`), predicts totals at 0.80 of actual
spend -- outside the +/-15% the Decision Lab needs to weigh sales against discount costs. On 2026-10-20
the model in use was switched to `usual_only_linear` (an ordinary least-squares line on usual spend,
totals 0.95) in `src/lead/models_in_use.yaml`. This round tests whether anything beats that.

Every candidate is judged by the same Modeling + Evaluation Agent machinery, on the 8 open forward-chaining
test campaigns (113 rows, 90 households). The 3 locked campaigns (23, 24, 25; 54 rows) were **not** read.

## Rules, fixed before looking at results

1. Predicted total within 15% of actual total (the Evaluation Agent's calibration check).
2. Beats usual spend on average dollar error.
3. Beats the model in use (`usual_only_linear`) by at least $0.50 average error (`selection.min_gain` in
   `modeling_spec.yaml`); otherwise keep the model in use.

Only a candidate passing all three goes to the one-time locked check.

## Results

Gap = model in use's error minus the candidate's (positive = candidate better); 95% range from resampling
whole households (2,000 resamples), as the Evaluation Agent does.

| Candidate | What changed | Avg error | Total / actual | Gap vs. in use (95% range) | Rule 1 | Rule 3 |
|---|---|---|---|---|---|---|
| `owner_linear` | 10 columns, least squares (`model2_baseline.py`) | $8.25 | 0.82 | +0.46 (-0.68 to +1.56) | fail | fail |
| `owner_calibrated` | median line on usual spend (owner's default) | $8.32 | 0.80 | +0.39 (-0.15 to +0.90) | fail | fail |
| `usual_plus_campaign_linear` (new) | usual spend + duration, type B | $8.35 | 0.84 | +0.36 (-0.31 to +1.01) | fail | fail |
| **`owner_calibrated_totals`** (new) | **median line, rescaled to training totals** | **$8.42** | **0.92** | **+0.29 (-0.21 to +0.79)** | **pass** | **fail** |
| `usual_only_linear` (**in use**) | least-squares line on usual spend | $8.71 | 0.95 | -- | pass | -- |
| `usual_plus_coupon_linear` (new) | usual spend + coupon use, share on discount | $8.75 | 0.94 | -0.04 (-0.72 to +0.57) | pass | fail |
| `owner_tweedie` (new) | Tweedie (power 1.5, log link) on usual spend | $8.83 | 0.96 | -0.12 (-0.60 to +0.35) | pass | fail |
| `owner_forest` | 10 columns, forest | $9.75 | 0.90 | -1.04 (-2.41 to -0.02) | pass | fail |
| `deeper_forest` | 10 columns, deeper forest | $9.87 | 0.91 | -1.16 (-2.58 to -0.12) | pass | fail |
| `shallower_forest` | 10 columns, shallower forest | $10.28 | 0.96 | -1.57 (-3.06 to -0.38) | pass | fail |
| Usual spend (rule) | -- | $9.22 | **0.61** | -- | -- | -- |

## Decision: keep `usual_only_linear`

No candidate passes all three rules, so the model in use stays and the locked campaigns stay unread.

- **`owner_calibrated_totals`** comes closest: it fixes the totals (0.80 to 0.92) while keeping most of
  the median line's per-household accuracy, but its $0.29 edge is below the $0.50 bar and its range
  includes zero. Kept as a tracked candidate.
- **`owner_tweedie`** gets totals right (0.96) but is no more accurate than the line in use.
- **Feature groups** (mentor follow-up): adding coupon habit changes nothing (-$0.04); adding campaign
  information lowers error but pulls totals down to 0.84, failing rule 1.

## What this tells us

- **The totals problem is real and large.** Usual spend alone adds up to only 61% of what redeeming
  households actually spend; any model that stays close to it, or fits the median, under-predicts the
  total. Fitting the mean (least squares, Tweedie) or rescaling fixes it.
- **The test set cannot separate close candidates.** With 113 rows, even the model in use's edge over plain
  usual spend (+$0.50) has a range that includes zero. Differences under about $1 are not distinguishable
  here; the data size, not the modeling, is the limit.
