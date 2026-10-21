# Model 3: full candidate comparison

Every approach tried, judged by the same Modeling + Evaluation Agent machinery, on the same metric
(`ipw_stratified_qini`: extra buyers per mailed household vs. random targeting, within campaign,
propensity-weighted). "Open" = the 15 regular test campaigns, reused across iteration. "Locked" =
campaigns 22, 23, 24, 25, held back as a one-time check (by the time of this report, checked 3
times total across this effort -- see the caveat at the end).

## Results

| Candidate | What changed | Open: gap vs. owner | Open status | Locked: gap vs. owner | Locked status |
|---|---|---|---|---|---|
| `owner_t_forest` (current, before this effort) | -- | -- | -- | -- | -- |
| `owner_t_logistic` | learner | +0.0139 | inconclusive | -0.0213 | inconclusive |
| `extended_t_forest` | columns | +0.0083 | pass | +0.0076 | inconclusive |
| `shallower_t_forest` | settings | +0.0165 | pass | -0.0057 | inconclusive |
| `confound_aware_t_forest` | columns (`campaign_promo_share`) | +0.0039 | inconclusive | -0.0037 | inconclusive |
| `owner_t_forest_weighted` | trained with IPW weights | +0.0027 | inconclusive | -0.0027 | inconclusive |
| `owner_t_forest_entropy_balanced` | trained with entropy-balance weights | +0.0042 | inconclusive | -0.0040 | inconclusive |
| **`owner_x_learner`** | **learner (X-learner)** | **+0.0187** | **pass** | **+0.0103** | inconclusive, but the only one still positive on both checks |
| `owner_x_learner_confound_aware` | learner + columns | +0.0223 | pass | +0.0068 | inconclusive (weaker on locked than plain X-learner) |
| `owner_dr_learner` | learner (DR/AIPW) | -0.0130 | inconclusive | +0.0042 | inconclusive |

Own score (`ipw_stratified_qini`, higher is better; 0 = exactly as good as mailing at random):
`owner_t_forest` -0.0111, `owner_t_logistic` +0.0028, `extended_t_forest` -0.0028, `shallower_t_forest`
+0.0055, `confound_aware_t_forest` -0.0072, `owner_t_forest_weighted` -0.0084,
`owner_t_forest_entropy_balanced` -0.0069, `owner_x_learner` +0.0077, `owner_x_learner_confound_aware`
+0.0112, `owner_dr_learner` -0.0241.

## What won, and why

**`owner_x_learner`** -- the only candidate that's positive on its own, real on the open campaigns,
and still directionally positive (not flipped negative) on the locked ones. It's built specifically
for this dataset's actual problem: a ~12:1 treated/control imbalance, where a plain T-learner's two
models are fit on wildly different sample sizes and their errors don't cancel. Adopted as Model 3's
model in use (`src/lead/models_in_use.yaml`), 2026-10-21.

**Why not the others:**
- `extended_t_forest` / `shallower_t_forest`: passed on open campaigns, but both failed to hold up on
  locked -- the two false leads that motivated checking everything else this carefully.
- `owner_t_forest_weighted` / `owner_t_forest_entropy_balanced`: both real, tested fixes for the
  train/evaluate mismatch and the measured-covariate balance gap respectively -- both gave small,
  not-statistically-significant nudges. Aligning training with evaluation, and fixing balance on the
  columns we can measure, aren't enough by themselves; the real gain came from a different learner
  entirely.
- `owner_x_learner_confound_aware`: looked like the best option on open campaigns (+0.0223, the
  single highest gap recorded), but weakened on the locked check (+0.0068, below plain X-learner's
  +0.0103) -- an honest example of a result that didn't replicate. Kept as a tracked candidate, not
  adopted.
- `owner_dr_learner`: own score worse than the owner's model -- the AIPW pseudo-outcome's variance
  with a treated group this small (3,229 rows) likely outweighs its robustness guarantee here.

## The honest caveat

The locked-campaign check exists to catch exactly the kind of false lead `extended_t_forest` and
`shallower_t_forest` turned out to be -- but it's a finite resource on this fixed, closed historical
dataset (no new campaigns will ever finish to refresh it). By the time `owner_x_learner_confound_aware`
was checked, the set had already been read twice; the Evaluation Agent's own output says so plainly:
*"these campaigns were already read 2 time(s)... this is no longer a clean test."* Nothing here should
be presented as a definitively proven effect -- X-learner is the most honest, best-supported bet among
everything tried, not a confirmed result. An E-value of 1.35 (`src/evaluation/metrics.py: e_value()`)
adds one more number to that same honesty: a fairly weak hidden confounder could fully explain the
remaining effect away.
