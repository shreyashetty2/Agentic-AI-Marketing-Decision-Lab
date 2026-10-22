# Model 1: which features add something beyond coupon use?

John Won, 10/19. Follow-up to the 10/16 mentor meeting ("test features beyond coupon use for Model 1, including demographic groups; keep demographics; test feature-group iterations").

## Short answer

- **Coupon use alone is still as good as the current model.** On the 13 open test campaigns, ranking by coupon use catches 56.5% of redeemers in each campaign's top 20%; Anna's random forest catches 56.4%.
- **No single existing group adds much to coupon use**, and no single new column does either.
- **Four new groups do add something together:** coupon habit in more detail, coupon use on the campaign's own products and commodities, redemption history in more detail, and demographics as numbers.
  - With them, the same forest catches **59.0%** of redeemers (+2.7 points over the current model, 95% range +0.6 to +4.7).
  - Logistic regression on log-scaled columns catches **60.1%** (+3.8, range +0.2 to +5.5).
  - These are the first Model 1 alternatives the Evaluation Agent marks as reliably better.
- **This is not proven yet.** It holds in 10 of 13 campaigns and in both the earlier and later halves. But the open campaigns have been reused, and an earlier round of Model 1 gains disappeared on the held-back campaigns. One final check there is the team's call (below).

## How it was tested

- **Rows:** Model 1's 7,208 mailed household × campaign rows. Every column uses only data from before the campaign's cutoff (start day − 7).
- **Test:** campaign by campaign. Each test campaign is predicted by a model trained only on campaigns that had ended before its cutoff. This is the same forward chaining the Modeling and Evaluation Agents use.
- **Which campaigns:** only the 13 open test campaigns (8 to 20). The 5 locked campaigns (21 to 25) were never read.
- **Scores:**
  - Within-campaign AUC: how well each campaign's list is ordered. It is smooth, so it is good for choosing.
  - Redeemers caught in each campaign's top 20%: the team's headline number.
  - Every gap has a 95% range from resampling whole households.
- **Fair comparisons:** all models are random forests with Anna's settings unless a learner is named. So a gain comes from the columns alone.
- **Code:** `src/modeling/model1_feature_search.py`, which wrote every number below. Full tables are in `data/processed/model1_feature_search/report.md`. The official Modeling and Evaluation Agent numbers come from `agent.py` and `evaluation/agent.py` with the new candidates in `modeling_spec.yaml`.

## What we found

### 1. Coupon use alone is the bar

| Ranking | Within-campaign AUC | Redeemers caught (top 20%) |
|---|---|---|
| "Redeemed before" rule | 0.685 | 47.6% |
| Coupon use alone (no model) | 0.804 | 56.5% |
| Anna's forest (11 columns) | 0.804 | 56.4% |
| Forest, 21 already-built columns | 0.810 | 56.9% |

The forest uses past redemption rate and nine other columns, yet it does no better than coupon use alone.

### 2. Adding one existing group to coupon use: nothing clear (step 2)

These are forests on coupon use plus one group, compared with a coupon-only forest. They are now Modeling Agent candidates (`coupon_*_forest`).

| Added to coupon use | AUC gap (pts) | Capture gap (pts) |
|---|---|---|
| Redemption history (rate, counts) | +0.7 [−0.3, +1.7] | +2.0 [−1.2, +4.6] |
| Offer fit (spend on campaign products) | −0.2 [−0.9, +0.8] | +0.6 [−1.7, +2.5] |
| Campaign (type, length) | −0.2 [−0.9, +0.4] | +0.8 [−1.1, +2.9] |
| Shopping activity | −0.5 [−1.5, +0.4] | +0.4 [−2.1, +2.6] |
| Discount share | −0.8 [−1.5, −0.1] | −0.7 [−2.2, +2.2] |

Redemption history is the only group that leans positive.

### 3. Where the signal beyond coupon use is

To see what is left once coupon habit is known, I compared households in the same campaign **and the same fifth of coupon use**. Within those groups, does a column still put redeemers on top? An AUC of 0.5 means "nothing beyond coupon use".

- **Strongest:** redemption history.
  - Past redemptions: 0.63
  - Past redemptions of coupons in this campaign's commodities: 0.63
  - Campaigns redeemed: 0.62
- **Next:** coupon use on this campaign's own products (0.62), coupon use over all history (0.61), and coupon use in the campaign's commodities (0.61).
- **Then:** recent (8-week) coupon use and coupon breadth (about 0.60), and some demographic labels (0.58 to 0.60).
- **Close to nothing:** trips, weeks of history, other live campaigns, private-label share, the main store's coupon culture, and flyer/display habits (all within 0.04 of 0.5).

### 4. The new groups help, but only together

Here each new group is added to Anna's 11 columns, compared with her forest:

| New group | AUC gap (pts) | Capture gap (pts) | Campaigns better |
|---|---|---|---|
| Coupon habit, refined | +1.0 [+0.0, +1.9] | +0.6 [−1.5, +2.4] | 8/13 |
| Offer fit, refined | +0.6 [−0.3, +1.3] | +2.0 [−0.3, +3.4] | 9/13 |
| Redemption history, refined | +0.5 [+0.0, +1.0] | +0.8 [−1.6, +3.2] | 9/13 |
| Demographics | +0.5 [−0.2, +1.4] | +1.4 [−1.0, +3.4] | 8/13 |
| Brand and store | −0.2 | −0.3 | 4/13 |
| Campaign size | −0.1 | −0.3 | 6/13 |
| In-store flyer/display habit | −0.1 | 0.0 | 5/13 |

The most consistent single column is recent coupon use (`coupon_lines_per_week_8w`). It improves the ordering in all 13 campaigns, but only by 0.7 AUC points.

**Together:**

- All 59 columns in one forest: +1.8 AUC [+0.9, +2.7] and +2.5 capture [+0.0, +4.2].
- The trimmed set (`extended_plus`: the 21 built columns plus the 18 kept, 39 in total) does as well or better.
- Dropping any one group from all the columns costs at most 0.3 AUC points. The groups overlap, so no single one is essential, but together they move the model.

### 5. The learner matters less than the columns, with one exception

With the 39 columns, the forest, a slow LightGBM and logistic regression all land within 0.5 AUC points of each other (0.821 to 0.826).

The exception is logistic regression on **log-scaled** columns. Coupon counts and dollars are very skewed: a few heavy coupon users sit far out, and a straight-line model handles that badly.

- Anna's logistic regression on her 11 columns catches 53.9%.
- The same columns log-scaled catch **59.3%** (+2.8 [+0.1, +4.5] over her forest).
- C = 0.05 was set once before testing, not tuned.

### 6. Demographics

- **Coverage is higher than we thought.** About 58% of Model 1's mailed rows have demographics (48% of mailed households), not 32%.
  - The 32% in the 10/16 notes is across all 2,500 households.
  - The 22% is Model 3's B/C rows, which are mostly unmailed households: 19% of unmailed rows have demographics, against 63% of mailed ones. The mailer went mostly to households with a demographic record.
- **The gain doesn't come only from demographics.** Households with demographics redeem much more often: 16% of their mailings, against 7% without. But the new columns help both groups.

  | Households | Anna's forest AUC | `extended_plus` forest AUC |
  |---|---|---|
  | With demographics | 0.786 | 0.800 |
  | Without demographics | 0.796 | 0.824 |

- **On their own they add little.** Coupon use plus demographics adds +0.2 AUC (range −0.9 to +1.5).
- **Among all the columns, they help a little.** Dropping them costs 0.3 AUC points, and the model is worse without them in 12 of 13 campaigns.

## What changed in the repo

- **Feature Agent 2A builds 18 more columns** (`feature_2a_spec.yaml`, `profile.py`):
  - `coupon_lines_per_week_8w`, `coupon_lines_per_week_all`, `coupon_trip_share_26w`, `coupon_commodities_26w`
  - `campaign_product_coupon_lines_per_week_26w`, `campaign_commodity_coupon_lines_per_week_26w`, `past_redemptions_in_campaign_commodities`, `redeemed_last_campaign`
  - Ten demographic number columns (0 = no record, so they are never blank)
  - The 48 earlier columns are byte-identical. All 10 real-data acceptance checks pass, including the leak test on every column. Each new column has a hand-checked unit test.
- **Modeling Agent** (`modeling_spec.yaml`):
  - Feature-group candidates: coupon only, and coupon plus one group.
  - The investigation's result as candidates: new columns only (`extended_plus_forest`), learner only (`owner_cols_logistic_log`), and both (`extended_plus_logistic_log`).
  - A new learner, `logistic_log` (`learners.py`), which the brain may also propose.
- **Lead agent** (Pranav's, merged 10/20), so the new options work through it:
  - `logistic_log` has a plain-language name in the run summary (`src/lead/describe.py`).
  - The four new everyday coupon columns are watched for unusual moves (`lead_spec.yaml`). On our two years, no campaign-to-campaign move trips the check.
  - Any of the three options can be made Model 1's model in use with `src/lead/in_use.py`. I tested this on a scratch copy, not the team's real file (decision 2 below).
- **Not adopted:** private-label share, the main store's coupon culture, campaign size, in-store display/flyer habits, and 14 other coupon, offer-fit and redemption variants. `model1_feature_search.py` still builds them, so the result can be reproduced.

## Official result (Modeling and Evaluation Agents, open campaigns)

| Model | Redeemers caught | vs. Anna's forest (95% range) | Evaluation Agent |
|---|---|---|---|
| Logistic on log-scaled columns, 39 columns | 60.1% | +3.8 [+0.2, +5.5] | reliably better |
| Logistic on log-scaled columns, Anna's 11 | 59.3% | +3.0 [+0.1, +4.5] | reliably better |
| Forest, 39 columns | 59.0% | +2.7 [+0.6, +4.7] | reliably better |
| Anna's forest, 11 columns (current) | 56.3% | | |
| Coupon-only forest | 55.6% | −0.7 [−3.2, +1.4] | can't tell |
| "Redeemed before" rule | 47.6% | | |

- **The agent's recommendation to Anna** is logistic regression on log-scaled columns with the 39 columns. Its own per-campaign selection picked it in 10 of 18 folds.
- **Forest randomness alone moves capture by about half a point.** The forest scores differ by up to 0.6 points between the agent and the search script: same columns, different column order, so different random trees. Logistic regression is not random, and the two agree exactly.

## Caveats

- **The open campaigns have been reused.** Several rounds of ideas have been judged on them. I also chose which 18 columns to keep after seeing results there, so the 39-column numbers lean optimistic. The all-59-columns result, where nothing was picked, is the cautious version: +1.8 AUC and +2.5 capture.
- **History says be careful.** In the first locked-campaign check, every Model 1 option that looked better on the open campaigns fell back within noise.
- **13 campaigns, 707 redeemers.** The ranges are wide, and the lower ends sit close to zero for capture.
- **Shreya's `campaign_promo_share` (merged 10/21) can't help Model 1.** It's the same for every household in a campaign, and blank for Type A, while Model 1 ranks households within a campaign.

## Decisions for the team

1. **Run the one locked-campaign check now?**
   - Command: `python src/evaluation/agent.py --model 1 --unlock`
   - It would be the second look at campaigns 21 to 25. The agent flags a second look as less clean, and with 811 rows it can catch a failure but not confirm a 3-point gain.
   - My suggestion: run it once for `extended_plus_forest` and `extended_plus_logistic_log` before Anna decides, and make no more Model 1 changes after it.
   - **Lead agent replays.** A replay holds nothing back, so one as of day 657 or later grades campaigns 21 and 22, and one as of day 720 grades all five. Pranav's examples use day 660. Any replay of Model 1 on this branch past day 656 would show the new options' results on held-back campaigns. Hold off on those replays until this decision is made.
2. **Anna:** adopt `extended_plus_logistic_log`, `extended_plus_forest`, or keep the current forest? Switching is the owner's decision.
   - With Pranav's lead agent, this needs no script edits: `python src/lead/in_use.py set --model 1 --idea extended_plus_logistic_log --by "Yuxin Cai" --reason "..."`, or ask the lead agent (`/lead-agent`).
   - The forest stays in as an option for comparison, and `in_use.py clear --model 1` switches back.
3. **Pranav and the team:** OK with 2A building the 18 columns?
   - On 10/16 we agreed the Feature Agent stays mechanical and new feature ideas live in the Modeling Agent. Here the ideas came from the modeling side, and 2A only computes them, because the Modeling Agent can't build columns.
   - The demographics recode settles a loose end. The 2A spec said turning labels into numbers was the Modeling Agent's job, while the Modeling Agent sends text columns back to the Feature Agent.
4. **Is the brain useful?** Evidence from Model 1:
   - Its round-2 Model 1 ideas didn't beat Anna's forest.
   - Its one new-column idea (coupon use on the campaign's own products) adds +0.5 AUC on its own, within noise.
   - The gain here needed many new columns built and tested together. The brain can only name a column for the Feature Agent to build. So far, the brain is most useful for trying settings on existing columns, and a person still had to build and screen new features.
