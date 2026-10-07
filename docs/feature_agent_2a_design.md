# Feature Agent, Step 2A: Design

## What 2A is

The Feature Agent turns clean data into the tables our models train on. It has two steps:

- **2A (this document):** builds one shared table for all three models.
- **2B:** filters that table into one table per model (which rows, which answer column).

2A starts from the Data Agent's household × campaign table (75,000 rows: 2,500 households × 30 campaigns) and adds two kinds of columns:

- **Profile columns:** what the household did *before* the campaign (spend, trips, coupon habits, past redemptions, interest in the campaign's products).
- **Outcome columns:** what happened *during* the campaign, one per model.

The exact list, settings and rules live in `src/features/feature_2a_spec.yaml`. The code reads that file, so the two cannot disagree. This document explains why each choice was made.

## The cutoff rule: no peeking at the future

Models are used before a campaign starts, to decide whom to mail. If a profile column uses anything from after that moment, the model looks excellent in testing and fails in real use. This is called leakage.

- **Cutoff day = start day − 7.** In practice, the mailing list is picked before the campaign starts. We tested gaps of 0, 7, 14 and 28 days: the test score stayed between 0.82 and 0.83, so the more realistic 7-day gap costs nothing.
- **Cut by day, never by week.** A start day sits inside a week that also contains later days.
- **No whole-period totals.** For example, "number of campaigns ever mailed" counts campaigns that come later.
- **Past redemptions are counted by date.** A campaign still running at the cutoff only contributes redemptions made before the cutoff.
- **One function builds every profile:** `features(household, cutoff_day)`. Training uses each campaign's cutoff; the Decision Lab uses "today". The rule exists in one place.
- **Outcome columns use the real campaign window** (start day to end day). The 7 days between cutoff and start are used by neither side.

## Which profile columns, and why

The original proposal named four: weekly spend, visits, past redemptions and share of discounted purchases. We tested each candidate on the 7,208 mailed rows. Each model was trained on campaigns starting before day 587 (4,890 rows) and tested on later ones (2,318 rows).

- **The added columns raise the test score.** Boosting goes from 0.790 to 0.828; logistic regression from 0.805 to 0.830.
- **Coupon use at the till is the strongest single column** (0.78 on its own).
  - It counts receipt lines where the household used any coupon, not just campaign coupons.
  - We checked it is not leakage: only 12.6% of those lines fall on days the household redeemed a campaign coupon, and it still scores 0.76 without them.
- **Past redemptions** is the other column that clearly helps even when everything else is present. Households that redeemed before redeem 29% of the time, against 8% for those that never did.
- **Spend on this campaign's products before it started** (0.69) is the concrete form of the README's "category affinity".
- **Most other columns repeat each other.** Removing any one alone changed nothing measurable, and no column made the model worse. 2A keeps them anyway: the table serves three models, and a column redundant for Model 1 may matter for Model 2 or 3. Choosing columns per model is the Modeling Agent's job.

Three details settled by testing:

- **Short history: weekly averages.**
  - 5,530 rows have less than 26 weeks of history before the cutoff. Most of them (3,592) are non-mailed TypeB/C rows in Model 3's comparison group.
  - Raw totals make these households look like light shoppers. Weekly averages describe them correctly, and Model 1's score is unchanged (0.817 either way).
  - A "weeks of history" column keeps the raw totals recoverable.
- **Coupon habit: one version.**
  - Excluding campaign-coupon days changed neither Model 1's score (0.817 vs 0.816) nor how much the column relates to who gets mailed (0.726 vs 0.717).
  - The column most related to who gets mailed is spend (0.84): the retailer targets big spenders. That is Model 3's central challenge, not this column's.
- **Conflicting display/flyer rows:** only 9 of 40,291 campaign-product purchases hit one, and no Model 3 outcome changes under any tie-break rule. We use "promoted if any row says so".

Demographics stay as raw labels. Turning them into numbers depends on the model type, so the Modeling Agent does it.

## Outcome columns

2A does not define outcomes. It computes them exactly as each model's owner defined them, and builds every variant they listed so that 2B can choose.

- **Model 1:** `redeemed_flag`, already in the Data Agent's table.
- **Model 2 (Yuxin's definition):**
  - Main column: spend on the campaign's products during the campaign.
  - Sensitivity column: spend on products of coupons the household actually redeemed.
  - For TypeA we only know the pool of about 200 coupons, not the 16 each household received. So the main column uses the full pool, and Model 2 handles TypeA separately.
- **Model 3 (Shreya's definition):**
  - Main column: bought any campaign product during the campaign, excluding purchases made while that product was on display or in the store flyer (that store and week).
  - Variant: excluding display only, as in the 10/2 deck. The two differ on 1,397 of 62,500 rows.
  - TypeB/C only; TypeA is blank.
- **Filter for Model 3:** `mailed_overlapping_campaign` marks households mailed another campaign overlapping this one, for Model 3's comparison group.
- **Week mapping:** the display/flyer file is by week, so purchases are mapped with `week = ceil((day + 2) / 7)`. This matches all 40,291 campaign-product purchases.

Each column carries a role in the spec:

- `profile` and `context` columns are safe model inputs.
- `outcome` and `filter` columns must never be inputs.

## What the agent decides, and what the code does

Most of 2A has one right answer, so plain code does it:

- loading and checking inputs
- computing every column
- the cutoff rule
- row flags
- the hard checks

The agent's one decision is whether each column is trustworthy.

- **Code measures each column's health:** share blank, value range, and how strongly it predicts each outcome.
- **Known patterns are handled by fixed rules.** Example: "blank because the household was never mailed before" is accepted with a note.
- **The agent judges anything else** and picks accept, accept with a note, quarantine, or stop, with a written reason.
- **High scores trigger the leak test automatically.** Honest habits can score high, so a score above 0.90 is not treated as a leak by itself. The code runs the scramble test on that column: if nothing changes, the column is accepted with a note; if anything changes, it is a real leak and the run stops for a human.
  - On the real data this fired once: spend on this campaign's products before the campaign scored 0.92 against Model 2's answer (spend on those products during it). The leak test showed no change, so it is habit, not leakage. Within each campaign type the relationship is weaker (TypeA 0.85, TypeB 0.57, TypeC 0.36); the pooled 0.92 is partly TypeA's large product pools making both numbers big.
- **Worked example:** the coupon-habit column scores suspiciously high. The agent runs the leak check described above, then accepts it with the note "genuine coupon habit, not leakage".

Limits on the agent:

- Quarantine keeps the column and only marks it, so the agent applies it without approval.
- Stopping the run always goes to a human.
- The agent writes the plain-language report. Code verifies that every number in it matches the health numbers.
- The agent never changes settings, values or columns.

## How we know it is done

The full checklist is under `done_when` in the spec. The most important checks:

- **Leakage test:** scramble every purchase and redemption on or after each cutoff; every profile column must stay identical.
- **Known numbers:**
  - 889 redeemed mailed rows
  - Model 3 outcome counts of 11,838 (display + flyer) and 13,235 (display only)
  - household 208's campaign 18 spend on campaign products = $209.58
- **Every blank has a known reason.**
- **Same input gives the same output,** and the run fits on an 8 GB laptop.
