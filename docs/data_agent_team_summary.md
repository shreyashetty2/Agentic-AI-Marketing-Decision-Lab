# Data Agent: Team Summary

*A 5-minute read. The full details are in `data_agent_guideline.md`. Numbers marked (verify) are expected values we haven't yet checked against the real files.*

## The idea in one paragraph

The Data Agent is Step 1 of our pipeline. It takes the 8 raw dunnhumby files and turns them into clean, linked tables that the Feature Agent can build on. We want it to serve **all three** prediction questions (response probability, expected value, uplift), not just Question 1, so we build the shared foundation once instead of three times.

## Workflow

```
8 raw CSV files
      |
1. Load + validate      (right columns, right types, keys not null)
      |
2. Integrity checks     (do the tables agree with each other?)
      |
3. Decide per issue     (auto-fix / flag / halt, every decision logged)
      |
4. Build shared tables  (campaign, household, spine, redemptions, transactions)
      |
5. Report               (data quality report + decision log)
      |
Feature Agent (Step 2)
```

## What it hands to the next agent

| Output | One row per | Why we need it |
|---|---|---|
| `dim_campaign` | campaign (30) | type, start/end day, order in time, overlaps |
| `dim_household` | household (2,500) | demographics + a "has demographics" flag |
| **`spine_household_campaign`** | household x campaign (75,000) | the backbone: mailed or not, redeemed or not, for all 3 questions |
| `bridge_campaign_product` | campaign x coupon x product | which products each campaign targeted |
| `fact_redemption` | redemption event | every redemption, with validity flags |
| `fact_transaction` | receipt line (~2.6M, verify) | cleaned purchases with derived prices and flags |
| Data quality report + decision log | (report) | what we found and what the agent did about it |

**How each question uses it:** Q1 uses the spine's mailed rows (7,208, of which 889 redeemed). Q2 uses transactions in the campaign window for redeemed pairs. Q3 uses targeted products plus the *non-mailed* rows of the spine as a comparison group.

## Four ground rules

1. **The Data Agent never aggregates household behavior.** Things like "spend in the 8 weeks before the campaign" belong to the Feature Agent, so we only have to police "no data from after `START_DAY`" in one place.
2. **Flag, don't delete.** Odd rows stay in the data with a flag column. Later agents decide what to exclude.
3. **`DAY` is the master clock.** Campaign windows are in days.
4. **Every decision is logged.** That is what makes it an agent that decides, rather than a script.

## Fix, flag, or halt (proposed default)

- **Auto-fix and log:** type casts, column names, whitespace, exact duplicates in the coupon mapping table, filling missing demographic labels with "Unknown".
- **Flag and keep:** zero/negative quantities or sales, price outliers, redemptions that don't match a mailed campaign or fall outside the campaign dates, products missing from the product table.
- **Halt:** missing files or columns, null keys, unexpected campaign IDs, or the spine not reproducing 7,208 mailed pairs / 889 redeemed pairs.
- **Never:** deduplicate redemptions or transactions, or drop households without demographics.

## Things we found that need a team decision

| # | Issue | Why it matters | Suggested direction |
|---|---|---|---|
| 1 | **Net revenue definition (Q2).** The proposal says `SALES_VALUE - COUPON_MATCH_DISC`, but the guide shows `sales_value` is already after the match discount, and discounts are stored as negatives. | Read literally, the formula adds the discount back instead of subtracting a cost. | Keep all components raw, confirm with the mentor, then fix one definition. |
| 2 | **TypeA campaigns (Q3).** Each household got 16 coupons from a pool, and which 16 is not in the data. | We only know the *pool* of targeted products for TypeA. | Use the pool and flag it, or analyze TypeA separately. |
| 3 | **Uplift design (Q3).** Comparing before vs during a campaign can't separate the campaign from seasonality. | Non-mailed households give a same-period baseline. | Build the full 75,000-row spine so we can do either. Note TypeA mailing was based on past behavior, so some bias remains. |
| 4 | **Campaigns aren't in date order and they overlap heavily** (74 of 435 pairs; e.g. campaign 26 runs before campaign 8). | A split by campaign ID would leak the future. | Split by start date, and consider a gap between train and test. |
| 5 | **Only 801 of 2,500 households have demographics.** | Joining carelessly would drop about two thirds of our data. | Keep everyone, add a "has demographics" flag, report the demographic subset separately. |
| 6 | **Q2 spend scope:** total basket, or only targeted products? | Changes the target variable a lot. | Derivable both ways; agree on one primary definition. |
| 7 | **Redemptions are clean:** 2,318 rows make up the 889 redeemed pairs; no orphans, no rows outside campaign dates, no exact duplicates. | We can also count redemptions per pair if we want. | Keep every row; use "at least one" for Q1. |
| 8 | **`week_no` = `ceil((day + 2) / 7)`** on every row (not `ceil(day/7)`). | Matters if we use the in-store display/mailer table. | Use this rule and check it on every run. |
| 9 | **Is the display/mailer table (Q4d omnichannel) in scope?** | It is the largest table and not needed for Q1 to Q3. | Defer to Phase 2. |

**Questions for the mentor:** what "net revenue" should mean (#1); how to handle TypeA in Q3 (#2); whether a before/after uplift is acceptable or a control group is expected (#3).

## Work split

| Package | What | Depends on |
|---|---|---|
| WP1 | Schema contract, loaders, clean base tables | start here |
| WP2 | Integrity checks, fix/flag/halt rules, decision log | WP1 |
| WP3 | Shared tables (campaign, household, spine, bridge, redemptions) | WP1 |
| WP4 | Transaction enrichment (derived prices, flags, product attributes) | WP1 |
| WP5 | Report generation, tests, optional LLM summary layer | WP2, WP3 |

Owners: *to be assigned in the meeting.*

## Demo target for the mentor meeting

WP1 + WP2 + the spine from WP3 + a first data quality report. That shows a running agent that reproduces **7,208 mailed pairs and 889 redeemed pairs (12.3%)** and explains any discrepancy.

## Definition of done

- All 8 files load and row counts reconcile with the decision log.
- Spine has 75,000 rows, unique on household x campaign; 7,208 mailed; 889 redeemed.
- 2,500 households in the household table, 801 with demographics.
- Rerunning gives identical outputs.
- Report covers every integrity check with pass / flagged / fail.

## What we need from everyone in the meeting

1. Agree on (or take to the mentor) decisions #1 to #3 above.
2. Approve the fix / flag / halt defaults.
3. Agree on naming (lowercase snake_case) and file format (parquet).
4. Assign owners to WP1 to WP5.
