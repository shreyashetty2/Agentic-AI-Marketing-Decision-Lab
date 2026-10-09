# Data Agent Guideline (Step 1)

**Purpose:** tell whoever builds the Data Agent *how* to process the dunnhumby dataset, *why* it should be processed that way, and *what the outputs should look like*, so the same outputs serve all three prediction questions (not just Question 1).

**Status:** research draft for team review, updated 28 Sep 2026 after running `verify_assumptions.py` on the real files (7 of 8 tables; `causal_data` not yet checked). Items tagged **[Verified]** were confirmed on the real data. Items still tagged **[Verify]** are unchecked. Section 11 summarizes the results.

**Source tags used below**

| Tag | Meaning |
|---|---|
| **[Guide]** | stated in the dunnhumby user guide (or visible in its example tables) |
| **[Proposal]** | stated in our project proposal ("ML Question and System Flow") |
| **[Recommend]** | my suggestion, open for team discussion |
| **[Verified]** | confirmed against the real files |
| **[Verify]** | expected value or assumption not yet checked against the actual files |

---

## 0. TL;DR

1. All three prediction questions share one grain: **household × campaign**. The Data Agent should build a single, clean **spine** at that grain, plus the shared tables around it, so no question needs its own data pipeline.
2. The Data Agent **cleans, standardizes, links and flags. It never aggregates household behavior.** Anything like "weekly spend before the campaign" belongs to the Feature Agent, because that is where the leakage rule (use only data before `START_DAY`) gets enforced.
3. **Flag, don't delete.** Rows that look odd stay in the data with a flag column. Only the Feature and Modeling Agents should decide what to exclude.
4. `DAY` is the master clock (campaign windows are defined in days). `WEEK_NO` is only needed for `causal_data`; on the real data it follows `week_no = ceil((day + 2) / 7)` exactly (see section 4.1).
5. The spine should cover **all 2,500 households × all 30 campaigns**, with a `mailed_flag`, not just the 7,208 mailed pairs. Question 3 (uplift) needs non-mailed households as a comparison group.
6. Four open definitions need a team (and mentor) decision before Question 2 and Question 3 can be built: net revenue, Type A targeted products, uplift design, and spend scope (section 8).
7. Definition of done for the demo: the spine reproduces **7,208 mailed pairs and 889 redeemed pairs (12.3%)** exactly, and the data quality report explains every discrepancy.

---

## 1. Scope: what the Data Agent owns

**Owns**
- Loading the 8 raw files and enforcing a schema contract (columns, types, keys).
- Cross-table integrity checks (does every redemption match a mailed household-campaign pair, and so on).
- Deciding, per issue, whether to auto-fix, flag, or halt (the proposal's "Decides" line for Step 1).
- Building shared, question-agnostic tables: campaign dimension, household dimension, household-campaign spine, campaign-product bridge, redemption fact, enriched transaction fact.
- Producing a data quality report and a decision log.

**Does not own**
- Rolling-window behavioral features (spend, visits, redemption history before `START_DAY`). That is the Feature Agent.
- Model training, splits, or evaluation. That is Modeling and Evaluation.
- Business metric definitions that are still undecided (net revenue, uplift). The Data Agent should keep the raw components available so those definitions can be settled later without reprocessing.

---

## 2. What each prediction question needs from the data

| Need | Q1 Response probability | Q2 Expected financial value | Q3 Incremental uplift | Tentative Q4 business questions |
|---|---|---|---|---|
| Unit of analysis | household × campaign | household × campaign | household × campaign | segment / household level |
| Outcome source | `coupon_redempt` (any redemption) | `transaction_data` spend in campaign window | purchases of targeted products, campaign window vs pre-window | redemption rates by segment; payouts vs sales |
| Campaign window (`START_DAY`, `END_DAY`) | yes | yes | yes | yes |
| Targeted products per campaign (`coupon` table) | for affinity features | for "targeted spend" if used | **essential** | explainability (commodity-level reasons) |
| Non-mailed households as comparison | no | no | **yes, for a clean baseline** | scenario "what if we target more" |
| Discount components (`retail_disc`, `coupon_disc`, `coupon_match_disc`) | share-of-discounted-purchases feature | **essential** | optional | cost of payouts (Q4c) |
| Demographics | feature | feature | feature | segment cuts (Q4a) |
| Product hierarchy (`commodity_desc`, etc.) | category affinity | optional | targeted-product grouping | explanations (Q4b) |
| `causal_data` (display / mailer) | optional | optional | confounder control | Q4d omnichannel |

**Consequence:** if the Data Agent produces the shared tables in section 7, none of the three questions has to go back to raw data.

---

## 3. Design principles (the "why")

1. **One grain, one spine.** Every question is asked per household per campaign. A single spine table means labels, features, and uplift comparisons all join on the same key `(household_key, campaign)` and can't drift apart. **[Recommend]**
2. **The Data Agent never aggregates behavior.** Our prediction timing rule is that features use only data before `START_DAY` **[Proposal]**. Any household-level rollup computed over the full 2 years silently leaks the future into earlier campaigns. Keeping all behavioral aggregation in the Feature Agent (with an explicit `< START_DAY` filter) puts leakage control in one place. **[Recommend]**
3. **Flag, don't delete.** We can't know yet which odd rows Question 2 or 3 will need (returns, zero-value lines, out-of-window redemptions). Deletion is irreversible for everyone downstream; a flag column costs nothing. **[Recommend]**
4. **`DAY` is the master clock.** Campaign windows are in days **[Guide]**. Convert to weeks only where a table forces it (`causal_data`). **[Recommend]**
5. **Derive, don't overwrite.** Keep raw discount columns and add derived columns beside them, so any later redefinition (e.g., of net revenue) stays possible. **[Recommend]**
6. **Every decision is logged.** The agent must record what it fixed, what it flagged, and why, in a machine-readable decision log plus a readable report. This is what makes it an *agent that decides* rather than a script that transforms. **[Proposal]**
7. **Local and reproducible.** Process locally, write parquet, make reruns deterministic (same input gives identical output hashes). **[Recommend]**

---

## 4. Table-by-table guide

For each table: what it is, what to check, known quirks, and how to treat it. Sizes are expected values **[Verify]**.

### 4.1 `transaction_data`

- **Grain:** one row per product line on a receipt (like a line on a store receipt **[Guide]**). No natural single key; `basket_id` groups lines into a trip. 2,595,732 rows, 2,500 households, days 1 to 711 **[Verified]**.
- **Why it matters:** source of all behavior features, Question 2 outcomes, and Question 3 purchase frequencies.
- **Checks**
  - `household_key` is within the 2,500-household universe; `product_id` exists in `product`.
  - `day` and `week_no` ranges (weeks should be 1 to 102 **[Guide]**); null keys.
  - `quantity` and `sales_value`: count of zero or negative values (returns, voids, coupon-only lines).
  - Extreme `quantity` or implied unit price outliers.
  - `trans_time` is an HHMM integer (0 to 2359).
  - **Discount sign convention:** in the guide's worked examples the three discount columns are stored as **negative numbers** (e.g., `retail_disc` -1.34, `coupon_disc` -0.55, `coupon_match_disc` -0.45). **[Verified]** `coupon_disc` and `coupon_match_disc` are always ≤ 0; `retail_disc` is ≤ 0 except 36 rows (max +3.99), which should be flagged. No row has negative `sales_value`; 18,850 rows have `sales_value` = 0, 14,466 have `quantity` ≤ 0, and 23,136 have `quantity` > 100 (max 89,638), which needs a look before we use quantity-based features.
  - **[Verified] `day` to `week_no` mapping:** it is *not* `ceil(day / 7)` (only 73% of rows match). It is exactly `week_no = ceil((day + 2) / 7)` on every row, and each day maps to a single week (so week 1 covers days 1 to 5). Keep this as a check in the agent and report any row that breaks the rule.
- **Treatment**
  - Auto-fix: dtype casts, column renaming, converting `trans_time` to an hour-of-day column.
  - Derive (keep raw columns): `shelf_price_loyalty = (sales_value - (retail_disc + coupon_match_disc)) / quantity` and `shelf_price_nonloyalty = (sales_value - coupon_match_disc) / quantity` per the guide's formulas, guarding against `quantity = 0`; `total_discount`; `has_coupon_flag`; `has_loyalty_disc_flag`.
  - Flag only: non-positive quantity or sales value, outliers, unknown products.
  - Attach `department`, `commodity_desc`, `brand` from `product` so downstream agents don't re-join 2.6M rows (also needed for explanations like "no purchases in that `COMMODITY_DESC`" **[Proposal]**).
- **Why not drop odd rows:** returns and zero-value lines may matter for Question 2 (net value) and are harmless if flagged.

### 4.2 `hh_demographic`

- **Grain:** one row per household, **only 801 of 2,500 households** **[Proposal]**.
- **[Verified] Column names.** The guide's page-4 variable table lists the *transaction_data* column names for this table, which is a documentation error. The real file has anonymized columns: `classification_1` to `classification_5`, `homeowner_desc`, `kid_category_desc`, `household_key`. (An earlier draft of this guideline listed `AGE_DESC`, `INCOME_DESC` and similar names from another version of the dataset. That was wrong for our files.) Values seen: `classification_1` = Age Group1 to Age Group6; `classification_2` = X / Y / Z; `classification_3` = Level1 to Level12; `classification_4` = 1, 2, 3, 4, 5+; `classification_5` = Group1 to Group6; `homeowner_desc` = Homeowner, Renter, Probable Owner, Probable Renter, Unknown; `kid_category_desc` = 1, 2, 3+, None/Unknown. The guide says the values are ordered (except X/Y/Z) but never says what each classification measures, so use neutral names and do not call them income or household size in the dashboard unless the mentor confirms.
- **Checks:** `household_key` unique; every key exists in the household universe; category label sets (list distinct values per column and compare to what the guide says: ordered groups, with meaningful ordering **[Guide]**).
- **Treatment**
  - Keep the raw text labels and add an ordinal code for the ordered fields (`classification_1`, `_3`, `_4`, `_5`, `kid_category_desc`; treat `classification_2` and `homeowner_desc` as nominal) from **one shared mapping config**, so encoding is defined once for the whole team.
  - Treat values like `Unknown` / `None/Unknown` as missing in the ordinal code but keep the original label. Note that `kid_category_desc` merges "no kids" with "unknown" (558 of 801 households), and `homeowner_desc` has 233 Unknown.
  - Do **not** drop households without demographics. Add `has_demographics` to the household dimension. Downstream agents then choose between "train on all with a missing indicator" and "demographic subset only".
- **Why:** 68% of households have no demographics, so any join that inner-joins this table silently removes most of the data.

### 4.3 `product`

- **Grain:** one row per `product_id`. Expected on the order of ~90k products **[Verify]**.
- **Checks:** `product_id` unique; share of missing `manufacturer`, `brand`, `curr_size_of_product` (the size field is *not available for all products* **[Guide]**); share of `transaction_data` product IDs missing here.
- **Treatment:** standardize whitespace and casing on `department`, `commodity_desc`, `sub_commodity_desc`; keep `brand` (Private / National **[Guide]**); flag rather than drop products missing from this table.
- **Why:** this is the taxonomy for category-affinity features and for plain-language explanations in the Decision Lab.

### 4.4 `campaign_table`

- **Grain:** one row per household × campaign mailed. Expected **7,208 rows** **[Proposal]**.
- **Checks:** `(household_key, campaign)` unique; campaign IDs within 1 to 30; `description` in {TypeA, TypeB, TypeC} **[Guide]**; `description` agrees with `campaign_desc` for the same campaign; households in the universe; **how many of the 2,500 households never appear here** (these are natural never-mailed comparison households) **[Verify]**.
- **Treatment:** this table defines `mailed_flag` in the spine. Do not filter the spine to it (see section 7).

### 4.5 `campaign_desc`

- **Grain:** one row per campaign (30 rows) with `start_day`, `end_day` **[Guide]**.
- **Checks:** `start_day ≤ end_day`; days within the observed transaction day range; type label consistent with `campaign_table`.
- **Known quirks from the guide's case study (household 208)**
  - **Campaign IDs are not chronological.** Campaign 26 runs days 224 to 264, before campaign 8 (days 412 to 460). Therefore any "train on earlier, test on later" split must sort by `start_day`, never by campaign ID.
  - **Campaigns overlap.** Campaign 18 (587 to 642) overlaps campaign 22 (624 to 656); campaign 29 (281 to 334) overlaps campaign 30 (323 to 369).
- **[Verified]** Overlap is heavy, not occasional: 74 of the 435 possible campaign pairs overlap. Campaigns 21 and 22 have identical windows (days 624 to 656), campaigns 11 and 12 start on the same day, and so do 24 and 25. Campaign 24 ends on day 719 but transactions stop at day 711, so its window is cut short by 8 days.
- **Treatment:** derive `duration_days`, `start_week`, `end_week` (using the verified week rule), `chrono_rank` (by `start_day`, then `end_day`, then campaign ID as the tie-break), `n_overlapping_campaigns`, and `window_truncated_flag`.
- **Why overlap matters:** pre-campaign features for one campaign can sit inside another campaign's active window, and a household can be exposed to two campaigns at once. Feature and uplift logic must be aware of this; the Data Agent's job is to make it visible.

### 4.6 `coupon`

- **Grain:** campaign × coupon × product; a coupon can apply to many products **[Guide]**. Expected on the order of ~125k rows **[Verify]**.
- **Critical semantic:** for **TypeA**, the table gives the *pool* of possible coupons; each household received 16 chosen from that pool, and **which 16 is not in the data** **[Guide]**. For **TypeB and TypeC**, every participating household got all of that campaign's coupons **[Guide]**. This limits Question 3 (see section 8, D2).
- **[Verified]** 124,548 rows, of which 5,164 are exact duplicates (safe to remove). 171 of 1,135 `coupon_upc` values appear under more than one campaign, so always join on `(campaign, coupon_upc)`. TypeA campaigns have a pool of 181 to 209 distinct coupons (each household got 16, about 8% of the pool); TypeB campaigns have 2 to 33 coupons and TypeC 1 to 34, and every household got all of them.
- **Checks:** campaigns exist in `campaign_desc`; product coverage in `product`.
- **Treatment:** treat `coupon_upc` as a **string** (long identifiers). Join to redemptions on `(campaign, coupon_upc)`, not `coupon_upc` alone. Exact duplicates in this mapping table are safe to remove (logged). Build the `bridge_campaign_product` table from it.

### 4.7 `coupon_redempt`

- **Grain:** one row per redemption event.
- **Do not deduplicate.** The guide's example for household 208 shows two identical rows, so identical rows could be genuine repeat redemptions. **[Verified]** In the real file there are 2,318 rows and zero exact duplicates; all of them match a mailed pair, fall inside the campaign dates, and match a coupon in the `coupon` table. Keep the no-dedupe rule anyway, since one household can redeem several coupons in a campaign (889 redeemed pairs come from 2,318 redemption rows). Q1's label uses "at least one", but Q2 and the business questions may need counts.
- **Checks (these are the proposal's own example checks)**
  - Every `(household_key, campaign)` in redemptions exists in `campaign_table` (orphan redemptions).
  - Redemption `day` falls inside `[start_day, end_day]` of its campaign (coupons are valid within campaign dates **[Guide]**).
  - `(campaign, coupon_upc)` exists in `coupon`.
- **Treatment:** add a synthetic `redemption_id`; attach `pair_was_mailed`, `day_in_window`, `coupon_in_pool` flags; never drop rows. Report the number of orphan and out-of-window redemptions explicitly.

### 4.8 `causal_data`

- **Grain:** product × store × week; very large (tens of millions of rows) **[Verify]**. Not needed for Q1 to Q3; only for the tentative omnichannel question (Q4d).
- **Codes are strings, not numbers.** `display`: 0, 1, 2, 3, 4, 5, 6, 7, 9, A. `mailer`: 0, A, C, D, F, H, J, L, P, X, Z **[Guide]**. Keep as strings; add decoded label columns and flags `on_display` (`display != '0'`) and `in_mailer` (`mailer != '0'`).
- **Checks:** duplicate `(product_id, store_id, week_no)`; codes outside the documented sets.
- **Treatment:** process with DuckDB or Polars (lazy) rather than loading into pandas; write partitioned parquet. **Treat as Phase 2** unless the team decides Q4d is in scope. **[Recommend]**
- **Dependency:** joining this to transactions needs the day-to-week mapping from 4.1.

---

## 5. Cross-table integrity checks

| ID | Check | Tables | Expected | If it fails |
|---|---|---|---|---|
| X1 | Every transaction household is in the 2,500 universe | transaction_data | 100% | halt |
| X2 | Every transaction product exists in `product` | transaction_data, product | ~100% | flag; report % |
| X3 | Every campaign in `campaign_table` exists in `campaign_desc` | campaign_table, campaign_desc | 100% | halt |
| X4 | Campaign type agrees across campaign tables | campaign_table, campaign_desc | 100% | flag |
| X5 | Every campaign in `coupon` exists in `campaign_desc` | coupon, campaign_desc | 100% | halt |
| X6 | Every redemption's `(household, campaign)` was mailed | coupon_redempt, campaign_table | 100% | flag orphans; report count |
| X7 | Redemption day within campaign window | coupon_redempt, campaign_desc | ~100% | flag; report count |
| X8 | Redeemed `(campaign, coupon_upc)` exists in `coupon` | coupon_redempt, coupon | ~100% | flag |
| X9 | Demographic households are in the universe | hh_demographic | 100% | flag |
| X10 | `campaign_table` has exactly 7,208 rows | campaign_table | 7,208 **[Proposal]** | halt if different |
| X11 | Distinct mailed pairs with at least one redemption = 889 | spine | 889 **[Proposal]** | halt; explain via X6, X7 |
| X12 | Transaction `day` range vs campaign day range consistent | transaction_data, campaign_desc | overlap | flag |
| X13 | Empirical `day` to `week_no` map is single-valued | transaction_data | mostly | flag; report mismatch rate |
| X14 | Households whose first transaction is on or after a campaign's `start_day` | dim_household, spine | few | flag via `history_days_before_start` |

---

## 6. Issue-handling policy and where the agent "decides"

The proposal asks which issues agents handle automatically and which get flagged for team review **[Proposal, note 3]**. Suggested default policy **[Recommend]**, to be approved by the team:

| Category | Examples | Action |
|---|---|---|
| **Auto-fix and log** | dtype casts; column name standardization; whitespace/case cleanup; exact duplicate rows in *mapping* tables (`coupon`); converting `trans_time` to hour; filling missing demographic labels with `Unknown`; deriving week and duration columns | apply, write to decision log |
| **Flag, keep the data** | non-positive quantity or sales value; outlier prices/quantities; products missing from `product`; orphan redemptions; out-of-window redemptions; positive discount values; `history_days_before_start ≤ 0` | add flag column, count, include in report, human review |
| **Halt and escalate** | missing file or required column; null keys in fact tables; unexpected campaign IDs; spine row count ≠ 7,208 or redeemed pairs ≠ 889 without a fully explained cause; any row-count change between raw and clean fact tables | stop the run, print the cause |
| **Never do** | dedupe `coupon_redempt` or `transaction_data`; drop households without demographics; aggregate household behavior | (design rule) |

**Where an LLM helps (optional, but this is the "agent" layer).** Keep the checks and transformations deterministic. Use an LLM only to (a) read the structured findings and write the plain-language data quality report, and (b) suggest a recommended action for findings that no rule covers. An LLM suggestion is written to the decision log as *proposed*, and it must never execute a transformation that isn't in the whitelist above. This keeps the risky part testable and reproducible while still giving the agent a real decision role. **[Recommend]**

**Finding format** (one record per issue, drives both the report and the decision log):
`check_id, table, severity (info/warn/error), n_rows_affected, pct_rows_affected, example_keys (max 5), action_taken (auto_fixed / flagged / halted / proposed), rationale`

---

## 7. Output specification

Naming convention **[Recommend]**: lowercase snake_case for all columns in every output table, one convention for the whole pipeline. Files are parquet under `data/processed/`.

### Tier 1: clean base tables (same grain as raw)

`clean_transaction_data`, `clean_hh_demographic`, `clean_product`, `clean_campaign_table`, `clean_campaign_desc`, `clean_coupon`, `clean_coupon_redempt`, `clean_causal_data` (Phase 2). Row counts equal raw, except for logged auto-fixes such as exact duplicates in `coupon`.

### Tier 2: shared derived tables (question-agnostic)

**2.1 `dim_campaign`** (one row per campaign, 30 rows)
`campaign` (int, PK), `campaign_type` (A/B/C), `start_day`, `end_day`, `duration_days`, `start_week`, `end_week`, `chrono_rank`, `n_households_mailed`, `n_distinct_coupons`, `n_target_products`, `n_overlapping_campaigns`, `window_truncated_flag`

**2.2 `dim_household`** (one row per household, 2,500 rows)
`household_key` (PK), `has_demographics`, `classification_1` to `classification_5`, `homeowner_desc`, `kid_category_desc` (raw labels; `Unknown` where missing), `*_ord` ordinal codes for the ordered fields, `n_campaigns_mailed`, `first_txn_day`, `last_txn_day`
*Only time-invariant attributes here. No full-period spend or visit totals (leakage).*

**2.3 `spine_household_campaign`** (one row per household × campaign, 2,500 × 30 = 75,000 rows)
- Keys: `household_key`, `campaign` (composite PK)
- Context: `campaign_type`, `start_day`, `end_day`, `chrono_rank`, `has_demographics`
- `mailed_flag` (1 if the pair is in `campaign_table`)
- `history_days_before_start` (`start_day - first_txn_day`; ≤ 0 means no prior history)
- **Outcome columns** (must never be used as features): `n_redemptions`, `n_redemptions_in_window`, `redeemed_flag` (1 if `n_redemptions ≥ 1`)
- Q1 is the subset `mailed_flag = 1` (7,208 rows; 889 with `redeemed_flag = 1`).
- Q3 uses non-mailed rows as the comparison group; Q2 uses redeemed rows.
*Why the full grid:* it lets Q3 build a real control group, and it means the mailed subset is one filter, not a separate table.

**2.4 `bridge_campaign_product`** (campaign × coupon × product, deduplicated)
`campaign`, `coupon_upc` (string), `product_id`, `campaign_type`, `department`, `commodity_desc`, `sub_commodity_desc`, `brand`, `is_pool_only` (1 for TypeA, where the household-specific 16 coupons are unknown)

**2.5 `fact_redemption`** (one row per raw redemption record, no dedupe)
`redemption_id`, `household_key`, `day`, `coupon_upc`, `campaign`, `campaign_type`, `start_day`, `end_day`, flags: `pair_was_mailed`, `day_in_window`, `coupon_in_campaign`

**2.6 `fact_transaction`** (one row per raw transaction line, ~2.6M rows)
All cleaned raw columns, plus: `week_no`, `hour_of_day`, `department`, `commodity_desc`, `brand`, `shelf_price_loyalty`, `shelf_price_nonloyalty`, `total_discount`, `has_coupon_flag`, `has_loyalty_disc_flag`, and quality flags (`flag_nonpositive_qty`, `flag_nonpositive_sales`, `flag_unknown_product`, `flag_outlier_qty`).
*Raw `retail_disc`, `coupon_disc`, `coupon_match_disc`, `sales_value` stay untouched so Q2's net-value definition can be finalized later.*

**2.7 (Phase 2, only if Q4d is in scope) `fact_causal`**
Cleaned `causal_data` with decoded `display_label`, `mailer_label`, `on_display`, `in_mailer`, partitioned by week; plus the empirical `day_to_week` lookup table.

### Tier 3: reports

- `data_quality_report.json` (machine-readable) and `data_quality_report.md` (readable): for each table the raw and clean row counts, the findings from sections 5 and 6, and the acceptance-test results.
- `decision_log.csv`: one row per finding with the action taken.
- `schema_contract.yaml`: expected columns, dtypes, keys, ordinal maps. The single source of truth for the team.

---

## 8. Open decisions for the team (with my recommendation)

| ID | Decision | Why it matters | Recommendation |
|---|---|---|---|
| **D1** | **Definition of "net revenue" for Q2.** The proposal says `SALES_VALUE - COUPON_MATCH_DISC`. But per the guide, `sales_value` is *already after* the coupon match and loyalty discounts, and the discount columns are stored as negatives. Read literally, the formula would *add* the match discount back rather than subtract a cost. | Q2's target could be off by the entire discount amount. | Keep all four components raw in `fact_transaction`, verify the sign in the data, then agree on the definition with the mentor. Write the chosen formula into this guide. |
| **D2** | **Targeted products for TypeA in Q3.** For TypeA the household-specific 16 coupons are unknown **[Guide]**. | Q3 compares purchases of *targeted* products; for TypeA we only know the pool. | Use the campaign's full coupon product pool for all types and record `is_pool_only`. Consider restricting Q3 to TypeB/TypeC or report Type A separately. Alternative for TypeA: use the household's own redeemed products, but that conditions on the outcome, so avoid it for the main analysis. |
| **D3** | **Uplift design for Q3.** The proposal compares during-campaign vs prior-weeks frequencies for the same household (a pre/post design). | Pre/post confuses campaign effect with seasonality and trend. Non-mailed households give a same-period baseline (difference-in-differences). TypeA mailing was based on prior behavior **[Guide]**, so selection bias remains. | Build the full 75,000-row spine so both designs are possible, and state the limitation in the write-up. |
| **D4** | **Spend scope for Q2:** total basket spend in the campaign window, or only targeted-product spend? | Changes the target variable materially. | Both are derivable from `fact_transaction` + `bridge_campaign_product`; agree on one primary definition. |
| **D5** | **Train/test split and overlapping campaigns.** | Splitting by campaign ID would leak time; overlaps mean a test campaign's pre-window can include a training campaign's window. | Split by `chrono_rank`; consider a gap of one campaign between train and test. Owned by Modeling/Evaluation, but the Data Agent supplies `chrono_rank` and overlap counts. |
| **D6** | **Demographics coverage (801 of 2,500).** | Inner joining would discard most data. | Train on all households with `has_demographics` as a feature, and report performance separately on the demographic subset. |
| **D7** | **Lookback window lengths** (e.g., 8, 13, 26 weeks before `start_day`). | Early campaigns may not have enough history. | Owned by the Feature Agent; the Data Agent exposes `history_days_before_start` so short-history pairs can be handled or flagged. |
| **D8** | **Is `causal_data` (Q4d) in scope?** **[Proposal: tentative]** | It is the heaviest table and not needed for Q1 to Q3. | Defer to Phase 2. |
| **D9** | **Naming convention and file format.** | Silent case mismatches break joins. | Lowercase snake_case, parquet. |
| **D10** | **Approve the fix/flag/halt policy** in section 6. | It defines what "agent decision" means in practice. | Approve as default; revise after the first data quality report. |

**Questions worth asking the mentor:** the intended definition of net revenue (D1); whether Q3 should be restricted or reported separately for TypeA (D2); whether a pre/post uplift definition is acceptable or a control group is expected (D3).

---

## 9. Build plan, work packages, and acceptance tests

**Suggested module layout**
```
agents/data_agent/
  config/schema_contract.yaml     # columns, dtypes, keys, ordinal maps
  loaders.py                      # read raw files, enforce contract (Tier 1)
  checks.py                       # section 5 checks, return structured findings
  policy.py                       # section 6 fix/flag/halt rules
  builders.py                     # Tier 2 tables
  report.py                       # data quality report + decision log
  run_data_agent.py               # orchestrates the steps
  tests/                          # acceptance tests below
```
The earlier `ingest.py` skeleton corresponds to `loaders.py` and part of `checks.py`.

**Work packages (can be combined depending on team size)**

| WP | Content | Depends on |
|---|---|---|
| WP1 | Schema contract + loaders + Tier 1 clean tables | none (start here) |
| WP2 | Integrity checks (section 5) + policy engine + decision log | WP1 |
| WP3 | Tier 2 builders: `dim_campaign`, `dim_household`, `bridge_campaign_product`, `fact_redemption`, `spine_household_campaign` | WP1 |
| WP4 | `fact_transaction` enrichment (derived prices, flags, product attributes) + day-to-week analysis; Phase 2 `fact_causal` | WP1 |
| WP5 | Report generation (+ optional LLM summarization layer) and tests | WP2, WP3 |

**Demo target for the mentor meeting:** WP1 + WP2 + the spine from WP3 + a first data quality report. That is enough to show a running agent that reproduces 7,208 / 889 and explains every discrepancy.

**Acceptance tests (definition of done)**
1. All 8 raw files load; each clean fact table's row count equals raw (or every difference is in the decision log).
2. `spine_household_campaign` has 75,000 rows, unique on `(household_key, campaign)`; `mailed_flag = 1` on exactly **7,208** rows.
3. Among mailed rows, exactly **889** have `redeemed_flag = 1` (12.3%). Any gap is fully explained by orphan or out-of-window redemptions in the report.
4. `dim_household` has 2,500 rows, of which **801** have `has_demographics = 1`.
5. `dim_campaign` has 30 rows with unique `chrono_rank`; overlapping campaigns are counted.
6. Primary keys are unique for every Tier 2 dimension/bridge table; `fact_redemption` row count equals raw `coupon_redempt`.
7. No Tier 2 table contains a full-period behavioral aggregate (checklist review).
8. Rerunning the agent on the same input produces identical output files (compare hashes).
9. The report lists every check X1 to X14 with a pass/fail/flagged status.

---

## 10. Appendix: reference values to verify against the real files

| Item | Expected | Source |
|---|---|---|
| Households | 2,500 | Guide / Proposal |
| Households with demographics | 801 | Proposal |
| Campaigns | 30 (IDs 1 to 30), types A/B/C | Guide |
| `campaign_table` rows (household-campaign mailings) | 7,208 | Proposal |
| Mailed pairs with at least one redemption | 889 (12.3%) | Proposal |
| `week_no` range | 1 to 102 | Guide |
| Transaction rows | ~2.6M | my recollection of the public dataset **[Verify]** |
| Distinct products | ~90k | my recollection **[Verify]** |
| Day range | roughly 1 to 711 | my recollection **[Verify]** |
| Households present in `campaign_table` | fewer than 2,500 | inferred **[Verify]** |
| Discount columns sign | ≤ 0 | Guide examples **[Verify]** |
| Whether `coupon_upc` repeats across campaigns | unknown | **[Verify]** |


---

## 11. Verification results (real files, 28 Sep 2026)

`verify_assumptions.py` ran on 7 of the 8 tables (`causal_data` not yet checked). Full output is in `verification_results.txt`.

| Topic | Result | Consequence |
|---|---|---|
| Headline counts | 2,500 households; 801 with demographics; 7,208 mailings; 30 campaigns; 1,584 households mailed at least once, 916 never mailed | the 916 never-mailed households are a ready comparison group for Q3 |
| Spine reproduction | 7,208 mailed pairs, 889 with a redemption (12.33%); 0 redeemed pairs that were not mailed | acceptance tests 2 and 3 work as written |
| Redemption rate by type | TypeA 16.0% (635 of 3,979); TypeB 7.9% (210 of 2,655); TypeC 7.7% (44 of 574) | TypeA is 55% of mailings but 71% of redeemed pairs; campaign type will be a strong signal, so evaluate per type |
| Redemption integrity | 2,318 rows; 0 orphans, 0 outside the campaign dates, 0 missing coupons, 0 exact duplicates | the "every redemption matches a mailed campaign" check passes with zero exceptions |
| Net revenue formula (D1) | `SALES_VALUE - COUPON_MATCH_DISC` is higher than `SALES_VALUE` on 100% of the 17,449 lines with a match discount | confirmed: the literal formula adds the match discount back, so it gives the amount *before* the match, not net of it. Still needs a mentor decision on what Q2 should measure |
| Discount signs | `coupon_disc` and `coupon_match_disc` always ≤ 0; `retail_disc` ≤ 0 except 36 rows | flag the 36 rows |
| Day to week | `week_no = ceil((day + 2) / 7)` on every row | resolves the day/week question for `causal_data` |
| Transactions | 2,595,732 rows; 92,339 products, all present in `product`; 18,850 zero-value lines; no negative sales; 14,466 rows with quantity ≤ 0; 23,136 rows with quantity > 100 | look into the quantity outliers before using unit counts |
| Campaign timing | IDs not chronological; 74 of 435 pairs overlap; campaigns 21 and 22 identical; campaign 24 extends past the last transaction day | split by start day with tie-breaks; treat campaign 24's window as truncated |
| Coupon table | 5,164 exact duplicates; 171 of 1,135 UPCs reused across campaigns; TypeA pool 181 to 209 coupons | join on (campaign, coupon_upc); TypeA targeted products would be about 8% of the pool for any household |
| History before campaigns | earliest campaign starts day 224; minimum history 134 days, median 433; 108 pairs have under 26 weeks | 8- and 13-week lookbacks have full history for every pair; a 26-week lookback leaves 108 pairs (1.5%) short |
| Demographics | anonymized `classification_*` columns (section 4.2) | neutral names; 233 households have Unknown homeowner status |

Still unverified: `causal_data` (run the script with `--causal`).
