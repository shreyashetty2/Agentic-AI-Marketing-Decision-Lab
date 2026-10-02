"""
Model 3 -- Slide Numbers Audit Trail
=====================================
Every number used in the Model 3 slides and shreya-master-reference-model3.md,
traced back to the exact CSV file, column(s), and pandas operation that produced it.

Run this top to bottom. Each section prints: the number, which slide/section it
supports, and the precise source + operation. Nothing here is estimated or
remembered from a prior run -- it's recomputed from the raw CSVs every time.
"""
import pandas as pd
import numpy as np

DATA_DIR = '/Users/shreyashetty/Documents/Projects_and_Assignments/DSI_Capstone/Agentic-AI-Marketing-Decision-Lab/data/dunnhumby_The-Complete-Journey CSV/'

pd.set_option('display.width', 200)


def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


# ---------------------------------------------------------------------------
section("1. DATASET SCALE NUMBERS (Section 4 of master reference)")
# ---------------------------------------------------------------------------
tx = pd.read_csv(DATA_DIR + 'transaction_data.csv')
product = pd.read_csv(DATA_DIR + 'product.csv')
causal = pd.read_csv(DATA_DIR + 'causal_data.csv', dtype={'display': str, 'mailer': str})
demo = pd.read_csv(DATA_DIR + 'hh_demographic.csv')
camp_desc = pd.read_csv(DATA_DIR + 'campaign_desc.csv')
camp_table = pd.read_csv(DATA_DIR + 'campaign_table.csv')
coupons = pd.read_csv(DATA_DIR + 'coupon.csv')
coupon_redempt = pd.read_csv(DATA_DIR + 'coupon_redempt.csv')

print(f"[transaction_data.csv] len(df)                        = {len(tx):,}  (total transaction line items)")
print(f"[transaction_data.csv] tx['household_key'].nunique()   = {tx['household_key'].nunique():,}  (total households)")
print(f"[product.csv]          product['PRODUCT_ID'].nunique() = {product['PRODUCT_ID'].nunique():,}  (total unique products)")
print(f"[hh_demographic.csv]   demo['household_key'].nunique() = {demo['household_key'].nunique():,}  (households WITH demographics)")
print(f"[causal_data.csv]      len(df)                         = {len(causal):,}  (display/mailer records)")
print(f"[campaign_desc.csv]    camp_desc['DESCRIPTION'].value_counts():")
print(camp_desc['DESCRIPTION'].value_counts().to_string())
print(f"[campaign_table.csv]   len(df)                         = {len(camp_table):,}  (total household-campaign mailings)")

# Redemption rate: 889/7208 = 12.3%
redeemed_pairs = coupon_redempt[['household_key', 'CAMPAIGN']].drop_duplicates()
print(f"\n[coupon_redempt.csv]   drop_duplicates(household_key, CAMPAIGN).shape[0] = {len(redeemed_pairs):,}  "
      f"(household-campaign pairs with >=1 redemption)")
print(f"REDEMPTION RATE = {len(redeemed_pairs):,} / {len(camp_table):,} = {len(redeemed_pairs)/len(camp_table)*100:.1f}%")


# ---------------------------------------------------------------------------
section("2. TYPEA PRODUCT POOL SIZES (Section 5 -- why TypeA is excluded)")
# ---------------------------------------------------------------------------
# coupon.csv: CAMPAIGN -> PRODUCT_ID (many rows per campaign, one per eligible product)
# groupby CAMPAIGN, count distinct PRODUCT_ID = how many products that campaign's "pool" covers
prod_counts = coupons.groupby('CAMPAIGN')['PRODUCT_ID'].nunique().rename('n_products').reset_index()
merged = camp_desc.merge(prod_counts, on='CAMPAIGN', how='left')
total_catalog = product['PRODUCT_ID'].nunique()
merged['pct_of_catalog'] = (merged['n_products'] / total_catalog * 100).round(1)
print("[coupon.csv groupby CAMPAIGN -> nunique(PRODUCT_ID)], shown against [campaign_desc.csv] for type:")
print(merged[merged['DESCRIPTION'] == 'TypeA'][['CAMPAIGN', 'DESCRIPTION', 'n_products', 'pct_of_catalog']]
      .sort_values('n_products', ascending=False).to_string(index=False))
print(f"\n(divided by total catalog size {total_catalog:,} from product.csv nunique(PRODUCT_ID))")

# Campaign 8 sanity check: actual redemption rate vs the inflated "eligible product" rate
c8_treated = set(camp_table[camp_table['CAMPAIGN'] == 8]['household_key'].unique())
c8_redeemers = set(coupon_redempt[coupon_redempt['CAMPAIGN'] == 8]['household_key'].unique())
print(f"\n[campaign_table.csv filtered CAMPAIGN==8] nunique(household_key) = {len(c8_treated)} (treated households)")
print(f"[coupon_redempt.csv filtered CAMPAIGN==8] nunique(household_key)  = {len(c8_redeemers)} (ACTUAL redeemers)")
print(f"ACTUAL redemption rate for campaign 8 = {len(c8_redeemers)}/{len(c8_treated)} = {len(c8_redeemers)/len(c8_treated)*100:.1f}%  "
      f"(compare to the 96.56% 'eligible product purchase rate' the broken v1 diagnostic produced)")


# ---------------------------------------------------------------------------
section("3. SCOPE: TypeB/TypeC campaign and mailing counts")
# ---------------------------------------------------------------------------
valid_campaigns = camp_desc[camp_desc['DESCRIPTION'].isin(['TypeB', 'TypeC'])].sort_values('START_DAY')
typeB_C_mailings = camp_table[camp_table['CAMPAIGN'].isin(valid_campaigns['CAMPAIGN'])]
print(f"[campaign_desc.csv filtered DESCRIPTION in (TypeB,TypeC)] count = {len(valid_campaigns)} campaigns")
print(f"[campaign_table.csv filtered to those CAMPAIGN ids] len(df)    = {len(typeB_C_mailings):,} mailings")
print(f"Share of all mailings = {len(typeB_C_mailings):,} / {len(camp_table):,} = {len(typeB_C_mailings)/len(camp_table)*100:.1f}%")


# ---------------------------------------------------------------------------
section("4. BUILDING THE POOLED HOUSEHOLD x CAMPAIGN PANEL (43,396 rows)")
# ---------------------------------------------------------------------------
all_hhs = set(tx['household_key'].unique())
tx_small = tx[['household_key', 'PRODUCT_ID', 'DAY', 'WEEK_NO', 'STORE_ID', 'SALES_VALUE', 'BASKET_ID']]

rows = []
for _, camp_info in valid_campaigns.iterrows():
    campaign_id = camp_info['CAMPAIGN']
    camp_type = camp_info['DESCRIPTION']
    start_day, end_day = camp_info['START_DAY'], camp_info['END_DAY']
    start_week, end_week = start_day // 7, end_day // 7

    # Treatment pool for THIS campaign: [coupon.csv] filtered to this CAMPAIGN -> unique PRODUCT_ID
    target_products = coupons[coupons['CAMPAIGN'] == campaign_id]['PRODUCT_ID'].unique()
    if len(target_products) == 0:
        continue

    # Treatment group: [campaign_table.csv] filtered to this CAMPAIGN -> unique household_key
    treated_hhs = set(camp_table[camp_table['CAMPAIGN'] == campaign_id]['household_key'].unique())

    # "Clean" control: exclude households mailed by any OTHER campaign whose [START_DAY,END_DAY]
    # window (from campaign_desc.csv) overlaps this campaign's window
    overlapping = camp_desc[(camp_desc['CAMPAIGN'] != campaign_id) &
                             (camp_desc['START_DAY'] <= end_day) &
                             (camp_desc['END_DAY'] >= start_day)]['CAMPAIGN'].unique()
    contaminated_hhs = set(camp_table[camp_table['CAMPAIGN'].isin(overlapping)]['household_key'].unique())
    control_hhs = all_hhs - treated_hhs - contaminated_hhs

    # hist_spend / hist_visits: [transaction_data.csv] filtered to DAY < start_day (strictly BEFORE
    # the campaign), grouped by household_key, SALES_VALUE summed / BASKET_ID counted distinct
    pre_tx = tx_small[tx_small['DAY'] < start_day]
    hh_spend = pre_tx.groupby('household_key')['SALES_VALUE'].sum()
    hh_visits = pre_tx.groupby('household_key')['BASKET_ID'].nunique()

    # hist_target_affinity: same pre-period transactions, further filtered to PRODUCT_ID in this
    # campaign's eligible-product pool, SALES_VALUE summed per household
    pre_target_tx = pre_tx[pre_tx['PRODUCT_ID'].isin(target_products)]
    hh_target_affinity = pre_target_tx.groupby('household_key')['SALES_VALUE'].sum()

    # Outcome: [transaction_data.csv] filtered to WEEK_NO in [start_week,end_week] AND PRODUCT_ID
    # in the eligible pool, LEFT-JOINED to [causal_data.csv] on (PRODUCT_ID, STORE_ID, WEEK_NO),
    # then exclude any row where display != '0' or mailer != '0' (i.e. a recorded in-store/mailer
    # promotion existed for that exact product+store+week)
    window_tx = tx_small[(tx_small['WEEK_NO'] >= start_week) & (tx_small['WEEK_NO'] <= end_week)]
    target_tx = window_tx[window_tx['PRODUCT_ID'].isin(target_products)]
    merged_tx = pd.merge(target_tx, causal, on=['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'], how='left')
    merged_tx['is_confounded'] = (
        ((merged_tx['display'].notna()) & (merged_tx['display'] != '0')) |
        ((merged_tx['mailer'].notna()) & (merged_tx['mailer'] != '0'))
    )
    pure_purchasers = set(merged_tx.loc[~merged_tx['is_confounded'], 'household_key'].unique())

    study_hhs = treated_hhs | control_hhs
    df = pd.DataFrame({'household_key': list(study_hhs)})
    df['CAMPAIGN'] = campaign_id
    df['campaign_type'] = camp_type
    df['n_eligible_products'] = len(target_products)
    df['campaign_length_days'] = end_day - start_day
    df['is_treated'] = df['household_key'].isin(treated_hhs).astype(int)
    df['hist_spend'] = df['household_key'].map(hh_spend).fillna(0)
    df['hist_visits'] = df['household_key'].map(hh_visits).fillna(0)
    df['hist_target_affinity'] = df['household_key'].map(hh_target_affinity).fillna(0)
    df['outcome_purchased'] = df['household_key'].isin(pure_purchasers).astype(int)
    rows.append(df)

panel = pd.concat(rows, ignore_index=True)
print(f"panel.shape = {panel.shape}  <- one row per (household, campaign) pair, for 25 TypeB/TypeC campaigns")
print(f"(panel['is_treated']==1).sum() = {(panel['is_treated']==1).sum():,}   <- treated rows (slide: '3,229 mailings')")
print(f"(panel['is_treated']==0).sum() = {(panel['is_treated']==0).sum():,}  <- control rows")

treated_rate = panel[panel.is_treated == 1]['outcome_purchased'].mean()
control_rate = panel[panel.is_treated == 0]['outcome_purchased'].mean()
print(f"\npanel[panel.is_treated==1]['outcome_purchased'].mean() = {treated_rate*100:.2f}%  (slide: 'raw treated rate')")
print(f"panel[panel.is_treated==0]['outcome_purchased'].mean() = {control_rate*100:.2f}%  (slide: 'raw control rate')")


# ---------------------------------------------------------------------------
section("5. THE 4.64x SELECTION-BIAS NUMBER -- exact provenance")
# ---------------------------------------------------------------------------
print("Computed on the FULL POOLED PANEL (all 25 campaigns, both the eventual train AND test split,")
print("43,396 rows) -- this is the number in Section 6 / the 'selection bias check' cell, NOT the")
print("same subset as the 'sure things vs persuadables' number in section 7 below (that one is")
print("computed on the held-out TEST campaigns ONLY -- different denominator, similar-looking result).")
print()
print("Code: panel.groupby('is_treated')[['hist_spend','hist_visits','hist_target_affinity']].mean()")
bias_check = panel.groupby('is_treated')[['hist_spend', 'hist_visits', 'hist_target_affinity']].mean()
bias_check['n'] = panel.groupby('is_treated').size()
print(bias_check)
ratio_full_panel = bias_check.loc[1, 'hist_spend'] / bias_check.loc[0, 'hist_spend']
print(f"\nRATIO = treated hist_spend / control hist_spend = {bias_check.loc[1,'hist_spend']:.2f} / "
      f"{bias_check.loc[0,'hist_spend']:.2f} = {ratio_full_panel:.2f}x  <-- THE '4.64x' SLIDE NUMBER")
print("hist_spend itself = sum of transaction_data['SALES_VALUE'] for each household, restricted to")
print("DAY < that campaign's START_DAY (campaign_desc.csv), computed separately per campaign, then")
print("every household-campaign row is stacked into the panel above and averaged by group.")


# ---------------------------------------------------------------------------
section("6. OUT-OF-TIME TRAIN/TEST SPLIT -- exactly which campaigns are in each")
# ---------------------------------------------------------------------------
campaign_order = panel[['CAMPAIGN']].drop_duplicates()
cutoff = int(len(campaign_order) * 0.7)
train_campaigns = set(campaign_order['CAMPAIGN'].iloc[:cutoff])
test_campaigns = set(campaign_order['CAMPAIGN'].iloc[cutoff:])
print("valid_campaigns sorted by START_DAY (campaign_desc.csv); first 70% = train, last 30% = test.")
print(f"\nTRAIN campaign IDs (17): {sorted(train_campaigns)}")
print(f"TEST campaign IDs (8):  {sorted(test_campaigns)}")
print("\n*** THE QINI CURVE AND ALL 'HELD-OUT' RESULTS ON THE SLIDES ARE COMPUTED ON THESE 8 TEST ***")
print("*** CAMPAIGNS ONLY (7 TypeB + 1 TypeC: campaign 20) -- NOT on all 25 TypeB/TypeC campaigns. ***")
print("The model is TRAINED on the 17 earliest campaigns and SCORED on the 8 latest, exactly like")
print("the out-of-time split planned for Model 1.")

train = panel[panel['CAMPAIGN'].isin(train_campaigns)].copy()
test = panel[panel['CAMPAIGN'].isin(test_campaigns)].copy()
print(f"\ntrain.shape = {train.shape}   test.shape = {test.shape}")


# ---------------------------------------------------------------------------
section("7. T-LEARNER TRAINING + 'SURE THINGS vs PERSUADABLES' NUMBERS")
# ---------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

features = ['hist_spend', 'hist_visits', 'hist_target_affinity', 'n_eligible_products', 'campaign_length_days']
train_t = train[train['is_treated'] == 1]
train_c = train[train['is_treated'] == 0]
train_c_sampled = train_c.sample(n=min(len(train_c), len(train_t) * 5), random_state=42)

model_t = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42)
model_c = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42)
model_t.fit(train_t[features], train_t['outcome_purchased'])
model_c.fit(train_c_sampled[features], train_c_sampled['outcome_purchased'])

print(f"Model T trained on train_t = train[train.is_treated==1]  -> {len(train_t)} rows")
print(f"Model C trained on a random sample of train[train.is_treated==0] -> {len(train_c_sampled)} rows "
      f"(of {len(train_c)} available control rows; subsampled 5x the treated count to balance)")
print(f"Model T train AUC = roc_auc_score(train_t.outcome_purchased, model_t.predict_proba(...)) "
      f"= {roc_auc_score(train_t['outcome_purchased'], model_t.predict_proba(train_t[features])[:,1]):.3f}")
print(f"Model C train AUC = {roc_auc_score(train_c_sampled['outcome_purchased'], model_c.predict_proba(train_c_sampled[features])[:,1]):.3f}")

test = test.copy()
test['p_treated'] = model_t.predict_proba(test[features])[:, 1]
test['p_control'] = model_c.predict_proba(test[features])[:, 1]
test['predicted_uplift'] = test['p_treated'] - test['p_control']

print("\n--- 'Sure Things vs Persuadables' numbers (Section 7 of master reference) ---")
print("Computed on the TEST SET ONLY (13,653 rows, the 8 held-out campaigns) -- note this is a")
print("DIFFERENT subset than the 4.64x number above, which used the FULL 43,396-row panel.")
behavior_by_actual = test.groupby('is_treated')[['hist_spend', 'hist_visits', 'hist_target_affinity']].mean()
print(behavior_by_actual.round(1))
ratio_test_only = behavior_by_actual.loc[1, 'hist_spend'] / behavior_by_actual.loc[0, 'hist_spend']
print(f"(On test-set-only, the ratio is {ratio_test_only:.2f}x -- similar magnitude to the 4.64x full-panel number, "
      f"but NOT the identical calculation -- different rows going into the average.)")

model_outputs_by_actual = test.groupby('is_treated')[['p_treated', 'p_control', 'predicted_uplift']].mean()
print("\nmodel outputs (p_treated, p_control, predicted_uplift) averaged by ACTUAL is_treated status:")
print(model_outputs_by_actual.round(4))
print("-> is_treated==1 (ACTUAL treated / 'Sure Things'): p_treated and p_control are nearly identical")
print("-> is_treated==0 (ACTUAL control / 'Persuadables'): bigger gap between p_treated and p_control")


# ---------------------------------------------------------------------------
section("8. QUINTILE VALIDITY CHECK + SPEARMAN CORRELATION (held-out test set)")
# ---------------------------------------------------------------------------
from scipy.stats import spearmanr

test['quintile'] = pd.qcut(test['predicted_uplift'], 5, labels=False, duplicates='drop')
qrows = []
for q in sorted(test['quintile'].dropna().unique()):
    b = test[test['quintile'] == q]
    t_rate = b[b.is_treated == 1]['outcome_purchased'].mean()
    c_rate = b[b.is_treated == 0]['outcome_purchased'].mean()
    qrows.append({'quintile': int(q), 'n_treated': (b.is_treated == 1).sum(), 'n_control': (b.is_treated == 0).sum(),
                  'actual_treated_rate': round(t_rate, 4), 'actual_control_rate': round(c_rate, 4),
                  'actual_observed_uplift': round(t_rate - c_rate, 4)})
qdf = pd.DataFrame(qrows)
print("test['quintile'] = pd.qcut(test['predicted_uplift'], 5)  -- buckets the 13,653 test rows by")
print("predicted uplift, then for each bucket: mean(outcome_purchased) separately for is_treated==1")
print("rows and is_treated==0 rows, subtracted. This is all on REAL, OBSERVED outcomes -- not predictions.")
print(qdf.to_string(index=False))
rho, p = spearmanr(qdf['quintile'], qdf['actual_observed_uplift'])
print(f"\nspearmanr(qdf['quintile'], qdf['actual_observed_uplift']) = rho={rho:.3f}, p={p:.3f}")

bottom20 = test[test['predicted_uplift'] <= test['predicted_uplift'].quantile(0.2)]
top20 = test[test['predicted_uplift'] >= test['predicted_uplift'].quantile(0.8)]
for name, b in [('Bottom 20%', bottom20), ('Top 20%', top20)]:
    t_rate = b[b.is_treated == 1]['outcome_purchased'].mean()
    c_rate = b[b.is_treated == 0]['outcome_purchased'].mean()
    print(f"{name} predicted uplift: n_treated={(b.is_treated==1).sum()}, "
          f"actual_observed_uplift = {t_rate:.3f} - {c_rate:.3f} = {t_rate-c_rate:.3f}")


# ---------------------------------------------------------------------------
section("9. FORMAL QINI CURVE (on the 8 held-out test campaigns -- see Section 6)")
# ---------------------------------------------------------------------------
test_sorted = test.sort_values('predicted_uplift', ascending=False).reset_index(drop=True)
N_t = (test_sorted['is_treated'] == 1).sum()
N_c = (test_sorted['is_treated'] == 0).sum()
cum_t = (test_sorted['is_treated'] * test_sorted['outcome_purchased']).cumsum()
cum_c = ((1 - test_sorted['is_treated']) * test_sorted['outcome_purchased']).cumsum()
frac = (np.arange(len(test_sorted)) + 1) / len(test_sorted)
qini = cum_t - cum_c * (N_t / N_c)
random_line = frac * qini.iloc[-1]
qini_coefficient = np.trapezoid(qini - random_line, frac)
print(f"N_treated (test set) = {N_t}, N_control (test set) = {N_c}  (sums to {N_t+N_c} = test.shape[0])")
print(f"Qini(phi) = cumsum(is_treated*outcome) - cumsum((1-is_treated)*outcome) * (N_t/N_c), "
      f"sorted by predicted_uplift descending")
print(f"Qini coefficient = area under (qini - random_line) via np.trapezoid = {qini_coefficient:.2f}")

# Common-support trim
t_spend = test[test.is_treated == 1]['hist_spend']
c_spend = test[test.is_treated == 0]['hist_spend']
lo = max(t_spend.quantile(0.05), c_spend.quantile(0.05))
hi = min(t_spend.quantile(0.95), c_spend.quantile(0.95))
trimmed = test[(test['hist_spend'] >= lo) & (test['hist_spend'] <= hi)].copy()
trimmed = trimmed.sort_values('predicted_uplift', ascending=False).reset_index(drop=True)
N_t2, N_c2 = (trimmed.is_treated == 1).sum(), (trimmed.is_treated == 0).sum()
cum_t2 = (trimmed['is_treated'] * trimmed['outcome_purchased']).cumsum()
cum_c2 = ((1 - trimmed['is_treated']) * trimmed['outcome_purchased']).cumsum()
frac2 = (np.arange(len(trimmed)) + 1) / len(trimmed)
qini2 = cum_t2 - cum_c2 * (N_t2 / N_c2)
random_line2 = frac2 * qini2.iloc[-1]
qini_coef2 = np.trapezoid(qini2 - random_line2, frac2)
print(f"\nCommon-support band on hist_spend (5th-95th percentile overlap of BOTH groups): ${lo:.0f}-${hi:.0f}")
print(f"Rows after trim: {len(trimmed)} (treated={N_t2}, control={N_c2})")
print(f"Qini coefficient on trimmed sample = {qini_coef2:.2f}  (compare to {qini_coefficient:.2f} on full test set)")

print("\n" + "=" * 100)
print("AUDIT COMPLETE. Every number above should match the slides / master reference file exactly.")
print("=" * 100)
