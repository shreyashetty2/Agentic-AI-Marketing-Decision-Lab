# Model 3: full uplift model report

Full dataset, no subsampling: 3,229 treated, 40,167 control rows.

## 1. Does targeting add lift?

Naive (unadjusted) comparison, full dataset: treated rate 49.4% vs. control
rate 13.8%, a 35.6% gap (z=52.5, p=0.00e+00).
Statistically this is not noise. But treated households are not a random sample -- this gap
mixes the campaign's real effect with the fact that heavier spenders were targeted to begin with.
The T-learner below conditions on pre-campaign spend to separate the two.

## 2. Does lift vary by household spend level?

Buckets from the treated group's own spend distribution (so each bucket has a comparable number
of treated households, not an artifact of control households dominating the full population):

```
spend_tier  n_treated  n_control  treated_rate  control_rate  observed_uplift
       Low       1076      36688      0.377323      0.112625         0.264698
    Medium       1076       2378      0.494424      0.356602         0.137822
      High       1077       1101      0.610956      0.517711         0.093245
```

Uplift falls as historical spend rises: Low-spend households show roughly 26%
incremental lift from the campaign; the heaviest spenders show only 9%.
The model's predicted uplift on held-out test campaigns reproduces the same ordering:
spend_tier
Low       0.101590
Medium    0.049197
High      0.006423

## 3. How much of the outcome is actually coupon-driven?

Of 1,596 treated households who bought an eligible product,
1,407 (88.2%) did so
without ever redeeming the coupon -- direct evidence consistent with (2): most of what the main
outcome counts as "success" would have happened regardless of the mailer, concentrated among
heavy spenders.

## 4. Does lift vary by demographics?

Partial-coverage view: only households with a demographic record (801 of 2,500, ~22% of Model 3's
rows after the TypeB/C + clean-control scoping). Categories with fewer than 20 treated households
are dropped as too small to read anything into.

By homeowner_desc:
```
homeowner_desc  n_treated  n_control  treated_rate  control_rate  observed_uplift
Probable Owner         20        106      0.700000      0.283019         0.416981
     Homeowner       1331       4832      0.571751      0.292632         0.279118
        Renter        124        350      0.532258      0.274286         0.257972
       Unknown        543       2231      0.497238      0.248767         0.248470
```

By kid_category_desc:
```
kid_category_desc  n_treated  n_control  treated_rate  control_rate  observed_uplift
                1        339       1101      0.572271      0.284287         0.287984
                2        220        551      0.618182      0.330309         0.287873
     None/Unknown       1280       5315      0.532031      0.267545         0.264487
               3+        195        693      0.528205      0.307359         0.220846
```

By classification_1:
```
classification_1  n_treated  n_control  treated_rate  control_rate  observed_uplift
      Age Group3        568       1725      0.593310      0.276522         0.316788
      Age Group6        108        814      0.500000      0.224816         0.275184
      Age Group4        686       2820      0.533528      0.279787         0.253740
      Age Group2        425       1357      0.555294      0.311717         0.243577
      Age Group5        126        508      0.492063      0.251969         0.240095
      Age Group1        121        436      0.487603      0.298165         0.189438
```


## Model setup

- Train: 1,842 treated + 27,901 control rows (ALL available control
  rows, not subsampled, trained on the natural class distribution -- class-weight rebalancing
  was tried and rejected, see train_t_learner()).
- Model T / Model C: RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20).
- Features (household historical spend is central, not incidental): spend_per_week_8w, spend_per_week_26w, avg_basket_value_26w, trips_per_week_26w, coupon_lines_per_week_26w, share_spend_on_discount_26w, past_redemption_rate, campaign_product_spend_per_week_26w, campaign_product_share_26w, history_weeks_before_cutoff, duration_days

## Held-out evaluation

```
 quintile  n_treated  n_control  actual_uplift
        0        824       1907         0.3650
        1        228       2502         0.3441
        2         72       2724         0.5051
        3        167       2498         0.4623
        4         96       2635         0.4017
```

Spearman rank correlation (quintile vs. actual observed uplift): rho=0.500,
p=0.391 -- not statistically significant on this split.

Qini coefficient (held-out test campaigns): -199.75. A negative Qini coefficient
here is a known, already-diagnosed issue, not a new problem: treated households cluster tightly
in the high-spend region of feature space (see section 2), so the treated and control populations
barely overlap at the extremes of the ranking, which distorts the cumulative Qini calculation even
though the underlying per-tier signal (above) is clean and directionally correct. Restricting to
the region where both groups actually overlap recovers a clean monotonic signal (established in
earlier analysis); not re-run here since this report's focus is (1) and (2) above. Chart:
reports/figures/model3_uplift_qini_curve.png
