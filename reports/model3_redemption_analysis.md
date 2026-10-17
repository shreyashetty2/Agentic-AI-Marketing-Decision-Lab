# Model 3: redemption-segmentation analysis


## Headline result

Of 3,229 treated TypeB/C households, 1,596 bought an
eligible product during the campaign window.

- Bought **via an actually-redeemed coupon**: 189
  (11.8%)
- Bought **without ever touching the coupon**: 1,407
  (88.2%)

**88.2% of what Model 3's main outcome currently counts as a
campaign "success" is purchase behavior that happened regardless of the mailer.** This is direct,
observed evidence for the "sure things vs. persuadables" pattern the T-learner baseline inferred
from spend levels -- not an inference this time, a direct count from redemption records.

## By campaign (campaigns with at least 20 treated purchasers)

```
          n_bought  n_without_redemption  pct_without_redemption
campaign                                                        
11              62                  59.0                    95.2
22             168                 156.0                    92.9
24              79                  72.0                    91.1
20             155                 141.0                    91.0
12             100                  91.0                    91.0
4               33                  30.0                    90.9
14             158                 142.0                    89.9
21              25                  22.0                    88.0
25             129                 113.0                    87.6
9               95                  83.0                    87.4
10              39                  34.0                    87.2
17             107                  93.0                    86.9
29              78                  67.0                    85.9
7               27                  23.0                    85.2
23             122                 102.0                    83.6
5               36                  30.0                    83.3
16              63                  51.0                    81.0
19              59                  47.0                    79.7
```

## What this means for Model 3 going forward

The current main outcome (`bought_campaign_product_excl_display_flyer`) overstates how often the
mailer is actually what drove a purchase. `bought_campaign_product_without_redemption` is now
available as a second, stricter outcome (Model 3's sensitivity label in
`model3_table.parquet`) for training a model that specifically targets persuadable households,
rather than ones who would have converted regardless.
