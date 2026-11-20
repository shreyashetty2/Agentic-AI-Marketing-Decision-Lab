# Decision Lab: sample questions and what the chat should answer

Every question below was run against the live chat logic. "Should say" is the point the answer must make.
If an answer ever drifts from these, treat it as a bug.

## The basics
| Question | Should say |
|---|---|
| What percent of coupons actually get redeemed? | 889 of 7,208 mailings (12.3%, about 1 in 8). Type A 16.0%, Type B 7.9%, Type C 7.7%. Real counts, not model scores. |
| What data is this built on? | dunnhumby Complete Journey, anonymised grocery loyalty data, about 2,500 households, 30 campaigns, 7,208 mailings to 1,584 households, no real dates. |
| What does campaign Type A mean? | Broad, storewide: a pool of about 180 to 210 coupons, each household got 16, which 16 is not recorded. Type B and C: small coupon sets, every household got all of them. |
| What is the difference between the three tools? | Redemption Likelihood (will they use it), Spend Forecast (what is it worth), Campaign Uplift (did it make a difference), plus sure things. |

## Targeting and results
| Question | Should say |
|---|---|
| Which households should we target for the most uplift? | Asks which campaign first, because the best households differ by campaign. Then gives a ranked list. |
| How confident are you that this beats random mailing? | +2.0 extra buyers per 100 mailed, range -0.6 to +4.5 includes zero, so it could be luck. |
| Which products do people buy without using the coupon? | About 88% (1,407 of 1,596) never redeemed. Examples: bananas 519 of 519 in campaign 8 (Type A, illustration only), eggs, pasta sauce and juice in targeted campaigns. |
| What about household 323? | Its uplift by campaign (about 21% in campaign 20, 10.5% in campaign 17), no demographics on file, never mailed, and the verdict caveat. |
| Which segment should we avoid mailing? | Lowest uplift groups with the "human needed" caveat, and the sure things tool. |

| Which households in campaign 17 would buy anyway, so we should not mail them? | Top 10 sure things with their chance of buying anyway (about 60 to 65%), from the sure-things classifier. Covers only the held-out campaigns 17, 19, 20 and 21. |
| Who should we target in campaign 17 to get the most uplift? | Top 10 by uplift (1984, 2325, 208 lead, about 15%), how it was ranked, and the "human needed" caveat. Without "uplift" it asks which measure you want. |
| Compare campaign 17 and campaign 8 | Type B (32 days, 15 categories) vs Type A (48 days, broad pool). Only Redemption Likelihood covers campaign 8. |
| Compare household 323 and household 373 | Very similar uplift in every campaign (campaign 17: 10.5% vs 11.1%). No demographics, neither mailed. |

| We are planning a new Type B campaign. Who should we target and who should we avoid? | Two sections. Target: kids 3+ and 2 (about 13 to 14% redeem), Age Group 6 and 1 for uplift. Avoid: Age Group 1 (lowest redemption), heavy past spenders on the promoted products (sure things). States it is an analogy to 8 past Type B campaigns, that gaps are small, and that habits matter more than demographics. |

## Honest "not available" answers
| Question | Should say |
|---|---|
| What month is best to mail? | The data has no real calendar dates. Offers campaign length and Type instead. |
| What is the ROI, or how many households for 1,000 coupons? | Not computed yet. Combining chance of redeeming times expected spend is the next step. The tools rank households and the marketer picks the depth. |
| Does the discount size matter? | Not measured. Only the share of spend on discounts is a model input. |
| Does it work for other stores? | Not known. Only one chain's data. |

## Trust and method
| Question | Should say |
|---|---|
| What does human needed mean? | Automatic checks could not settle it, so a person should judge. Approve means it passed. |
| What does causal mean here? | Did the mailing cause the buying, comparing mailed with not mailed households. |
| Why is mailed vs not mailed hard to compare? | Mailing was not random, a gap remains after weighting (0.20 vs the 0.10 target). |
| Why did you pick the X-learner? | The mailed to unmailed split is about 12 to 1, and the X-learner is built for lopsided groups. Gain is real on open test campaigns, not clean on the locked one. |
| Can I trust the chat? | Every number comes from the agents' output. Redemption Likelihood is approved, the other two need human judgment. |
