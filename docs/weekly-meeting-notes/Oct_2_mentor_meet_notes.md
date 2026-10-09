## October 2, 2026 Meeting Notes (by Pranav Jain)

The team presented progress on the Agentic AI Marketing Decision Lab: validated data findings, scopes for three models, and one shared agent architecture. Reviewers approved the plan to build Model 1 first.


### Data Agent and Data Validation

- Presented by Claudia

- 8 raw files are cleaned, validated, and joined into one household-by-campaign table that feeds all 3 models.

- Confirmed 7,208 household campaign mailings, 889 with a redemption, a 12.33% redemption rate matching the proposal.

- First-draft ingestion script built, covering 2.6 million transaction rows and 36.8 million display and mailer rows, plus a data quality report.

- GitHub repo scaffolded with a README and a folder structure matching the 5-agent pipeline.



### Model 1: Response Probability

- Presented bu John

- Predicts the 0 to 100% probability a household redeems at least 1 coupon from a campaign (binary classification).

- About 88% of recipients redeem nothing; around 2,500 households over a little over 2 years, demographics for roughly 30-32%.

- 30 campaigns of types A, B, C; Type A redemption is much higher, so the model must add value within each type.

- Transaction features use only pre-campaign days to avoid data leakage.



### Models 2 and 3: Expected Value and Uplift

- Model 2 presented by Anna, is a regression on redeeming households: sum of sales value for campaign-related products during the campaign window.

- Open question for Model 2: how to define campaign-related products for Type A, where exact coupon exposure is unknown.

- Model 3 presented by Shreya, estimates incremental mail effect using treatment vs. control groups, scoped to Type B and C campaigns only.

- Purchases of products on in-store display in the same window are removed as a confounder

- Two-model approach (treatment minus control), evaluated with a Qini-style uplift curve instead of ROC-AUC

- Mailers skew toward heavy spenders, so this is uplift adjusted for observed confounders, not a random experiment.



### Shared Agent Architecture

- Presented by Pranav

- One pipeline serves all 3 models; only per-model instruction documents change, not the agent process.

- Feature agent splits into shared step 2A transformations and model-specific step 2B filtering.

- Modeling and evaluation agents follow the same generic process, guided by per-question instruction sets.



### Reviewer Questions

- In-store promotion group looked small in initial tests; methodology may be adjusted if it proves large.

- Speaker 4 questioned excluding Type A; for Model 1 it stays in, though personalization may add some bias.

- Overlapping campaigns mean a redemption can't be attributed to one campaign; any response counts as one observation.



### Next Steps

- (Speaker 1) Review the presentation and send over anything the team should consider this week

- (Speaker 2) Email the presentation document along with the proposed deliverables for next week

- (Speaker 2) Meet as a team to finalize a plan for what to accomplish over the next week



### Decisions Made

- Model 1 includes all campaign types, including Type A

- One shared agent architecture for all 3 models, with Model 1 built end-to-end first

- Model 3 is scoped to Type B and C campaigns only