# Modeling Agent, Evaluation Agent and the Brain: Design

## What these agents are

Before these agents existed, each model was a hand-written script, and every choice in it was made by a person: `model1_baseline.py`, `model2_baseline.py` and `model3_uplift.py`. Each script has its own columns, its own model and its own 70/30 split.

The README's plan is one shared **Modeling Agent** (pipeline step 3) and one shared **Evaluation Agent** (step 4). These agents are that plan, built for all three models, plus a **brain** that suggests what to try next:

- **Modeling Agent** (`src/modeling/agent.py`, instructions in `src/modeling/modeling_spec.yaml`): the test bench.
  - trains every option for a model the same honest way
  - chooses between them by a written rule
  - hands its predictions to the Evaluation Agent
- **Evaluation Agent** (`src/evaluation/agent.py`, instructions in `src/evaluation/evaluation_spec.yaml`): the referee.
  - judges each model on campaigns it never trained on, by its own measures
  - shows where any gain comes from: columns, learner or settings
  - decides what happens next: approve, send back to modelling, send back to features, or ask a person
  - lists every open question for a person, naming who should answer it
- **Brain** (`src/modeling/brain.py`, rules in `src/modeling/proposals.py`): the advisor.
  - reads the latest results and explains why they came out that way
  - proposes up to 5 changes to test next, each with a reason and the result that would show it worked
  - never decides anything: a person approves each proposal before it is tested

**They orchestrate; they do not re-model.** The models belong to their owners:

- Model 1: Yuxin, evaluated by John
- Model 2: Yuxin. Her current model is `model2_value.py` (usual spend, calibrated); `model2_baseline.py` stays in as a comparison.
- Model 3: Shreya

Each model's default option *is* the owner's current code, called unchanged through `src/modeling/owners.py`.

**Staying on the owners' latest model.** The agent calls the owners' code live, so an edit to an owner's script is picked up on the next run. A *new* script is different: the agent would keep using the one it knows. So the instruction file lists every model script it knows (`scripts`), and every run compares that list with the scripts committed to the repo. A new one is flagged for a person (MOD-09): it may be the owner's newer model. If the agents prefer an alternative, or the verdict is negative, that is written up as a question for the owner (`decided_by: human_needed`). Nothing is applied automatically.

**Why they exist:** to make model development better, not to save work. Every genuine question (an inconclusive result, a disagreement with an owner, a correction that did not fully work) is flagged to a person instead of being settled silently.

## The loop, and who does what

1. **Modeling Agent** runs the owner's model and the alternatives on every test campaign.
2. **Evaluation Agent** judges them and writes a verdict plus "Questions for a person".
3. **Brain** reads that verdict and proposes changes.
4. **A data scientist** reads the proposals and sets `approved: true` on the ones worth testing. They can also add their own.
5. Back to step 1 with the approved proposals as extra options (`--proposals`).
6. When a choice looks settled, the Evaluation Agent's **`--unlock`** checks it once on campaigns that were held back the whole time.

The commands:

```
python src/modeling/agent.py --model all --mode both        # round 1: reproduce owners' numbers, test every option
python src/evaluation/agent.py --model all                  # verdicts and questions
python src/modeling/brain.py --model 1                      # proposals (needs ANTHROPIC_API_KEY; --manual works without)
#   ...a person approves proposals in data/processed/modeling_agent/model1/proposals.yaml...
python src/modeling/agent.py --model 1 --mode forward --proposals --iteration 2
python src/evaluation/agent.py --model 1
python src/evaluation/agent.py --model 1 --unlock           # final check on the held-back campaigns, once
```

**Inputs:**

- the Feature Agent's per-model tables (`data/processed/model{n}_table.parquet`)
- the two instruction files: which options to try, how to choose, how to judge
- the owners' own functions, through a fixed mapping (`owners.py`); the agents do not read the rest of the repo
- for the brain: the latest verdict, the column descriptions from the Feature Agent's spec, and the proposals made so far

**Outputs**, per model, under `data/processed/modeling_agent/model{n}/` and `data/processed/evaluation_agent/model{n}/`:

- **Modeling Agent:** `predictions.parquet` (open test campaigns), `locked_predictions.parquet` (held-back campaigns), `manifest.json` (instruction-file hashes, folds, choices and scores), `decisions.json` / `.csv`, `final_model.joblib`, `report.md`
- **Evaluation Agent:** `verdict.json`, `decisions.*`, `report.md`; after `--unlock`: `unlock_report.md`, `unlock.json`, `unlock_history.json`
- **Brain:** `brief.md` (exactly what it was told), `brain_diagnosis.md` (its reading of the results), `brain_response.json`, `proposals.yaml`
- **Shared:** `data/processed/modeling_agent/proposal_ledger.csv`, every proposal ever made, approved or not, with its results

Code writes every number, so the reports are exact, and identical for identical runs.

## How every model is tested: forward chaining and locked campaigns

**Forward chaining:** each test campaign is predicted by models trained only on campaigns that had **ended** before its cutoff (its start day minus 7). This is John's `forward_chain_folds`, used for all three models. The old 70/30 split is kept only as a reproduction check.

**Locked campaigns:** each model's last few test campaigns are predicted the same way but written to a separate file, which the Evaluation Agent reads only on `--unlock`.

- **Why:** every round of proposals is judged on the same open campaigns. Try enough ideas and one will look better by luck. The locked campaigns were never used to choose anything, so they give one clean check. (Pranav's Model 1 experiments showed this: new columns looked better on the campaigns used for choosing and the gain disappeared on held-back ones.)
- **Each unlock is recorded.** A second unlock after more changes is flagged as a "second look", because anything chosen after the first look has been partly fitted to those campaigns.

Test campaigns:

- **Model 1:** 18 test campaigns, 5,785 rows. Locked: 21–25 (811 rows), the same five the Model 1 experiments held back.
- **Model 2:** 11 test campaigns, 167 rows. Locked: 23–25 (54 rows). Its first 4 possible test campaigns have fewer than 80 training rows, so they are skipped and the skip is logged.
- **Model 3:** 15 test campaigns, 24,397 rows. Locked: 22–25 (7,905 rows).

## How the Modeling Agent chooses

For each test campaign, it uses only that campaign's training campaigns:

1. Score every option on the last 4 of them, each predicted out of time.
2. Measure each option with the model's selection measure:
   - Model 1: within-campaign AUC (a ranking score: how often a redeemer is ranked above a non-redeemer)
   - Model 2: average dollar error
   - Model 3: propensity-weighted within-campaign Qini (extra buyers per mailed household compared with mailing at random, after correcting for who got mailed)
3. **Rule:** an alternative replaces the owner's default only if it is better by at least `min_gain`. If several alternatives are within `min_gain` of each other, the simplest one wins.
4. If there is too little finished history to compare fairly, keep the default, and flag it.

**One change at a time.** The code works out what each option changes compared with the owner's model, from its columns, learner and settings: `columns`, `learner`, `settings`, or several. The instruction file includes options that change only one thing:

- other columns (`extended_forest`)
- another learner (`owner_logistic`, `owner_cols_lightgbm`)
- other settings: deeper and shallower trees (`deeper_forest`, `shallower_forest`)

So the Evaluation Agent can say where a gain comes from. This answers the mentors' question: is a model better because of features or because of tuning?

For the final model, the owner's default is refit on every campaign and saved for scoring (`final_model.joblib`). The agent's own preference uses the same rule over its open-campaign predictions, and it is only a recommendation.

## The brain

**What it reads** (`brief.md`, written fresh each time):

- the model's task, answer column and the owner's columns and settings
- every option's result on the open campaigns, and exactly what each option changes
- the verdict and the questions the Evaluation Agent raised
- every column it could add, with its description and a rough signal (rank correlation with the answer, open campaigns only)
- columns it cannot add as they are, and why (text that needs turning into numbers, timeline columns that would leak the future)
- the settings it may change, with allowed ranges
- the proposals already made and how they did

It never sees the locked campaigns.

**What it may propose:**

- add or drop columns
- another learner: forest, LightGBM, logistic, or linear for dollar answers
- other settings, within allowed ranges
- a column that does not exist yet, with an exact definition using only data from before the cutoff. These cannot be tested yet; they go to the Feature Agent.

**How proposals are checked** (`proposals.validate`), before a person sees them:

- no answer column, treatment column or timeline column; only profile/context columns
- no text columns, and no blanks that 0 does not describe
- a learner the task supports, and settings inside the allowed ranges
- not the owner's model unchanged, and not a repeat of an option already tested
- restating something unchanged (the owner's learner, a setting at its current value) is not a change

Each proposal is saved with its status (`ready`, `rejected` with the reason, or `needs_feature_agent`) and `approved: false`. A person flips `approved`; the Modeling Agent re-checks approved proposals before running them (a person may have edited them) and stops if one fails.

**Two ways to run it:**

- **With an AI** (`brain.py --model N`): one call to Claude per model per run, so a bug cannot loop and run up a bill. The key is read from the environment or the local `.env` file, which is never committed. One call is about 4,000 tokens in and 2,000 out: a few cents per model.
- **By hand** (`--manual` writes the brief only; a person writes proposals, then `--check`). Same checks, same ledger.

**The ledger** (`proposal_ledger.csv`) records every proposal: who made it, whether it passed the checks, whether a person approved it, and its result on the open and locked campaigns. Over the project it answers the research question "can agents improve model development?" with evidence, for example how many ideas the brain had, how many people tried, and how many held up.

## What the Evaluation Agent decides, and what the code does

**Code computes:**

- every measure, on the open test campaigns only
- each gap against the baseline, with a 95% interval from resampling whole households (one household appears in many campaigns)
- every option against the owner's model, and the best single-change option per kind (columns / learner / settings)
- a calibration check
- for Model 3: whether weighting really made mailed and unmailed households comparable (balance)

**Baselines:**

- Model 1: the "redeemed before" rule
- Model 2: usual spend
- Model 3: random targeting, plus a "lowest spenders first" rule

**Calibration checks:**

- Model 1: honest forecast ranges must cover at least 80% of campaigns
- Model 2: the predicted total must be within 15% of the actual total
- Model 3: predicted uplift on mailed households vs. the weighted estimate of the real effect

**Rules decide**, checked in order:

1. **Mailed and unmailed households not comparable (Model 3):** too little overlap, or still too different after weighting. A person decides.
2. **Round 3 or later:** a person decides. Round 1 is the owner's model; rounds 2–3 are proposals. After that the open campaigns have been reused enough.
3. **Beats its baseline and calibrated:** approve. For Model 3, approval goes to a person, because no rule can check for hidden confounders.
4. **Beats its baseline but not calibrated:** send back to modelling. For Model 1 the hint is "widen forecast ranges; do not recalibrate", because recalibrating on past errors made Model 1 worse when we tested it. Other models get no such hint: that lesson was learned on Model 1 only.
5. **Doesn't beat its baseline, and an alternative beats the owner's model, or the agent's choice keeps flipping between folds:** send back to modelling.
6. **Doesn't beat its baseline, and every option is within noise of the others:** send back to features. The signal is not in the columns.
7. **Anything else:** a person decides.

**Questions for a person** are listed in every report, each tagged with who should answer: inconclusive results, alternatives that beat the owner's model, numbers that don't match reality, measures that differ from the owner's, and everything the Modeling Agent flagged. Duplicates between the two agents are merged.

## Results (15 Oct 2026)

**Correction to the first run (14 Oct):** that run said Model 3 ranks "worse than random". That was wrong. Its correction for "the store mailed heavier spenders" used a straight-line (logistic) model, which left mailed and unmailed households far apart (balance gap 0.79). With boosted trees the gap is 0.20–0.28, and the owner's model comes out about equal to random. The agent now checks the correction on every run and flags it when it falls short.

### Round 1: the owners' models and the built-in options (open campaigns)

**Model 1: approve.**

- The owner's forest beats the "redeemed before" rule by **+8.7 points** of capture (range +5.2 to +12.4).
- Forecast ranges covered 92% of 13 open campaigns.
- No other option reliably beats it. Columns only: +1.1 points, can't tell. Settings only: +0.4, can't tell. LightGBM on the owner's columns: −3.3, reliably worse.

**Model 2: send back to modelling.**

- The owner's forest does not beat usual spend (−$0.54 average error, range −$2.05 to +$0.80).
- **Learner only:** the owner's own linear regression beats the forest by **$1.50** (range $0.40 to $2.93). Settings only (deeper or shallower trees): no gain. So the gain comes from the learner, not from columns or tuning: with about 200 rows, the forest flattens its predictions.

**Model 3: a person decides.**

- On the open campaigns, mailed and unmailed households still differ after weighting (gap 0.28 on campaign-product spend; 0.25 is the limit). Every uplift number is approximate.
- Taken at face value: the owner's model vs. random targeting +0.007 (range −0.020 to +0.030), can't tell.
- Columns only: +0.012, reliably better. Settings only (shallower trees): +0.017, reliably better.

### Model 2 update: Yuxin's calibrated model (later on 15 Oct)

Yuxin's new current Model 2, `model2_value.py`, is a straight line on usual spend fit through the middle of the answers. The agent now uses it as Model 2's current model (the numbers above judged her earlier random forest).

- On the agent's test it **beats usual spend by $0.90** of average error (range $0.32 to $1.59): her own finding holds under the stricter test, where training campaigns must have ended.
- Her earlier forest is reliably worse than it ($1.44); her linear regression ties it.
- **Verdict: send back to modelling**, because the predicted total is 20% below the actual total. Her own report notes the same: every model under-predicts the later, higher-spending campaigns.

### Round 2: the brain's proposals

The brain made 15 proposals (5 per model): 12 testable, 3 new-column ideas for the Feature Agent. All 12 were approved for this test run.

- **Model 1:** none of the 4 beats the owner's forest. Its new-column idea, coupon use on the campaign's own products, is the same column the Model 1 experiments prototyped by hand.
- **Model 2:** 3 linear variants beat the forest, but none beats the owner's own linear regression.
- **Model 3:** 3 of 4 beat the owner's model on the open campaigns. Best: shallower trees plus two extra columns, +0.026 (range +0.013 to +0.039).

### The locked-campaign check (first unlock)

**Nothing that looked better on the open campaigns held up.**

- **Model 1:** every option within noise of the owner's forest or worse. The owner's forest vs. the rule: +9.8 points, but with only 811 rows the range (−1.6 to +18.3) includes 0.
- **Model 2:** with 54 rows, almost nothing can be told apart. The linear regression's edge is +$0.41 (range −$1.24 to +$2.24).
- **Model 3:** the Modeling Agent's pick from round 2 fell from +0.024 to +0.001; the best open-campaign result fell from +0.026 to +0.009. Both can't tell.

**What this means:** the process works as intended. Gains that looked real on reused campaigns did not survive a clean test, and the agents said so instead of adopting them. But the locked sets are small, so they can catch a big failure, not confirm a small gain.

## Open questions for the team

- **Locked sets are small** (54 rows for Model 2). Is one clean check worth giving up those campaigns for the open comparisons?
- **Round limit:** a person decides from round 3. Is two rounds of proposals the right number?
- **Gains are measured against the owner's model only.** The brain wanted to test changes against Model 2's linear regression instead; that is not supported yet.
- **Choosing by AUC but judging by capture** (Model 1), and judging Model 3 on the weighted Qini rather than Shreya's unweighted one: both need sign-off.
- **The brain's key** is a personal key in a local `.env`. Teammates without one use `--manual`.

## How we know it is done

Checked by `tests/modeling/acceptance_agents.py` on the real data (27 of 27 passing), plus the unit tests in `tests/modeling` and `tests/evaluation` (168 passing):

- **The owners' own numbers are reproduced exactly:**
  - Model 1, 70/30: 61.1% capture, AUC 0.820
  - Model 1, John's forward chain: 57.0% vs. 48.2%, interval [0.055, 0.124]
  - Model 2, 70/30: average dollar error 13.18 / 11.71 / 11.68 / 14.16
  - Model 3, 70/30: Qini −199.83, plus the full quintile table
- **The default option equals the owner's code:** its predictions are identical to calling the owner's functions directly.
- **No look-ahead:** every training campaign ended before its test campaign's cutoff.
- **Locked campaigns stay locked:** they appear only in the locked file, and the recommendation does not move when their predictions are scrambled.
- **The owners' settings are recorded correctly** in the instruction file (a test reads them from the owners' scripts), so "settings only" really means settings only.
- **Proposals:** every unsafe or pointless proposal type is rejected with its reason; only approved proposals run; the brain makes exactly one structured call.
- **Repeatable:** two runs give identical predictions.
- **Every decision has a decider and a reason**, and Model 3 is never approved by a rule.
- **The owners' Model 1/2 scripts are unchanged.** Shreya's script has only a surgical fix: `.to_numpy(copy=True)` twice, so it runs on pandas 3 (it crashed before), plus a calculation-only `qini_points()` split out of `qini_curve()`. No numbers change.
- **Run time:** round 1 for all three models takes about 4 minutes; with 12 proposals, about 7.
