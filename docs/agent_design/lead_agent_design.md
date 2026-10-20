# Lead Agent: Design

This document explains the lead agent: the single agent a data scientist talks to, which runs the Data, Feature, Modeling and Evaluation Agents for them.

## 1. What the lead agent is

**A data scientist asks for what they want; the lead agent runs every step behind the scenes and reports back.**

- **The problem it solves:** today each agent is run separately, by hand, in the right order: six Data Agent scripts, two Feature Agent scripts, then the Modeling and Evaluation Agents. The person has to know that order, which files each step reads and writes, and what to do when one step sends work back to another.
- **What changes:** the data scientist only talks to the lead agent. They ask, for example, "Run Model 2 on the latest data." The lead agent runs the four agents in order, collects what each one reports, and gives the person one summary.
- **When it is used:** after the team has finished improving the data and the models. By then the known data questions have been answered and written into each agent's rules, so a normal run needs no input from anyone.
- **Who uses it:** a data scientist who did not build the system. Every message assumes the reader does not know the agents, the files or the models, and explains them in plain language.

## 2. How a run works

**The four agents pass files to each other; the lead agent decides what runs next, by written rules.**

```
Data scientist ("Run Model 2 on the latest data")   or   the daily timer
   ↓
Lead agent
   ↓  starts
Step 1: Data Agent        →  clean tables + what it handled + new questions
   ↓
Step 2: Feature Agent     →  feature tables + columns that moved unusually + new questions
   ↓
Step 3: Modeling Agent    →  every option's score + recommendation + the final model file
   ↓
Step 4: Evaluation Agent  →  verdict and reason + forecast check + new questions
   ↓
Lead agent  →  one summary to the data scientist
```

- **The agents never talk to each other directly.** Each one writes its outputs to files and the next one reads them, as they do today. This keeps every step traceable: anyone can open the files and see what was passed on.
- **The lead agent follows rules, not judgment,** to decide what runs next. For example: if the data has not changed since the last run, the Data and Feature Agents are skipped; if the Evaluation Agent sends a model back to features, the Feature Agent is next.
- **Two ways a run starts:**
  - **On a schedule**, with nobody involved. The lead agent only contacts a data scientist when something needs them.
  - **On request**, when a data scientist asks for something.
- **Two jobs, at two speeds:**
  - **Rescore every day.** New transactions update each household's scores, so the marketer's lists stay current. The models do not change.
  - **Retrain when a campaign finishes.** A model can only learn from a campaign once we know who redeemed, which is only known after the campaign ends.
- **For the prototype,** daily refreshes are simulated by replaying our two years of data: run as of day 500, then as of day 530, and so on. The design is for real daily data.

## 3. What each agent reports back

**Every agent reports the same four kinds of thing: whether it finished, what it produced, what changed, and any new question for a person.**

- **Data Agent**
  - Status: finished, or stopped and why.
  - What is new since the last run, e.g. "30 more days of transactions, 2 new campaigns."
  - One line on what it handled by its rules, e.g. "removed 12 duplicate rows, flagged 300 gas lines."
  - New questions: anything its rules cannot explain.
  - New check needed: a value it has never seen before (a new product department, a new campaign type) becomes a question.
- **Feature Agent**
  - Status: finished, or stopped and which check failed.
  - What it built, e.g. "three model tables, 2 new campaigns added."
  - Columns that moved far more than usual since the last run. For example, if average coupon use per week drops from 2.1 to 0.3 overnight, something probably broke upstream.
  - New questions, e.g. blanks it cannot explain.
  - Change needed: some of its checks compare against fixed counts from our dataset (75,000 rows, 7,208 mailed, 889 redeemers), which change with new data. They should compare against what the Data Agent reports it produced.
  - It also builds a new column when a data scientist approves one suggested by the brain; the lead agent passes the request on.
- **Modeling Agent**
  - Status: finished, or stopped.
  - Every option's score, best first.
  - A recommendation: keep the current model, or switch to an option that has beaten it consistently. Switching kinds of model stays the owner's decision.
  - New questions, e.g. a new model script in the repo that it does not know.
  - The final model file, for scoring.
  - It runs every option at each retrain (about 2 minutes for Model 1), so a current model slowly falling behind is noticed without anyone checking.
  - Change needed: its check that it reproduces the owners' numbers uses numbers recorded in October. It should instead check that it gives the same predictions as running the owner's script directly on the same data.
- **Evaluation Agent**
  - Status: finished, or stopped (for example, the test was not fair).
  - The verdict and its reason, in one line.
  - How the model did on the newest finished campaign, compared with its usual range. This is how a model slowly getting worse is caught.
  - Whether the forecast totals matched reality.
  - New questions for a person.
  - Change needed: the limit of 3 rounds of trying ideas exists so the same test campaigns are not reused until something passes by luck. Each newly finished campaign is fresh, so the count resets when one finishes.

**"Usual" is learned, not set by hand.** How much a column normally moves, and a model's usual range, are measured by replaying past data, with a minimum so that tiny changes are never flagged.

## 4. When the lead agent stops and asks

**Only something new stops a run; everything else goes in the summary.**

- **A surprise stops the run.** A surprise is something no rule covers: a never-seen value, a column that suddenly changed, broken data, a model far below its usual range. The lead agent stops and asks the data scientist, because results built on top of it cannot be trusted.
- **A normal outcome does not stop the run.** Scores, verdicts and recommendations are the system working as designed. For example, "Model 2's totals are 20% low, send back to modeling" is reported with next steps.
- **Every answer becomes a rule,** so the same question is never asked twice. The lead agent drafts the rule in plain words, the data scientist approves or edits it, and only then is it saved, with who decided, when, and why. For example: "I'll add a rule: 'Pharmacy Services' does not count as household spend, the same as gas. This applies to all three models from the next run. Okay to save?"

## 5. When a model is updated

**A retrained model replaces the current one only if it passes; changing the kind of model always needs a person.**

- **The retrained model passes the Evaluation Agent:** it replaces the current one automatically, and the summary says so.
- **It fails:** scoring keeps using the last approved model, and the lead agent explains what failed and why.
- **A different kind of model now scores best** (for example LightGBM instead of the random forest): the lead agent recommends it and asks the data scientist before switching.

## 6. How the lead agent explains things

**The person reading did not build the system, so every message explains itself.**

- **Name every model in plain words,** the current one and any new one. For example: "a random forest: 300 decision trees that each look at 11 facts about a household, then vote."
- **Say what changed:** only the data, the settings, the columns, or the kind of model.
- **Explain every improvement with evidence the system measured,** never with a bare "it got better." Acceptable reasons:
  - how much more training data the model now has
  - which columns the model relies on now, compared with before
  - which campaign types gained
  - whether the gain is larger than luck
- **If no reason can be shown, say so:** "the gain is small and could be luck."
- **End with a clear question when a decision is needed,** and a recommendation.

An example, when a different kind of model scores best (illustrative numbers):

> When I retrained Model 1, a different kind of model scored higher, and I'd like your input before switching.
>
> Current model: a random forest, 300 decision trees built separately that vote. Challenger: LightGBM, trees built one after another, each fixing the mistakes of the ones before.
>
> Over the last 4 finished campaigns, LightGBM caught 59% of redeemers in its top 20%, against 56% for the random forest. The likely range of that 3-point gap is +1 to +5, so it is unlikely to be luck.
>
> Why it likely improved: the model now trains on 24,000 rows, against 7,200 when the two were last compared, and LightGBM usually needs more data. The gain is in Type B and C campaigns (+5 points), where most of the new data arrived; Type A barely moved.
>
> My suggestion is to switch, since the gain is real and the reasons make sense. Because this changes the kind of model the team uses, should I switch, or keep the random forest?

## 7. Where it lives

**Two parts: an engine that runs everything, and a front door that talks to people.**

- **Engine** (plain code, no AI)
  - Runs the agents in order, applies the rules, stops on surprises.
  - Writes a run summary: what ran, the scores, what changed, the measured reasons, the open questions.
  - Being plain code, it gives the same result every time, and every number traces back to a calculation.
- **Front door** (a chat, the only part that uses AI)
  - Reads the run summary and explains it, following section 6.
  - Asks the questions, drafts new rules for approval, and starts the engine on request.
  - It only explains what the engine measured; it never produces a number or a reason of its own.
- **Prototype:**
  - Engine: a Python script, run on a laptop, with daily refreshes simulated by replaying the data.
  - Front door: a Claude Code skill, like the modeling agent's today.
- **Real version:**
  - Engine: on Google Cloud, started by a daily timer, paid for by the GCP credits.
  - Front door: Gemini, through Google Cloud's AI service (the Gemini Enterprise Agent Platform, formerly Vertex AI). GCP credits cover Google's own models but not third-party models such as Claude. The brain, which suggests improvements, needs a Gemini option for the same reason.
  - Only the front door and the timer differ from the prototype; the engine is the same.

## 8. The prototype

**In Claude Code, a data scientist types `/lead-agent run model 1` (or "as of day 660" for a replay) and gets the summary described above.**

- **Where it lives:**
  - `src/lead/engine.py`: runs the steps listed in `src/lead/lead_spec.yaml`, skips a step whose inputs are unchanged (by content, and it says which input changed when one did), and writes `run_summary.json` and `run_summary.md` to `data/processed/lead_agent/`.
  - `src/lead/asof.py`: the replay.
  - `src/lead/surprises.py`: the surprise checks.
  - `src/lead/rules.py` and `src/lead/lead_rules.yaml`: answers saved as rules.
  - `src/lead/describe.py`: the plain words for models, scores and columns.
  - `.claude/skills/lead-agent/SKILL.md`: the front door.
- **How the replay works:** every feature column is already computed from data before its campaign started, so the replay keeps the campaigns that had finished by the chosen day and leaves out those still running, whose answers are not known yet. The result is the same table a rebuild on data up to that day would give. Teammates' scripts run once, unchanged, on the full data. Each replay folder is named after the day the latest finished campaign ended, so days with no newly finished campaign share a folder and nothing is retrained. The one gap: the Data Agent's outlier cut-offs are learned from all two years; they only set flags, not model inputs.
- **Decisions made while building:**
  - **No held-back campaigns in a replay.** The Modeling Agent normally holds back its 5 newest test campaigns for a final check, so repeated idea-testing cannot pass by luck. In a replay every newly finished campaign is fresh, and the newest one is exactly what should be graded, so nothing is held back there.
  - **The surprise checks belong to the lead agent,** not to the Data and Feature Agents, so no teammate's files change. They run after the Feature Agent and before any model. Only everyday-shopping columns (spend, trips, coupon use, basket size) are watched for unusual moves: more than 4 times the usual campaign-to-campaign move, and at least 10%. Columns tied to the campaign itself, such as spend on its products, change between campaigns by design. On our two years nothing trips the check; an 80% drop in coupon use would be about 40 times the usual move.
  - **The newest campaign's usual range** counts only campaigns big enough to mean something (Model 1: at least 15 redeemers) and needs at least 5 earlier ones.
  - **Row counts:** the Modeling Agent checks the full dataset's exact counts only when reproducing the owners' numbers. Otherwise it records the rows it used, and the lead agent checks them against the table it was given.
  - **A replay never steps backwards:** replaying an earlier day never replaces the approved model with one that knows less.
  - **A rule never edits another agent's files.** It records the decision and says when that agent's owner needs to apply it.
- **Not built in the prototype:**
  - daily rescoring (it is Step 5 and belongs with the Decision Lab)
  - following a send-back automatically (the summary reports it and what to do next)
  - building a new column on request
  - reproducing the owners' numbers on new data (see section 3)

## 9. Open questions

- **What the lead agent hands to the Decision Lab.** This depends on what the Decision Lab does and needs, which is designed next.
- **Which GCP credits we have.** The terms above are Google's startup and free-trial terms; the exact terms of our credits should be confirmed when the account is set up.
