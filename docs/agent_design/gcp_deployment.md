# Deploying the Lead Agent to GCP

This is the first real step toward the Decision Lab: the Lead Agent engine (`src/lead/engine.py`),
running on Google Cloud instead of a laptop, on a daily timer, exactly as described in
`docs/agent_design/lead_agent_design.md`'s "Real version" section.

## What's built, as of this doc

- `Dockerfile` — packages the engine's code (not the data; see below) into a container image.
- `scripts/gcp_entrypoint.sh` — the container's entrypoint: pulls data down from a GCS bucket, runs
  `src/lead/engine.py`, pushes the results back up.
- This doc, with the exact commands to deploy it.

**Not built yet, deliberately out of scope for this step:** the Gemini front door (the design doc's
real-version front door is Gemini through Google Cloud's AI service, not Claude, since GCP credits
don't cover third-party models); anything marketer-facing.

## Why data syncs through a bucket, not the image

Raw data is never committed to this repo (see the README's Contributing rules), so it can't be
baked into a container image either. Instead, the entrypoint script pulls `data/raw/` and the last
run's `data/processed/` down from a GCS bucket at the start of every run, and pushes the new
`data/processed/` (including `run_summary.json`/`.md` and `state.json`) back up at the end. This
also means the Lead Agent's staleness check (only rerun a step if its inputs changed) keeps working
across days, since `state.json` persists in the bucket between runs.

## What has to happen outside this repo, and who does it

These steps need a real Google account logged in and billing/credit details, so they can't be done
from inside a coding session -- a person has to do them directly in the Google Cloud Console or
their own terminal with `gcloud auth login` already run.

1. **Redeem credits and create a GCP project.** Whichever credit program the team is using (GCP for
   Education, a startup program, etc.) -- follow that program's own redemption flow, then create a
   new project from the resulting billing account.
2. **Enable the APIs this needs:** Cloud Run, Cloud Scheduler, Cloud Storage, Artifact Registry.
   ```bash
   gcloud services enable run.googleapis.com cloudscheduler.googleapis.com \
     storage.googleapis.com artifactregistry.googleapis.com
   ```
3. **Create the bucket and upload the raw data once:**
   ```bash
   gcloud storage buckets create gs://YOUR-BUCKET-NAME --location=us-east1
   gcloud storage rsync data/raw gs://YOUR-BUCKET-NAME/raw --recursive
   ```
4. **Build and push the image, then deploy it as a Cloud Run Job** (a Job, not a Service: the
   engine runs once and exits, it doesn't serve HTTP requests):
   ```bash
   gcloud builds submit --tag us-east1-docker.pkg.dev/YOUR-PROJECT/lead-agent/engine:latest
   gcloud run jobs create lead-agent-daily \
     --image us-east1-docker.pkg.dev/YOUR-PROJECT/lead-agent/engine:latest \
     --region us-east1 \
     --set-env-vars GCS_BUCKET=gs://YOUR-BUCKET-NAME,MODEL=all \
     --memory 2Gi --task-timeout 1800
   ```
5. **Schedule it to run daily:**
   ```bash
   gcloud scheduler jobs create http lead-agent-trigger \
     --location us-east1 --schedule "0 6 * * *" \
     --uri "https://us-east1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/YOUR-PROJECT/jobs/lead-agent-daily:run" \
     --http-method POST --oauth-service-account-email YOUR-RUNNER-SERVICE-ACCOUNT
   ```
6. **Run it once by hand to check it works, before trusting the schedule:**
   ```bash
   gcloud run jobs execute lead-agent-daily --region us-east1
   ```

## Proposed path from here to the marketer-facing prototype

```
Today:        Lead Agent engine runs on a laptop, Claude Code is the front door
This step:    Lead Agent engine runs on Cloud Run, on a schedule, data in a GCS bucket
Next:         swap the front door for Gemini (Vertex AI / Gemini Enterprise), same engine underneath
Then:         a small web UI (the actual "Decision Lab") that a marketer opens, which shows the
              latest run_summary in plain language and lets them answer open questions and approve
              ideas -- the same actions `rules.py` / `in_use.py` already support from a terminal,
              just behind buttons instead of commands
Finally:      the UI also reads scores (not yet built -- step 5 in the original pipeline) so a
              marketer can pull a target list directly, with the Decision Support Agent (step 7)
              answering plain-language questions about why a household is or isn't on it
```

The engine itself does not change between these stages -- only the front door and, eventually, what
else reads its output.
