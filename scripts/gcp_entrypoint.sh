#!/bin/sh
# Cloud Run Job entrypoint for the Lead Agent.
#
# Raw data is never committed to the repo or baked into the image (see README's Contributing
# rules), so this syncs it from a GCS bucket at the start of every run, and syncs the processed
# outputs (including the Lead Agent's own run_summary.json/md and state.json) back up at the end,
# so the next day's run picks up from where this one left off and the results are inspectable
# without needing to exec into the container.
#
# Required env vars (set on the Cloud Run Job, never in the image): GCS_BUCKET, MODEL (default "all").
set -eu

: "${GCS_BUCKET:?GCS_BUCKET must be set (e.g. gs://td-decision-lab-data)}"
MODEL="${MODEL:-all}"

echo "Syncing data down from ${GCS_BUCKET}..."
mkdir -p data/raw data/processed
gcloud storage rsync "${GCS_BUCKET}/raw" data/raw --recursive
gcloud storage rsync "${GCS_BUCKET}/processed" data/processed --recursive || true   # first run: nothing to pull yet

echo "Running the Lead Agent (model ${MODEL})..."
python src/lead/engine.py --model "${MODEL}"
status=$?

echo "Syncing results back up to ${GCS_BUCKET}..."
gcloud storage rsync data/processed "${GCS_BUCKET}/processed" --recursive

exit "${status}"
