#!/bin/sh
# Cloud Run Service entrypoint for the Decision Lab interface (app/streamlit_app.py).
#
# The app only ever reads data/processed (never raw data -- see app/lib/data.py), so this syncs just
# that down from the same GCS bucket the Lead Agent engine writes to, then starts Streamlit. No sync
# back up: this container never produces new processed output, it only displays what the engine wrote.
#
# Required env vars (set on the Cloud Run service, never in the image): GCS_BUCKET.
# Vertex AI auth (the chat page) comes from the Cloud Run service's own identity -- no key file,
# no ANTHROPIC_API_KEY-style secret to manage.
#
# Sync uses the Python Storage client (scripts/sync_processed.py), not `gcloud storage`: the gcloud
# CLI keeps its own credential store, which a Cloud Run service's attached identity does not populate
# automatically (confirmed live: `gcloud storage rsync` failed here with "no active account", while
# google.auth -- what this script and the chat page both use -- picks up that same identity with no
# setup).
set -eu

: "${GCS_BUCKET:?GCS_BUCKET must be set (e.g. gs://capstone-decision-lab-data)}"

echo "Syncing processed data down from ${GCS_BUCKET}..."
python scripts/sync_processed.py "${GCS_BUCKET}"

echo "Starting the Decision Lab on port ${PORT:-8080}..."
exec streamlit run app/streamlit_app.py \
    --server.port="${PORT:-8080}" \
    --server.address=0.0.0.0 \
    --server.headless=true
