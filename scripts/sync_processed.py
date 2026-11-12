"""Pull data/processed/ down from GCS, via the Python Storage client (not `gcloud storage`).

The `gcloud` CLI binary keeps its own credential store, which Cloud Run's attached service-account
identity does not populate automatically -- confirmed by a failed deploy where `gcloud storage rsync`
errored "no active account" inside the container, while google.auth (what this script and the chat
page both use) picks up that same identity with no setup. Read-only: the app never writes new output,
so this never needs to sync back up.

Usage: python scripts/sync_processed.py gs://BUCKET_NAME
"""
import sys
from pathlib import Path

from google.cloud import storage

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if len(sys.argv) != 2 or not sys.argv[1].startswith("gs://"):
        raise SystemExit("usage: python scripts/sync_processed.py gs://BUCKET_NAME")
    bucket_name = sys.argv[1].removeprefix("gs://").rstrip("/")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    n = 0
    for blob in client.list_blobs(bucket, prefix="processed/"):
        if blob.name.endswith("/"):
            continue
        dest = ROOT / "data" / blob.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))
        n += 1
    print(f"Synced {n} file(s) from gs://{bucket_name}/processed to data/processed/")


if __name__ == "__main__":
    main()
