
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts --install-dir=/opt && \
    rm -rf /var/lib/apt/lists/*
ENV PATH="/opt/google-cloud-sdk/bin:${PATH}"

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/gcp_entrypoint.sh scripts/gcp_entrypoint.sh
RUN chmod +x scripts/gcp_entrypoint.sh

ENTRYPOINT ["scripts/gcp_entrypoint.sh"]
