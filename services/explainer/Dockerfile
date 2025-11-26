# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install deps
COPY services/explainer/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Pre-download spaCy model during build for faster cold starts
RUN python -m spacy download en_core_web_sm
# Default spaCy model (overridable at runtime)
ENV SPACY_MODEL=en_core_web_sm

# Copy source
COPY services/explainer/src ./src
ENV PYTHONPATH=/app/src

# Default to dev mode (single-run sample HIF)
ENV DEV_MODE=1
ENV S3_BUCKET=
ENV S3_PREFIX=traces

ENV DATABASE_URL=postgresql://postgres:postgres@postgres:5432/hypergraph
ENV EXPLAINER_TABLE=explanations_v2
ENV DB_CONNECT_TIMEOUT=5

# Run the worker (DEV_MODE=1 will process one envelope and exit)
CMD ["python", "-u", "src/worker.py"]