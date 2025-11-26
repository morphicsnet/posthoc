# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

# System setup
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python deps
COPY services/gateway/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY services/gateway/src ./src
ENV PYTHONPATH=/app/src
# Redis defaults
ENV REDIS_URL=redis://redis:6379/0
ENV REDIS_STREAM=hypergraph:completions
ENV REDIS_CONSUMER_GROUP=gateway

ENV DATABASE_URL=postgresql://postgres:postgres@postgres:5432/hypergraph
ENV EXPLAINER_TABLE=explanations_v2
ENV DB_CONNECT_TIMEOUT=5

# Runtime
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python - <<'PY'\nimport urllib.request,sys\ntry:\n  urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2)\nexcept Exception:\n  sys.exit(1)\nPY

# Launch FastAPI app
# Module path resolved via PYTHONPATH=/app/src to load app object from app.py
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]