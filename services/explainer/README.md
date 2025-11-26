# Explainer (Internal)

## Purpose
Consumes Redis Streams, performs concept extraction and optional shadow-model verification, and persists a Hypergraph (HIF).

Worker code: [services/explainer/src/worker.py](services/explainer/src/worker.py), core extraction in [concept_extraction()](services/explainer/src/worker.py:243)

## Operability
Env vars:
- REDIS_URL
- DEV_MODE (0 recommended in production)
- SPACY_MODEL (e.g., en_core_web_sm)
- DATABASE_URL (Postgres DSN; writes to EXPLAINER_TABLE)
- EXPLAINER_TABLE (e.g., explanations_v2)
- SHADOW_ENDPOINT (optional)
- VERIFY_MODEL, VERIFY_TEMPERATURE, VERIFY_TOP_K (optional)
- COST_* (optional accounting toggles/limits)

## Run and deploy
- docker run -d --name explainer --network hypergraph-net -e REDIS_URL=redis://redis:6379/0 -e DATABASE_URL=postgresql://postgres:postgres@postgres:5432/hypergraph -e DEV_MODE=0 -e SPACY_MODEL=en_core_web_sm yourco-explainer

## Observability
- Logs: per-trace lifecycle (started, extracted, verified, persisted).
- Metrics: consumer lag, extraction duration, verification success rate.

## Failure modes
- Poison messages: implement DLQ or max-retry policy; add guardrails around parsing.
- spaCy model issues: ensure model present; fallback to simpler tokenization when unavailable.
- Shadow verification failures: treat as non-fatal; record status and proceed with base graph.
