# Setup

Audience: Both

## Prerequisites
- Docker 24+ (optional: Docker Compose)
- Python 3.11+ (for tooling and validator)
- Network access to pull images and download spaCy models (if needed)

## Environment configuration
Recommended .env defaults:

```
# Ports
GATEWAY_PORT=8080
INTERCEPTOR_PORT=8081          # external
INTERNAL_INTERCEPTOR_PORT=8080 # container/internal

# Infrastructure
REDIS_URL=redis://redis:6379/0
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/hypergraph

# Upstream LLM proxy
LLM_PROXY_URL=http://llm-proxy:8080

# Explainer
EXPLAINER_TABLE=explanations_v2
SPACY_MODEL=en_core_web_sm
DEV_MODE=0

# Optional shadow verification
SHADOW_ENDPOINT=http://ollama:11434
VERIFY_MODEL=gpt-4o-mini
VERIFY_TEMPERATURE=0.0
VERIFY_TOP_K=3
```

## Environment variables (cross-cutting matrix)

The following cross-cutting environment variables apply across HIF v1 components: Gateway, Interceptor (Async Sidecar), and Explainer. Localhost examples assume Gateway at http://localhost:8080.

### Gateway
- AUTH_MODE (default: none) — auth mode: none|static. See [rbac.py](services/gateway/src/rbac.py:1)
- AUTH_TOKENS_JSON (default: unset) — JSON mapping token -> identity payload (tenant_id, scopes, subject). Exact shape documented in [secrets_and_keys.md](docs/runbooks/secrets_and_keys.md:1)
- AUTH_EXPECT_ISS (default: unset) — optional issuer pin check. See [rbac.py](services/gateway/src/rbac.py:120)
- AUTH_EXPECT_AUD (default: unset) — optional audience pin check. See [rbac.py](services/gateway/src/rbac.py:124)
- RATE_LIMIT_READ_RPS (default: 50), RATE_LIMIT_READ_BURST (default: 200) — read category. See [rate_limit.py](services/gateway/src/rate_limit.py:95)
- RATE_LIMIT_WRITE_RPS (default: 5), RATE_LIMIT_WRITE_BURST (default: 20) — write category. See [rate_limit.py](services/gateway/src/rate_limit.py:99)
- RL_COUNTERS_JSON_PATH (default: unset) — persistence path for limiter state. See [rate_limit.py](services/gateway/src/rate_limit.py:104)
- ENABLE_OTEL (default: 0) — enable metrics/tracing/logs. See [otel.py (gateway)](services/gateway/src/otel.py:330)
- OTEL_EXPORTER_OTLP_ENDPOINT (default: unset), OTEL_EXPORTER_OTLP_PROTOCOL (default: grpc|http) — tracing outputs. See [otel.py (gateway)](services/gateway/src/otel.py:64)
- STATUS_BACKEND (default: json) — shared status/index backend: json|ddb. See [status_store.py](services/explainer/src/status_store.py:208)
- STATUS_JSON_PATH (default: /tmp/hif/status.json) — JSON store path. See [status_store.py](services/explainer/src/status_store.py:63)
- TRACE_TTL_DAYS (default: 7) — TTL computation days. See [app.py](services/gateway/src/app.py:174)
- LLM_PROXY_URL (default: unset) — upstream LLM proxy base. Required for chat.
- REDIS_URL (default: redis://redis:6379/0), REDIS_STREAM (default: hypergraph:completions), REDIS_MAXLEN (default: unset) — work queue stream. See [app.py](services/gateway/src/app.py:103) and [interceptor](services/interceptor/src/capture.py:100)
- DATABASE_URL (default: unset) — Postgres connection for historical explanation lookups. See [app.py](services/gateway/src/app.py:423)
- EXPLAINER_TABLE (default: explanations_v2) — table name. See [app.py](services/gateway/src/app.py:114)
- GATEWAY_REDACT (default: 0) — redact message copies in enqueue payloads. See [app.py](services/gateway/src/app.py:189)
- GATEWAY_VERSION (default: unknown) — used in metrics/log labels. See [otel.py (gateway)](services/gateway/src/otel.py:289)

### Explainer
- ENABLE_OTEL (default: 0) — enable metrics/tracing. See [otel.py (explainer)](services/explainer/src/otel.py:274)
- EXPLAINER_METRICS_HOST (default: 0.0.0.0), EXPLAINER_METRICS_PORT (default: 9090) — /metrics server. See [otel.py (explainer)](services/explainer/src/otel.py:256)
- REDIS_URL, REDIS_STREAM, REDIS_MAXLEN — same semantics as Gateway; consumer side.
- S3_BUCKET (default: unset), S3_PREFIX (default: traces), S3_KMS_KEY_ID (default: unset) — SSE-KMS if key set. Local fallback when bucket unset. See [s3_store.py](services/explainer/src/s3_store.py:62)
- CHAOS_CONTROL_PATH (default: /tmp/hif/chaos.json) — chaos flags file used by tests; optional for demos (see [load_stress_chaos.md](docs/testing/load_stress_chaos.md:1))
- USE_SAE_LOADER / ENABLE_SAE_SERVICE / DICT_ROOT — optional SAE dictionary loading patterns; course-specific or internal usage. See [loader.py](libs/sae/loader.py:1)
- ATTR_* — attribution budgets/knobs (implementation-defined)
- BP_* — backpressure thresholds (implementation-defined). See [backpressure.py](services/explainer/src/backpressure.py:64)
- HG_* — hypergraph pruning knobs (implementation-defined)
- DEV_MODE (default: 0), SPACY_MODEL (default: en_core_web_sm) — if used in your build, see [docs/setup.md (existing vars)](docs/setup.md:27)
- SHADOW_ENDPOINT, VERIFY_MODEL, VERIFY_TEMPERATURE, VERIFY_TOP_K — optional verification path per your deployment (see [docs/setup.md](docs/setup.md:31))

### Interceptor (Async Sidecar)
- REDIS_URL, REDIS_STREAM, REDIS_MAXLEN — stream enqueue settings. See [capture.py](services/interceptor/src/capture.py:100)
- HOST (default: 0.0.0.0), PORT (default: 8080 external or as configured) — ingress server. See [capture.py](services/interceptor/src/capture.py:101)

### Observability/OTEL
- ENABLE_OTEL (default: 0) — enable JSON logs, Prometheus metrics, and optional OpenTelemetry tracing. See [otel.py (gateway)](services/gateway/src/otel.py:330), [otel.py (explainer)](services/explainer/src/otel.py:274)
- OTEL_EXPORTER_OTLP_ENDPOINT (default: unset), OTEL_EXPORTER_OTLP_PROTOCOL (default: grpc|http) — OTLP exporter endpoint and protocol. See [otel.py (gateway)](services/gateway/src/otel.py:64) and [otel.py (explainer)](services/explainer/src/otel.py:83)
- EXPLAINER_METRICS_HOST (default: 0.0.0.0), EXPLAINER_METRICS_PORT (default: 9090) — Prometheus /metrics server for Explainer. See [otel.py (explainer)](services/explainer/src/otel.py:256)
- GATEWAY_VERSION (default: unknown) — included on metrics/log labels. See [otel.py (gateway)](services/gateway/src/otel.py:289)
- EXPLAINER_VERSION (default: unknown) — included on metrics/log labels. See [otel.py (explainer)](services/explainer/src/otel.py:149)
### Security/RBAC
- AUTH_MODE (default: none) — auth mode: none|static. See [rbac.py](services/gateway/src/rbac.py:1)
- AUTH_TOKENS_JSON (default: unset) — JSON mapping token -> identity payload (tenant_id, scopes, subject). Exact shape documented in [secrets_and_keys.md](docs/runbooks/secrets_and_keys.md:1) and guidance in [SECURITY.md](docs/security/SECURITY.md:1)
- AUTH_EXPECT_ISS (default: unset) — optional issuer pin check. See [rbac.py](services/gateway/src/rbac.py:120)
- AUTH_EXPECT_AUD (default: unset) — optional audience pin check. See [rbac.py](services/gateway/src/rbac.py:124)
- GATEWAY_REDACT (default: 0) — redact message copies in enqueue payloads. See [app.py](services/gateway/src/app.py:189)

### Storage/Persistence
- STATUS_BACKEND (default: json) — shared status/index backend: json|ddb. See [status_store.py](services/explainer/src/status_store.py:208)
- STATUS_JSON_PATH (default: /tmp/hif/status.json) — JSON store path. See [status_store.py](services/explainer/src/status_store.py:63)
- RL_COUNTERS_JSON_PATH (default: unset) — persistence path for rate limiter state. See [rate_limit.py](services/gateway/src/rate_limit.py:104)
- DATABASE_URL (default: unset) — Postgres connection for historical explanation lookups. See [app.py](services/gateway/src/app.py:423)
- EXPLAINER_TABLE (default: explanations_v2) — table name. See [app.py](services/gateway/src/app.py:114)
- S3_BUCKET (default: unset), S3_PREFIX (default: traces), S3_KMS_KEY_ID (default: unset) — SSE-KMS if key set. See [s3_store.py](services/explainer/src/s3_store.py:62)
- REDIS_URL (default: redis://redis:6379/0), REDIS_STREAM (default: hypergraph:completions), REDIS_MAXLEN (default: unset) — work queue stream configuration. See [capture.py](services/interceptor/src/capture.py:100) and [app.py](services/gateway/src/app.py:103)
### Observability and Delivery cross-links
- OTEL Collector reference: [otel-collector.yaml](manifests/otel/otel-collector.yaml:1)
- KEDA scaler settings: [keda-scalers.yaml](manifests/helm/hypergraph/templates/keda-scalers.yaml:1) with values under [values.yaml](manifests/helm/hypergraph/values.yaml:49)
- Security and IAM: [SECURITY.md](docs/security/SECURITY.md:1), [iam-policies.md](docs/security/iam-policies.md:1)
- Runbooks:
  - Deploy: [deploy.md](docs/runbooks/deploy.md:1)
  - Scale & Recovery: [scale_and_recovery.md](docs/runbooks/scale_and_recovery.md:1)
  - Secrets & Keys: [secrets_and_keys.md](docs/runbooks/secrets_and_keys.md:1)
  - Attach Rate & Capacity: [attach_rate_and_capacity.md](docs/runbooks/attach_rate_and_capacity.md:1)
  - Microbenchmark & Autoscaling: [microbenchmark_autoscaling.md](docs/runbooks/microbenchmark_autoscaling.md:1)

## Build images
From the repo root:

- docker build -t yourco-gateway -f services/gateway/Dockerfile services/gateway
- docker build -t yourco-interceptor -f services/interceptor/Dockerfile services/interceptor
- docker build -t yourco-explainer -f services/explainer/Dockerfile services/explainer

## Run locally (Docker)
Create a network:
- docker network create hypergraph-net || true

Start infra:
- docker run -d --name redis --network hypergraph-net -p 6379:6379 redis:7-alpine
- docker run -d --name postgres --network hypergraph-net -p 5432:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_USER=postgres -e POSTGRES_DB=hypergraph postgres:16-alpine

Optional shadow model (example):
- docker run -d --name ollama --network hypergraph-net -p 11434:11434 ollama/ollama:latest

Start services:
- docker run -d --name interceptor --network hypergraph-net -p 8081:8081 -e REDIS_URL=redis://redis:6379/0 -e HOST=0.0.0.0 -e PORT=${INTERCEPTOR_PORT:-8081} yourco-interceptor
- docker run -d --name explainer --network hypergraph-net -e REDIS_URL=redis://redis:6379/0 -e DATABASE_URL=postgresql://postgres:postgres@postgres:5432/hypergraph -e DEV_MODE=${DEV_MODE:-0} -e SPACY_MODEL=${SPACY_MODEL:-en_core_web_sm} -e SHADOW_ENDPOINT=${SHADOW_ENDPOINT:-http://ollama:11434} -e VERIFY_MODEL=${VERIFY_MODEL:-gpt-4o-mini} yourco-explainer
- docker run -d --name gateway --network hypergraph-net -p 8080:8080 -e REDIS_URL=redis://redis:6379/0 -e DATABASE_URL=postgresql://postgres:postgres@postgres:5432/hypergraph -e LLM_PROXY_URL=${LLM_PROXY_URL:-http://llm-proxy:8080} yourco-gateway

## Verify
- Note: If AUTH_MODE=static is enabled, include `-H "Authorization: Bearer $TOKEN"` on requests. See [rbac.py](services/gateway/src/rbac.py:85).
Gateway health (may be unimplemented; check logs if non-200):
- curl -sS http://localhost:8080/healthz || echo "Note: /healthz may not be implemented"

Minimal chat completion:
- curl -sS -H "Content-Type: application/json" -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Hello"}],"stream":false}' http://localhost:8080/v1/chat/completions

Redis ping:
- docker exec -it redis redis-cli PING

Postgres connectivity:
- docker exec -it postgres psql -U postgres -d hypergraph -c '\dt'

## Stop and clean up
- docker rm -f gateway explainer interceptor postgres redis || true
- docker rm -f ollama || true
- docker network rm hypergraph-net || true
