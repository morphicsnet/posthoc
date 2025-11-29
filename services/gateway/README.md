# Gateway (Internal)

## Purpose and responsibilities
- OpenAI-compatible endpoint for chat completions.
- Emits explainability headers and trace metadata.
- Orchestrates async lifecycle and integrates with Redis Streams and (optionally) Postgres.

Entrypoint: [services/gateway/src/app.py](services/gateway/src/app.py) — handler [create_chat_completion()](services/gateway/src/app.py:369)

## Operability
Environment variables:
- LLM_PROXY_URL: upstream provider/proxy URL.
- REDIS_URL: e.g., redis://redis:6379/0
- DATABASE_URL: optional Postgres for storing explanations/metadata.
- LOG_LEVEL: debug | info | warning | error (default: info)
- PORT: external port (default 8080)

## Run and deploy
- docker run -d --name gateway --network hypergraph-net -p 8080:8080 -e REDIS_URL=redis://redis:6379/0 -e DATABASE_URL=postgresql://postgres:postgres@postgres:5432/hypergraph -e LLM_PROXY_URL=http://llm-proxy:8080 yourco-gateway

## Observability
- Logs: request/response metadata, trace ids, error paths.
- Metrics (if enabled): request rate, p50/p95 latency, error rate.

## Failure modes
- Redis down: chat continues but trace/explanation pipeline stalls; return degraded headers.
- Proxy down (LLM_PROXY_URL): 502/504 upstream errors; exponential backoff and circuit breaker recommended.
- DB issues: non-blocking for live responses; persistence may fail—emit warnings and surface degraded state.

## SLOs and scaling
- Latency: pass-through latency should be within provider tolerance; target <250ms overhead.
- Concurrency: horizontally scale using stateless containers behind a load balancer.

## Access control (RBAC)
- AUTH_MODE=none|static (default: none)
- AUTH_TOKENS_JSON: static mapping of bearer token to identity and scopes, for example:
  {"tokenA":{"tenant_id":"t1","scopes":["traces:read","traces:write"]}}
- Required scopes:
  - POST /v1/chat/completions -> traces:write
  - GET /v1/traces/{trace_id}/status -> traces:read
  - GET /v1/traces/{trace_id}/graph -> traces:read
  - GET /v1/traces/{trace_id}/stream -> traces:read
  - POST /v1/traces/{trace_id}/webhooks -> traces:write
  - DELETE /v1/traces/{trace_id} -> traces:write

## Security

See the comprehensive guide: [docs/security/SECURITY.md](docs/security/SECURITY.md)

Environment variables (gateway):
- AUTH_MODE=none|static (default: none)
- AUTH_TOKENS_JSON='{"token":{"tenant_id":"t1","scopes":["traces:read","traces:write"],"subject":"user"}}'
- AUTH_EXPECT_ISS, AUTH_EXPECT_AUD (optional issuer/audience checks for static identities)
- AUDIT_LOG_ENABLE=1 to enable audit events
- AUDIT_LOG_PATH=/var/log/hypergraph/audit.log (falls back to stdout if unwritable)
- PII_MASK_EMAIL=1, PII_MASK_PHONE=1, PII_MASK_NUMBERS=0 (defaults). Only affects log copies, not data path.

Notes:
- Do not log Authorization headers. The gateway’s structured audit logs never include Authorization and only include a short scrubbed snippet of user content when x-explain-mode is used.
- Scope coverage:
  - POST /v1/chat/completions → traces:write
  - POST /v1/traces/{trace_id}/webhooks → traces:write
  - DELETE /v1/traces/{trace_id} → traces:write
  - GET /v1/traces/{trace_id}/status → traces:read
  - GET /v1/traces/{trace_id}/graph → traces:read
  - GET /v1/traces/{trace_id}/stream → traces:read

## Rate limits
Per-tenant token bucket categories:
- write: chat completions, webhook registration, cancel
- read: status, graph, stream

Defaults (env override):
- RATE_LIMIT_READ_RPS=50, RATE_LIMIT_READ_BURST=200
- RATE_LIMIT_WRITE_RPS=5, RATE_LIMIT_WRITE_BURST=20
Optional local persistence of limiter counters:
- RL_COUNTERS_JSON_PATH=/tmp/hif/rl_counters.json

## Status backend and TTL
If STATUS_BACKEND is set (json|ddb), the gateway uses the shared StatusStore from [services/explainer/src/status_store.py](services/explainer/src/status_store.py) to read/write trace status.
TTL controlled by TRACE_TTL_DAYS (default 7). When using the shared store, GET graph returns 410 if the item is expired.

## Example curl
- Using static auth:
  export AUTH_MODE=static
  export AUTH_TOKENS_JSON='{"tokenA":{"tenant_id":"t1","scopes":["traces:read","traces:write"]}}'
  curl -H "Authorization: Bearer tokenA" -H "x-explain-mode: hypergraph" -H "x-trace-id: trc_demo" \
       -H "Content-Type: application/json" -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}' \
       http://localhost:8080/v1/chat/completions

  curl -H "Authorization: Bearer tokenA" http://localhost:8080/v1/traces/trc_demo/status

## References
- App: [services/gateway/src/app.py](services/gateway/src/app.py)
- Handler: [create_chat_completion()](services/gateway/src/app.py:369)


## Observability (metrics, logs, traces)

This service emits low-overhead, production-ready telemetry guarded by ENABLE_OTEL.

- Enablement
  - ENABLE_OTEL=1 to turn on metrics and optional tracing
  - OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317 (gRPC) or http://otel-collector:4318 (HTTP)
  - OTEL_EXPORTER_OTLP_PROTOCOL=grpc|http/protobuf (optional; default gRPC if unspecified)
  - GATEWAY_VERSION=1.0.0 (optional version label on metrics/logs)
- Metrics
  - Exposed in Prometheus format at: http://localhost:8080/metrics
  - Implemented via middleware in [services/gateway/src/otel.py](services/gateway/src/otel.py:1) with low-cardinality labels:
    - http_requests_total{path,method,status,tenant_id,service,version}
    - http_request_duration_seconds{path,method,status,service,version}
    - chat_requests_total{tenant_id,granularity,featureset,service,version}
    - traces_status_get_total{tenant_id,service,version}
    - traces_graph_get_total{tenant_id,service,version}
    - rate_limit_rejections_total{tenant_id,category,service,version}
    - rbac_denied_total{tenant_id,service,version}
  - Path labels are templated (e.g., /v1/traces/{trace_id}/status → /v1/traces/_/status) to bound cardinality.
- Logs
  - Structured JSON with correlation fields using get_logger() from [get_logger()](services/gateway/src/otel.py:124)
  - Fields include ts, level, service, version, msg, and when available: trace_id, tenant_id, request_id, path, method, status
- Traces (optional)
  - If OTEL_EXPORTER_OTLP_ENDPOINT is set, tracing is initialized via OTLP exporter (HTTP or gRPC) in [setup_otel()](services/gateway/src/otel.py:286)
- Dashboards and Collector
  - Deploy the example OpenTelemetry Collector: kubectl apply -f [manifests/otel/otel-collector.yaml](manifests/otel/otel-collector.yaml:1)
  - Import the Grafana dashboard: [dashboards/grafana/sidecar-overview.json](dashboards/grafana/sidecar-overview.json:1)
  - The Collector exposes its own Prometheus endpoint at http://otel-collector:8889/metrics

Common labels across lanes and artifacts (kept low-cardinality for metrics):
- tenant_id, granularity, featureset, service, version for metrics
- plus trace_id and request_id in logs
- avoid high-cardinality labels like raw trace_id or model_name in metrics; those appear in logs/traces

Test
- Run: python -m [tests.gateway.test_metrics](tests/gateway/test_metrics.py:1)
- The test enables ENABLE_OTEL=1, exercises a few endpoints, and asserts the presence of key metric names on /metrics.
