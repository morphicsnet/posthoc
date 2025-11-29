# System architecture

Audience: Both (internal + external)

## Overview
Your Company’s Hypergraph Explainability Stack turns model I/O into an auditable, graph-structured explanation (HIF). Requests enter the Gateway, traces are ingested via the Interceptor onto Redis Streams, and the Explainer constructs and optionally verifies a hypergraph that can be fetched via the Hypergraph API.

- Gateway: OpenAI-compatible entrypoint; see [services/gateway/src/app.py](services/gateway/src/app.py) and [create_chat_completion()](services/gateway/src/app.py:540).
- Interceptor: Accepts trace/events and writes them to Redis Streams; see [services/interceptor/src/capture.py](services/interceptor/src/capture.py).
- Explainer: Consumes streams, runs concept extraction/verification, persists results; see [services/explainer/src/worker.py](services/explainer/src/worker.py).

## Components (1–2 lines each)
- Gateway: Front-door API that forwards chat completions to a provider/proxy while emitting explainability headers and trace context.
- Interceptor: Minimal HTTP ingress that normalizes and enqueues events into Redis Streams for asynchronous processing.
- Explainer: Stream worker that performs concept extraction and optional shadow-model verification, materializing a HIF hypergraph.

## HIF design and limits
- HIF v1 structure and field semantics are documented in [docs/hif-schema.md](docs/hif-schema.md:1). See limits and pruning defaults in the `meta.limits` section (min_edge_weight, max_nodes, max_incidences), and the version marker `version: "hif-1"`.

## Asynchronous tracing model
- The Gateway returns an `explanation_metadata.trace_id` from [create_chat_completion()](services/gateway/src/app.py:540). Clients poll or subscribe for lifecycle updates:
  - GET /v1/traces/{trace_id}/status — running states and progress
  - GET /v1/traces/{trace_id}/graph — HIF graph when complete (404 until ready, 410 if expired)
  - GET /v1/traces/{trace_id}/stream — SSE updates
- See the full API details and cURL snippets in [docs/api-reference.md](docs/api-reference.md:1).

### Trace and correlation IDs
- Gateway injects correlation headers on every response: `x-trace-id` and `x-request-id`. See middleware in [gateway otel](services/gateway/src/otel.py:421).
- Clients may also supply `x-trace-id` on POST /v1/chat/completions; the server echoes this in audit/metrics and lifecycle.

## Diagram node IDs
- client: Client App
- sdk: Python SDK
- gateway: Gateway (Hypergraph API)
- interceptor: Interceptor (Capture)
- explainer: Explainer Worker
- redis: Redis Streams
- postgres: Postgres (explanations_v2)
- llmproxy: LLM Provider / Proxy
- shadow: Shadow Model (optional)
- webhook: External Webhook Consumer (optional)

## Data flow and control flow
1) client → sdk → gateway: POST /v1/chat/completions handled by [create_chat_completion()](services/gateway/src/app.py:540); returns 200 with explanation_metadata.trace_id.
2) gateway → llmproxy: forwards chat; gateway → redis: XADD hypergraph:completions with envelope and trace.
3) client → interceptor: optional POST /ingest; interceptor → redis: XADD hypergraph:completions; contract handled by the capture service.
4) redis → explainer: XREADGROUP; explainer → redis: XACK. Pipeline persists hypergraph to Postgres when configured.
5) explainer → postgres: upsert HIF; gateway → postgres: select HIF by request_id/trace_id for retrieval endpoints.
6) client → gateway: GET /v1/traces/{trace_id}/status, GET /v1/traces/{trace_id}/graph, SSE /v1/traces/{trace_id}/stream; gateway → webhook: POST complete|failed (if configured).

## Persistence and streams
- Redis Streams: primary transport for traces/events and work queues.
- Postgres (optional): durable store for completed explanations; table name via EXPLAINER_TABLE (e.g., explanations_v2).
- HIF schema: see [libs/hif/schema.json](libs/hif/schema.json); validator utilities live in [libs/hif/validator.py](libs/hif/validator.py:1).

## Extensibility points
- Webhooks: emit lifecycle events (trace started/completed/failed) to external systems.
- Shadow model verification: configure SHADOW_ENDPOINT, VERIFY_MODEL, VERIFY_TEMPERATURE, VERIFY_TOP_K to compare/complement the primary model.

## Diagram
See the Mermaid source at [docs/diagrams/architecture.mmd](docs/diagrams/architecture.mmd).

## Observability overview
- Gateway metrics: GET /metrics on port 8080 when OTEL is enabled via [setup_otel()](services/gateway/src/otel.py:330). Core counters validated in tests: [test_metrics.py](tests/gateway/test_metrics.py:70).
- Explainer metrics: standalone HTTP server on EXPLAINER_METRICS_PORT (default 9090) started by [start_http_server](services/explainer/src/otel.py:256). Scrape /metrics (Prometheus format).
- Tracing: optional OTLP exporter controlled by ENABLE_OTEL=1 and OTEL_EXPORTER_OTLP_ENDPOINT (HTTP or gRPC), see [otel.py](services/explainer/src/otel.py:73) and [gateway otel](services/gateway/src/otel.py:64).

## Backpressure and degradation
Backpressure controller determines level and actions based on backlog seconds and queue lengths. See [backpressure.py](services/explainer/src/backpressure.py:1).
- Levels: normal | soft | hard
- Degradation ladder (applied deterministically):
  - token->sentence, reduce-samples, reduce-topk, reduce-layers, saliency-fallback, drop
- Metrics (low-cardinality): backpressure_level, backpressure_actions_total, backlog_seconds
- Status fields (when present from worker/store): bp_level, bp_actions, granularity_downgraded

## Persistence pipeline
- Transport: Redis Streams for envelopes and events
- Explainer: SAE decode → attribution → hypergraph materialization
- Storage:
  - S3 artifacts with optional SSE-KMS: key layout {prefix}/{yyyy}/{mm}/{dd}/{trace_id}/{granularity}-{sae_version}-{model_hash}.hif.json.gz via [S3Store.put_json_gz()](services/explainer/src/s3_store.py:104)
  - Local fallback when S3 is disabled or unavailable: file:///tmp/hif/<key> (same key layout rooted under /tmp/hif)
  - Optional DB envelope for historic lookups via GET /v1/chat/completions/{id}/explanation ([get_completion_explanation()](services/gateway/src/app.py:1169))
  - Optional StatusStore (JSON or DDB stub) with TTL and tenant immutability ([get_status_store_from_env](services/explainer/src/status_store.py:208))

## Cross-links
- API reference: [docs/api-reference.md](docs/api-reference.md:1)
- HIF Schema: [docs/hif-schema.md](docs/hif-schema.md:1)
