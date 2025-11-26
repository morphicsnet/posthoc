# System architecture

Audience: Both (internal + external)

## Overview
Your Company’s Hypergraph Explainability Stack turns model I/O into an auditable, graph-structured explanation (HIF). Requests enter the Gateway, traces are ingested via the Interceptor onto Redis Streams, and the Explainer constructs and optionally verifies a hypergraph that can be fetched via the Hypergraph API.

- Gateway: OpenAI-compatible entrypoint; see [services/gateway/src/app.py](services/gateway/src/app.py) and [create_chat_completion()](services/gateway/src/app.py:369).
- Interceptor: Accepts trace/events and writes them to Redis Streams; see [services/interceptor/src/capture.py](services/interceptor/src/capture.py) and [ingest()](services/interceptor/src/capture.py:193).
- Explainer: Consumes streams, runs concept extraction/verification, persists results; see [services/explainer/src/worker.py](services/explainer/src/worker.py) and [concept_extraction()](services/explainer/src/worker.py:243).

## Components (1–2 lines each)
- Gateway: Front-door API that forwards chat completions to a provider/proxy while emitting explainability headers and trace context.
- Interceptor: Minimal HTTP ingress that normalizes and enqueues events into Redis Streams for asynchronous processing.
- Explainer: Stream worker that performs concept extraction and optional shadow-model verification, materializing a HIF hypergraph.

## Data flow and control flow
1) Client calls POST /v1/chat/completions on Gateway, handled by [create_chat_completion()](services/gateway/src/app.py:369).
2) Gateway forwards to an LLM provider/proxy with x-explain-* headers and attaches a trace identifier. It emits events to Redis Streams.
3) Producers (Gateway or callers) may also POST to the Interceptor, which accepts input via [ingest()](services/interceptor/src/capture.py:193) and enqueues into Redis.
4) Explainer consumes from Redis; within its pipeline, [concept_extraction()](services/explainer/src/worker.py:243) builds a hypergraph and may run verification.
5) Clients fetch status and graph via the Hypergraph API (HIF), e.g., GET /v1/traces/{trace_id}/graph.

## Persistence and streams
- Redis Streams: primary transport for traces/events and work queues.
- Postgres (optional): durable store for completed explanations; table name via EXPLAINER_TABLE (e.g., explanations_v2).
- HIF schema: see [libs/hif/schema.json](libs/hif/schema.json); validator utilities live in [libs/hif/validator.py](libs/hif/validator.py:1).

## Extensibility points
- Webhooks: emit lifecycle events (trace started/completed/failed) to external systems.
- Shadow model verification: configure SHADOW_ENDPOINT, VERIFY_MODEL, VERIFY_TEMPERATURE, VERIFY_TOP_K to compare/complement the primary model.

## Diagram
See the Mermaid source at [docs/diagrams/architecture.mmd](docs/diagrams/architecture.mmd).
