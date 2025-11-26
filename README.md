# Your Company Hypergraph Explainability Stack

## What is this?

One stack to capture, explain, and validate model behavior end to end. It includes:
- Gateway: an OpenAI-compatible HTTP layer that forwards chat completions to a provider/proxy and emits explainability metadata. See entrypoint [services/gateway/src/app.py](services/gateway/src/app.py) and handler [create_chat_completion()](services/gateway/src/app.py:369).
- Interceptor: a lightweight ingest service that accepts model I/O and traces, placing events onto Redis Streams for downstream processing. See [services/interceptor/src/capture.py](services/interceptor/src/capture.py) and handler [ingest()](services/interceptor/src/capture.py:193).
- Explainer: a worker that consumes traces, performs concept extraction and verification, and materializes a Hypergraph. See [services/explainer/src/worker.py](services/explainer/src/worker.py) and [concept_extraction()](services/explainer/src/worker.py:243).
- Hypergraph API (HIF): a vendor-neutral JSON schema and HTTP surface for explainability. OpenAPI lives at [api/openapi/hypergraph-api.yaml](api/openapi/hypergraph-api.yaml).

## Key capabilities

- OpenAI-compatible chat completions (POST /v1/chat/completions)
- Explainability via x-explain-* headers and async trace lifecycle
- End-to-end Hypergraph generation and retrieval (HIF)
- HIF validation utilities for CI and runtime safety

## System architecture at a glance

Gateway receives the request, emits trace context, Interceptor buffers events, and Explainer builds the hypergraph and optional verifications before the graph is fetched by clients. See the high-level overview in [docs/architecture.md](docs/architecture.md). Diagram source (Mermaid) is at [docs/diagrams/architecture.mmd](docs/diagrams/architecture.mmd).

## Quick start (Docker)

Minimal local run with sensible defaults. For a complete setup (including Postgres and optional shadow verification), see [docs/setup.md](docs/setup.md).

1) Create a network and start Redis:

- docker network create hypergraph-net || true
- docker run -d --name redis --network hypergraph-net -p 6379:6379 redis:7-alpine

2) Build service images:

- docker build -t yourco-gateway ./services/gateway
- docker build -t yourco-interceptor ./services/interceptor
- docker build -t yourco-explainer ./services/explainer

3) Run the services (basic defaults):

- docker run -d --name interceptor --network hypergraph-net -p 8081:8081 -e REDIS_URL=redis://redis:6379/0 yourco-interceptor
- docker run -d --name explainer --network hypergraph-net -e REDIS_URL=redis://redis:6379/0 -e DEV_MODE=0 yourco-explainer
- docker run -d --name gateway --network hypergraph-net -p 8080:8080 -e REDIS_URL=redis://redis:6379/0 -e LLM_PROXY_URL=http://llm-proxy:8080 yourco-gateway

4) Verify:

- curl -sS http://localhost:8080/healthz || echo "Note: /healthz may not yet be implemented; check container logs."
- curl -sS -H "Content-Type: application/json" -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Hello"}],"stream":false}' http://localhost:8080/v1/chat/completions

## API and SDK

- API reference: [docs/api-reference.md](docs/api-reference.md) (generated from [api/openapi/hypergraph-api.yaml](api/openapi/hypergraph-api.yaml))
- Python SDK: [sdks/python/your_company_explainability/README.md](sdks/python/your_company_explainability/README.md)

## Learn by doing

Follow the end-to-end tutorial at [docs/tutorials/e2e.md](docs/tutorials/e2e.md).

## Support and license

- Issues: open a ticket in this repository with logs and reproduction steps.
- License: Provided for evaluation; replace this note with your company’s license terms.