# Documentation

This folder contains the official docs for the Your Company Hypergraph Explainability Stack. If you are new, start with the overview, then run a local stack, and finally try the end‑to‑end tutorial.

- Architecture overview: [docs/architecture.md](docs/architecture.md)
- Setup (Docker + local): [docs/setup.md](docs/setup.md)
- API reference (HIF + endpoints): [docs/api-reference.md](docs/api-reference.md)
- Tutorials (hands-on): [docs/tutorials/e2e.md](docs/tutorials/e2e.md)
- Troubleshooting: [docs/troubleshooting.md](docs/troubleshooting.md)
- FAQ: [docs/faq.md](docs/faq.md)
- Workshop outline: [docs/workshop/workshop-60min-outline.md](docs/workshop/workshop-60min-outline.md)

Key repo references used throughout:
- Gateway entrypoint and handler [create_chat_completion()](services/gateway/src/app.py:369)
- Interceptor ingest handler [ingest()](services/interceptor/src/capture.py:193)
- Explainer worker and [concept_extraction()](services/explainer/src/worker.py:243)
- OpenAPI spec: [api/openapi/hypergraph-api.yaml](api/openapi/hypergraph-api.yaml)

## Who should read what

- External (users, partners)
  - Start with: Architecture, Setup, API reference, Tutorials, FAQ, Troubleshooting.
  - Skip internal runbooks; those live under each service directory.
- Internal (operators, developers)
  - Start with: Architecture, Setup, Service READMEs, Troubleshooting.
  - Deep dive code in [services/gateway/src/app.py](services/gateway/src/app.py), [services/interceptor/src/capture.py](services/interceptor/src/capture.py), [services/explainer/src/worker.py](services/explainer/src/worker.py).

## Terminology

- Hypergraph API (HIF): A vendor-neutral JSON schema and HTTP contract for explainability artifacts and graphs.
- HIF: Abbreviation for Hypergraph Interchange Format, defined in [libs/hif/schema.json](libs/hif/schema.json).
- Gateway: OpenAI-compatible front door; see [create_chat_completion()](services/gateway/src/app.py:369).
- Interceptor: Ingests model I/O to Redis Streams; see [ingest()](services/interceptor/src/capture.py:193).
- Explainer: Consumes traces, extracts concepts, verifies; see [concept_extraction()](services/explainer/src/worker.py:243).