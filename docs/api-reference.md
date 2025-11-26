# API reference

This document summarizes auth, headers, and pointers to generated reference material.

- OpenAPI source: [api/openapi/hypergraph-api.yaml](api/openapi/hypergraph-api.yaml)
- Python SDK: [sdks/python/your_company_explainability/README.md](sdks/python/your_company_explainability/README.md) and [Client](sdks/python/your_company_explainability/your_company_explainability/client.py:1)

## Authentication
- Bearer token via `Authorization: Bearer <token>`.

## Explainability headers (x-explain-*)
The Gateway accepts an extensible family of headers:
- x-explain-mode: e.g., `hypergraph`
- x-explain-granularity: `sentence` | `token`
- x-explain-features: feature bundle name
- x-explain-budget: integer milliseconds
- x-trace-id: optional client-supplied trace identifier
See the SDK examples for end-to-end usage of these headers.

## Endpoints (high level)
- POST /v1/chat/completions: OpenAI-compatible chat. Implemented in Gateway; see [create_chat_completion()](services/gateway/src/app.py:369).
- GET /v1/traces/{trace_id}/status: poll explanation lifecycle.
- GET /v1/traces/{trace_id}/graph: retrieve the HIF hypergraph.

## Generated sections (placeholder)
The detailed REST surface can be generated from [api/openapi/hypergraph-api.yaml](api/openapi/hypergraph-api.yaml). A lightweight script will produce markdown sections for routes, schemas, and examples.

### Update instructions
Run:
```
python scripts/gen_api_md.py api/openapi/hypergraph-api.yaml > docs/api-reference.md
```
