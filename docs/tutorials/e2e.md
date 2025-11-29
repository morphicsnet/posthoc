# End-to-end tutorial

Audience: Both

Goal: From prompt to hypergraph (HIF), running locally.

See also:
- Python Quickstart: [docs/tutorials/python-quickstart.md](docs/tutorials/python-quickstart.md)
- TypeScript/React Quickstart: [docs/tutorials/typescript-quickstart.md](docs/tutorials/typescript-quickstart.md)

## Step 1: Launch the stack
Follow [docs/setup.md](docs/setup.md) to run Redis, Postgres, Interceptor, Explainer, and Gateway.

## Step 2: Send a chat request
Use the SDK quickstart:
- python sdks/python/your_company_explainability/examples/quickstart.py
- or run the example script: [examples/python/e2e_quickstart.py](examples/python/e2e_quickstart.py)

Or curl:
```
curl -sS -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \# omit if AUTH_MODE=none
  -H "x-explain-mode: hypergraph" \
  -H "x-explain-granularity: sentence" \
  -H "x-explain-features: sae-gpt4-2m" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Hello"}],"stream":false}' \
  http://localhost:8080/v1/chat/completions
```

## Step 3: Follow the trace
The SDK exposes a lazy poller (`get_explanation(...)`). To manually inspect:
- GET /v1/traces/{trace_id}/status
- GET /v1/traces/{trace_id}/graph
- SSE: GET /v1/traces/{trace_id}/stream

Gateway handler reference: [create_chat_completion()](services/gateway/src/app.py:540)

## Step 4: Explore results
Interpret the nodes and incidences/hyperedges:
- Use the SDK’s structure or visualize downstream.
- Validate against HIF schema using [libs/hif/validator.py](libs/hif/validator.py:1).

Example:
```python
from libs.hif.validator import validate_hypergraph
hg = {"nodes": [], "hyperedges": []}
validate_hypergraph(hg)
```

## Cleanup
Stop containers and remove the Docker network as described in [docs/setup.md](docs/setup.md).
