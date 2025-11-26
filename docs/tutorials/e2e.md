# End-to-end tutorial

Audience: Both

Goal: From prompt to hypergraph (HIF), running locally.

## Step 1: Launch the stack
Follow [docs/setup.md](docs/setup.md) to run Redis, Postgres, Interceptor, Explainer, and Gateway.

## Step 2: Send a chat request
Use the SDK quickstart:
- python sdks/python/your_company_explainability/examples/quickstart.py

Or curl:
```
curl -sS -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Hello"}],"stream":false}' \
  http://localhost:8080/v1/chat/completions
```

## Step 3: Follow the trace
The SDK exposes a lazy poller (`get_explanation(...)`). To manually inspect:
- GET /v1/traces/{trace_id}/status
- GET /v1/traces/{trace_id}/graph

Gateway handler reference: [create_chat_completion()](services/gateway/src/app.py:369)

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
