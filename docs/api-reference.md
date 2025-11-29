# Public API Reference (v1)

This document defines the public Async Sidecar API surface, including OpenAI-compatible Chat Completions and the Trace APIs for explainability. For the authoritative specification, see [api/openapi/hypergraph-api.yaml](api/openapi/hypergraph-api.yaml:1). SDKs and UI components:
- Python SDK [README](sdks/python/your_company_explainability/README.md) and [Client](sdks/python/your_company_explainability/your_company_explainability/client.py:143)
- TypeScript UI [ExplainabilityClient](sdks/typescript/explainability-ui/src/client.ts:19) and [HypergraphViewer](sdks/typescript/explainability-ui/src/HypergraphViewer.tsx:233)

Base URL (local): http://localhost:8080

## Authentication
- Send bearer token: Authorization: Bearer <token>
- Required when AUTH_MODE=static; omit if AUTH_MODE=none.

## Explainability and Observability Headers
The Gateway accepts an extensible family of headers:
- x-explain-mode: hypergraph to request explainability
- x-explain-granularity: sentence or token
- x-explain-features: feature bundle identifier (e.g., sae-gpt4-2m)
- x-explain-budget: integer milliseconds budget for explain step (best-effort)
- x-trace-id: optional client-supplied trace identifier
- x-idempotency-key: optional idempotency key for POST semantics
- x-provider: optional upstream provider hint (e.g., "openai"); forwarded to proxy if supported

Headers are forwarded to upstream/proxy when appropriate. See [create_chat_completion()](services/gateway/src/app.py:540).

## Endpoints
- POST /v1/chat/completions — OpenAI-compatible Chat Completions. See [create_chat_completion()](services/gateway/src/app.py:540)
- GET /v1/traces/{trace_id}/status — Poll trace lifecycle. See [get_trace_status()](services/gateway/src/app.py:886)
- GET /v1/traces/{trace_id}/graph — Retrieve HIF hypergraph. See [get_trace_graph()](services/gateway/src/app.py:922)
- GET /v1/traces/{trace_id}/stream — Server-Sent Events (SSE) for status. See [stream_trace()](services/gateway/src/app.py:965)
- POST /v1/traces/{trace_id}/webhooks — Register webhooks. See [register_webhook()](services/gateway/src/app.py:1021)
- DELETE /v1/traces/{trace_id} — Cancel a running trace. See [cancel_trace()](services/gateway/src/app.py:1062)

An additional DB-backed explanation endpoint is provided for historic lookups:
- GET /v1/chat/completions/{id}/explanation — DB lookup of explanation envelope. See [get_completion_explanation()](services/gateway/src/app.py:1169)

---

## POST /v1/chat/completions

Semantics: Forwards an OpenAI-compatible chat completion request to a configured LLM proxy. If explain headers are provided (x-explain-mode=hypergraph), the Gateway begins an asynchronous explainability flow and returns a handle to the trace.

Example request body (OpenAI-compatible):
```json
{
  "model": "gpt-4o-mini",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Why might a model refuse to answer a dangerous request?"}
  ],
  "stream": false,
  "temperature": 0.2,
  "top_p": 1.0,
  "max_tokens": 200
}
```

Important headers:
- x-explain-mode: hypergraph
- x-explain-granularity: sentence | token
- x-explain-features: sae-gpt4-2m
- x-explain-budget: 1500
- x-trace-id: trc_client_supplied_optional
- x-idempotency-key: idemp-1234-abc

cURL:
```bash
curl -sS -X POST "http://localhost:8080/v1/chat/completions" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "x-provider: openai" \
  -H "x-explain-mode: hypergraph" \
  -H "x-explain-granularity: sentence" \
  -H "x-explain-features: sae-gpt4-2m" \
  -H "x-explain-budget: 1500" \
  -H "x-trace-id: trc_client_supplied_optional" \
  -H "x-idempotency-key: idemp-1234-abc" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role":"user","content":"classify: prompt injection risks"}],
    "stream": false
  }'
```

Response (non-streaming) example:
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "choices": [
    {"index": 0, "message": {"role": "assistant", "content": "…"}, "finish_reason": "stop"}
  ],
  "usage": {"prompt_tokens": 18, "completion_tokens": 42, "total_tokens": 60},
  "explanation_metadata": {
    "trace_id": "trc_5e1a9d0c3baf",
    "status": "processing",
    "estimated_wait": "PT2S",
    "stream_endpoint": "/v1/traces/trc_5e1a9d0c3baf/stream",
    "granularity": "sentence",
    "featureset": "sae-gpt4-2m"
  }
}
```

Response (streaming) note:
- When stream=true, the Gateway streams upstream SSE. Explanation is still queued best-effort; use returned id in streamed events or the x-trace-id header to correlate with trace APIs.

---

## GET /v1/traces/{trace_id}/status

Returns current status for a trace. States: queued, running, complete, failed, canceled, expired.

cURL:
```bash
curl -sS "http://localhost:8080/v1/traces/trc_5e1a9d0c3baf/status" \
  -H "Authorization: Bearer $TOKEN"
```

Response example:
```json
{
  "trace_id": "trc_5e1a9d0c3baf",
  "state": "running",
  "progress": 60.0,
  "stage": "attribution",
  "updated_at": 1732800000.123,
  "granularity": "sentence",
  "featureset": "sae-gpt4-2m"
}
```

Notes:
- 404 if no such trace is known.
- If a TTL-backed store is configured and the trace has expired, subsequent Graph requests return 410.

---

## GET /v1/traces/{trace_id}/graph

Retrieves the HIF graph for a completed trace (HIF v1 legacy structure). See HIF reference in [docs/hif-schema.md](docs/hif-schema.md:1).

cURL (with compression negotiation):
```bash
# Generic compression negotiation
curl -sS --compressed \
  "http://localhost:8080/v1/traces/trc_5e1a9d0c3baf/graph" \
  -H "Authorization: Bearer $TOKEN"

# Prefer zstd when supported by your gateway/proxy
curl -sS \
  "http://localhost:8080/v1/traces/trc_5e1a9d0c3baf/graph" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Accept-Encoding: zstd,gzip"
```

Responses:
- 200: JSON HIF graph document
- 404: Graph not ready (continue polling status)
- 410: Trace expired (no graph available)

---

## GET /v1/traces/{trace_id}/stream

Server-Sent Events channel emitting status updates for a given trace.

cURL:
```bash
curl -N "http://localhost:8080/v1/traces/trc_5e1a9d0c3baf/stream" \
  -H "Authorization: Bearer $TOKEN"
```

Events:
- event: status_update — data is a status object (see status endpoint)
- event: complete — data: {"trace_id": "…"} when the trace completes

See server implementation [stream_trace()](services/gateway/src/app.py:965).

---

## POST /v1/traces/{trace_id}/webhooks

Registers a webhook for trace events.

Request:
```json
{
  "url": "https://example.org/hooks/trace",
  "secret": "optional-shared-secret",
  "events": ["status_update", "complete"]
}
```

cURL:
```bash
curl -sS -X POST \
  "http://localhost:8080/v1/traces/trc_5e1a9d0c3baf/webhooks" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.org/hooks/trace","events":["complete"]}'
```

Response:
```json
{"id":"a1b2c3d4","trace_id":"trc_5e1a9d0c3baf","url":"https://example.org/hooks/trace"}
```

---

## DELETE /v1/traces/{trace_id}

Attempts to cancel a running trace.

cURL:
```bash
curl -sS -X DELETE \
  "http://localhost:8080/v1/traces/trc_5e1a9d0c3baf" \
  -H "Authorization: Bearer $TOKEN"
```

Possible outcomes:
- 202: Accepted; returns current canceled state
- 404: Unknown trace
- 409: Cannot cancel (already complete/expired/failed/canceled)

---

## RBAC and Rate Limiting

Required scopes per endpoint:
- traces:write — POST /v1/chat/completions, POST /v1/traces/{trace_id}/webhooks, DELETE /v1/traces/{trace_id}
- traces:read — GET /v1/traces/{trace_id}/status, GET /v1/traces/{trace_id}/graph, GET /v1/traces/{trace_id}/stream, GET /v1/chat/completions/{id}/explanation

429 behavior:
- Requests may be rate-limited. A 429 Too Many Requests response includes a Retry-After header (integer seconds). Clients should back off and retry after the indicated window.
Example:
```http
HTTP/1.1 429 Too Many Requests
Retry-After: 2
Content-Type: application/json

{"code":"rate_limited","message":"Rate limit exceeded"}
```

See enforcement in [rbac.py](services/gateway/src/rbac.py:1) and rate-limit dependency used in [create_chat_completion()](services/gateway/src/app.py:540), [get_trace_status()](services/gateway/src/app.py:886), [get_trace_graph()](services/gateway/src/app.py:922), [stream_trace()](services/gateway/src/app.py:965), [register_webhook()](services/gateway/src/app.py:1021), [cancel_trace()](services/gateway/src/app.py:1062).

---

## Error Model

Common HTTP status codes:
- 400 — Invalid request (e.g., bad JSON body)
- 401/403 — Authentication/authorization failure (missing/invalid token or insufficient scopes)
- 404 — Not found (unknown trace, or graph not ready)
- 409 — Conflict (e.g., cancel when terminal)
- 410 — Gone/Expired (trace TTL exceeded; graph unavailable)
- 429 — Rate limited (observe Retry-After if present)
- 5xx — Upstream/provider, database, or internal server error

Error body shape is JSON with an error or detail message when available, for example:
```json
{"error": "LLM_PROXY_URL is not configured"}
```
or
```json
{"detail":"Trace not found"}
```

Auth and rate limit examples:
- 401 Unauthorized (missing Authorization under AUTH_MODE=static)
```http
HTTP/1.1 401 Unauthorized
WWW-Authenticate: Bearer
Content-Type: application/json

{"code":"unauthorized","message":"Missing Authorization header"}
```

- 403 Forbidden (token known but insufficient scopes)
```http
HTTP/1.1 403 Forbidden
Content-Type: application/json

{"code":"missing_scope","message":"Missing required scopes: [\"traces:write\"]"}
```

- 429 Too Many Requests (per-tenant token bucket; note Retry-After)
```http
HTTP/1.1 429 Too Many Requests
Retry-After: 2
Content-Type: application/json

{"code":"rate_limited","message":"Rate limit exceeded"}
```

---

## SDK Usage (Quick Look)

Python:
- Use [Client](sdks/python/your_company_explainability/your_company_explainability/client.py:143) and [ChatCompletions.create()](sdks/python/your_company_explainability/your_company_explainability/client.py:97). After sending a chat request with explain headers, poll explanation via [ChatResponse.get_explanation()](sdks/python/your_company_explainability/your_company_explainability/client.py:31), which calls GET /v1/traces/{trace_id}/graph and handles 404/410 semantics.

TypeScript (browser/React):
- Use [ExplainabilityClient](sdks/typescript/explainability-ui/src/client.ts:19) to fetch the graph, and render with [HypergraphViewer](sdks/typescript/explainability-ui/src/HypergraphViewer.tsx:233). Ensure CORS is enabled on the Gateway for your origin.

---

## Notes on Compression and Large Graphs

If you expect large graphs, use HTTP compression. Many deployments negotiate gzip automatically when the client sends --compressed (curl) or Accept-Encoding. Some ingress/proxies support zstd. The server may respond with Content-Encoding: gzip or zstd when configured.

---

## OpenAPI

The full reference schemas and operations are defined in [api/openapi/hypergraph-api.yaml](api/openapi/hypergraph-api.yaml:1). Contract tests should be derived from the examples in this document.
