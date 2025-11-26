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

## References
- App: [services/gateway/src/app.py](services/gateway/src/app.py)
- Handler: [create_chat_completion()](services/gateway/src/app.py:369)
