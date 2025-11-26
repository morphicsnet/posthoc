# Troubleshooting

Audience: Both

## Startup issues
- Port conflicts: ensure 8080 (Gateway), 8081 (Interceptor), 6379 (Redis), 5432 (Postgres) are free.
- Environment: verify all required env vars; see sample in [docs/setup.md](docs/setup.md).
- Docker network: `docker network create hypergraph-net` and ensure all containers join it.

## Runtime failures
- Gateway 500/502:
  - Missing/invalid LLM_PROXY_URL -> set a reachable provider proxy.
  - Redis connect errors -> ensure REDIS_URL is correct and Redis is up.
  - DB issues -> confirm DATABASE_URL and Postgres availability.
- spaCy download/model:
  - Ensure SPACY_MODEL is available to the Explainer container (or disable advanced extraction in DEV_MODE).

## Explanation not appearing
- Worker not consuming: check Explainer logs for stream lag or connection errors.
- Timeouts too low: increase `get_explanation(max_wait_seconds=...)` in SDK.
- DEV_MODE: if set, pipeline stages may be altered; set DEV_MODE=0 for full flow.

## Debug aids
- Increase LOG_LEVEL to debug on Gateway/Interceptor/Explainer.
- curl checks:
  - `curl -i http://localhost:8080/healthz` (if implemented)
  - `curl -i http://localhost:8080/v1/traces/<trace_id>/status`
- Redis Streams:
  - `docker exec -it redis redis-cli XINFO STREAM your_stream_name`
- Postgres:
  - `docker exec -it postgres psql -U postgres -d hypergraph -c '\dt'`
