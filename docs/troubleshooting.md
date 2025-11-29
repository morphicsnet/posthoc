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

## RBAC 401/403
- 401 Unauthorized: missing/invalid Authorization header under AUTH_MODE=static. The server returns `{"code":"unauthorized"}`; add a valid token to Authorization: Bearer.
- 403 Forbidden: token known but missing required scopes (e.g., POST write without traces:write). Detail payload includes `{"code":"missing_scope"}`. See [rbac.py](services/gateway/src/rbac.py:62).

## Rate limit 429
- Per-tenant token bucket. Respect Retry-After header (integer seconds). Tune defaults with RATE_LIMIT_* envs; see [rate_limit.py](services/gateway/src/rate_limit.py:180).

## SSE timeouts (GET /v1/traces/{id}/stream)
- Ensure client keeps the connection open and accepts text/event-stream. If idle, server emits only state transitions (no keepalives).

## CORS issues (browser)
- Use a dev proxy (Vite) or enable development CORS middleware near [app = FastAPI(...)](services/gateway/src/app.py:333). Restrict origins in production.

## KEDA not scaling
- Verify ScaledObject present and Active: `kubectl -n <ns> get scaledobject`.
- Confirm metrics source reachable and query valid in [keda-scalers.yaml](manifests/helm/hypergraph/templates/keda-scalers.yaml:1).
- Check Prometheus target and label filters; adjust `.Values.explainer.keda.backlogSecondsTarget` and min/max replicas.

## S3 access denied (SSE-KMS)
- Ensure bucket policy enforces and permits SSE-KMS with the configured key; see [iam-policies.md](docs/security/iam-policies.md:1).
- Confirm role permissions include kms:Encrypt/Decrypt and S3 PutObject/GetObject.

## StatusStore errors
- STATUS_BACKEND=json requires write access to STATUS_JSON_PATH (default `/tmp/hif/status.json`). JSONPath errors often indicate a truncated file; delete and let it be recreated. See [LocalJSONStatusStore](services/explainer/src/status_store.py:52).

## Chaos flags file effects
- Chaos is toggled via `/tmp/hif/chaos.json` by the test harness; flags may degrade performance (e.g., slow-sae, drop-s3). See [load_stress_chaos.md](docs/testing/load_stress_chaos.md:1).
