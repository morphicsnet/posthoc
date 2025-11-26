# Interceptor (Internal)

## Purpose
Accepts ingestion payloads and enqueues them to Redis Streams for asynchronous processing by the Explainer.

Key entry: [services/interceptor/src/capture.py](services/interceptor/src/capture.py) — handler [ingest()](services/interceptor/src/capture.py:193)

## API surface
- POST /ingest
  - Content-Type: application/json
  - Minimal contract: includes a trace identifier and event body.
  - Example:
    ```
    {
      "trace_id": "trace-123",
      "event_type": "completion",
      "payload": { "model": "gpt-4o-mini", "text": "..." }
    }
    ```

## Operability
Env vars:
- HOST (default 0.0.0.0), PORT (external default 8081)
- REDIS_URL (e.g., redis://redis:6379/0)
- LOG_LEVEL (default: info)

## Run and deploy
- docker run -d --name interceptor --network hypergraph-net -p 8081:8081 -e REDIS_URL=redis://redis:6379/0 yourco-interceptor

## Health note
No explicit /healthz. As an interim liveness check, `POST /ingest` with an invalid/malformed body should return HTTP 400, confirming process and routing are alive.

## Failure modes and remediation
- Redis unavailable: returns 503/500; retry with backoff; ensure Redis reachable.
- Payload schema errors: 400; validate against shared schema prior to submission.
- Backpressure: monitor stream lag; autoscale or increase consumer throughput.
