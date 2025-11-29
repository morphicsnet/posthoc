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

---

## Hooks API (for future vLLM integration)

A lightweight capture pipeline is provided to be invoked from model layer outputs per token.

Key types:
- Capture configuration: `CaptureConfig(model_name, model_hash, topk=256, window_size=16, compress="json", namespace="activations")`
- Pipeline lifecycle:
  - start a trace: `CapturePipeline.start_trace(trace_id, params)`
  - per-token capture: `CapturePipeline.capture_token(trace_id, token_idx, tensor_summary)`
  - finalize: `CapturePipeline.flush(trace_id) -> envelope`

Adapter:
- Register a per-token hook function: `register_layer_hook(pipeline, trace_id) -> (token_idx, tensor_summary) -> None`

Minimal integration example (pseudo):
```
from services.interceptor.src.hooks import CaptureConfig, CapturePipeline, register_layer_hook

cfg = CaptureConfig(model_name="mymodel", model_hash="abc123", topk=256, window_size=16)
pipe = CapturePipeline(cfg)
pipe.start_trace(trace_id)
hook = register_layer_hook(pipe, trace_id)

for t in range(num_tokens):
    acts = layer_output_vector  # list[float] or numpy array
    hook(t, acts)

envelope = pipe.flush(trace_id)
# envelope["shards"] holds shard keys written to the store
```

Notes:
- Tensors are not required; pass a simple list[float] or JSON-encoded bytes. NumPy is optional.
- The current store is an in-memory stub; later swap for real Redis and Kafka producers without changing call sites.

## Microbenchmark: Inference capture overhead

A small harness simulates per-token capture cost to validate the target overhead budget (≤5% p95 vs model token time).

Run:
```
python benchmarks/inference_capture_bench.py \
  --tokens 128 --dim 4096 --topk 256 \
  --model-ms-per-token 60 --compress json --repeat 3
```

Outputs per run:
- per-token capture ms: mean, p50, p95
- overhead ratio vs model_ms_per_token
- PASS/FAIL if p95 capture_ms ≤ 5% of the provided model ms/token
