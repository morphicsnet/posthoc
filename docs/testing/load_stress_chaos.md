# Load, Stress, and Chaos Testing Runbook

This runbook provides a repeatable harness to exercise the Gateway and Async Sidecar (Interceptor/Explainer) under high concurrency, soak patterns, and chaos conditions, and to generate reports from test artifacts.

Artifacts produced:
- JSONL per-request logs:
  - tests/load/results/chat_results.jsonl
  - tests/load/results/explanation_results.jsonl
- Summary JSON: tests/load/results/summary.json
- Soak time‑series: tests/load/results/soak_summary.json
- Final report: tests/load/results/report.json and tests/load/results/report.md

If the module form (-m) does not work in your environment, you can run the scripts directly via python path instead (e.g., python tests/load/load_runner.py).

## 1) Basic load test

Example: 10 minutes, 500 concurrency, 30% explanation attach, sentence granularity.

```
python -m tests.load.load_runner \
  --base-url http://localhost:8080 \
  --concurrency 500 \
  --duration-seconds 600 \
  --attach-rate 0.3 \
  --granularity sentence \
  --features sae-gpt4-2m \
  --timeout-seconds 15
```

Options:
- --auth-token TOKEN: send Authorization: Bearer TOKEN (RBAC)
- --tenant-id TID: include x-tenant-id for low-cardinality labeling
- --rps-limit N: global token bucket across all workers (0 = unlimited)
- --granularity sentence|token|mix (mix = 80% sentence / 20% token)
- --features FEATURESET (default: sae-gpt4-2m)

What it measures:
- Chat submission latency p50/p95/p99 (excludes streamed content; stream=False)
- Success/error rates (HTTP status + error text)
- Collects trace_id for attached explanations
- For attached traces: polls /v1/traces/{id}/status and /graph until complete/expired/failed; computes explanation SLA p95 split by sentence/token

Outputs:
- JSONL lines for each chat and each explanation polling result
- tests/load/results/summary.json + printed single‑line JSON on stdout

## 2) Soak test orchestrator

Run alternating segments over hours to surface leaks/instabilities. Profiles:
- default: moderate concurrency (≈400)
- heavy: high concurrency (≈1000), unlimited RPS
- spiky: 10‑minute alternation of high RPS (≈800 conc, unlimited) and low RPS (≈200 conc, limited)

Example (2 hours, spiky):
```
python -m tests.load.soak_runner \
  --base-url http://localhost:8080 \
  --hours 2 \
  --profile spiky
```

Behavior:
- 10‑minute segments, rotating:
  - attach_rate in [0.30, 0.50]
  - granularity "mix" with periodic token bursts (every 5th run uses token)
- Aggregates time‑series SLOs to tests/load/results/soak_summary.json:
  - Chat: RPS, p50/p95, 4xx/5xx rates
  - Explanations: p95 (sentence/token), expired/failed percentages

## 3) Chaos injectors

Chaos is controlled via a local JSON file at /tmp/hif/chaos.json, read by the Explainer worker (import‑guarded; no public API changes). Use the helper CLI to toggle flags:

Enable slow SAE (50% of requests, ~150ms jitter):
```
python -m tests.chaos.chaos_injector --enable slow-sae --percent 50 --jitter-ms 150
```

Disable slow SAE:
```
python -m tests.chaos.chaos_injector --disable slow-sae
```

Additional flags:
- drop-s3: simulate occasional persistence write failures
  - Enable 10% failure:
    ```
    python -m tests.chaos.chaos_injector --enable drop-s3 --percent 10
    ```
  - Disable:
    ```
    python -m tests.chaos.chaos_injector --disable drop-s3
    ```
- fail-attribution: randomly fail attribution
  - Soft (fallback to heuristic): 25%
    ```
    python -m tests.chaos.chaos_injector --enable fail-attribution --percent 25 --mode fallback
    ```
  - Hard (mark failed): 10%
    ```
    python -m tests.chaos.chaos_injector --enable fail-attribution --percent 10 --mode fail
    ```
- rate-limit-spike: escalate backpressure severity
  - Hard spike:
    ```
    python -m tests.chaos.chaos_injector --enable rate-limit-spike --severity hard
    ```
  - Soft spike:
    ```
    python -m tests.chaos.chaos_injector --enable rate-limit-spike --severity soft
    ```
Disable all:
```
python -m tests.chaos.chaos_injector --disable-all
```

Explainer behavior under chaos (import‑guarded):
- slow-sae: introduces artificial latency prior to SAE decode
- fail-attribution: raises exception to either trigger fallback or mark failed
- drop-s3: simulates persistence failures safely (caught and reflected in status/metrics)
- rate-limit-spike: temporarily escalates backpressure (soft→hard) during evaluation

All injections are contained via try/except and reflected in status/error fields; defaults to no effect if the control file is missing.

## 4) Report generation

After running load or soak:
```
python -m tests.load.report --results-dir tests/load/results
```

Optional Prometheus snapshots (if metrics endpoints are enabled):
```
python -m tests.load.report \
  --results-dir tests/load/results \
  --metrics-url-gateway http://localhost:9091/metrics \
  --metrics-url-explainer http://localhost:9090/metrics
```

Outputs:
- tests/load/results/report.json
- tests/load/results/report.md

Includes:
- Chat lane: RPS, p50/p95/p99, 4xx/5xx rates
- Explanation SLA: sentence p95, token p95, expired/failed percentages
- Backpressure: action counters and level counts (soft/hard) from metrics
- Chaos snapshot: current flags from /tmp/hif/chaos.json
- Time‑bucket SLOs (pre/during/post) to visualize chaos impact deltas across the run

## 5) Interpreting results and SLO alignment

- Chat SLOs:
  - p95 submission latency remains within target budget (e.g., < 500ms)
  - Error budget: (4xx + 5xx)/requests within the agreed % over interval
- Explanation SLOs:
  - sentence p95 within the designed budget (e.g., < 1.5s)
  - token p95 within the designed budget (e.g., < 3.5s)
  - Expired/failed % should stay within error budget allocation for explanation lane
- Chaos impact:
  - Compare pre/during/post p95 deltas in report.md; spikes should correspond to backpressure activations
  - Use backpressure_actions_total to confirm degradation ladder usage
  - For rate-limit-spike, expect higher soft/hard counts and increased downgrades (reduce-topk/layers)

## 6) Grafana mapping

The included dashboard JSON at dashboards/grafana/sidecar-overview.json maps to:
- Backpressure: backpressure_level, backpressure_actions_total
- Explainer stages: explainer_stage_duration_seconds (extract/decode/attribution/persist)
- Queue lag: explainer_queue_lag_seconds
- Job outcomes: explainer_jobs_total{state}

Correlate these with time‑bucket SLOs from the report to understand chaos-induced degradations and recovery.

## Notes

- Dependencies: prefers httpx; falls back to requests when needed (no heavy libs added).
- The loaders handle failures robustly; all per‑request failures are recorded to JSONL and do not stop the harness.
- To set a custom chaos control path, export CHAOS_CONTROL_PATH to point the Explainer and CLI to the same JSON file.

## Quick commands

- Basic load:
  ```
  python -m tests.load.load_runner --base-url http://localhost:8080 --concurrency 500 --duration-seconds 600 --attach-rate 0.3
  ```

- Soak:
  ```
  python -m tests.load.soak_runner --base-url http://localhost:8080 --hours 2 --profile spiky
  ```

- Chaos (enable slow SAE):
  ```
  python -m tests.chaos.chaos_injector --enable slow-sae --percent 50 --jitter-ms 150
  ```

- Chaos (disable):
  ```
  python -m tests.chaos.chaos_injector --disable slow-sae
  ```

- Report:
  ```
  python -m tests.load.report --results-dir tests/load/results