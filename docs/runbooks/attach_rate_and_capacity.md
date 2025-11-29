# Attach-rate baseline and capacity reservations

This runbook explains how to measure the attach-rate (fraction of chats requested with x-explain-mode=hypergraph) and translate that into capacity reservations for the Explainer.

Inputs
- StatusStore JSON (default: /tmp/hif/status.json) produced by the system’s LocalJSONStatusStore
- Optional Gateway audit JSON lines (stdout fallback or file), see [tests/security/test_rbac_audit.py](tests/security/test_rbac_audit.py)
- Observation window and bucket size
- Throughput assumptions (explanations/sec/GPU) for sentence and token paths

Analyzer
- Script: [tools/analysis/attach_rate_analyzer.py](tools/analysis/attach_rate_analyzer.py)
- Outputs:
  - JSON summary: [tools/analysis/results/attach_rate_summary.json](tools/analysis/results/attach_rate_summary.json)
  - Markdown: [tools/analysis/results/attach_rate_report.md](tools/analysis/results/attach_rate_report.md)

Commands
```bash
# Minimal (StatusStore only)
python3 tools/analysis/attach_rate_analyzer.py \
  --status-json /tmp/hif/status.json \
  --output tools/analysis/results/attach_rate_summary.json

# With audit JSONL cross-check
python3 tools/analysis/attach_rate_analyzer.py \
  --status-json /tmp/hif/status.json \
  --audit-log /var/log/hypergraph/audit.log \
  --window-minutes 120 \
  --bucket-seconds 60 \
  --token-mix 0.1 \
  --throughput-sentence-range 3,6 \
  --throughput-token-range 0.4,0.8 \
  --concurrency 1000,2000 \
  --output tools/analysis/results/attach_rate_summary.json \
  --report-output tools/analysis/results/attach_rate_report.md
```

Interpreting outputs
- baseline_attach_rate_mean/median: rolling window central tendency
- p95_attach_rate: bucket-level spike rate
- capacity_recommendations: GPU replica counts at target concurrencies (1000/2000 by default) using conservative (min throughput) and optimistic (max throughput) assumptions

Capacity reservation table (example)
Assuming token_mix=10%, throughputs (sentence 3–6/s/GPU, token 0.4–0.8/s/GPU)

| Concurrency | Attach Rate | Explanation QPS | GPUs (conservative) | GPUs (optimistic) |
|-------------|-------------|-----------------|---------------------|-------------------|
| 1000        | 20%         | 200             | 69 (60 sentence + 9 token) | 34 (30 + 4) |
| 1000        | 30%         | 300             | 103 (90 + 13)             | 50 (50 + 5) |
| 1000        | 50%         | 500             | 172 (150 + 22)            | 84 (84 + 8) |
| 2000        | 20%         | 400             | 138 (120 + 18)            | 68 (60 + 8) |
| 2000        | 30%         | 600             | 206 (180 + 26)            | 100 (100 + 10) |
| 2000        | 50%         | 1000            | 344 (300 + 44)            | 168 (168 + 16) |

How to enact capacity changes
- Karpenter: ensure GPU-capable node pools sized for peak (p95) plus buffer
- KEDA autoscaling (Explainer):
  - Template: [manifests/helm/hypergraph/templates/keda-scalers.yaml](manifests/helm/hypergraph/templates/keda-scalers.yaml:1)
  - Values: [manifests/helm/hypergraph/values.yaml](manifests/helm/hypergraph/values.yaml:1)
  - Update:
    - .Values.explainer.keda.minReplicaCount
    - .Values.explainer.keda.maxReplicaCount
    - .Values.explainer.keda.backlogSecondsTarget (see microbenchmark runbook)

Backpressure signals and degradation
- Metrics to watch (Explainer):
  - backpressure_level (0=normal, 1=soft, 2=hard)
  - backpressure_actions_total{action}
  - backlog_seconds{granularity}
- Controller and ladder:
  - See [BackpressureConfig](services/explainer/src/backpressure.py:64) thresholds and ladder defaults (token->sentence, reduce-samples, reduce-topk, reduce-layers, saliency-fallback, drop) around [degrade_ladder](services/explainer/src/backpressure.py:69).
  - Evaluation entrypoint: [BackpressureController.evaluate()](services/explainer/src/backpressure.py:263).
- Operational impact:
  - Token requests may be downgraded to sentence granularity under load (token->sentence). Plan capacity to keep downgrade incidence within SLOs.
  - Use attach-rate scenarios from this runbook together with ladder effects to size GPUs.
