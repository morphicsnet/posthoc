# Microbenchmark plan and autoscaling thresholds (KEDA/MSK)

Goal: derive backlogSecondsTarget and initial min/max replicas for the Explainer from observed service times and attach-rate, and optionally compute MSK lag thresholds.

Inputs
- Observed service times JSON: [tools/benchmarks/results/service_times.json](tools/benchmarks/results/service_times.json) (create from prior runs; see sample at [tools/benchmarks/results/service_times.sample.json](tools/benchmarks/results/service_times.sample.json))
- SLOs for backlog seconds (sentence/token)
- Target chat RPS, attach-rate, and token-mix
- GPU throughput assumptions (explanations/sec/GPU)

Script
- [tools/benchmarks/microbench_plan.py](tools/benchmarks/microbench_plan.py)

Commands
```bash
# Use sample service times as a starting point
cp tools/benchmarks/results/service_times.sample.json tools/benchmarks/results/service_times.json

# Derive thresholds (example)
python3 tools/benchmarks/microbench_plan.py \
  --backlog-seconds-slo 1.0,3.0 \
  --observed-service-times-json tools/benchmarks/results/service_times.json \
  --target-rps 500 \
  --attach-rate 0.30 \
  --token-mix 0.10 \
  --gpu-throughput-sentence 5.0 \
  --gpu-throughput-token 0.6 \
  --output-json tools/benchmarks/results/keda_thresholds.json \
  --output-md tools/benchmarks/results/keda_thresholds.md
```

Outputs
- JSON: [tools/benchmarks/results/keda_thresholds.json](tools/benchmarks/results/keda_thresholds.json)
- Markdown: [tools/benchmarks/results/keda_thresholds.md](tools/benchmarks/results/keda_thresholds.md)

Apply to Helm/KEDA
- Template: [manifests/helm/hypergraph/templates/keda-scalers.yaml](manifests/helm/hypergraph/templates/keda-scalers.yaml)
- Values file: [manifests/helm/hypergraph/values.yaml](manifests/helm/hypergraph/values.yaml)

Set:
```yaml
explainer:
  keda:
    enabled: true
    backlogSecondsTarget: <weighted from keda_thresholds.json>
    minReplicaCount: <minReplicaCount>
    maxReplicaCount: <maxReplicaCount>
```

MSK notes
- If using Kafka/MSK for work dispatch, prefer a KEDA Kafka trigger (see commented example in [manifests/helm/hypergraph/templates/keda-scalers.yaml](manifests/helm/hypergraph/templates/keda-scalers.yaml)).
- Partitions: ensure enough to parallelize consumers across replicas.
- Consumer lag threshold: approximate per-partition lag as RPS_per_partition * backlogSecondsTarget.
  This is emitted by the plan when `kafka.enabled=true` in service_times.json.

Tuning guidance
- If backlog seconds persistently exceed targets, increase maxReplicaCount and/or lower backlogSecondsTarget.
- If replicas oscillate, increase cooldownPeriod/pollingInterval in the ScaledObject and widen max:min ratio.
