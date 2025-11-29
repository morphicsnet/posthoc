# Attach-rate analysis report

- Window start: 2025-11-29T00:56:16+00:00
- Now:          2025-11-29T01:56:16+00:00
- Bucket size:  60 seconds
- Buckets:      0

## Baseline attach-rate

- Mean:   0.0
- Median: 0.0
- P95:    0.0

## Assumptions

- Token mix: 0.1 (sentence=0.9)
- Sentence throughput (explanations/sec/GPU) min=3.0 max=6.0
- Token throughput    (explanations/sec/GPU) min=0.4 max=0.8
- Note: StatusStore JSON missing; results derived from audit only.

## Capacity recommendations

| Concurrency | Mean total GPUs (cons/opt) | P95 total GPUs (cons/opt) |
|-------------|-----------------------------|----------------------------|
| 300 | 0/2 | 0/2 |

Apply via Karpenter node groups and KEDA min/max replicas. See:
- Helm KEDA scaler: [manifests/helm/hypergraph/templates/keda-scalers.yaml](manifests/helm/hypergraph/templates/keda-scalers.yaml)
- Helm values: [manifests/helm/hypergraph/values.yaml](manifests/helm/hypergraph/values.yaml)
