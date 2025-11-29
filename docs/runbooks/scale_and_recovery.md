# Scale and Recovery Runbook

KEDA Scaling (Explainer)
- Primary trigger: backlog-seconds metric. Target configured via .Values.explainer.keda.backlogSecondsTarget.
- Adjust:
  - minReplicaCount/maxReplicaCount based on SLOs and budget.
  - backlogSecondsTarget to tune responsiveness vs. cost.
- Alternative Kafka scaler:
  - Use lagThreshold for topic "envelopes" if enabling Kafka, see commented stub in keda-scalers.yaml.

Backpressure and Degradation
- Interpret backpressure signals (queue depth, backlog-seconds).
- Implement degradation ladder in services to prefer timely responses under load.
- Validate with load/chaos testing; see docs/testing/load_stress_chaos.md.

Node Provisioning with Karpenter
- CPU pool: NodePool hypergraph-cpu under manifests/karpenter/karpenter-provisioners.yaml.
- GPU pool: NodePool hypergraph-gpu with accelerator requirements.
- Customize:
  - instance families, Spot/On-Demand mix, budgets, TTLSecondsAfterEmpty (via expiration/ttlSecondsAfterEmpty).
  - subnet/security group selectors, IAM roles in EC2NodeClass.
- Verify scale events:
  - Watch Karpenter controller logs.
  - Confirm nodes join and pods schedule quickly.

Recovery Procedures
- S3 Failures:
  - Check IRSA and bucket policy/KMS key permissions.
  - Temporarily disable S3 via .Values.global.s3.enabled=false if blocking.
- Attribution Errors:
  - Inspect explainer logs; roll back to last known good image via ArgoCD rollback or helm rollback.
- Kafka Lag (if enabled):
  - Increase maxReplicaCount temporarily.
  - Investigate broker health and consumer offsets. Consider backfill windows.

Operational Tips
- Use PDBs to avoid full outage during maintenance.
- Prefer GitOps changes via PRs to maintain audit history.
- Coordinate disruptive changes (GPU migration, KMS rotation) during maintenance windows.

## Troubleshooting KEDA not scaling
- Check ScaledObject status and HPA:
  - `kubectl -n <ns> get scaledobject,hpa`
- Validate metric source and query in [keda-scalers.yaml](manifests/helm/hypergraph/templates/keda-scalers.yaml:1):
  - serverAddress correct
  - metricName matches Prometheus series
  - query filters namespace/app labels that actually exist
- Ensure Explainer metrics endpoint is up: `curl -sS http://<explainer-svc>:9090/metrics | head`
- Tune `.Values.explainer.keda.backlogSecondsTarget`, minReplicaCount, maxReplicaCount in [values.yaml](manifests/helm/hypergraph/values.yaml:1) and increase cooldownPeriod if oscillation observed.
