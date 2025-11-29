# Async Sidecar E2E - Verbose Report

Source: [runner.py](tests/e2e/runner.py:1)

## Environment

- base_url: http://localhost:8080
- metrics_gateway: http://localhost:8080/metrics
- metrics_explainer: http://localhost:9090/metrics
- auth_write_set: False
- auth_read_set: False
- tenant_a: tA
- tenant_b: tB
- status_json: /tmp/hif/status.json
- audit_log: /var/log/hypergraph/audit.log
- s3_check: file:///tmp/hif
- helm_chart: manifests/helm/hypergraph
- argocd_manifest: manifests/argocd/argocd-apps.yaml
- keda_template: manifests/helm/hypergraph/templates/keda-scalers.yaml
- karpenter_file: manifests/karpenter/karpenter-provisioners.yaml
- load_duration: 120
- concurrency: 300
- attach_rate: 0.3
- token_mix: 0.1
- chaos_control: /tmp/hif/chaos.json
- tools:
  - helm: not found
  - kubectl: /usr/local/bin/kubectl (v1.32.2)
  - argocd: not found
  - git: /opt/local/bin/git (rev 7bd11f2)

## Summary

- Total: 12
- Passed: 4
- Failed: 1
- Skipped: 7

## Results table

| Test | Status | Duration(ms) | Reason |
|------|--------|--------------|--------|
| test_gateway_rbac.py | SKIP |  | AUTH_MODE appears disabled (probe without token did not return 401/403) |
| test_trace_workflow.py | SKIP | 82 | chat upstream error 599 (connection refused) |
| test_security_tenant_isolation.py | SKIP |  | AUTH_MODE appears disabled (no 401/403 on unauthenticated probe) |
| test_observability.py | SKIP |  | Gateway /metrics unavailable or missing core metrics |
| test_backpressure_under_load.py | FAIL |  | no graphs retrievable for sampled attached traces |
| test_slo_adherence.py | PASS |  |  |
| test_deploy_artifacts.py | SKIP |  | helm not found in PATH |
| test_autoscaling_policies.py | PASS |  |  |
| test_chaos_recovery.py | SKIP |  | baseline polling timeout (environment not ready) |
| test_capacity_attach_rate.py | PASS |  |  |
| test_audit_pii.py | SKIP |  | audit log not found at /var/log/hypergraph/audit.log |
| test_api_versioning.py | PASS |  |  |

## Detailed results

### test_gateway_rbac.py

- Status: SKIP
- Reason: AUTH_MODE appears disabled (probe without token did not return 401/403)
- Endpoint references: [rbac_dependency()](services/gateway/src/rbac.py:62), [create_chat_completion()](services/gateway/src/app.py:540)
- Action: Set AUTH_MODE=static and provide --auth-token-write/--auth-token-read to enable RBAC coverage.

### test_trace_workflow.py

- Status: SKIP
- Reason: chat upstream error 599: [Errno 61] Connection refused
- Likely cause: Gateway lacks LLM upstream (LLM_PROXY_URL not configured).
- References: [create_chat_completion()](services/gateway/src/app.py:540), [get_trace_status()](services/gateway/src/app.py:886), [get_trace_graph()](services/gateway/src/app.py:922), [validate_hif()](libs/hif/validator.py:117)
- Action: Set LLM_PROXY_URL and re-run.

### test_security_tenant_isolation.py

- Status: SKIP
- Reason: AUTH_MODE appears disabled (no 401/403 on unauthenticated probe)
- References: [put_status()](services/explainer/src/status_store.py:100)
- Action: Enable static tokens and pass --tenant-a/--tenant-b tokens.

### test_observability.py

- Status: SKIP
- Reason: Gateway /metrics unavailable
- References: [setup_otel()](services/gateway/src/otel.py:330), [setup_otel()](services/explainer/src/otel.py:273)
- Action: Set ENABLE_OTEL=1 for Gateway/Explainer and expose /metrics.

### test_backpressure_under_load.py

- Status: FAIL
- Reason: no graphs retrievable for sampled attached traces
- Chat summary: count=27579, 5xx=27579 (error rate 1.0) -> upstream unavailable
- Explanation results: count=8268, p95 sentence/token = 0 (no completed graphs)
- Metrics BP signals: none observed
- Actions:
  - Configure LLM upstream (LLM_PROXY_URL), run Explainer/Interceptor.
  - Ensure STATUS_BACKEND=json and Explainer writes to /tmp/hif/status.json.
  - Verify Gateway emits /metrics with ENABLE_OTEL=1.

### test_slo_adherence.py

- Status: PASS
- Note: p95=0ms due to no completed explanations; not a meaningful pass.
- Action: Re-run after enabling upstream to get real SLA measurements.

### test_deploy_artifacts.py

- Status: SKIP
- Reason: helm not found in PATH
- References: [Chart.yaml](manifests/helm/hypergraph/Chart.yaml:1)
- Action: Install Helm and re-run.

### test_autoscaling_policies.py

- Status: PASS
- KEDA: min=2, max=16, backlogSecondsTarget=1.0; template references present.
- Karpenter: CPU/GPU NodePools and EC2NodeClass present.
- References: [keda-scalers.yaml](manifests/helm/hypergraph/templates/keda-scalers.yaml:1), [values.yaml](manifests/helm/hypergraph/values.yaml:1), [karpenter-provisioners.yaml](manifests/karpenter/karpenter-provisioners.yaml:1)

### test_chaos_recovery.py

- Status: SKIP
- Reason: baseline polling timeout (environment not ready)
- References: [chaos_injector.py](tests/chaos/chaos_injector.py:1)
- Action: Enable upstream first; then chaos should induce failure and recovery.

### test_capacity_attach_rate.py

- Status: PASS
- Recommendation vs Helm min/max: satisfied (mean.opt >= min, p95.cons <= max + slack)
- References: [attach_rate_analyzer.py](tools/analysis/attach_rate_analyzer.py:396), [values.yaml](manifests/helm/hypergraph/values.yaml:1)

### test_audit_pii.py

- Status: SKIP
- Reason: audit log not found at /var/log/hypergraph/audit.log
- References: [sanitize_message()](libs/sanitize/pii.py:72)
- Action: Enable audit emission in Gateway and pass --audit-log path.

### test_api_versioning.py

- Status: PASS
- OpenAPI paths constrained to /v1; change-management doc OK.
- Runtime HIF version check skipped due to upstream error.
- References: [hypergraph-api.yaml](api/openapi/hypergraph-api.yaml:1), [docs/change-management.md](docs/change-management.md:1)

## Next actions to move suite to PASS

1) Configure LLM upstream for Gateway (LLM_PROXY_URL).
2) Enable metrics: set ENABLE_OTEL=1 for Gateway/Explainer.
3) Enable RBAC: AUTH_MODE=static and provide --auth-token-write/--auth-token-read.
4) Ensure StatusStore backend is active: STATUS_BACKEND=json and /tmp/hif/status.json writable.
5) Install Helm to enable chart templating checks.
6) Enable audit logging and pass --audit-log path.

## Re-run command

python3 tests/e2e/runner.py --base-url http://localhost:8080 --metrics-gateway http://localhost:8080/metrics --metrics-explainer http://localhost:9090/metrics --auth-token-write TOKEN_WRITE --auth-token-read TOKEN_READ --tenant-a tA --tenant-b tB --status-json /tmp/hif/status.json --audit-log /var/log/hypergraph/audit.log --s3-check file:///tmp/hif --helm-chart manifests/helm/hypergraph --argocd-manifest manifests/argocd/argocd-apps.yaml --keda-template manifests/helm/hypergraph/templates/keda-scalers.yaml --karpenter-file manifests/karpenter/karpenter-provisioners.yaml --load-duration 120 --concurrency 300 --attach-rate 0.3 --token-mix 0.1 --chaos-control /tmp/hif/chaos.json --output tests/e2e/results/e2e_report.json