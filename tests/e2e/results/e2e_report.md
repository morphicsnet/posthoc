# Async Sidecar E2E Report

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
- tools: {'helm': {'path': None, 'version': None}, 'kubectl': {'path': '/usr/local/bin/kubectl', 'version': 'clientVersion:\n  buildDate: "2025-02-12T21:26:09Z"\n  compiler: gc\n  gitCommit: 67a30c0adcf52bd3f56ff0893ce19966be12991f\n  gitTreeState: clean\n  gitVersion: v1.32.2\n  goVersion: go1.23.6\n  major: "1"\n  minor: "32"\n  platform: darwin/arm64\nkustomizeVersion: v5.5.0'}, 'argocd': {'path': None, 'version': None}, 'git': {'path': '/opt/local/bin/git', 'revision': '7bd11f2'}}

## Results

| Test | Status | Duration(ms) | Reason |
|------|--------|--------------|--------|
| test_gateway_rbac.py | SKIP |  | AUTH_MODE appears disabled (probe without token did not return 401/403) |
| test_trace_workflow.py | SKIP | 82 |  |
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

