# Async Sidecar E2E Test Plan

This document describes how to configure and run the stdlib-only end-to-end (E2E) test harness for the production-ready Async Sidecar system. It includes prerequisites, environment configuration, execution instructions, result interpretation, and mappings to operational dashboards and runbooks.

Key code anchors:
- Orchestrator: [runner.py](tests/e2e/runner.py)
- Shared utilities: [utils.py](tests/e2e/utils.py)
- RBAC dependency: [rbac_dependency()](services/gateway/src/rbac.py:62)
- Gateway endpoints: [create_chat_completion()](services/gateway/src/app.py:540), [get_trace_status()](services/gateway/src/app.py:886), [get_trace_graph()](services/gateway/src/app.py:922), [register_webhook()](services/gateway/src/app.py:1021), [cancel_trace()](services/gateway/src/app.py:1062)
- Observability setup (Gateway/Explainer): [setup_otel()](services/gateway/src/otel.py:330), [setup_otel()](services/explainer/src/otel.py:273)
- HIF validation entrypoint: [validate_hif()](libs/hif/validator.py:117)
- Chaos injector CLI: [main()](tests/chaos/chaos_injector.py:90)
- Load harness: [run_load_async()](tests/load/load_runner.py:455)
- Attach-rate analyzer: [main()](tools/analysis/attach_rate_analyzer.py:396)
- OpenAPI: [hypergraph-api.yaml](api/openapi/hypergraph-api.yaml:1)
- Change management policy: [docs/change-management.md](docs/change-management.md:1)

## 1) Scope

The E2E harness validates the workflow from Gateway ingress through Interceptor processing to Explainer execution, focusing on:

- RBAC validation and scope matrix
- End-to-end trace lifecycle and HIF schema compliance (v1)
- Tenant isolation and StatusStore immutability (local JSON or DDB stub)
- Observability (metrics presence, low-cardinality correlation)
- Backpressure behavior under stress and HIF continuity
- SLO adherence for explanation latencies
- Deployment artifacts (Helm template) and GitOps sanity (ArgoCD Applications)
- Autoscaling policies (KEDA scaler alignment with values.yaml; Karpenter CPU/GPU presence)
- Chaos injection and recovery outcomes
- Attach-rate/capacity recommendations vs configured replicas
- Audit logging integrity and PII scrubbing
- API v1 path/versioning compliance and change-management guardrails

No service code or behavior is modified by these tests.

## 2) Prerequisites

Required:
- Python 3.9+ (stdlib-only tests; optional imports are guarded)
- Running Gateway (local or remote), reachable at `--base-url`
- If RBAC is enabled (`AUTH_MODE=static`), provide tokens and implied tenants via CLI

Optional (tests skip gracefully if absent):
- Helm CLI for templating chart (`helm template`)
- ArgoCD Applications YAML
- Explainer metrics endpoint (Prometheus exposition)
- kubectl, argocd CLIs (not required by default)
- Local StatusStore JSON (`/tmp/hif/status.json`) or configured via `--status-json`
- Audit log JSONL (`/var/log/hypergraph/audit.log`) for audit checks
- Chaos control file path for chaos tests
- Prometheus-scrapable Gateway/Explainer metrics endpoints

## 3) Tokens and RBAC configuration

The Gateway can enforce RBAC with static tokens. See the reference dependency [rbac_dependency()](services/gateway/src/rbac.py:62). If `AUTH_MODE=none`, RBAC tests skip.

Environment example for static tokens:
```
AUTH_MODE=static
AUTH_TOKENS_JSON={"token_write":{"tenant_id":"tenantA","scopes":["traces:write","traces:read"],"subject":"userA"},"token_read":{"tenant_id":"tenantB","scopes":["traces:read"],"subject":"userB"}}
```

## 4) Orchestrator CLI

Run the orchestrator from the repository root:
```
python3 tests/e2e/runner.py \
  --base-url http://localhost:8080 \
  --metrics-gateway http://localhost:8080/metrics \
  --metrics-explainer http://localhost:9090/metrics \
  --auth-token-write TOKEN_WRITE \
  --auth-token-read TOKEN_READ \
  --tenant-a tenantA --tenant-b tenantB \
  --status-json /tmp/hif/status.json \
  --audit-log /var/log/hypergraph/audit.log \
  --s3-check file:///tmp/hif \
  --helm-chart manifests/helm/hypergraph \
  --argocd-manifest manifests/argocd/argocd-apps.yaml \
  --keda-template manifests/helm/hypergraph/templates/keda-scalers.yaml \
  --karpenter-file manifests/karpenter/karpenter-provisioners.yaml \
  --load-duration 120 --concurrency 300 --attach-rate 0.3 --token-mix 0.1 \
  --chaos-control /tmp/hif/chaos.json \
  --output tests/e2e/results/e2e_report.json
```

Useful selection flags:
- `--only "test_*slo*,test_*rbac*"` to run a subset
- `--skip "test_*chaos*"` to skip chaos tests

Outputs:
- JSON: `tests/e2e/results/e2e_report.json`
- Markdown: `tests/e2e/results/e2e_report.md`

The orchestrator will skip tests with clear reasons when external dependencies are not present.

## 5) Subtests overview

- RBAC: [test_gateway_rbac.py](tests/e2e/test_gateway_rbac.py)
  - Missing token -> 401; invalid -> 403; valid scope matrix
- Trace workflow: [test_trace_workflow.py](tests/e2e/test_trace_workflow.py)
  - POST with `x-explain-mode: hypergraph`, poll status, GET graph, validate HIF (meta.version = "hif-1")
- Tenant isolation: [test_security_tenant_isolation.py](tests/e2e/test_security_tenant_isolation.py)
  - Cross-tenant denial (403/404) and StatusStore tenant immutability
- Observability: [test_observability.py](tests/e2e/test_observability.py)
  - Metrics scraping; presence checks (http_requests_total, chat_requests_total, explainer_jobs_total, etc.)
- Backpressure under load: [test_backpressure_under_load.py](tests/e2e/test_backpressure_under_load.py)
  - Invoke [run_load_async()](tests/load/load_runner.py:455), look for BP signals and HIF continuity
- SLO adherence: [test_slo_adherence.py](tests/e2e/test_slo_adherence.py)
  - Evaluate p95 sentence/token vs thresholds (≤2s, ≤8s) from load summary
- Deploy artifacts: [test_deploy_artifacts.py](tests/e2e/test_deploy_artifacts.py)
  - `helm template` sanity for Deployments/Services/PDB/NetworkPolicies/ScaledObject; ArgoCD optional checks
- Autoscaling policies: [test_autoscaling_policies.py](tests/e2e/test_autoscaling_policies.py)
  - KEDA `minReplicaCount`/`maxReplicaCount`/`backlogSecondsTarget` consistent with [values.yaml](manifests/helm/hypergraph/values.yaml:1); Karpenter CPU/GPU presence
- Chaos recovery: [test_chaos_recovery.py](tests/e2e/test_chaos_recovery.py)
  - Toggle fail-attribution and verify failures occur then recover after disable
- Capacity attach-rate: [test_capacity_attach_rate.py](tests/e2e/test_capacity_attach_rate.py)
  - Run [attach_rate_analyzer.py](tools/analysis/attach_rate_analyzer.py:396) and compare recommendations vs Helm min/max replicas
- Audit & PII: [test_audit_pii.py](tests/e2e/test_audit_pii.py)
  - Authorization headers absent; snippets PII-scrubbed per [sanitize_message()](libs/sanitize/pii.py:72)
- API versioning: [test_api_versioning.py](tests/e2e/test_api_versioning.py)
  - All OpenAPI paths under `/v1`; change-management doc mentions ADR + freeze; optional runtime HIF version check

## 6) Configuration and environment variables

- Gateway:
  - `AUTH_MODE=static|none`
  - `AUTH_TOKENS_JSON` (for `static`)
  - `STATUS_BACKEND=json|ddb`, `STATUS_JSON_PATH` (default: `/tmp/hif/status.json`)
  - `ENABLE_OTEL=1` to expose `/metrics`
- Explainer:
  - `ENABLE_OTEL=1`, `EXPLAINER_METRICS_PORT` (default: `9090`)
- Chaos:
  - `CHAOS_CONTROL_PATH=/tmp/hif/chaos.json` or pass via `--chaos-control`

## 7) Interpreting results

The orchestrator emits a consolidated JSON and Markdown report with:
- Per-test PASS/FAIL/SKIP status and reasons
- Environment snapshot (tools found, versions, git rev)
- Aggregated SLO results and notes about capacity and autoscaling checks
- Chaos and recovery outcomes
- Helm/ArgoCD linting presence

Common SKIP reasons (non-fatal):
- RBAC disabled (`AUTH_MODE=none`)
- Helm/jq/argocd absent
- Audit log or StatusStore path unavailable
- Metrics endpoints not enabled
- Chat upstream error (LLM proxy not configured)

## 8) Operational runbooks and dashboards

- Deployment and rollout: [deploy.md](docs/runbooks/deploy.md:1)
- Scale and recovery: [scale_and_recovery.md](docs/runbooks/scale_and_recovery.md:1)
- Attach rate & capacity: [attach_rate_and_capacity.md](docs/runbooks/attach_rate_and_capacity.md:1)
- Secrets and keys: [secrets_and_keys.md](docs/runbooks/secrets_and_keys.md:1)
- Troubleshooting: [docs/troubleshooting.md](docs/troubleshooting.md:1)
- Grafana dashboard JSON: [sidecar-overview.json](dashboards/grafana/sidecar-overview.json:1)

## 9) SLOs and acceptance criteria

Target SLOs validated by the suite:
- Explanation latency p95:
  - Sentence: ≤ 2s
  - Token:    ≤ 8s
- Attach-rate-based capacity guidance is within bounds of Helm KEDA min/max (with small slack)
- Presence-based backpressure signals baked into metrics when under load
- API v1 compliance: HIF meta.version == `hif-1`, OpenAPI paths under `/v1`, and change-management guardrails

## 10) Tips

- For local smoke:
  - Start Gateway with `AUTH_MODE=none` to bypass tokens. RBAC tests will SKIP; others still run.
  - Set `ENABLE_OTEL=1` to expose `/metrics` and unlock observability checks.
  - Use `--only` filter to iterate quickly on a specific test.
- For CI:
  - Configure tokens and tenants via CI secrets to enable RBAC and tenant isolation tests.
  - Optionally stage the Helm chart and point orchestrator to a rendered manifest bundle.

## 11) FAQ

Q: Tests are failing due to chat upstream 500 errors.  
A: Configure `LLM_PROXY_URL` on the Gateway or accept SKIPs for chat-dependent tests until upstream is available. See the chat handler [create_chat_completion()](services/gateway/src/app.py:540).

Q: HIF validation fails due to missing `jsonschema`.  
A: The E2E suite will fall back to minimal HIF shape checks when `jsonschema` is unavailable. For strict validation, install `jsonschema>=4.18` and ensure [validate_hif()](libs/hif/validator.py:117) is importable.

Q: Metrics checks are skipping.  
A: Ensure `ENABLE_OTEL=1` is set and that `/metrics` endpoints are reachable. Gateway’s metrics endpoint is configured by [setup_otel()](services/gateway/src/otel.py:330); Explainer’s by [setup_otel()](services/explainer/src/otel.py:273).

Q: Helm checks are failing due to missing CLI.  
A: Install Helm or skip `test_deploy_artifacts.py`. Tests skip gracefully when Helm is unavailable.
