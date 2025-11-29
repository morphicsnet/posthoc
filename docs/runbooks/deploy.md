# Hypergraph Deploy Runbook

Prerequisites
- AWS EKS cluster (OIDC enabled), cluster-admin access.
- IAM roles for service accounts (IRSA) for services requiring AWS access (S3/KMS).
- Karpenter installed and configured with controller and CRDs (NodePool/EC2NodeClass or Provisioner/AWSNodeTemplate variant).
- Observability stack:
  - Prometheus Operator (for PodMonitor) and/or metrics pipeline.
  - OpenTelemetry Collector deployed. Reference: manifests/otel/otel-collector.yaml
- ArgoCD installed for GitOps (optional, recommended).

Configuration
- AUTH_TOKENS_JSON:
  - Provide plaintext JSON or base64-encoded JSON of static tokens for RBAC.
  - Set via .Values.global.rbacTokensJson or external secret (see secrets_and_keys.md).
- STATUS_BACKEND: "json" or other supported backends.
- S3 (optional):
  - Enable with .Values.global.s3.enabled=true
  - Configure bucket, prefix, kmsKeyId.
  - Ensure IAM permissions (PutObject, GetObject, kms:Encrypt/Decrypt if using CMK).
- OpenTelemetry:
  - Set .Values.global.otel.enabled=true
  - Set .Values.global.otel.otlpEndpoint to your collector service (e.g. http://otel-collector:4317).
- Rate Limits (gateway):
  - Set readRps/burst and writeRps/burst under .Values.global.rateLimits.

Install with Helm (local)
- Render templates:
  - helm template hypergraph manifests/helm/hypergraph
- Lint:
  - helm lint manifests/helm/hypergraph
- Install/upgrade:
  - helm upgrade --install hypergraph manifests/helm/hypergraph -n hypergraph --create-namespace \
      --set global.rbacTokensJson='{"tokens":[{"token":"REDACTED","scopes":["read"]}]}' \
      --set global.otel.otlpEndpoint='http://otel-collector.otel.svc.cluster.local:4317'

Install with ArgoCD (GitOps)
- Create Application from manifests/argocd/argocd-apps.yaml.
- Sync the Application; ArgoCD will create namespace and deploy all resources.

Post-Deploy Validation
- Health:
  - kubectl -n hypergraph get pods
  - Gateway/Explainer/Interceptor pods Ready=1/1
- Endpoints:
  - Gateway: curl http://<svc>:8080/healthz
  - Explainer: curl http://<svc>:9090/healthz
- Metrics:
  - /metrics endpoints reachable, scraped by Prometheus if configured.
- Traces:
  - Send a test request through Gateway; verify spans in your tracing backend via OTEL Collector.
- HIF retrieval:
  - Exercise explainer workflow and verify retrieval/attribution paths function.

Environment variables (quick matrix)
- Gateway
  - AUTH_MODE=none|static; AUTH_TOKENS_JSON (see shape in [rbac_dependency](services/gateway/src/rbac.py:62)); AUTH_EXPECT_ISS; AUTH_EXPECT_AUD
  - RATE_LIMIT_READ_RPS/RATE_LIMIT_READ_BURST; RATE_LIMIT_WRITE_RPS/RATE_LIMIT_WRITE_BURST; RL_COUNTERS_JSON_PATH
  - ENABLE_OTEL=1; OTEL_EXPORTER_OTLP_ENDPOINT; OTEL_EXPORTER_OTLP_PROTOCOL=http|grpc
  - STATUS_BACKEND=json|ddb; STATUS_JSON_PATH; TRACE_TTL_DAYS
  - LLM_PROXY_URL; REDIS_URL/REDIS_STREAM/REDIS_MAXLEN; DATABASE_URL; EXPLAINER_TABLE
  - GATEWAY_REDACT=0|1; GATEWAY_VERSION
- Explainer
  - ENABLE_OTEL=1; EXPLAINER_METRICS_HOST/EXPLAINER_METRICS_PORT (default 9090)
  - S3_BUCKET; S3_PREFIX; S3_KMS_KEY_ID (SSE‑KMS). Local fallback persists to file:///tmp/hif/<key>
  - CHAOS_CONTROL_PATH (see chaos runbook)
  - Optional domain knobs (if enabled in your build): USE_SAE_LOADER/ENABLE_SAE_SERVICE/DICT_ROOT; ATTR_* (budgets); BP_* (backpressure); HG_* (pruning)
- Interceptor
  - REDIS_URL; REDIS_STREAM; REDIS_MAXLEN; HOST; PORT

Notes
- Prefer setting these via Helm values and envFrom secret references. See [values.yaml](manifests/helm/hypergraph/values.yaml:1)
- For CORS during development, use a Vite dev proxy or temporarily enable CORSMiddleware near [app = FastAPI(...)](services/gateway/src/app.py:333)
