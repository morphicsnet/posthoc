# Your Company Hypergraph Explainability Stack

## What is this?

Production-ready Async Sidecar pattern that decouples inference and explanation lanes. The Gateway serves inference (OpenAI-compatible), while the Explainer runs asynchronously using trace_id propagation with polling/streaming APIs for hypergraph retrieval. One stack to capture, explain, and validate model behavior end to end. It includes:
- Gateway: an OpenAI-compatible HTTP layer that forwards chat completions to a provider/proxy and emits explainability metadata. See entrypoint [services/gateway/src/app.py](services/gateway/src/app.py) and handler [create_chat_completion()](services/gateway/src/app.py:540).
- Interceptor: a lightweight ingest service that accepts model I/O and traces, placing events onto Redis Streams for downstream processing. See [services/interceptor/src/capture.py](services/interceptor/src/capture.py) and handler [ingest()](services/interceptor/src/capture.py:265).
- Explainer: a worker that consumes traces, performs concept extraction and verification, and materializes a Hypergraph (HIF). See [services/explainer/src/worker.py](services/explainer/src/worker.py).
- Hypergraph API (HIF): a vendor-neutral JSON schema and HTTP surface for explainability. OpenAPI lives at [api/openapi/hypergraph-api.yaml](api/openapi/hypergraph-api.yaml).

## Key capabilities

- OpenAI-compatible chat completions (POST /v1/chat/completions)
- Explainability via x-explain-* headers and async trace lifecycle
- End-to-end Hypergraph generation and retrieval (HIF)
- HIF validation utilities for CI and runtime safety

## System architecture at a glance

Gateway receives the request, emits trace context, Interceptor buffers events, and Explainer builds the hypergraph and optional verifications before the graph is fetched by clients. See the high-level overview in [docs/architecture.md](docs/architecture.md). Diagram source (Mermaid) is at [docs/diagrams/architecture.mmd](docs/diagrams/architecture.mmd).

## Quickstart matrix

- Local
  - Docker: bring up Redis, Gateway, Interceptor, Explainer. See [docs/setup.md](docs/setup.md:1)
  - Python: run end-to-end script [e2e_quickstart.py](examples/python/e2e_quickstart.py:1)
- EKS
  - Helm: chart and values at [manifests/helm/hypergraph](manifests/helm/hypergraph/Chart.yaml:1), scalers [keda-scalers.yaml](manifests/helm/hypergraph/templates/keda-scalers.yaml:1)
  - GitOps: ArgoCD apps at [manifests/argocd/argocd-apps.yaml](manifests/argocd/argocd-apps.yaml:1)
  - Compute: Karpenter NodePools [karpenter-provisioners.yaml](manifests/karpenter/karpenter-provisioners.yaml:1)
- SDKs
  - Python SDK quickstart [docs/tutorials/python-quickstart.md](docs/tutorials/python-quickstart.md:1) using [Client](sdks/python/your_company_explainability/your_company_explainability/client.py:143)
  - TypeScript/React quickstart [docs/tutorials/typescript-quickstart.md](docs/tutorials/typescript-quickstart.md:1) using [ExplainabilityClient](sdks/typescript/explainability-ui/src/client.ts:19) and [HypergraphViewer](sdks/typescript/explainability-ui/src/HypergraphViewer.tsx:233)

## Quick start (Docker)

Minimal local run with sensible defaults. For a complete setup (including Postgres and optional shadow verification), see [docs/setup.md](docs/setup.md).

1) Create a network and start Redis:

- docker network create hypergraph-net || true
- docker run -d --name redis --network hypergraph-net -p 6379:6379 redis:7-alpine

2) Build service images:

- docker build -t yourco-gateway ./services/gateway
- docker build -t yourco-interceptor ./services/interceptor
- docker build -t yourco-explainer ./services/explainer

3) Run the services (basic defaults):

- docker run -d --name interceptor --network hypergraph-net -p 8081:8081 -e REDIS_URL=redis://redis:6379/0 yourco-interceptor
- docker run -d --name explainer --network hypergraph-net -e REDIS_URL=redis://redis:6379/0 -e DEV_MODE=0 yourco-explainer
- docker run -d --name gateway --network hypergraph-net -p 8080:8080 -e REDIS_URL=redis://redis:6379/0 -e LLM_PROXY_URL=http://llm-proxy:8080 yourco-gateway

4) Verify:

- curl -sS http://localhost:8080/healthz || echo "Note: /healthz may not yet be implemented; check container logs."
- curl -sS -H "Content-Type: application/json" -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Hello"}],"stream":false}' http://localhost:8080/v1/chat/completions
## API, HIF, and quick links

- API Reference: [docs/api-reference.md](docs/api-reference.md:1) (authoritative spec [hypergraph-api.yaml](api/openapi/hypergraph-api.yaml:1))
- HIF Schema Reference (v1): [docs/hif-schema.md](docs/hif-schema.md:1) and canonical schema [schema.json](libs/hif/schema.json:1)
- Tutorials:
  - Python Quickstart: [docs/tutorials/python-quickstart.md](docs/tutorials/python-quickstart.md:1)
  - TypeScript Quickstart: [docs/tutorials/typescript-quickstart.md](docs/tutorials/typescript-quickstart.md:1)
  - End-to-end: [docs/tutorials/e2e.md](docs/tutorials/e2e.md:1)
- Security and Compliance:
  - Security guide: [docs/security/SECURITY.md](docs/security/SECURITY.md:1)
  - IAM policies: [docs/security/iam-policies.md](docs/security/iam-policies.md:1)
- Runbooks (delivery/ops):
  - Deploy: [docs/runbooks/deploy.md](docs/runbooks/deploy.md:1)
  - Scale & Recovery: [docs/runbooks/scale_and_recovery.md](docs/runbooks/scale_and_recovery.md:1)
  - Secrets & Keys: [docs/runbooks/secrets_and_keys.md](docs/runbooks/secrets_and_keys.md:1)
  - Attach Rate & Capacity: [docs/runbooks/attach_rate_and_capacity.md](docs/runbooks/attach_rate_and_capacity.md:1)
  - Microbenchmark & Autoscaling: [docs/runbooks/microbenchmark_autoscaling.md](docs/runbooks/microbenchmark_autoscaling.md:1)
- Delivery artifacts:
  - Helm chart: [manifests/helm/hypergraph](manifests/helm/hypergraph/Chart.yaml:1)
  - KEDA ScaledObject template: [keda-scalers.yaml](manifests/helm/hypergraph/templates/keda-scalers.yaml:1)
  - Karpenter NodePools: [karpenter-provisioners.yaml](manifests/karpenter/karpenter-provisioners.yaml:1)
  - ArgoCD Applications: [argocd-apps.yaml](manifests/argocd/argocd-apps.yaml:1)
- Observability:
  - Gateway metrics: GET http://localhost:8080/metrics (enable via [setup_otel()](services/gateway/src/otel.py:330))
  - Explainer metrics: GET http://localhost:9090/metrics (enable via [otel.py](services/explainer/src/otel.py:249))
  - Grafana dashboard: [dashboards/grafana/sidecar-overview.json](dashboards/grafana/sidecar-overview.json:1)
- Change Management:
  - Policy: [docs/change-management.md](docs/change-management.md:1)
  - Checklist: [docs/change-management-checklist.md](docs/change-management-checklist.md:1)


## Support and license

- Issues: open a ticket in this repository with logs and reproduction steps.
- License: Provided for evaluation; replace this note with your company’s license terms.