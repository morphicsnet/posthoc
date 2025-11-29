# Documentation

This folder contains the official docs for the Your Company Hypergraph Explainability Stack. If you are new, start with the overview, then run a local stack, and finally try the end‑to‑end tutorial.

- Architecture overview: [docs/architecture.md](docs/architecture.md:1)
- Setup (Docker + local): [docs/setup.md](docs/setup.md:1)
- API reference (endpoints, headers, examples): [docs/api-reference.md](docs/api-reference.md:1)
- HIF Schema Reference: [docs/hif-schema.md](docs/hif-schema.md:1)
- Quickstarts:
  - Python: [docs/tutorials/python-quickstart.md](docs/tutorials/python-quickstart.md:1)
  - TypeScript/React: [docs/tutorials/typescript-quickstart.md](docs/tutorials/typescript-quickstart.md:1)
  - End-to-end: [docs/tutorials/e2e.md](docs/tutorials/e2e.md:1)
- Delivery & Ops:
  - Deploy: [docs/runbooks/deploy.md](docs/runbooks/deploy.md:1)
  - Scale & Recovery: [docs/runbooks/scale_and_recovery.md](docs/runbooks/scale_and_recovery.md:1)
  - Secrets & Keys: [docs/runbooks/secrets_and_keys.md](docs/runbooks/secrets_and_keys.md:1)
  - Attach Rate & Capacity: [docs/runbooks/attach_rate_and_capacity.md](docs/runbooks/attach_rate_and_capacity.md:1)
  - Microbenchmark & Autoscaling: [docs/runbooks/microbenchmark_autoscaling.md](docs/runbooks/microbenchmark_autoscaling.md:1)
  - Helm/KEDA/Karpenter/ArgoCD: [Chart](manifests/helm/hypergraph/Chart.yaml:1) | [KEDA](manifests/helm/hypergraph/templates/keda-scalers.yaml:1) | [Karpenter](manifests/karpenter/karpenter-provisioners.yaml:1) | [ArgoCD](manifests/argocd/argocd-apps.yaml:1)
- Testing:
  - Load/Chaos: [docs/testing/load_stress_chaos.md](docs/testing/load_stress_chaos.md:1)
- Security & Compliance:
  - Guide: [docs/security/SECURITY.md](docs/security/SECURITY.md:1)
  - IAM: [docs/security/iam-policies.md](docs/security/iam-policies.md:1)
- Change Management:
  - Plan: [docs/change-management.md](docs/change-management.md:1)
  - Checklist: [docs/change-management-checklist.md](docs/change-management-checklist.md:1)
- Troubleshooting: [docs/troubleshooting.md](docs/troubleshooting.md:1)
- FAQ: [docs/faq.md](docs/faq.md:1)
- Workshop outline: [docs/workshop/workshop-60min-outline.md](docs/workshop/workshop-60min-outline.md:1)

Key repo references used throughout:
- Gateway entrypoint and handler [create_chat_completion()](services/gateway/src/app.py:540)
- Interceptor ingest: [services/interceptor/src/capture.py](services/interceptor/src/capture.py)
- Explainer worker: [services/explainer/src/worker.py](services/explainer/src/worker.py)
- OpenAPI spec: [api/openapi/hypergraph-api.yaml](api/openapi/hypergraph-api.yaml)
- HIF schema: [libs/hif/schema.json](libs/hif/schema.json)
- HIF validator CLI: [libs/hif/validator.py](libs/hif/validator.py:1)

## Who should read what

- External (users, partners)
  - Start with: Architecture, Setup, API Reference, HIF Schema, Quickstarts, FAQ, Troubleshooting.
  - Skip internal runbooks; those live under each service directory.
- Internal (operators, developers)
  - Start with: Architecture, Setup, Service READMEs, Troubleshooting.
  - Deep dive code in [services/gateway/src/app.py](services/gateway/src/app.py), [services/interceptor/src/capture.py](services/interceptor/src/capture.py), [services/explainer/src/worker.py](services/explainer/src/worker.py).

## Terminology

- Hypergraph API (HIF): A vendor-neutral JSON schema and HTTP contract for explainability artifacts and graphs.
- HIF (v1): Legacy interchange format documented in [docs/hif-schema.md](docs/hif-schema.md:1) and defined in [libs/hif/schema.json](libs/hif/schema.json).
- Gateway: OpenAI-compatible front door; see [create_chat_completion()](services/gateway/src/app.py:540).
- Interceptor: Ingests model I/O to Redis Streams; see [services/interceptor/src/capture.py](services/interceptor/src/capture.py).
- Explainer: Consumes traces, extracts concepts, verifies; see [services/explainer/src/worker.py](services/explainer/src/worker.py).