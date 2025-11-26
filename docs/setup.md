# Setup

Audience: Both

## Prerequisites
- Docker 24+ (optional: Docker Compose)
- Python 3.11+ (for tooling and validator)
- Network access to pull images and download spaCy models (if needed)

## Environment configuration
Recommended .env defaults:

```
# Ports
GATEWAY_PORT=8080
INTERCEPTOR_PORT=8081          # external
INTERNAL_INTERCEPTOR_PORT=8080 # container/internal

# Infrastructure
REDIS_URL=redis://redis:6379/0
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/hypergraph

# Upstream LLM proxy
LLM_PROXY_URL=http://llm-proxy:8080

# Explainer
EXPLAINER_TABLE=explanations_v2
SPACY_MODEL=en_core_web_sm
DEV_MODE=0

# Optional shadow verification
SHADOW_ENDPOINT=http://ollama:11434
VERIFY_MODEL=gpt-4o-mini
VERIFY_TEMPERATURE=0.0
VERIFY_TOP_K=3
```

## Build images
From the repo root:

- docker build -t yourco-gateway -f services/gateway/Dockerfile services/gateway
- docker build -t yourco-interceptor -f services/interceptor/Dockerfile services/interceptor
- docker build -t yourco-explainer -f services/explainer/Dockerfile services/explainer

## Run locally (Docker)
Create a network:
- docker network create hypergraph-net || true

Start infra:
- docker run -d --name redis --network hypergraph-net -p 6379:6379 redis:7-alpine
- docker run -d --name postgres --network hypergraph-net -p 5432:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_USER=postgres -e POSTGRES_DB=hypergraph postgres:16-alpine

Optional shadow model (example):
- docker run -d --name ollama --network hypergraph-net -p 11434:11434 ollama/ollama:latest

Start services:
- docker run -d --name interceptor --network hypergraph-net -p 8081:8081 -e REDIS_URL=redis://redis:6379/0 -e HOST=0.0.0.0 -e PORT=${INTERCEPTOR_PORT:-8081} yourco-interceptor
- docker run -d --name explainer --network hypergraph-net -e REDIS_URL=redis://redis:6379/0 -e DATABASE_URL=postgresql://postgres:postgres@postgres:5432/hypergraph -e DEV_MODE=${DEV_MODE:-0} -e SPACY_MODEL=${SPACY_MODEL:-en_core_web_sm} -e SHADOW_ENDPOINT=${SHADOW_ENDPOINT:-http://ollama:11434} -e VERIFY_MODEL=${VERIFY_MODEL:-gpt-4o-mini} yourco-explainer
- docker run -d --name gateway --network hypergraph-net -p 8080:8080 -e REDIS_URL=redis://redis:6379/0 -e DATABASE_URL=postgresql://postgres:postgres@postgres:5432/hypergraph -e LLM_PROXY_URL=${LLM_PROXY_URL:-http://llm-proxy:8080} yourco-gateway

## Verify
Gateway health (may be unimplemented; check logs if non-200):
- curl -sS http://localhost:8080/healthz || echo "Note: /healthz may not be implemented"

Minimal chat completion:
- curl -sS -H "Content-Type: application/json" -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Hello"}],"stream":false}' http://localhost:8080/v1/chat/completions

Redis ping:
- docker exec -it redis redis-cli PING

Postgres connectivity:
- docker exec -it postgres psql -U postgres -d hypergraph -c '\dt'

## Stop and clean up
- docker rm -f gateway explainer interceptor postgres redis || true
- docker rm -f ollama || true
- docker network rm hypergraph-net || true
