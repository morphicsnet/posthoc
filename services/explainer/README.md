# Explainer (Internal)

## Purpose
Consumes Redis Streams, performs concept extraction and optional shadow-model verification, and persists a Hypergraph (HIF).

Worker code: [services/explainer/src/worker.py](services/explainer/src/worker.py), core extraction in [concept_extraction()](services/explainer/src/worker.py:243)

## Operability
Env vars:
- REDIS_URL
- DEV_MODE (0 recommended in production)
- SPACY_MODEL (e.g., en_core_web_sm)
- DATABASE_URL (Postgres DSN; writes to EXPLAINER_TABLE)
- EXPLAINER_TABLE (e.g., explanations_v2)
- SHADOW_ENDPOINT (optional)
- VERIFY_MODEL, VERIFY_TEMPERATURE, VERIFY_TOP_K (optional)
- COST_* (optional accounting toggles/limits)

## Run and deploy
- docker run -d --name explainer --network hypergraph-net -e REDIS_URL=redis://redis:6379/0 -e DATABASE_URL=postgresql://postgres:postgres@postgres:5432/hypergraph -e DEV_MODE=0 -e SPACY_MODEL=en_core_web_sm yourco-explainer

## Security

See: [docs/security/SECURITY.md](docs/security/SECURITY.md)

- Audit logging
  - AUDIT_LOG_ENABLE=1
  - AUDIT_LOG_PATH=/var/log/hypergraph/audit.log (falls back to stdout if unwritable)
  - Events: trace.running, trace.complete, trace.failed, trace.canceled (low-cardinality, no secrets)
- PII scrubbing
  - Sanitization utilities live in [libs/sanitize/pii.py](libs/sanitize/pii.py:1). Only use for log copies of user-provided strings.
- S3 persistence
  - S3_BUCKET, S3_PREFIX (default traces)
  - S3_KMS_KEY_ID enables SSE-KMS at write
  - Keys are path-safe slugified; traversal is refused. See inline SECURITY notes in [services/explainer/src/s3_store.py](services/explainer/src/s3_store.py:1).
- StatusStore
  - Tenant immutability enforced in Local JSON backend; attempts to change tenant_id are ignored and logged.

## Observability (metrics, logs, traces)

This service emits low-overhead telemetry guarded by ENABLE_OTEL.

- Enablement
  - ENABLE_OTEL=1 to turn on metrics and optional tracing
  - OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317 (gRPC) or http://otel-collector:4318 (HTTP)
  - OTEL_EXPORTER_OTLP_PROTOCOL=grpc|http/protobuf (optional; default gRPC if unspecified)
  - EXPLAINER_VERSION=1.0.0 (optional version label on logs/metrics)

- Metrics
  - Exposed in Prometheus format via an embedded HTTP server (default: host 0.0.0.0, port 9090)
  - Scrape target: http://explainer:9090/metrics
  - Families implemented in [services/explainer/src/otel.py](services/explainer/src/otel.py:1):
    - explainer_jobs_total{state,granularity,featureset}
    - explainer_stage_duration_seconds{stage}
    - explainer_queue_lag_seconds
    - sae_decode_requests_total{device,layer}
    - sae_decode_latency_seconds{device,layer}
    - sae_decode_fallbacks_total{device}
    - sae_layer_cache_size (gauge)
    - sae_decode_batch_size
    - hif_nodes_count, hif_incidences_count (gauges)
    - hif_prune_ratio (gauge)
    - attribution_method_total{method,granularity}
    - attribution_latency_seconds{method,granularity}
    - attribution_early_stop_total{method}
  - The worker wires these in [services/explainer/src/worker.py](services/explainer/src/worker.py:29) and records stage timings, queue lag, and HIF sizes.

- Logs
  - Structured JSON via [get_logger()](services/explainer/src/otel.py:119) with fields: ts, level, service, version, msg, plus optional trace_id, tenant_id, request_id, stage.

- Traces (optional)
  - If OTEL_EXPORTER_OTLP_ENDPOINT is set, tracing is initialized via OTLP exporter (HTTP or gRPC) in [setup_otel()](services/explainer/src/otel.py:209)

- Dashboards and Collector
  - Deploy the example OpenTelemetry Collector: kubectl apply -f [manifests/otel/otel-collector.yaml](manifests/otel/otel-collector.yaml:1)
  - Import Grafana dashboard: [dashboards/grafana/sidecar-overview.json](dashboards/grafana/sidecar-overview.json:1)
  - The Collector exposes its own Prometheus endpoint at http://otel-collector:8889/metrics

Prometheus scrape summary
- Gateway: scrape http://gateway:8080/metrics (FastAPI)
- Explainer: scrape http://explainer:9090/metrics (embedded server)

Common label guidance
- Use bounded labels to avoid high-cardinality: tenant_id, granularity, featureset, service, version
- Prefer logs/traces for highly variable identifiers (e.g., trace_id, model_name)
## Observability
- Logs: per-trace lifecycle (started, extracted, verified, persisted).
- Metrics: consumer lag, extraction duration, verification success rate.

## Failure modes
- Poison messages: implement DLQ or max-retry policy; add guardrails around parsing.
- spaCy model issues: ensure model present; fallback to simpler tokenization when unavailable.
- Shadow verification failures: treat as non-fatal; record status and proceed with base graph.

## SAE decode microservice

High-throughput SAE feature projection with batching, per-layer queues, and warm LRU caches.

- Service module: [services/explainer/src/sae_service.py](services/explainer/src/sae_service.py:1)
- Worker integration: see sae_decode_stub inside [services/explainer/src/worker.py](services/explainer/src/worker.py)
- GPU import guard: torch is optional. If not installed or CUDA not available, the CPU path using [project_topk()](libs/sae/loader.py:195) is used automatically.

Configuration (env):
- ENABLE_SAE_SERVICE
- DICT_ROOT
- DICT_NAME
- SAE_DEVICE
- SAE_TOPK
- SAE_BATCH_SIZE
- SAE_MAX_WAIT_MS
- SAE_CACHE_LAYERS
- SAE_TILE_ROWS

Example enablement:
- ENABLE_SAE_SERVICE=1 DICT_ROOT=/mnt/sae_dictionaries DICT_NAME=sae-gpt4-2m SAE_DEVICE=auto python -m services.explainer.src.worker


## Attribution configuration

Production-ready, budgeted proxy attribution is available without GPU dependencies. The strategies are implemented in [services/explainer/src/attribution.py](services/explainer/src/attribution.py) and integrated in the legacy DEV pipeline within [`process_envelope()`](services/explainer/src/worker.py:2421) for non-breaking enablement.

- Sentence granularity (default method: "acdc"):
  - Approximates ACDC-style interventions by sampling small feature subsets (size 1–3) with a normalized co-activation heuristic. Bounded strictly by max_ms_budget and sample caps.
- Token granularity (default method: "shapley"):
  - Sampled Shapley approximation of marginal contributions for a specific output token using coalitions. Early-stops on low rolling delta or when hitting the time budget.

Environment variables:
- ATTR_METHOD_SENTENCE: acdc|auto (default: acdc)
- ATTR_METHOD_TOKEN: shapley|auto (default: shapley)
- ATTR_SENTENCE_BUDGET_MS: integer milliseconds (default: 900)
- ATTR_TOKEN_BUDGET_MS: integer milliseconds (default: 3500)
- ATTR_MAX_SAMPLES: integer (default: 512)
- ATTR_EARLY_STOP_DELTA: float (default: 0.01)
- ATTR_PER_TOKEN_CAP: integer cap on incident edges (default: 256)
- ATTR_MIN_EDGE_WEIGHT: float threshold in [0,1] (default: 0.01)
- ATTR_RANDOM_SEED: optional integer for deterministic runs (tests, reproducibility)

Method selection/fallback:
- "auto" is treated as the default per granularity ("acdc" for sentence; "shapley" for token).
- If budgets are exceeded early or the attribution module is unavailable, the worker falls back to an activation-weighted heuristic with strict caps and pruning.
- All attribution is proxy (no live model re-run), engineered for inference-side explainability under strict compute budgets.

Return format (legacy HIF v1 incidence subset):
- Each incidence (hyperedge) conforms to:
  {
    "id": "att_{...}",
    "node_ids": ["feat_XXXX", "token_out_YYY" or "token_Z"],
    "weight": float in [0,1],
    "metadata": { "type": "causal_circuit", "method": "acdc"|"shapley", "window": "sent-X"|"tok-i" }
  }

Operational notes:
- CPU-only safe; stdlib only.
- Deterministic under fixed seeds.
- Strict caps on candidates and incident edges to avoid hairball graphs.
- The worker’s public API is unchanged. The legacy DEV path integrates attribution behind environment switches and falls back safely.

Testing:
- Assert-based tests live at [tests/attribution/test_attribution.py](tests/attribution/test_attribution.py). Run:
  - python -m tests.attribution.test_attribution
- Smoke/benchmark guard (skipped by default):
  - ATTR_TEST_SMOKE=1 python -m tests.attribution.test_attribution

## Backpressure and Degradation

A centralized backpressure controller protects inference latency by monitoring queue depth, backlog seconds, and job age hints, and applies per-tenant quotas and global shedding when necessary. It also drives graceful degradations across explanation components.

- Controller module: [services/explainer/src/backpressure.py](services/explainer/src/backpressure.py)
  - Config (env):
    - BP_MAX_BACKLOG_SENTENCE (default 1.5)
    - BP_MAX_BACKLOG_TOKEN (default 3.0)
    - BP_MAX_QUEUE_TENANT (default 200)
    - BP_MAX_QUEUE_GLOBAL (default 5000)
    - BP_TENANT_MIN_GUARANTEE (default 2)
    - BP_BURST_MULTIPLIER (default 1.5)
    - BP_DEGRADE_LADDER (comma list: token->sentence,reduce-samples,reduce-topk,reduce-layers,saliency-fallback,drop)
  - Levels: "normal" | "soft" | "hard"
  - Actions ladder (applied deterministically):
    - token->sentence: downgrade token requests to sentence when overloaded
    - reduce-samples: halve attribution samples, relax early_stop_delta
    - reduce-topk: reduce SAE top-k and hypergraph per-token incident cap
    - reduce-layers: reduce SAE cache layers and raise pruning threshold
    - saliency-fallback: force heuristic attribution
    - drop: shed work when far beyond thresholds (respects tenant minimums unless global cap is hit)

- Integration points:
  - Worker: evaluates backpressure at the beginning of [`process_envelope()`](services/explainer/src/worker.py:2584), optionally downgrades granularity (token->sentence), sets SAE runtime overrides, and may shed early.
  - Attribution: configuration is degraded via controller.advise_attribution().
  - SAE decode: per-decode runtime overrides for topk and cache_layers are honored by [SAEDecodeService.decode()](services/explainer/src/sae_service.py:172).
  - Hypergraph: pruning is strengthened via controller.advise_hypergraph().

- Observability:
  - StatusStore fields (additive):
    - bp_level: "normal" | "soft" | "hard"
    - bp_actions: actions applied
    - granularity_downgraded: boolean
  - Metrics (Prometheus):
    - backpressure_level{tenant,granularity}
    - backpressure_actions_total{action}
    - backlog_seconds{granularity}
    - queue_len_global
    - queue_len_tenant{tenant}

Note: Inference lane remains unaffected; only the explanation lane is degraded.

## Hypergraph construction and pruning

The explainer includes a standardized HIF v1 hypergraph constructor and pruner implemented in [services/explainer/src/hypergraph.py](services/explainer/src/hypergraph.py:1). The worker integrates it in the legacy DEV pipeline inside [`process_envelope()`](services/explainer/src/worker.py:2421), replacing the previous ad-hoc pruning with calls to:
- [HypergraphConfig](services/explainer/src/hypergraph.py:13)
- [prune_and_group()](services/explainer/src/hypergraph.py:203)
- [build_hif()](services/explainer/src/hypergraph.py:421)
- [validate_hif()](services/explainer/src/hypergraph.py:478) (optional)

Overview
- Input: legacy v1-style nodes (sae_feature/input_token/output_token) and incidences (node_ids + weight).
- Output: HIF v1 JSON object matching the legacy schema in [libs/hif/schema.json](libs/hif/schema.json:185).
- Features:
  - Thresholding by edge weight.
  - Per-token incident cap (top-k by weight).
  - Optional supernode grouping of SAE features by label prefix.
  - Global caps on nodes and incidences with deterministic tie-breaking.
  - Safety guardrails and consistent meta.limits population.

Configuration (env)
- HG_MIN_EDGE_WEIGHT (float, default 0.01)
- HG_PER_TOKEN_CAP (int, default 256)
- HG_MAX_NODES (int, default 5000)
- HG_MAX_INCIDENCES (int, default 20000)
- HG_GROUPING ("supernode"|"none", default "supernode")
- HG_SUPERNODE_MIN_GROUP (int, default 3)
- HG_LABEL_DELIM (str, default ":")

Behavior
1) Thresholding
   - Drop incidences with weight &lt; HG_MIN_EDGE_WEIGHT.
   - Weights are clamped into [0,1] during normalization.

2) Per-token cap
   - For each output_token (and input_token), keep only the top HG_PER_TOKEN_CAP incidences by weight, tie-breaking by lexicographic id for determinism.
   - Tokens are preserved preferentially during node truncation.

3) Supernode grouping (default enabled)
   - Group sae_feature nodes by the label prefix before HG_LABEL_DELIM (e.g., "Biology: ..." → "Biology").
   - Only prefixes with at least HG_SUPERNODE_MIN_GROUP members form a supernode.
   - A supernode has:
     - id: "super_{prefix_slug}"
     - type: "circuit_supernode"
     - label: "{prefix} (cluster)"
     - attributes.member_count and attributes.members (members list truncated to 200).
   - Incidences referencing grouped members are remapped to the supernode id.
   - Parallel incidences that become identical are merged by taking MAX(weight) with stable id tie-breaks.

4) Global caps and guardrails
   - Incidences are sorted by weight desc, then id asc and truncated to HG_MAX_INCIDENCES.
   - Unreferenced nodes are dropped.
   - Nodes are truncated to HG_MAX_NODES by priority: tokens first, then supernodes, then highest-activation features (falling back to id for ties).
   - Ordering is deterministic.

Meta and validation
- [build_hif()](services/explainer/src/hypergraph.py:421) assembles:
  - "network-type": "directed"
  - normalized "nodes" and "incidences"
  - "meta": includes "version": "hif-1" and "limits" populated from [HypergraphConfig](services/explainer/src/hypergraph.py:13).
- Optional schema validation is performed via [libs/hif/validator.py](libs/hif/validator.py:1) if available. Validation failures are logged and do not crash the worker by default.
- Note: HIF v1 legacy schema has strict meta additionalProperties=false; grouping summaries are emitted in logs, not embedded into meta.

Integration points
- The legacy DEV pipeline uses the standardized constructor/pruner in [`process_envelope()`](services/explainer/src/worker.py:2421). The modern Redis stream pipeline continues producing HIF v2 structures separately.

Tests
- Minimal assert-based tests are included: [tests/hypergraph/test_hypergraph.py](tests/hypergraph/test_hypergraph.py:1)
- Run:
  - python -m tests.hypergraph.test_hypergraph

These cover:
- Edge thresholding.
- Per-token cap enforcement.
- Supernode grouping and remapping semantics.
- Global caps with deterministic ordering and no dangling nodes.
- Optional schema validation hook (skips gracefully if jsonschema is not installed).
