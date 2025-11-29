# Security Guide

Hardened posture across tenant isolation, token scopes, PII scrubbing, encryption, and audit logging. This document describes trust boundaries, data flows, controls, and operator guidance.

## Threat model and trust boundaries

Components and boundaries:
- External clients → Gateway (FastAPI) over TLS. Gateway enforces RBAC scopes and emits structured audit logs.
- Internal async lane: Redis Streams payloads → Explainer Worker → optional Postgres / S3 artifact store → optional StatusStore (JSON or DDB stub).
- Persistence:
  - S3 artifacts with optional SSE-KMS encryption.
  - Optional local JSON StatusStore for DEV; DynamoDB-like stub for prod wiring.
- Observability: structured JSON logs, optional OTEL metrics/traces.

Assumptions and non-goals:
- TLS termination is provided by the ingress/proxy (use modern TLS; HTTP/2 preferred; mTLS optional for east-west).
- No public API shape changes for compatibility; safeguards are additive and low-overhead.
- PII scrubbing applies to log copies only (never mutates data path or storage payloads).

## Tenant isolation

Controls:
- Gateway enforces presence of tenant_id for authenticated requests:
  - AUTH_MODE=static requires token identity to carry a non-empty tenant_id; otherwise 403.
  - AUTH_MODE=none allows anonymous/testing and sets tenant_id="anon".
- StatusStore disallows tenant_id changes once a trace exists (immutable tenant binding) to prevent cross-tenant hijack attempts.
  - Local backend checks on put/update reject or ignore tenant_id mutations (warning is logged). See [put_status()](services/explainer/src/status_store.py:100) and [update_fields()](services/explainer/src/status_store.py:133).
- Rate limits are per-tenant across read/write categories.

References:
- Gateway RBAC: services/gateway/src/rbac.py
- Tenant immutability: services/explainer/src/status_store.py

## Authentication and scopes

Modes:
- AUTH_MODE: none | static
  - none: anonymous, tenant_id="anon" for testing.
  - static: Authorization: Bearer <token> matched against AUTH_TOKENS_JSON map.

Static identities:
- AUTH_TOKENS_JSON example:
  {"tokenA":{"tenant_id":"t1","scopes":["traces:read","traces:write"],"subject":"userA"}}

Optional token checks (defense-in-depth):
- AUTH_EXPECT_ISS, AUTH_EXPECT_AUD: If present and token identity includes iss or aud, issuer/audience must match (403 on mismatch).

Scope coverage (no API shape changes):
- POST /v1/chat/completions → traces:write
- POST /v1/traces/{trace_id}/webhooks → traces:write
- DELETE /v1/traces/{trace_id} → traces:write
- GET /v1/traces/{trace_id}/status → traces:read
- GET /v1/traces/{trace_id}/graph → traces:read
- GET /v1/traces/{trace_id}/stream → traces:read

Error bodies:
- 401 uses code=unauthorized (no secret leakage).
- 403 uses code=missing_scope for scope failures; code=forbidden for other token issues.

## PII scrubbing (logs only)

Sanitizers (stdlib-only regex; deterministic):
- Emails → [EMAIL]
- Common phone patterns → [PHONE]
- Long digit runs (min digits configurable; default ≥9) → [NUMBER]

Configuration (read once at import):
- PII_MASK_EMAIL=1 (default 1)
- PII_MASK_PHONE=1 (default 1)
- PII_MASK_NUMBERS=0 (default 0)
- PII_MASK_MIN_DIGITS=9 (default 9)

Usage:
- Gateway scrubs short user message snippet in audit events for chat submissions when x-explain-mode is set.
- Explainer should scrub any user-provided text if included in logs (current code avoids logging raw user messages by default; keep this practice).

References:
- Sanitizers: libs/sanitize/pii.py
- Gateway audit emission: services/gateway/src/app.py

## Audit logging

Low-cardinality, structured JSON events for security-relevant actions. Never include Authorization or raw secret values.

Events (Gateway):
- chat.submit (on explain mode), webhook.register, trace.cancel
- rbac.deny (401/403), rate_limited (429)

Events (Explainer):
- trace.running, trace.complete, trace.failed, trace.canceled (e.g., backpressure shedding)

Schema (fields):
- ts, service, event, tenant_id, trace_id, subject, scopes, path, method, status, reason, plus small scalar extras (e.g., mode, model, has_secret).

Example (single line JSON):
```json
{"ts":"2025-01-01T12:00:00Z","service":"gateway","event":"chat.submit","tenant_id":"t1","trace_id":"trc_abc123def456","subject":"userA","scopes":["traces:read","traces:write"],"path":"/v1/chat/completions","method":"POST","status":null,"mode":"hypergraph","model":"gpt-4o-mini","has_secret":false}
```
Note: Authorization header is never logged. User content is sanitized via [sanitize_message()](libs/sanitize/pii.py:72) and truncated in audit extras.

Enablement:
- AUDIT_LOG_ENABLE=1
- AUDIT_LOG_PATH=/var/log/hypergraph/audit.log
  - If file not writable, logger falls back to stdout.

## Data minimization

- Do not log raw Authorization or user message content. Only short scrubbed snippets (PII-sanitized) are permitted for audit context.
- Metrics/traces use bounded labels (tenant_id, granularity, featureset, service, version). Avoid high-cardinality values in metrics.

## Data in transit (TLS)

- Require TLS for all external-to-Gateway traffic.
- For service-to-service traffic, use mTLS or private network segmentation where appropriate.
- Enforce modern ciphers/suites at the ingress (not shown in this repo).

## Data at rest (S3 SSE-KMS)

- Artifacts are uploaded to S3 with Server-Side Encryption using AWS KMS if S3_KMS_KEY_ID is set.
- Key names are constructed using path-safe slugs; traversal attempts rejected.
  - Key format: {prefix}/{YYYY}/{MM}/{DD}/{trace_id}/{granularity}-{sae_version}-{model_hash}.hif.json.gz
  - All dynamic parts are strictly slugified; unknown fallback when missing.

References:
- S3 persistence: services/explainer/src/s3_store.py

## Secrets handling

- Gateway never logs Authorization or secret headers; passthrough occurs only for upstream needs.
- Dotenv support is optional for DEV. Production secrets should come from environment/injected secret stores.
- Database URLs are masked in config logs (password redacted).

## Key rotation guidance

- Prefer short-lived tokens or rotate static tokens regularly.
- For SSE-KMS, rotate AWS KMS keys via AWS KMS key rotation; applications do not need changes if aliases are used.
- Ensure the S3 bucket policy enforces SSE-KMS and only allows the intended KMS key (see iam-policies.md).

## Suggested IAM policies

See docs/security/iam-policies.md for:
- S3 bucket policy that requires SSE-KMS and denies unencrypted writes.
- KMS key policy that permits the service principal to use the key for put/get.
- DynamoDB policy placeholder if using a managed StatusStore.

## Hardening checklist

- [ ] Set AUTH_MODE=static, AUTH_TOKENS_JSON, and RBAC scopes per endpoint.
- [ ] Optionally set AUTH_EXPECT_ISS and AUTH_EXPECT_AUD to pin identity.
- [ ] Enable AUDIT_LOG_ENABLE=1 and route audit logs to a secure file/stream.
- [ ] Configure PII masking flags (PII_MASK_EMAIL/PHONE/NUMBERS) based on org policy.
- [ ] Use TLS termination with modern cipher suites at the ingress; prefer mTLS internally.
- [ ] Configure S3 bucket with SSE-KMS; set S3_KMS_KEY_ID and bucket policy to require KMS.
- [ ] Limit S3 access to write-only where appropriate; do not allow public access.
- [ ] Use least-privilege IAM roles for Gateway/Explainer workloads.
- [ ] Monitor audit events (rbac.deny, rate_limited) for abuse detection.
- [ ] Enforce StatusStore TTLs and ensure tenant_id immutability is not bypassed.