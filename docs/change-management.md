# Change Management Plan (API v1)

Audience: maintainers and contributors proposing changes to the public-facing API, schemas, and SDKs.

Canonical specs and references:
- OpenAPI: [api/openapi/hypergraph-api.yaml](api/openapi/hypergraph-api.yaml:1)
- API reference: [docs/api-reference.md](docs/api-reference.md:1)
- HIF schema (legacy v1 + v2 objects): [libs/hif/schema.json](libs/hif/schema.json:1)
- Gateway reference implementation: [create_chat_completion()](services/gateway/src/app.py:540), [get_trace_status()](services/gateway/src/app.py:886), [get_trace_graph()](services/gateway/src/app.py:922), [stream_trace()](services/gateway/src/app.py:965), [register_webhook()](services/gateway/src/app.py:1021), [cancel_trace()](services/gateway/src/app.py:1062)
- Python SDK entry point: [Client](sdks/python/your_company_explainability/your_company_explainability/client.py:143)
- TypeScript client: [ExplainabilityClient](sdks/typescript/explainability-ui/src/client.ts:19)

## 1) Versioning policy

Scope
- This plan governs the public API v1 surface:
  - Path versioning: `/v1/...`
  - HIF legacy interchange format: `meta.version = "hif-1"`
  - OpenAI-compatible chat endpoint and Trace endpoints documented in [docs/api-reference.md](docs/api-reference.md:1)
- Internal service-to-service contracts may iterate faster but must not break the public surface.

Frozen guarantees for API v1
- HIF v1 (hif-1) shape is frozen. Only additive changes are allowed (new optional fields, new metadata keys).
- HTTP endpoints under `/v1` are stable; no breaking changes to request/response required fields, status codes, or semantics.
- New endpoints MAY be added under `/v1` when strictly additive and orthogonal.

Additive (non-breaking) changes allowed
- New optional response fields in existing JSON objects
- New headers (optional usage)
- New endpoints that do not alter existing ones
- Expanded enums with default/unknown-safe behavior (consumers must ignore unknown enum values gracefully)
- Documentation clarifications and example expansions

Breaking changes (prohibited in v1)
- Removing fields or changing requiredness (optional → required)
- Renaming fields, changing types, or structuring
- Changing success/erroneous status code classes (e.g., 200 → 202, 404 → 410) for existing endpoints
- Removing or changing the meaning of headers with established semantics

If you must break v1:
- Introduce `/v2` endpoints while keeping `/v1` working for the deprecation window.
- HIF v2 objects should be served only on v2 endpoints (or via explicit format negotiation on new endpoints).

Deprecation window
- Minimum: 90 calendar days
- Recommended: dual-run/dual-serve during the entire window (serve both old and new), publish migration guidance on day 0.

Preview features (non-contract)
- Expose experimental behavior via:
  - Header gates like `X-API-Preview: feature-name`
  - Beta paths like `/v1beta/...` (not for production)
- Preview features are exempt from stability guarantees and can change without notice. Clearly document preview status.

## 2) Deprecation process

Phases
1. Proposal
   - Open an ADR (Architecture Decision Record) titled “Deprecate X in v1” including rationale, impact, alternatives, and migration strategy.
   - Link to the precise spec locations (OpenAPI paths, schema sections).
2. Announcement (Day 0)
   - Update [docs/api-reference.md](docs/api-reference.md:1) and related guides with deprecation notices and the Sunset date (+90 days).
   - Add CHANGELOG entry (Unreleased → Released).
   - If available, send an org-level webhook “deprecation.notice” (see sample below).
   - Optionally emit HTTP response headers (if feasible in deployment):
     - `Deprecation: true`
     - `Sunset: Wed, 01 Mar 2025 00:00:00 GMT`
     - `Link: <migration-guide-url>; rel="deprecation"`
3. Dual-serve window (Day 0 → Day 90)
   - Maintain both old and new behaviors.
   - Provide compatibility shims in SDKs if practical.
4. Removal (≥ Day 90)
   - Remove deprecated behavior from code.
   - Keep the artifacts discoverable in documentation history; mark as removed with version/date.
   - Finalize CHANGELOG with a “Removed” note.

Webhook example (org-level release channel; not per-trace)
```json
{
  "event": "deprecation.notice",
  "component": "api",
  "resource": "/v1/traces/{trace_id}/graph",
  "sunset": "2025-03-01T00:00:00Z",
  "migration": "https://example.org/docs/change-management#graph-endpoint-migration",
  "notes": "Graph response stays in HIF v1; use new /v2/... for v2 objects."
}
```

## 3) Release workflow

All API-affecting changes must:
1. Update OpenAPI
   - Edit [api/openapi/hypergraph-api.yaml](api/openapi/hypergraph-api.yaml:1)
   - Reflect request/response body, parameters, headers, and status codes
   - Include at least one example per endpoint showcasing headers and representative payload
2. Update docs
   - API Reference: [docs/api-reference.md](docs/api-reference.md:1)
   - HIF Schema reference (if fields/semantics affected): [docs/hif-schema.md](docs/hif-schema.md:1)
   - Tutorials/Quickstarts if developer workflow changes: [docs/tutorials/python-quickstart.md](docs/tutorials/python-quickstart.md:1), [docs/tutorials/typescript-quickstart.md](docs/tutorials/typescript-quickstart.md:1)
3. Update SDKs (SemVer)
   - Python SDK: bump version per SemVer; reflect new fields/headers
     - Backwards compatibility: tolerate unknown fields; avoid raising on additive changes
     - Ensure examples and docstrings reference new behavior where applicable
   - TypeScript package: bump version per SemVer and update types/guards to accept additive fields
4. Add CHANGELOG entry
   - Unreleased section → enumerate additions/changes/fixes
   - On release, cut a tag and move entries under the new version
5. Contract tests and fixtures
   - Add/refresh contract tests (see section below) using the examples in [docs/api-reference.md](docs/api-reference.md:1)
6. Review & Approvals
   - Product + Platform approvals required (see approval matrix)
   - Security review if headers touch auth, PII, or observability data flows

SemVer guidance
- SDKs:
  - MAJOR: removal or breaking behavior (should be rare; aligned with API v2)
  - MINOR: new features, new optional fields/headers, new endpoints
  - PATCH: bug fixes, docs-only, perf/no-op changes
- API:
  - Keep `/v1` stable. If a breaking change is unavoidable, add `/v2` and keep `/v1` during deprecation window.

## 4) Change control gates

ADR (Architecture Decision Record)
- Required for any public API addition, modification, or deprecation.
- Contents: context, decision, alternatives, migration, risks, rollout plan.
- Place under a future `docs/adr/` directory or team ADR system (link in PR).

Approval matrix
- Platform (owning team) approval
- Product approval (customer impact, messaging)
- Security approval (if auth/headers/PII or schema carries sensitive implicatures)
- SRE approval (if affecting SLAs, rate limits, or backpressure behavior)

## 5) Communication

- Release notes: summarize changes, who is impacted, and action required.
- Migration guides: “Before/After” diffs in payloads; cURL and code snippets included.
- Webhooks:
  - Use an org-level release/deprecation webhook channel (not per-trace) to notify consumers (sample payload above).
- Docs home updates:
  - Update [docs/README.md](docs/README.md:1) to point at change areas for quick discoverability.

## 6) Testing and freeze window

Contract tests
- Tests should assert that examples in [docs/api-reference.md](docs/api-reference.md:1) are accepted and returned by the system:
  - Validate example requests via OpenAPI schema
  - Validate example responses (status codes, headers, body shapes)
  - Store fixtures derived from examples and run them in CI
- HIF validation:
  - Use [libs/hif/validator.py](libs/hif/validator.py:1) in CI to check any example HIF payloads
  - Ensure v1 graphs declare `"version": "hif-1"` and match the legacy sections

Freeze window (pre-release)
- Minimum 48h “code freeze” for API-impacting changes before cutting a release tag
- During freeze:
  - Only docs fixes, test fixture updates, or critical corrections allowed
  - Re-run contract tests and smoke tests against a staging Gateway
- Post-freeze:
  - Tag release, publish SDKs, and ship release notes

## 7) Mapping common proposals to this policy

- Add a new optional field to status payload
  - Allowed in v1; update OpenAPI, docs, and contract tests; bump SDK MINOR if surfaced
- Tighten a required field or change types
  - Breaking; must target `/v2` and run 90-day deprecation for `/v1`
- Introduce new headers for observability
  - Allowed when optional; document in API reference and examples; bump SDK MINOR
- Change default compression or content encoding
  - Non-breaking if content negotiation remains; document in API reference (Accept-Encoding and Content-Encoding) and ensure legacy behavior still honored

## 8) Appendix: sample checklists

Proposal checklist
- [ ] ADR opened with alternatives and migration plan
- [ ] OpenAPI updated: [api/openapi/hypergraph-api.yaml](api/openapi/hypergraph-api.yaml:1)
- [ ] API Reference updated: [docs/api-reference.md](docs/api-reference.md:1)
- [ ] HIF docs updated if relevant: [docs/hif-schema.md](docs/hif-schema.md:1)
- [ ] SDKs updated: [Client](sdks/python/your_company_explainability/your_company_explainability/client.py:143), [ExplainabilityClient](sdks/typescript/explainability-ui/src/client.ts:19)
- [ ] Contract tests added/refreshed
- [ ] CHANGELOG entry written
- [ ] Approvals: Product, Platform, Security (as needed), SRE (as needed)

Deprecation checklist (90-day window)
- [ ] ADR marked “Deprecation”
- [ ] Deprecation notice in docs with Sunset date
- [ ] Release notes + migration guide published
- [ ] Webhook notification (org-level) sent
- [ ] Dual-serve verified by contract tests
- [ ] Removal after window and final CHANGELOG update

## 9) Notes for HIF versions

- API v1 continues to serve HIF v1 (“hif-1”) from GET `/v1/traces/{trace_id}/graph`.
- v2 objects exist within [libs/hif/schema.json](libs/hif/schema.json:1) for forward compatibility but are not returned from v1 endpoints.
- Any move to serve v2 must go through `/v2` or explicit negotiation on new endpoints with preview headers, adhering to the deprecation process above.
## 10) Operations execution checklist

For day-to-day operations during the API v1 freeze, use the actionable checklist:
- [docs/change-management-checklist.md](docs/change-management-checklist.md)

This checklist enumerates gates (ADR requirement, approvals, contract tests), the 90-day deprecation schedule, release calendar, communication plan, and a PR template snippet aligned with this policy.
