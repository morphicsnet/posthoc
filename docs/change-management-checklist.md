# API v1 Change Management – Operations Checklist

This checklist operationalizes the policy in [docs/change-management.md](docs/change-management.md) for the API v1 freeze and ongoing changes.

Gates (must be true before merge/deploy)
- Freeze date declared and on calendar
- ADR opened and approved for any change (additions, docs clarifications, deprecations)
- Contract tests green (schema and examples in [docs/api-reference.md](docs/api-reference.md))
- Approvals captured:
  - Platform (owning team)
  - Product
  - Security (if touching auth/headers/PII)
  - SRE (if SLAs, rate limits, or backpressure)
- Deprecations posted with 90-day schedule and migration guide

Release calendar template
- Week -1: Freeze announcement (Slack/email) + calendar block for tag day
- Freeze window (48h minimum): code freeze for API-impacting changes; docs/tests only
- Tag day: cut release tag, publish SDKs, post release notes
- Week +1: retrospective and metrics review

Communication plan
- Slack: #eng-announcements + #cust-success with summary, impact, action required
- Email: release notes to tenant contacts
- Webhooks (optional): org-level “release.notice” or “deprecation.notice”
- Docs home update: highlight changes on [docs/README.md](docs/README.md)

PR template snippet
```
## API v1 impact
- [ ] None (docs/tests only)
- [ ] Additive (new optional fields/headers)
- [ ] Deprecation (90-day window)
- [ ] Breaking (targeting v2 only)

## Artifacts updated
- [ ] OpenAPI (api/openapi/hypergraph-api.yaml)
- [ ] Docs (docs/api-reference.md, tutorials)
- [ ] SDKs (python/typescript)
- [ ] Contract tests

## Approvals
- [ ] Platform
- [ ] Product
- [ ] Security (if required)
- [ ] SRE (if required)
```

CHANGELOG format (per release)
- Added: new optional fields/endpoints, examples
- Changed: docs clarifications, non-breaking behaviors
- Deprecated: start window (Sunset header/date)
- Removed: only after window closed (reference to migration guide)
- Fixed: bug fixes
- Security: notes if applicable
