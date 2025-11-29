# Secrets and Keys Runbook

AUTH_TOKENS_JSON
- Recommended to manage via external secret store (e.g., AWS Secrets Manager or External Secrets Operator).
- The chart exposes a ConfigMap for simple bootstrap; DO NOT store real secrets in ConfigMaps for production.
- Options:
  1) Use ExternalSecret to inject AUTH_TOKENS_JSON as an env var into pods.
  2) Mount as a secret volume and point services to file.
- Format (exact shape consumed by [rbac_dependency](services/gateway/src/rbac.py:62)):
{
  "good":   {"tenant_id":"t1","scopes":["traces:read","traces:write"],"subject":"userA"},
  "reader": {"tenant_id":"t2","scopes":["traces:read"],"subject":"svcR"}
}
- Helm example (.Values.global.rbacTokensJson as a JSON string):
  --set global.rbacTokensJson='{"good":{"tenant_id":"t1","scopes":["traces:read","traces:write"],"subject":"userA"},"reader":{"tenant_id":"t2","scopes":["traces:read"],"subject":"svcR"}}'

KMS and S3
- Rotate CMKs on a regular cadence; ensure key policy allows service roles to use Encrypt/Decrypt.
- S3 bucket policy:
  - Restrict to VPC endpoints where possible.
  - Enforce SSL, SSE-KMS, and least-privilege access.
- Update .Values.global.s3.kmsKeyId and bucket/prefix; deploy via GitOps.

RBAC Scope Updates
- Adjust scopes in AUTH_TOKENS_JSON as teams/products evolve.
- Roll tokens regularly and decommission unused tokens promptly.
- Keep an access register tied to on-call rotations and service ownership.

Notes
- For Helm, prefer referencing secrets via envFrom.secretRef or valueFrom.secretKeyRef per Deployment.
- For ArgoCD, avoid committing plaintext secrets; use SOPS or External Secrets patterns.
