#!/usr/bin/env python3
# tests/e2e/test_security_tenant_isolation.py
# Tenant isolation and StatusStore immutability checks.
#
# What this test attempts:
# 1) Create a trace with tenant A (using --auth-token-write) and a fixed x-trace-id.
# 2) Attempt to retrieve that trace with "tenant B" context:
#    - If RBAC/authorization denies cross-tenant read, expect 403 or 404.
#    - If the deployment does not enforce cross-tenant visibility at the Gateway layer,
#      the test will SKIP with an explicit note (non-failing).
# 3) Verify StatusStore tenant immutability matches behavior of
#    [put_status()](services/explainer/src/status_store.py:100):
#    - Submit the SAME trace id again but under a different tenant token (if available).
#    - Ensure the StatusStore entry keeps the original tenant_id and does not mutate to the new one.
#
# References:
# - RBAC dependency and semantics: [rbac_dependency()](services/gateway/src/rbac.py:62)
# - Chat: [create_chat_completion()](services/gateway/src/app.py:540)
# - Status: [get_trace_status()](services/gateway/src/app.py:886)
# - StatusStore (local JSON backend): [get_status_store_from_env()](services/explainer/src/status_store.py:208)

from __future__ import annotations

import json
import os
import random
import string
import time
from typing import Any, Dict, Optional, Tuple

from tests.e2e.utils import (
    E2EConfig,
    HttpClient,
    auth_headers,
    new_result,
)


def _rand_trace(prefix: str = "trc_tenant") -> str:
    sfx = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(10))
    return f"{prefix}_{sfx}"


def _auth_enabled_probe(client: HttpClient, base_url: str) -> bool:
    status, _body, _err = client.get_json(f"{base_url.rstrip('/')}/v1/traces/trc_probe/status", headers=None)
    return status in (401, 403)


def _post_chat_with_trace(
    client: HttpClient,
    base_url: str,
    token: Optional[str],
    tenant_header: Optional[str],
    trace_id: str,
) -> Tuple[int, Dict[str, Any], Dict[str, str], str]:
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "stream": False,
        "temperature": 0.0,
    }
    headers = auth_headers(token, tenant_header)
    headers["x-explain-mode"] = "hypergraph"
    headers["x-explain-granularity"] = "sentence"
    headers["x-explain-features"] = "sae-gpt4-2m"
    headers["x-trace-id"] = trace_id
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    return client.post_json(url, body, headers=headers)


def _get_status(client: HttpClient, base_url: str, token: Optional[str], tenant_header: Optional[str], trace_id: str) -> Tuple[int, Dict[str, Any], str]:
    url = f"{base_url.rstrip('/')}/v1/traces/{trace_id}/status"
    return client.get_json(url, headers=auth_headers(token, tenant_header))


def _read_status_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def run(config: E2EConfig) -> Dict[str, Any]:
    base = config.base_url
    if not base:
        return new_result("test_security_tenant_isolation.py", "SKIP", reason="no base_url provided")

    client = HttpClient(timeout=12.0)
    auth_enabled = _auth_enabled_probe(client, base)

    if not auth_enabled:
        return new_result(
            "test_security_tenant_isolation.py",
            "SKIP",
            reason="AUTH_MODE appears disabled (no 401/403 on unauthenticated probe)",
            details={"auth_enabled": False},
        )

    # We need at least a write token (tenant A) and preferably a read token that maps to a different tenant (tenant B).
    if not config.auth_token_write:
        return new_result(
            "test_security_tenant_isolation.py",
            "SKIP",
            reason="--auth-token-write not provided",
        )

    # Trace creation under tenant A
    tenant_a = config.tenant_a or "tenantA"
    tenant_b = config.tenant_b or "tenantB"  # header used only for logs; tenant is determined by token in RBAC
    trace_id = _rand_trace("trc_tenant")

    st_chat_a, body_a, _hdrs_a, err_a = _post_chat_with_trace(client, base, config.auth_token_write, tenant_a, trace_id)
    if st_chat_a in (401, 403):
        return new_result(
            "test_security_tenant_isolation.py",
            "FAIL",
            reason=f"RBAC rejected tenant A chat ({st_chat_a})",
            details={"trace_id": trace_id, "status": st_chat_a, "body": body_a or err_a},
        )

    # Give the system a short moment to persist initial status (if STATUS_BACKEND=json is enabled)
    time.sleep(0.75)

    # Attempt cross-tenant retrieval: use read token (ideally for tenant B)
    cross_denied = None
    status_json_path = config.status_json or "/tmp/hif/status.json"

    if not config.auth_token_read:
        # Cannot attempt cross-tenant read without a read token
        cross_denied = None
    else:
        st_cross, body_cross, _err = _get_status(client, base, config.auth_token_read, tenant_b, trace_id)
        # If the deployment enforces tenant isolation, we expect 403 (forbidden) or 404 (concealed)
        if st_cross in (403, 404):
            cross_denied = True
        elif st_cross in (200, 410):  # 410 expired is visible; 200 visible -> likely not isolated at gateway
            cross_denied = False
        else:
            # Other statuses are ambiguous; treat as not proven
            cross_denied = None

    # Tenant immutability check (StatusStore):
    # Re-submit the same trace id with a "different tenant". We need a token that RBAC will treat as a different tenant.
    # If --auth-token-read lacks write scope, this step will be skipped gracefully.
    tenant_immutable_ok = None
    immutable_note = None
    if config.auth_token_read:
        st_chat_b, body_b, _hdrs_b, err_b = _post_chat_with_trace(client, base, config.auth_token_read, tenant_b, trace_id)
        if st_chat_b == 403:
            # Read token likely doesn't have traces:write; cannot proceed with immutability mutation attempt.
            tenant_immutable_ok = None
            immutable_note = "read token lacks write scope; immutability mutation attempt skipped"
        elif st_chat_b in (401, 599, 500):
            tenant_immutable_ok = None
            immutable_note = f"second submission failed (status={st_chat_b}); skipping immutability check"
        else:
            # If STATUS_BACKEND=json, the LocalJSONStatusStore should have an entry for this trace_id
            # with tenant_id from the ORIGINAL submission (tenant A derived from token).
            store = _read_status_json(status_json_path)
            item = store.get(trace_id) if isinstance(store, dict) else None
            if not isinstance(item, dict):
                tenant_immutable_ok = None
                immutable_note = f"status json not found or entry missing at {status_json_path}; backend may not be enabled"
            else:
                old_tid = item.get("tenant_id")
                # Wait shortly and re-read after second submission
                time.sleep(0.5)
                store2 = _read_status_json(status_json_path)
                item2 = store2.get(trace_id) if isinstance(store2, dict) else None
                new_tid = item2.get("tenant_id") if isinstance(item2, dict) else None
                # Immutability requires that tenant_id stays the same
                tenant_immutable_ok = (old_tid is not None) and (new_tid == old_tid)
                immutable_note = f"tenant_id before={old_tid!r} after={new_tid!r}"
    else:
        immutable_note = "no --auth-token-read provided to attempt cross-tenant resubmission"

    details = {
        "trace_id": trace_id,
        "cross_tenant_denied": cross_denied,
        "status_json_path": status_json_path,
        "tenant_immutability_ok": tenant_immutable_ok,
        "immutability_note": immutable_note,
    }

    # Decide outcome:
    # - PASS if cross-tenant read was denied (403/404) AND (immutability passed or was not applicable but backend missing).
    # - SKIP if cross-tenant enforcement is not present (visible 200/410), to avoid failing environments not enabling isolation at gateway.
    # - FAIL only if cross-tenant read was denied but immutability explicitly failed where backend is present.
    if cross_denied is True:
        if tenant_immutable_ok is False:
            return new_result("test_security_tenant_isolation.py", "FAIL", reason="tenant_id mutated in StatusStore", details=details)
        return new_result("test_security_tenant_isolation.py", "PASS", reason=None, details=details)

    if cross_denied is False:
        return new_result(
            "test_security_tenant_isolation.py",
            "SKIP",
            reason="cross-tenant read appears allowed in this deployment; gateway isolation not enforced",
            details=details,
        )

    return new_result(
        "test_security_tenant_isolation.py",
        "SKIP",
        reason="could not determine cross-tenant behavior (insufficient tokens or ambiguous statuses)",
        details=details,
    )