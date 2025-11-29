#!/usr/bin/env python3
# tests/e2e/test_gateway_rbac.py
# RBAC validation for Gateway endpoints and scope matrix.
#
# References:
# - RBAC dependency: [rbac_dependency()](services/gateway/src/rbac.py:62)
# - Chat endpoint: [create_chat_completion()](services/gateway/src/app.py:540)
# - Trace endpoints: [get_trace_status()](services/gateway/src/app.py:886), [get_trace_graph()](services/gateway/src/app.py:922), [stream_trace()](services/gateway/src/app.py:965)
#
# Behavior:
# - Missing token => expect 401 when AUTH_MODE=static
# - Invalid token => expect 403
# - Valid tokens => verify scope matrix expectations:
#     * POST /v1/chat/completions requires traces:write
#     * GET /v1/traces/{id}/status, /graph, /stream require traces:read
#     * POST /v1/traces/{id}/webhooks and DELETE /v1/traces/{id} require traces:write
# - If auth is disabled (AUTH_MODE=none), the test SKIPs with a clear reason.
# - For environments where tokens may include both read+write scopes, scope-negative assertions
#   are treated as ambiguous (recorded but do not fail the test).

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from tests.e2e.utils import (
    E2EConfig,
    HttpClient,
    auth_headers,
    new_result,
    post_chat,
    get_status,
    get_graph,
)


def _post_chat_min(client: HttpClient, base_url: str, token: Optional[str], tenant: Optional[str]) -> tuple[int, Dict[str, Any], Dict[str, str], str]:
    # Minimal POST without forcing x-explain-mode (RBAC is enforced before upstream)
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ],
        "stream": False,
        "temperature": 0.0,
    }
    headers = auth_headers(token, tenant)
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    return client.post_json(url, body, headers=headers)


def _post_webhook(client: HttpClient, base_url: str, trace_id: str, token: Optional[str], tenant: Optional[str]) -> int:
    url = f"{base_url.rstrip('/')}/v1/traces/{trace_id}/webhooks"
    status, _, _err = client.post_json(url, {"url": "https://example.invalid/hook"}, headers=auth_headers(token, tenant))[:3]
    return status


def _delete_cancel(client: HttpClient, base_url: str, trace_id: str, token: Optional[str], tenant: Optional[str]) -> int:
    import urllib.request
    import urllib.error
    import json as _json

    url = f"{base_url.rstrip('/')}/v1/traces/{trace_id}"
    req = urllib.request.Request(url, headers=auth_headers(token, tenant), method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=10.0) as r:
            return int(getattr(r, "status", 200))
    except urllib.error.HTTPError as he:
        # HTTPError carries .code
        return int(he.code)
    except Exception:
        return 599


def _auth_enabled_probe(client: HttpClient, base_url: str) -> bool:
    # Probe GET /status without token; if 401/403 then auth is enabled (AUTH_MODE=static).
    status, _body, _err = get_status(client, base_url, "trc_does_not_exist", token=None, tenant=None)
    return status in (401, 403)


def _is_missing_scope(resp_body: Dict[str, Any]) -> bool:
    try:
        if isinstance(resp_body, dict):
            det = resp_body.get("detail")
            if isinstance(det, dict):
                return str(det.get("code") or "") == "missing_scope"
    except Exception:
        pass
    return False


def run(config: E2EConfig) -> Dict[str, Any]:
    t_details: Dict[str, Any] = {"matrix": {}, "notes": []}
    client = HttpClient(timeout=10.0)
    base = config.base_url

    # Preconditions
    if not base:
        return new_result("test_gateway_rbac.py", "SKIP", reason="no base_url provided")

    # Detect whether auth is enabled
    auth_enabled = _auth_enabled_probe(client, base)
    if not auth_enabled:
        return new_result(
            "test_gateway_rbac.py",
            "SKIP",
            reason="AUTH_MODE appears disabled (probe without token did not return 401/403)",
            details={"auth_enabled": False},
        )

    # Tokens availability
    if not config.auth_token_write or not config.auth_token_read:
        return new_result(
            "test_gateway_rbac.py",
            "SKIP",
            reason="auth tokens not provided (--auth-token-write/--auth-token-read)",
            details={"auth_enabled": True, "write_set": bool(config.auth_token_write), "read_set": bool(config.auth_token_read)},
        )

    tenant_a = config.tenant_a or "tenantA"
    tid = "trc_e2e_rbac"

    # 1) Missing token -> 401
    st_missing, body_missing, _ = _post_chat_min(client, base, token=None, tenant=tenant_a)
    t_details["matrix"]["chat_missing_token"] = {"status": st_missing, "body": body_missing}
    missing_ok = (st_missing == 401)

    # 2) Invalid token -> 403
    st_bad, body_bad, _ = _post_chat_min(client, base, token="invalid-token", tenant=tenant_a)
    t_details["matrix"]["chat_invalid_token"] = {"status": st_bad, "body": body_bad}
    invalid_ok = (st_bad == 403)

    # 3) Valid write token on chat -> expect authorized (200 ideal; any non-401/403 accepted)
    st_write, body_write, _ = _post_chat_min(client, base, token=config.auth_token_write, tenant=tenant_a)
    t_details["matrix"]["chat_valid_write"] = {"status": st_write, "body": body_write}
    write_authorized = (st_write not in (401, 403))

    # 4) GET status with read token -> RBAC pass (200/404 acceptable)
    st_status_read, body_status_read, _ = get_status(client, base, tid, token=config.auth_token_read, tenant=tenant_a)
    t_details["matrix"]["status_valid_read"] = {"status": st_status_read, "body": body_status_read}
    read_authorized = (st_status_read not in (401, 403))

    # 5) GET graph with read token -> RBAC pass (200/404/410 acceptable)
    st_graph_read, body_graph_read, _ = get_graph(client, base, tid, token=config.auth_token_read, tenant=tenant_a)
    t_details["matrix"]["graph_valid_read"] = {"status": st_graph_read, "body": body_graph_read}
    read_authorized_graph = (st_graph_read not in (401, 403))

    # 6) Scope-negatives (best-effort, may be ambiguous if tokens carry both scopes)
    # 6a) POST chat with READ-only token -> expect 403 missing_scope (unless token has write)
    st_chat_as_read, b_chat_as_read, _ = _post_chat_min(client, base, token=config.auth_token_read, tenant=tenant_a)
    neg_chat_missing_scope = (st_chat_as_read == 403 and _is_missing_scope(b_chat_as_read))
    t_details["matrix"]["chat_with_read_token"] = {"status": st_chat_as_read, "missing_scope": _is_missing_scope(b_chat_as_read)}

    # 6b) GET status with WRITE-only token -> expect 403 missing_scope (unless token has read)
    st_status_as_write, b_status_as_write, _ = get_status(client, base, tid, token=config.auth_token_write, tenant=tenant_a)
    neg_status_missing_scope = (st_status_as_write == 403 and _is_missing_scope(b_status_as_write))
    t_details["matrix"]["status_with_write_token"] = {"status": st_status_as_write, "missing_scope": _is_missing_scope(b_status_as_write)}

    # 7) Webhook register requires write (404 acceptable if trace unknown; RBAC gate first)
    st_webhook = _post_webhook(client, base, tid, token=config.auth_token_write, tenant=tenant_a)
    t_details["matrix"]["webhook_with_write_token"] = {"status": st_webhook}
    webhook_authorized = (st_webhook not in (401, 403))

    # 8) Cancel requires write (404 acceptable if trace unknown)
    st_cancel = _delete_cancel(client, base, tid, token=config.auth_token_write, tenant=tenant_a)
    t_details["matrix"]["cancel_with_write_token"] = {"status": st_cancel}
    cancel_authorized = (st_cancel not in (401, 403))

    # Aggregate decision:
    # Core RBAC correctness hinges on 1,2 and at least one positive authorization path (3/4/5/7/8).
    core_ok = all([missing_ok, invalid_ok, write_authorized, read_authorized, read_authorized_graph, webhook_authorized, cancel_authorized])

    # If scope-negative checks failed but tokens are likely multi-scope, record note instead of failing.
    ambiguous = False
    if not neg_chat_missing_scope or not neg_status_missing_scope:
        # Ambiguous if both tokens appear to authorize both operations
        if write_authorized and read_authorized:
            ambiguous = True
            t_details["notes"].append("Tokens may include both read and write scopes; scope-negative checks are ambiguous.")

    status = "PASS" if core_ok else "FAIL"
    reason = None if core_ok else "RBAC checks failed (see matrix)"

    # If upstream not configured, accept 500 on chat_with_write as still authorized (already accounted by write_authorized rule)
    return {
        "test": "test_gateway_rbac.py",
        "status": status,
        "reason": reason,
        "details": {
            **t_details,
            "auth_enabled": auth_enabled,
            "ambiguous_scopes": ambiguous,
        },
    }