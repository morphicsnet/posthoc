#!/usr/bin/env python3
# tests/e2e/test_trace_workflow.py
# End-to-end trace workflow validation:
# - POST /v1/chat/completions with x-explain-mode: hypergraph for sentence and token
# - Poll /v1/traces/{id}/status until terminal state; track durations
# - GET /v1/traces/{id}/graph and validate HIF via [validate_hif()](libs/hif/validator.py:117) if jsonschema available
# - If configured, verify S3/local fallback artifact existence using status.s3_key
# - Cross-check explanation_metadata.trace_id (if present in chat response) with trace_id used/polled
#
# References:
# - Chat handler: [create_chat_completion()](services/gateway/src/app.py:540)
# - Trace APIs: [get_trace_status()](services/gateway/src/app.py:886), [get_trace_graph()](services/gateway/src/app.py:922)
# - Status persistence (tenant immutability and file layout hints): [put_status()](services/explainer/src/status_store.py:100)

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
    poll_until,
    read_json_maybe_gz,
)

# Optional HIF validator (requires jsonschema dependency at runtime)
try:
    from libs.hif.validator import validate_hif  # type: ignore
except Exception:
    validate_hif = None  # type: ignore


def _rand_trace(prefix: str = "trc_e2e") -> str:
    sfx = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(10))
    return f"{prefix}_{sfx}"


def _auth_enabled_probe(client: HttpClient, base_url: str) -> bool:
    # Probe GET status for a bogus id; if 401/403, RBAC static mode is likely enabled
    st, _body, _err = client.get_json(f"{base_url.rstrip('/')}/v1/traces/trc_probe/status", headers=None)
    return st in (401, 403)


def _post_chat_with_trace(
    client: HttpClient,
    base_url: str,
    token: Optional[str],
    tenant: Optional[str],
    trace_id: str,
    granularity: str,
    features: str = "sae-gpt4-2m",
) -> Tuple[int, Dict[str, Any], Dict[str, str], str]:
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello and mention one color."},
        ],
        "stream": False,
        "temperature": 0.2,
    }
    headers = auth_headers(token, tenant)
    headers["x-explain-mode"] = "hypergraph"
    headers["x-explain-granularity"] = granularity
    headers["x-explain-features"] = features
    headers["x-trace-id"] = trace_id
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    return client.post_json(url, body, headers=headers)


def _poll_status_until_done(
    client: HttpClient,
    base_url: str,
    trace_id: str,
    token: Optional[str],
    tenant: Optional[str],
    timeout_s: float = 90.0,
) -> Tuple[bool, Dict[str, Any], int]:
    t0 = time.monotonic()

    def _step():
        st, data, _err = client.get_json(f"{base_url.rstrip('/')}/v1/traces/{trace_id}/status", headers=auth_headers(token, tenant))
        # Accept 404 if not yet known
        if st == 200 and isinstance(data, dict):
            state = str(data.get("state") or "")
            if state in ("complete", "failed", "expired", "canceled"):
                return True, {"status": st, "body": data}
        elif st in (410,):  # expired
            return True, {"status": st, "body": {"state": "expired"}}
        return False, {"status": st, "body": data}

    done, val = poll_until(_step, timeout_s=timeout_s, interval_s=0.5)
    return done, val, int((time.monotonic() - t0) * 1000)


def _get_graph(client: HttpClient, base_url: str, trace_id: str, token: Optional[str], tenant: Optional[str]) -> Tuple[int, Dict[str, Any], str]:
    return client.get_json(f"{base_url.rstrip('/')}/v1/traces/{trace_id}/graph", headers=auth_headers(token, tenant))


def _validate_hif_payload(hif_obj: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    # Prefer strict validator when available; fallback to minimal shape checks.
    try:
        if validate_hif is not None:
            validate_hif(hif_obj)  # may raise
            return True, None
    except Exception as e:
        return False, f"validate_hif error: {e}"

    # Minimal shape checks for HIF v1 response
    try:
        if not isinstance(hif_obj, dict):
            return False, "graph not an object"
        if "nodes" not in hif_obj or "incidences" not in hif_obj or "meta" not in hif_obj:
            return False, "missing nodes/incidences/meta"
        meta = hif_obj.get("meta") or {}
        if not isinstance(meta, dict):
            return False, "meta not an object"
        # API v1 compliance requires meta.version == "hif-1"
        ver = meta.get("version")
        if ver != "hif-1":
            return False, f"unexpected meta.version={ver!r}"
        return True, None
    except Exception as e:
        return False, f"hif minimal check failed: {e}"


def _s3_local_check(config: E2EConfig, status_body: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    root = config.s3_check or ""
    if not root:
        return True, None  # not configured; treat as skipped-success
    try:
        s3_key = status_body.get("s3_key")
        if not s3_key or not isinstance(s3_key, str):
            return False, "no s3_key in status to verify"
        # file:// prefix handling
        if root.startswith("file://"):
            from urllib.parse import urlparse
            base = urlparse(root).path
        else:
            base = root
        candidate = os.path.join(base, s3_key)
        obj = read_json_maybe_gz(candidate)
        if obj is None:
            return False, f"artifact not found/readable: {candidate}"
        # Lightweight sanity that it's JSON and looks like a graph-ish object
        if not isinstance(obj, dict):
            return False, "artifact JSON is not an object"
        return True, None
    except Exception as e:
        return False, f"s3/local check error: {e}"


def _run_one_granularity(config: E2EConfig, granularity: str) -> Dict[str, Any]:
    client = HttpClient(timeout=15.0)
    base = config.base_url
    tenant = config.tenant_a or "tenantA"

    auth_enabled = _auth_enabled_probe(client, base)
    token = config.auth_token_write if auth_enabled else None
    if auth_enabled and not token:
        return new_result(f"trace_workflow[{granularity}]", "SKIP", reason="auth enabled but --auth-token-write not provided")

    trace_id = _rand_trace(f"trc_e2e_{granularity}")
    st_chat, body_chat, _hdrs, err_text = _post_chat_with_trace(client, base, token, tenant, trace_id, granularity)
    if st_chat in (401, 403):
        return new_result(f"trace_workflow[{granularity}]", "FAIL", reason=f"RBAC rejected chat ({st_chat})")
    if st_chat >= 500:
        # Upstream LLM not configured (LLM_PROXY_URL missing) -> skip trace validation but report clearly
        return new_result(f"trace_workflow[{granularity}]", "SKIP", reason=f"chat upstream error {st_chat}: {err_text or body_chat}")

    # Check explanation_metadata.trace_id consistency if present
    em_tid = None
    try:
        em = body_chat.get("explanation_metadata") if isinstance(body_chat, dict) else None
        if isinstance(em, dict):
            em_tid = em.get("trace_id")
    except Exception:
        em_tid = None

    em_consistent = (em_tid is None) or (str(em_tid) == str(trace_id))

    # Poll status
    done, val, dur_ms = _poll_status_until_done(client, base, trace_id, token if auth_enabled else None, tenant, timeout_s=120.0)
    status_code = int(val.get("status", 0))
    status_body = val.get("body") if isinstance(val, dict) else {}
    state = (status_body or {}).get("state")
    if not done:
        return new_result(f"trace_workflow[{granularity}]", "FAIL", reason="status polling timeout", details={"trace_id": trace_id, "last_status": status_code, "state": state, "duration_ms": dur_ms})

    # If failed/expired, still attempt to read graph, but mark as FAIL
    graph_ok = False
    hif_ok = False
    hif_err = None
    s3_ok = True
    s3_err = None

    st_graph, body_graph, _err = _get_graph(client, base, trace_id, token if auth_enabled else None, tenant)
    if st_graph == 200 and isinstance(body_graph, dict):
        graph_ok = True
        ok, herr = _validate_hif_payload(body_graph)
        hif_ok, hif_err = ok, herr
    elif st_graph in (404, 410):
        # Accept 410 when expired; 404 may occur if backend hasn't materialized yet.
        graph_ok = False
        hif_ok = False
        hif_err = f"graph status={st_graph}"

    # Optional artifact check when configured and we have a status document with s3_key
    try:
        s3_ok, s3_err = _s3_local_check(config, status_body if isinstance(status_body, dict) else {})
    except Exception as e:
        s3_ok, s3_err = False, str(e)

    # Build details summary
    details = {
        "trace_id": trace_id,
        "granularity": granularity,
        "chat_status": st_chat,
        "status_state": state,
        "status_code": status_code,
        "poll_duration_ms": dur_ms,
        "graph_status": st_graph,
        "hif_ok": hif_ok,
        "hif_err": hif_err,
        "artifact_ok": s3_ok,
        "artifact_err": s3_err,
        "explanation_metadata_trace_id_present": bool(em_tid),
        "explanation_metadata_consistent": em_consistent,
    }

    # Determine final outcome per granularity:
    # - Prefer PASS when state complete AND graph validated (hif_ok True)
    # - Allow PASS when state complete and validate_hif unavailable but minimal checks passed (hif_ok True covers both paths)
    # - Otherwise FAIL, except when upstream chat is 2xx but system legitimately expires (mark FAIL to surface SLO breach)
    if state == "complete" and hif_ok and em_consistent and (s3_ok or config.s3_check is None or config.s3_check == ""):
        return new_result(f"trace_workflow[{granularity}]", "PASS", reason=None, details=details)
    else:
        # Provide actionable reason
        reasons = []
        if state != "complete":
            reasons.append(f"state={state}")
        if not graph_ok:
            reasons.append("graph_unavailable")
        if not hif_ok:
            reasons.append(f"hif_invalid: {hif_err}")
        if not em_consistent:
            reasons.append("explanation_metadata.trace_id mismatch")
        if config.s3_check and not s3_ok:
            reasons.append(f"artifact_check_failed: {s3_err}")
        return new_result(f"trace_workflow[{granularity}]", "FAIL", reason=", ".join(reasons) or "trace workflow failed", details=details)


def run(config: E2EConfig) -> Dict[str, Any]:
    if not config.base_url:
        return new_result("test_trace_workflow.py", "SKIP", reason="no base_url provided")

    cases = ["sentence", "token"]
    results = []
    for g in cases:
        results.append(_run_one_granularity(config, g))

    # Summarize across both runs
    passed = [r for r in results if r.get("status") == "PASS"]
    failed = [r for r in results if r.get("status") == "FAIL"]

    overall = "PASS" if len(passed) >= 1 and len(failed) == 0 else ("FAIL" if failed else "SKIP")
    reason = None
    if overall == "FAIL":
        reason = "; ".join([str(r.get("reason")) for r in failed if r.get("reason")])

    # Merge details
    details = {"subtests": results}
    return {
        "test": "test_trace_workflow.py",
        "status": overall,
        "reason": reason,
        "details": details,
    }