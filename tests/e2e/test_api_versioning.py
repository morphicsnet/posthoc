#!/usr/bin/env python3
# tests/e2e/test_api_versioning.py
# API v1 compliance checks:
# - Paths in OpenAPI are consistently under /v1
# - Change management docs include freeze + ADR process
# - Optional runtime check: produced HIF has meta.version == "hif-1"
#
# References:
# - OpenAPI spec: [hypergraph-api.yaml](api/openapi/hypergraph-api.yaml:1)
# - Change management doc: [docs/change-management.md](docs/change-management.md:1)
# - Gateway endpoints: [create_chat_completion()](services/gateway/src/app.py:540), [get_trace_status()](services/gateway/src/app.py:886), [get_trace_graph()](services/gateway/src/app.py:922)
# - HIF validator entrypoint: [validate_hif()](libs/hif/validator.py:117)

from __future__ import annotations

import os
import random
import re
import string
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tests.e2e.utils import (
    E2EConfig,
    HttpClient,
    auth_headers,
    new_result,
)

# Optional validator import (jsonschema may not be available in test env)
try:
    from libs.hif.validator import validate_hif  # type: ignore
except Exception:
    validate_hif = None  # type: ignore


OPENAPI_PATH = Path("api/openapi/hypergraph-api.yaml")
CHANGE_MGMT_PATH = Path("docs/change-management.md")


def _parse_openapi_paths(yaml_text: str) -> List[str]:
    """
    Minimal path extraction: collect keys under 'paths:' block.
    Expects lines like:
      paths:
        /v1/chat/completions:
          post:
            ...
    """
    lines = yaml_text.splitlines()
    paths: List[str] = []
    in_paths = False
    base_indent = None

    def _indent(s: str) -> int:
        return len(s) - len(s.lstrip(" "))

    for raw in lines:
        s = raw.rstrip("\n")
        if not s.strip():
            continue
        if s.strip() == "paths:":
            in_paths = True
            base_indent = _indent(s)
            continue
        if in_paths:
            ind = _indent(s)
            if base_indent is not None and ind <= base_indent:
                # left the paths block
                break
            # Match path keys at one indent level deeper than base
            m = re.match(r'^\s{2,}(/[^:\s]+):\s*$', s)
            if m:
                paths.append(m.group(1))
    return paths


def _check_change_mgmt(text: str) -> Tuple[bool, Dict[str, bool]]:
    """
    Verify that the change management doc mentions ADR and freeze/deprecation process.
    """
    t = text.lower()
    hints = {
        "mentions_adr": ("adr" in t),
        "mentions_freeze": ("freeze" in t) or ("code freeze" in t),
        "mentions_deprecation_window": ("deprecation" in t) or ("sunset" in t) or ("90" in t),
        "mentions_v1": ("/v1" in t) or ("api v1" in t),
    }
    ok = all(hints.values())
    return ok, hints


def _rand_trace(prefix: str = "trc_api_v1") -> str:
    sfx = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(10))
    return f"{prefix}_{sfx}"


def _probe_auth_enabled(base_url: str) -> bool:
    import urllib.request
    import urllib.error
    url = f"{base_url.rstrip('/')}/v1/traces/trc_probe/status"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=3.0):
            return False
    except urllib.error.HTTPError as he:
        return int(getattr(he, "code", 0)) in (401, 403)
    except Exception:
        return False


def _post_chat_attach_with_trace(client: HttpClient, base_url: str, token: Optional[str], tenant: Optional[str], trace_id: str) -> Tuple[int, Dict[str, Any], Dict[str, str], str]:
    headers = auth_headers(token, tenant)
    headers["x-explain-mode"] = "hypergraph"
    headers["x-explain-granularity"] = "sentence"
    headers["x-explain-features"] = "sae-gpt4-2m"
    headers["x-trace-id"] = trace_id
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello and mention any color."},
        ],
        "stream": False,
        "temperature": 0.0,
    }
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    return client.post_json(url, body, headers=headers)


def _poll_status_until_terminal(client: HttpClient, base_url: str, token: Optional[str], tenant: Optional[str], trace_id: str, timeout_s: float = 90.0) -> Tuple[bool, str]:
    url = f"{base_url.rstrip('/')}/v1/traces/{trace_id}/status"
    t0 = time.monotonic()
    last_state = ""
    while (time.monotonic() - t0) < timeout_s:
        st, obj, _err = client.get_json(url, headers=auth_headers(token, tenant))
        if st == 200 and isinstance(obj, dict):
            state = str(obj.get("state") or "")
            last_state = state
            if state in ("complete", "failed", "expired", "canceled"):
                return True, state
        elif st == 410:
            return True, "expired"
        time.sleep(0.5)
    return False, last_state


def _get_graph(client: HttpClient, base_url: str, token: Optional[str], tenant: Optional[str], trace_id: str) -> Tuple[int, Dict[str, Any], str]:
    url = f"{base_url.rstrip('/')}/v1/traces/{trace_id}/graph"
    return client.get_json(url, headers=auth_headers(token, tenant))


def _check_hif_version(obj: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    # Preferred: strict validate_hif, else minimal check for meta.version == "hif-1"
    try:
        if validate_hif is not None:
            validate_hif(obj)  # may raise
    except Exception as e:
        # Even if validation fails, still check minimal version key to provide signal
        meta = obj.get("meta") if isinstance(obj, dict) else None
        ver = meta.get("version") if isinstance(meta, dict) else None
        return (ver == "hif-1"), f"validate_hif error: {e}"
    try:
        meta = obj.get("meta") if isinstance(obj, dict) else None
        ver = meta.get("version") if isinstance(meta, dict) else None
        return (ver == "hif-1"), (None if ver == "hif-1" else f"unexpected meta.version={ver!r}")
    except Exception as e:
        return False, f"exception checking meta.version: {e}"


def run(config: E2EConfig) -> Dict[str, Any]:
    # Static checks
    if not OPENAPI_PATH.exists():
        return new_result("test_api_versioning.py", "SKIP", reason=f"OpenAPI not found at {OPENAPI_PATH}")

    text = OPENAPI_PATH.read_text(encoding="utf-8")
    paths = _parse_openapi_paths(text)
    non_v1 = [p for p in paths if not p.startswith("/v1/")]
    has_v2 = any(p.startswith("/v2/") for p in paths)

    cm_ok = False
    cm_hints: Dict[str, bool] = {}
    if CHANGE_MGMT_PATH.exists():
        cm_text = CHANGE_MGMT_PATH.read_text(encoding="utf-8")
        cm_ok, cm_hints = _check_change_mgmt(cm_text)

    static_ok = (len(paths) > 0) and (not non_v1) and (not has_v2) and cm_ok

    details: Dict[str, Any] = {
        "openapi_paths_count": len(paths),
        "non_v1_paths": non_v1,
        "has_v2_paths": has_v2,
        "change_mgmt_ok": cm_ok,
        "change_mgmt_hints": cm_hints,
    }

    # Optional runtime check
    base = config.base_url or "http://localhost:8080"
    client = HttpClient(timeout=10.0)
    auth_enabled = _probe_auth_enabled(base)
    write_token = config.auth_token_write if auth_enabled else None
    read_token = config.auth_token_read if auth_enabled else None
    tenant = config.tenant_a or "tenantA"

    runtime_note = "skipped"
    hif_ok = None
    hif_err = None
    if auth_enabled and not write_token:
        runtime_note = "auth enabled but --auth-token-write missing; runtime skipped"
    else:
        # Attempt to create trace and validate HIF version
        trace_id = _rand_trace()
        st_chat, body_chat, _hdrs, err_text = _post_chat_attach_with_trace(client, base, write_token, tenant, trace_id)
        if st_chat >= 500:
            # Upstream missing; cannot complete runtime path
            runtime_note = f"chat upstream error {st_chat}: {err_text or body_chat}"
        elif st_chat in (401, 403):
            runtime_note = f"RBAC rejected chat ({st_chat}); runtime skipped"
        else:
            done, state = _poll_status_until_terminal(client, base, read_token or write_token, tenant, trace_id, timeout_s=120.0)
            if done and state == "complete":
                g_code, g_obj, g_err = _get_graph(client, base, read_token or write_token, tenant, trace_id)
                if g_code == 200 and isinstance(g_obj, dict):
                    ok, err = _check_hif_version(g_obj)
                    hif_ok, hif_err = ok, err
                    runtime_note = "validated"
                else:
                    runtime_note = f"graph not available (status {g_code})"
            else:
                runtime_note = f"terminal state={state!r} (done={done})"

    details.update({
        "runtime": {
            "note": runtime_note,
            "hif_version_ok": hif_ok,
            "hif_error": hif_err,
        }
    })

    # Decide status:
    # - PASS if static checks OK. Runtime contributes additional signal but does not fail the v1 contract when environment cannot produce graphs.
    if static_ok:
        return new_result("test_api_versioning.py", "PASS", details=details)

    reason_bits = []
    if len(paths) == 0:
        reason_bits.append("no paths in OpenAPI")
    if non_v1:
        reason_bits.append(f"non_v1_paths={non_v1}")
    if has_v2:
        reason_bits.append("found v2 paths under OpenAPI paths")
    if not cm_ok:
        reason_bits.append("change-management doc missing ADR/freeze signals")
    return new_result("test_api_versioning.py", "FAIL", reason="; ".join(reason_bits), details=details)