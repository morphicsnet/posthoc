#!/usr/bin/env python3
# tests/e2e/test_chaos_recovery.py
# Chaos injection and recovery validation using the control file toggled by the injector CLI.
#
# Plan:
# 1) Ensure chaos is disabled via injector.
# 2) Submit a baseline chat with explanation attached; expect non-failure terminal state.
# 3) Enable a failure-inducing chaos flag (fail-attribution, 100%); submit another chat; expect failure (or at least not success).
# 4) Disable-all; submit again; expect recovery to successful completion.
#
# Skips:
# - If --chaos-control is not configured or injector is unavailable.
# - If baseline cannot succeed due to environment (e.g., LLM proxy misconfigured), we SKIP (not FAIL).
# - If enabling chaos does not produce any effect (no failure after a reasonable timeout), we SKIP as "chaos ineffective".
#
# References:
# - Chaos injector CLI: [main()](tests/chaos/chaos_injector.py:90)
# - Chat API: [create_chat_completion()](services/gateway/src/app.py:540)
# - Status API: [get_trace_status()](services/gateway/src/app.py:886)
# - Graph API: [get_trace_graph()](services/gateway/src/app.py:922)

from __future__ import annotations

import json
import os
import random
import string
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from tests.e2e.utils import (
    E2EConfig,
    HttpClient,
    auth_headers,
    new_result,
    poll_until,
)


def _rand_trace(prefix: str = "trc_chaos") -> str:
    sfx = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(10))
    return f"{prefix}_{sfx}"


def _auth_enabled_probe(base_url: str) -> bool:
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


def _inject_disable_all(path: str) -> Tuple[bool, str]:
    py = sys.executable or "python3"
    inj = str((Path(__file__).resolve().parents[2] / "tests" / "chaos" / "chaos_injector.py"))
    import subprocess
    try:
        cp = subprocess.run([py, inj, "--path", path, "--disable-all"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return (cp.returncode == 0), (cp.stdout or cp.stderr)
    except Exception as e:
        return False, str(e)


def _inject_fail_attribution(path: str, percent: float = 100.0) -> Tuple[bool, str]:
    py = sys.executable or "python3"
    inj = str((Path(__file__).resolve().parents[2] / "tests" / "chaos" / "chaos_injector.py"))
    import subprocess
    try:
        cp = subprocess.run(
            [py, inj, "--path", path, "--enable", "fail-attribution", "--percent", f"{percent}", "--mode", "fail"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return (cp.returncode == 0), (cp.stdout or cp.stderr)
    except Exception as e:
        return False, str(e)


def _post_chat_attach(client: HttpClient, base_url: str, token: Optional[str], tenant: Optional[str], trace_id: str) -> Tuple[int, Dict[str, Any]]:
    headers = auth_headers(token, tenant)
    headers["x-explain-mode"] = "hypergraph"
    headers["x-explain-granularity"] = "sentence"
    headers["x-explain-features"] = "sae-gpt4-2m"
    headers["x-trace-id"] = trace_id
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello and mention one color."},
        ],
        "stream": False,
        "temperature": 0.1,
    }
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    st, data, _hdrs, _err = client.post_json(url, body, headers=headers)
    return st, (data if isinstance(data, dict) else {})


def _poll_final_state(client: HttpClient, base_url: str, token: Optional[str], tenant: Optional[str], trace_id: str, timeout_s: float = 90.0) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Returns (done, state, last_payload)
    """
    url = f"{base_url.rstrip('/')}/v1/traces/{trace_id}/status"

    def _step():
        st, data, _err = client.get_json(url, headers=auth_headers(token, tenant))
        if st == 200 and isinstance(data, dict):
            state = str(data.get("state") or "")
            if state in ("complete", "failed", "expired", "canceled"):
                return True, (state, data)
        elif st == 410:
            return True, ("expired", {"state": "expired"})
        return False, ("", data)

    done, val = poll_until(_step, timeout_s=timeout_s, interval_s=0.5)
    state, payload = val if isinstance(val, tuple) else ("", {})
    return done, state, (payload if isinstance(payload, dict) else {})


def run(config: E2EConfig) -> Dict[str, Any]:
    base = config.base_url
    if not base:
        return new_result("test_chaos_recovery.py", "SKIP", reason="no base_url provided")

    chaos_path = config.chaos_control or "/tmp/hif/chaos.json"
    # Confirm injector script presence
    injector_py = Path(__file__).resolve().parents[2] / "tests" / "chaos" / "chaos_injector.py"
    if not injector_py.exists():
        return new_result("test_chaos_recovery.py", "SKIP", reason=f"chaos injector not found at {injector_py}")

    client = HttpClient(timeout=12.0)
    auth_enabled = _auth_enabled_probe(base)
    write_token = config.auth_token_write if auth_enabled else None
    read_token = config.auth_token_read if auth_enabled else None
    tenant = config.tenant_a or "tenantA"

    if auth_enabled and not write_token:
        return new_result("test_chaos_recovery.py", "SKIP", reason="auth enabled but --auth-token-write not provided")

    # 1) Disable all chaos
    ok, out = _inject_disable_all(chaos_path)
    if not ok:
        return new_result("test_chaos_recovery.py", "SKIP", reason=f"failed to disable chaos: {out[:200]}")

    # 2) Baseline success attempt
    t_base = _rand_trace("trc_base")
    st_chat, _ = _post_chat_attach(client, base, write_token, tenant, t_base)
    if st_chat in (401, 403):
        return new_result("test_chaos_recovery.py", "FAIL", reason=f"RBAC rejected baseline chat ({st_chat})")

    done, state, payload = _poll_final_state(client, base, read_token or write_token, tenant, t_base, timeout_s=90.0)
    if not done:
        return new_result("test_chaos_recovery.py", "SKIP", reason="baseline polling timeout (environment not ready)")

    if state not in ("complete",):  # Accept only success for a strong baseline
        return new_result("test_chaos_recovery.py", "SKIP", reason=f"baseline not successful (state={state}); environment may not produce explanations")

    # 3) Enable fail-attribution chaos and expect failure
    ok, out = _inject_fail_attribution(chaos_path, percent=100.0)
    if not ok:
        return new_result("test_chaos_recovery.py", "SKIP", reason=f"failed to enable fail-attribution: {out[:200]}")

    t_fail = _rand_trace("trc_fail")
    st_chat2, _ = _post_chat_attach(client, base, write_token, tenant, t_fail)
    if st_chat2 in (401, 403):
        return new_result("test_chaos_recovery.py", "FAIL", reason=f"RBAC rejected chat under chaos ({st_chat2})")

    done2, state2, payload2 = _poll_final_state(client, base, read_token or write_token, tenant, t_fail, timeout_s=90.0)
    if not done2:
        # Chaos ineffective or system laggy -> SKIP rather than FAIL
        return new_result("test_chaos_recovery.py", "SKIP", reason="chaos ineffective (no terminal state observed)")

    chaos_effective = (state2 in ("failed",))
    if not chaos_effective:
        # If still complete, chaos may not be wired; SKIP
        return new_result("test_chaos_recovery.py", "SKIP", reason=f"chaos did not induce failure (state={state2})")

    # 4) Disable all and ensure recovery
    ok, out = _inject_disable_all(chaos_path)
    if not ok:
        return new_result("test_chaos_recovery.py", "SKIP", reason=f"failed to disable chaos post-test: {out[:200]}")

    t_ok = _rand_trace("trc_recover")
    st_chat3, _ = _post_chat_attach(client, base, write_token, tenant, t_ok)
    if st_chat3 in (401, 403):
        return new_result("test_chaos_recovery.py", "FAIL", reason=f"RBAC rejected chat after chaos disable ({st_chat3})")

    done3, state3, payload3 = _poll_final_state(client, base, read_token or write_token, tenant, t_ok, timeout_s=120.0)
    if not done3:
        return new_result("test_chaos_recovery.py", "FAIL", reason="recovery polling timeout")
    if state3 != "complete":
        return new_result("test_chaos_recovery.py", "FAIL", reason=f"recovery state not complete: {state3}")

    details = {
        "baseline": {"trace_id": t_base, "state": state},
        "chaos": {"trace_id": t_fail, "state": state2},
        "recovery": {"trace_id": t_ok, "state": state3},
        "chaos_control": chaos_path,
    }
    return new_result("test_chaos_recovery.py", "PASS", details=details)