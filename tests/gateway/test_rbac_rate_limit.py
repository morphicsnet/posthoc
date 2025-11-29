from __future__ import annotations

import os
import sys
import json
import importlib
import importlib.util
from typing import Tuple, Optional

try:
    from fastapi.testclient import TestClient  # type: ignore
except Exception:
    try:
        from starlette.testclient import TestClient  # type: ignore
    except Exception:
        TestClient = None  # type: ignore


class EnvPatch:
    def __init__(self, **overrides: str) -> None:
        self.overrides = overrides
        self.prev: dict[str, Optional[str]] = {}

    def __enter__(self):
        for k, v in self.overrides.items():
            self.prev[k] = os.getenv(k)
            os.environ[k] = v
        return self

    def __exit__(self, exc_type, exc, tb):
        for k, old in self.prev.items():
            if old is None:
                try:
                    del os.environ[k]
                except KeyError:
                    pass
            else:
                os.environ[k] = old


def _fresh_app() -> Tuple[object, object]:
    """
    Load gateway app in a way that works without packages (__init__.py).
    Returns (app_module, fastapi_app).
    """
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    gw_src = os.path.join(ROOT, "services", "gateway", "src")
    if gw_src not in sys.path:
        sys.path.insert(0, gw_src)

    def _load_path(mod_name: str, path: str) -> object:
        spec = importlib.util.spec_from_file_location(mod_name, path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        sys.modules[mod_name] = mod
        return mod

    # Load helper modules first so app.py can fallback to 'rbac' and 'rate_limit'
    _load_path("rbac", os.path.join(gw_src, "rbac.py"))
    _load_path("rate_limit", os.path.join(gw_src, "rate_limit.py"))

    # Load the app module
    appmod = _load_path("gateway_app", os.path.join(gw_src, "app.py"))
    return appmod, appmod.app  # type: ignore[attr-defined]


def _client_for_current_env() -> Tuple[object, TestClient]:
    appmod, app = _fresh_app()
    client = TestClient(app)  # type: ignore[arg-type]
    return appmod, client


def _ensure_trace(appmod: object, tid: str) -> None:
    # Construct a minimal TraceStatus within the imported app module
    TraceStatus = getattr(appmod, "TraceStatus")
    TRACE_STATUS = getattr(appmod, "TRACE_STATUS")
    TRACE_STATUS[tid] = TraceStatus(trace_id=tid, state="queued", progress=0.0, stage="queued")


def test_auth_mode_none_allows_and_rl_applies() -> None:
    if TestClient is None:  # type: ignore[truthy-bool]
        print("SKIP: TestClient not available")
        return

    # Very tight read limits: 1 token burst, 1 rps
    with EnvPatch(AUTH_MODE="none", RATE_LIMIT_READ_RPS="1", RATE_LIMIT_READ_BURST="1"):
        appmod, client = _client_for_current_env()
        tid = "trc_test_anon"
        _ensure_trace(appmod, tid)

        # First read should pass
        r1 = client.get(f"/v1/traces/{tid}/status")
        assert r1.status_code == 200, f"expected 200, got {r1.status_code} body={r1.text}"

        # Second immediate read should be rate limited
        r2 = client.get(f"/v1/traces/{tid}/status")
        assert r2.status_code == 429, f"expected 429, got {r2.status_code} body={r2.text}"
        ra = r2.headers.get("Retry-After")
        assert ra is not None and int(ra) >= 1, "Retry-After header should be present and integer seconds"


def test_static_missing_and_invalid_tokens() -> None:
    if TestClient is None:  # type: ignore[truthy-bool]
        print("SKIP: TestClient not available")
        return

    tokens = json.dumps({"good": {"tenant_id": "t1", "scopes": ["traces:read", "traces:write"]}})
    with EnvPatch(AUTH_MODE="static", AUTH_TOKENS_JSON=tokens, RATE_LIMIT_READ_RPS="50", RATE_LIMIT_READ_BURST="50"):
        appmod, client = _client_for_current_env()
        tid = "trc_auth_missing"
        _ensure_trace(appmod, tid)

        # Missing Authorization header -> 401
        r_missing = client.get(f"/v1/traces/{tid}/status")
        assert r_missing.status_code == 401, f"expected 401, got {r_missing.status_code}"

        # Invalid token -> 403
        r_bad = client.get(f"/v1/traces/{tid}/status", headers={"Authorization": "Bearer bad"})
        assert r_bad.status_code == 403, f"expected 403, got {r_bad.status_code}"


def test_static_valid_but_missing_scope_for_write() -> None:
    if TestClient is None:  # type: ignore[truthy-bool]
        print("SKIP: TestClient not available")
        return

    # Token has only read scope; write endpoints should be forbidden
    tokens = json.dumps({"readOnly": {"tenant_id": "t1", "scopes": ["traces:read"]}})
    with EnvPatch(AUTH_MODE="static", AUTH_TOKENS_JSON=tokens):
        appmod, client = _client_for_current_env()
        tid = "trc_no_write_scope"
        _ensure_trace(appmod, tid)

        headers = {"Authorization": "Bearer readOnly"}
        body = {"url": "http://localhost/hook"}
        r = client.post(f"/v1/traces/{tid}/webhooks", headers=headers, json=body)
        assert r.status_code == 403, f"expected 403 for missing write scope, got {r.status_code}"


def test_rate_limit_write_exceeded() -> None:
    if TestClient is None:  # type: ignore[truthy-bool]
        print("SKIP: TestClient not available")
        return

    # Write category: burst=1, rps=0 => first allowed, second must be 429
    tokens = json.dumps({"w1": {"tenant_id": "tW", "scopes": ["traces:read", "traces:write"]}})
    with EnvPatch(
        AUTH_MODE="static",
        AUTH_TOKENS_JSON=tokens,
        RATE_LIMIT_WRITE_BURST="1",
        RATE_LIMIT_WRITE_RPS="0"
    ):
        appmod, client = _client_for_current_env()
        tid = "trc_write_rl"
        _ensure_trace(appmod, tid)

        headers = {"Authorization": "Bearer w1"}
        body = {"url": "http://localhost/hook"}

        r1 = client.post(f"/v1/traces/{tid}/webhooks", headers=headers, json=body)
        assert r1.status_code == 201, f"expected 201 for first write, got {r1.status_code} body={r1.text}"

        r2 = client.post(f"/v1/traces/{tid}/webhooks", headers=headers, json=body)
        assert r2.status_code == 429, f"expected 429 for second write, got {r2.status_code}"
        ra = r2.headers.get("Retry-After")
        assert ra is not None and int(ra) >= 1, "Retry-After header should be present and integer seconds"


def test_status_store_unavailability_fallback() -> None:
    if TestClient is None:  # type: ignore[truthy-bool]
        print("SKIP: TestClient not available")
        return

    # Enable STATUS_BACKEND to ddb stub; store.get_status returns None -> fallback to in-memory
    tokens = json.dumps({"reader": {"tenant_id": "tR", "scopes": ["traces:read"]}})
    with EnvPatch(AUTH_MODE="static", AUTH_TOKENS_JSON=tokens, STATUS_BACKEND="ddb"):
        appmod, client = _client_for_current_env()
        tid = "trc_fallback"
        _ensure_trace(appmod, tid)

        headers = {"Authorization": "Bearer reader"}
        r = client.get(f"/v1/traces/{tid}/status", headers=headers)
        assert r.status_code == 200, f"expected 200 fallback to in-memory, got {r.status_code}"
        js = r.json()
        assert js.get("trace_id") == tid, "fallback status should return the in-memory item"


def _run_all():
    if TestClient is None:  # type: ignore[truthy-bool]
        print("SKIP: TestClient not available")
        return
    test_auth_mode_none_allows_and_rl_applies()
    test_static_missing_and_invalid_tokens()
    test_static_valid_but_missing_scope_for_write()
    test_rate_limit_write_exceeded()
    test_status_store_unavailability_fallback()
    print("OK: gateway RBAC/rate-limit tests passed")


if __name__ == "__main__":
    _run_all()