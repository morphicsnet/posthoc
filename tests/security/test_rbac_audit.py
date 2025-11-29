from __future__ import annotations

import io
import os
import sys
import json
import importlib.util
from contextlib import redirect_stdout
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
    Mirrors approach used by other gateway tests.
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

    # Ensure local imports for rbac/rate_limit are resolvable
    _load_path("rbac", os.path.join(gw_src, "rbac.py"))
    _load_path("rate_limit", os.path.join(gw_src, "rate_limit.py"))

    appmod = _load_path("gateway_app_audit", os.path.join(gw_src, "app.py"))
    return appmod, appmod.app  # type: ignore[attr-defined]


def _ensure_trace(appmod: object, tid: str) -> None:
    TraceStatus = getattr(appmod, "TraceStatus")
    TRACE_STATUS = getattr(appmod, "TRACE_STATUS")
    TRACE_STATUS[tid] = TraceStatus(trace_id=tid, state="queued", progress=0.0, stage="queued")


def _capture_audit(fn):
    """
    Helper to capture stdout and return printed content.
    Assumes AUDIT_LOG_ENABLE=1 and AUDIT_LOG_PATH is not writable,
    causing audit sink to fallback to stdout.
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue()


def test_static_missing_token_401_emits_audit() -> None:
    if TestClient is None:  # type: ignore[truthy-bool]
        print("SKIP: TestClient not available")
        return

    # Use unwritable audit path to force stdout fallback
    with EnvPatch(
        AUTH_MODE="static",
        AUTH_TOKENS_JSON=json.dumps({"good": {"tenant_id": "t1", "scopes": ["traces:read"]}}),
        AUDIT_LOG_ENABLE="1",
        AUDIT_LOG_PATH="/root/denied/audit.log",  # typically unwritable in CI/local
    ):
        appmod, app = _fresh_app()
        client = TestClient(app)  # type: ignore[arg-type]
        tid = "trc_audit_missing"
        _ensure_trace(appmod, tid)

        def _do():
            r = client.get(f"/v1/traces/{tid}/status")
            assert r.status_code == 401, f"expected 401, got {r.status_code}"

        out = _capture_audit(_do)
        # Expect an audit line with event=rbac.deny and status=401
        assert "rbac.deny" in out, f"expected rbac.deny event in audit log, got: {out}"
        assert '"status":401' in out or '"status": 401' in out, "expected status=401 in audit log"
        # Reason code should be 'unauthorized' (from rbac detail)
        assert "unauthorized" in out, "expected reason 'unauthorized' in audit log payload"


def test_insufficient_scopes_403_emits_missing_scope_reason() -> None:
    if TestClient is None:  # type: ignore[truthy-bool]
        print("SKIP: TestClient not available")
        return

    tokens = json.dumps({"readOnly": {"tenant_id": "t1", "scopes": ["traces:read"]}})
    with EnvPatch(
        AUTH_MODE="static",
        AUTH_TOKENS_JSON=tokens,
        AUDIT_LOG_ENABLE="1",
        AUDIT_LOG_PATH="/root/denied/audit.log",
    ):
        appmod, app = _fresh_app()
        client = TestClient(app)  # type: ignore[arg-type]
        tid = "trc_audit_scope"
        _ensure_trace(appmod, tid)

        headers = {"Authorization": "Bearer readOnly"}
        body = {"url": "http://localhost/hook"}

        def _do():
            r = client.post(f"/v1/traces/{tid}/webhooks", headers=headers, json=body)
            assert r.status_code == 403, f"expected 403, got {r.status_code}"

        out = _capture_audit(_do)
        assert "rbac.deny" in out, f"expected rbac.deny event in audit log, got: {out}"
        # Our rbac returns code 'missing_scope' for this case, ensure reason is visible
        assert "missing_scope" in out, "expected reason 'missing_scope' in audit payload"


def test_authorized_write_emits_audit_event() -> None:
    if TestClient is None:  # type: ignore[truthy-bool]
        print("SKIP: TestClient not available")
        return

    tokens = json.dumps({"w1": {"tenant_id": "tW", "scopes": ["traces:read", "traces:write"]}})
    with EnvPatch(
        AUTH_MODE="static",
        AUTH_TOKENS_JSON=tokens,
        AUDIT_LOG_ENABLE="1",
        AUDIT_LOG_PATH="/root/denied/audit.log",
    ):
        appmod, app = _fresh_app()
        client = TestClient(app)  # type: ignore[arg-type]
        tid = "trc_audit_ok"
        _ensure_trace(appmod, tid)

        headers = {"Authorization": "Bearer w1"}
        body = {"url": "http://localhost/hook"}

        def _do():
            r = client.post(f"/v1/traces/{tid}/webhooks", headers=headers, json=body)
            assert r.status_code == 201, f"expected 201, got {r.status_code}"

        out = _capture_audit(_do)
        # webhook.register event should be present
        assert "webhook.register" in out, f"expected webhook.register event in audit log, got: {out}"


def _run_all():
    if TestClient is None:  # type: ignore[truthy-bool]
        print("SKIP: TestClient not available")
        return
    test_static_missing_token_401_emits_audit()
    test_insufficient_scopes_403_emits_missing_scope_reason()
    test_authorized_write_emits_audit_event()
    print("OK: security RBAC/audit tests passed")


if __name__ == "__main__":
    _run_all()