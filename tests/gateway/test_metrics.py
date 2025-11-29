from __future__ import annotations

import os
import sys
import importlib.util

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
        self.prev: dict[str, str | None] = {}

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


def _fresh_app() -> tuple[object, object]:
    """
    Load gateway app without package install. Returns (app_module, fastapi_app).
    Mirrors the loader used in other gateway tests.
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

    appmod = _load_path("gateway_app_metrics", os.path.join(gw_src, "app.py"))
    return appmod, appmod.app  # type: ignore[attr-defined]


def _ensure_trace(appmod: object, tid: str) -> None:
    TraceStatus = getattr(appmod, "TraceStatus")
    TRACE_STATUS = getattr(appmod, "TRACE_STATUS")
    TRACE_STATUS[tid] = TraceStatus(trace_id=tid, state="queued", progress=0.0, stage="queued")


def test_metrics_names_present() -> None:
    if TestClient is None:  # type: ignore[truthy-bool]
        print("SKIP: TestClient not available")
        return

    # Enable observability and disable auth for this test
    with EnvPatch(AUTH_MODE="none", ENABLE_OTEL="1"):
        appmod, app = _fresh_app()
        client = TestClient(app)  # type: ignore[arg-type]

        # Touch some endpoints
        r_health = client.get("/healthz")
        assert r_health.status_code == 200

        tid = "trc_metrics_test"
        _ensure_trace(appmod, tid)
        # Exercise a templated path for low-cardinality label mapping
        _ = client.get(f"/v1/traces/{tid}/status")

        # Scrape metrics
        r_metrics = client.get("/metrics")
        assert r_metrics.status_code == 200, f"/metrics not served, status={r_metrics.status_code}"
        body = r_metrics.text

        # Assert presence by name (not cardinality-sensitive)
        # Core HTTP metrics
        assert "http_requests_total" in body
        assert "http_request_duration_seconds" in body

        # Domain metrics
        assert "chat_requests_total" in body
        assert "traces_status_get_total" in body
        assert "traces_graph_get_total" in body
        assert "rate_limit_rejections_total" in body
        assert "rbac_denied_total" in body


if __name__ == "__main__":
    # Allow running as a script: python -m tests.gateway.test_metrics
    test_metrics_names_present()
    print("OK: metrics names present")