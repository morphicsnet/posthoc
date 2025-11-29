#!/usr/bin/env python3
# tests/e2e/test_observability.py
# Observability checks for Gateway and Explainer metrics endpoints.
#
# Validates:
# - Gateway Prometheus metrics served at --metrics-gateway
#   - Presence of http_requests_total, http_request_duration_seconds counters/histograms
#   - Presence of domain counters: chat_requests_total, traces_status_get_total, traces_graph_get_total
#   - Low-cardinality correlation: perform an operation and assert at least one relevant counter exists
# - Explainer Prometheus metrics served at --metrics-explainer (if reachable)
#   - Presence of explainer_jobs_total, explainer_stage_duration_seconds
#
# Skips gracefully if metrics endpoints are not reachable or ENABLE_OTEL is not enabled.
#
# References:
# - Gateway metrics setup: [setup_otel()](services/gateway/src/otel.py:330)
# - Explainer metrics setup: [setup_otel()](services/explainer/src/otel.py:273)

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from tests.e2e.utils import (
    E2EConfig,
    HttpClient,
    auth_headers,
    new_result,
    parse_prometheus_text,
)


def _fetch_text(url: str, timeout: float = 5.0) -> Optional[str]:
    """
    Minimal fetch using HttpClient.get_json path by bypassing JSON parsing:
    We'll use urllib directly for text to avoid depending on httpx.
    """
    import urllib.request
    import urllib.error

    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            charset = "utf-8"
            try:
                ctype = r.headers.get("content-type") or ""
                if "charset=" in ctype:
                    charset = ctype.split("charset=", 1)[1].split(";")[0].strip()
            except Exception:
                pass
            return r.read().decode(charset, errors="ignore")
    except urllib.error.HTTPError as he:
        try:
            body = he.read()
            return (body or b"").decode("utf-8", errors="ignore")
        except Exception:
            return None
    except Exception:
        return None


def _probe_auth_enabled(base_url: str) -> bool:
    import urllib.request
    import urllib.error
    url = f"{base_url.rstrip('/')}/v1/traces/trc_probe/status"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=3.0) as r:
            return False
    except urllib.error.HTTPError as he:
        return int(getattr(he, "code", 0)) in (401, 403)
    except Exception:
        return False


def run(config: E2EConfig) -> Dict[str, Any]:
    base = config.base_url or "http://localhost:8080"
    gw_metrics = config.metrics_gateway or (base.rstrip("/") + "/metrics")
    ex_metrics = config.metrics_explainer or "http://localhost:9090/metrics"
    client = HttpClient(timeout=6.0)

    # Touch a couple endpoints on Gateway to ensure some counters move
    # Health endpoint is lightweight; status path exercises trace-handling paths.
    try:
        _ = _fetch_text(base.rstrip("/") + "/healthz", timeout=3.0)
    except Exception:
        pass

    # Hit status for a bogus id to ensure traces_status_get_total increments
    auth_enabled = _probe_auth_enabled(base)
    tenant = config.tenant_a or "tenantA"
    token = config.auth_token_read if auth_enabled else None
    _ = client.get_json(f"{base.rstrip('/')}/v1/traces/trc_observe/status", headers=auth_headers(token, tenant))

    # Scrape Gateway metrics
    gw_text = _fetch_text(gw_metrics, timeout=5.0)
    if not isinstance(gw_text, str) or "http_requests_total" not in gw_text:
        return new_result("test_observability.py", "SKIP", reason="Gateway /metrics unavailable or missing core metrics", details={"metrics_gateway": gw_metrics})

    gw = parse_prometheus_text(gw_text)
    # Presence checks (names only; low-cardinality conditions)
    gw_required = ["http_requests_total", "http_request_duration_seconds", "chat_requests_total", "traces_status_get_total", "traces_graph_get_total"]
    missing = [m for m in gw_required if m not in gw]
    if missing:
        return new_result("test_observability.py", "FAIL", reason=f"gateway metrics missing: {missing}", details={"missing": missing, "endpoint": gw_metrics})

    # Correlation: ensure at least one traces_status_get_total sample exists
    has_status_counter = bool(gw.get("traces_status_get_total"))
    if not has_status_counter:
        return new_result("test_observability.py", "FAIL", reason="no traces_status_get_total samples after GET /v1/traces/{id}/status probe", details={"endpoint": gw_metrics})

    # Explainer metrics are optional; scrape and validate presence if reachable
    ex_text = _fetch_text(ex_metrics, timeout=5.0)
    ex_ok = False
    ex_reason = None
    if isinstance(ex_text, str) and ex_text.strip():
        ex = parse_prometheus_text(ex_text)
        if ("explainer_jobs_total" in ex) and ("explainer_stage_duration_seconds" in ex):
            ex_ok = True
        else:
            ex_reason = "explainer metrics present but required names missing"
    else:
        ex_reason = "explainer /metrics not reachable; skipping explainer checks"

    details = {
        "gateway_metrics": gw_metrics,
        "explainer_metrics": ex_metrics,
        "gateway_present": gw_required,
        "explainer_present": ex_ok,
        "explainer_note": ex_reason,
    }
    return new_result("test_observability.py", "PASS", reason=None, details=details)