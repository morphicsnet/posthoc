"""
Lightweight observability helpers for the Gateway service.

- Prometheus metrics on /metrics (guarded; no-op when libs missing)
- Optional OpenTelemetry tracing via OTLP (env-driven, import-guarded)
- JSON logging with correlation fields

Usage from the FastAPI app:
    from services.gateway.src.otel import setup_otel
    setup_otel(app, service_name="gateway")
"""

from __future__ import annotations

import logging
import re
import time
import uuid
import os
from typing import Any, Callable, Optional

# -----------------------
# Prometheus availability
# -----------------------
_PROM_AVAILABLE = True
try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        generate_latest,
        CollectorRegistry,
        CONTENT_TYPE_LATEST,
    )
except Exception:
    _PROM_AVAILABLE = False
    # Minimal no-op stand-ins so code paths do not break if lib is absent
    class _NoopLabels:
        def __init__(self) -> None:
            pass

        def inc(self, *args, **kwargs) -> None:
            return None

        def observe(self, *args, **kwargs) -> None:
            return None

        def set(self, *args, **kwargs) -> None:
            return None

    class _NoopMetric:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def labels(self, *args, **kwargs) -> _NoopLabels:
            return _NoopLabels()

    Counter = Histogram = Gauge = _NoopMetric  # type: ignore


# -----------------------
# Tracing availability
# -----------------------
def _init_tracing(service_name: str, version: str) -> None:
    """
    Initialize OTLP tracing if OTEL_EXPORTER_OTLP_ENDPOINT is set and
    OpenTelemetry libs are available. Graceful no-op otherwise.
    """
    endpoint = (os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "") or "").strip()
    if not endpoint:
        return
    try:
        from opentelemetry import trace  # type: ignore
        from opentelemetry.sdk.resources import Resource  # type: ignore
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore

        # Prefer protocol from env; default gRPC if not specified
        protocol = (os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "") or "").strip().lower()
        if protocol in ("http", "http/protobuf", "http_proto", "http_protobuf"):
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore
                OTLPSpanExporter as OTLPHTTPExporter,
            )

            exporter = OTLPHTTPExporter(endpoint=endpoint)
        else:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore
                OTLPSpanExporter as OTLPGRPCExporter,
            )

            # For gRPC endpoint, strip scheme if present
            cleaned = endpoint
            if cleaned.startswith("http://"):
                cleaned = cleaned[len("http://") :]
            elif cleaned.startswith("https://"):
                cleaned = cleaned[len("https://") :]
            exporter = OTLPGRPCExporter(endpoint=cleaned)

        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": version,
                "service.namespace": "hypergraph-sidecar",
                "telemetry.sdk.language": "python",
            }
        )
        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
    except Exception:
        # Any error -> tracing disabled
        return


# -----------------------
# JSON logging
# -----------------------
class _JsonFormatter(logging.Formatter):
    def __init__(self, service: str, version: str) -> None:
        super().__init__()
        self.service = service
        self.version = version

    def format(self, record: logging.LogRecord) -> str:
        # Minimal JSON to keep overhead low; avoid orjson dependency
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(getattr(record, "created", time.time())))
        base = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "service": self.service,
            "version": self.version,
            "msg": record.getMessage(),
        }
        # Pass-through common correlation fields if present
        for k in ("trace_id", "tenant_id", "request_id", "path", "method", "status"):
            v = getattr(record, k, None)
            if v is not None:
                base[k] = v
        # Render compact JSON without extra spaces
        try:
            import json as _json  # stdlib
            return _json.dumps(base, separators=(",", ":"), sort_keys=False)
        except Exception:
            return f"{ts} {record.levelname} {record.name} {record.getMessage()}"


def _setup_json_logging(service: str, version: str) -> None:
    try:
        root = logging.getLogger()
        if not root.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(_JsonFormatter(service, version))
            root.addHandler(handler)
        root.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO))
    except Exception:
        # Keep default logging if anything fails
        pass


def get_logger(name: str = "gateway") -> logging.Logger:
    """
    Return a logger (JSON formatted if ENABLE_OTEL=1).
    """
    service = "gateway"
    version = os.getenv("GATEWAY_VERSION", "unknown")
    if os.getenv("ENABLE_OTEL", "0") == "1":
        _setup_json_logging(service, version)
    return logging.getLogger(name)


# -----------------------
# Path templating
# -----------------------
_PATH_SUBSTITUTIONS: list[tuple[re.Pattern[str], str]] = [
    # Normalize trace_id segment
    (re.compile(r"^/v1/traces/[^/]+/status$"), "/v1/traces/_/status"),
    (re.compile(r"^/v1/traces/[^/]+/graph$"), "/v1/traces/_/graph"),
    (re.compile(r"^/v1/traces/[^/]+/stream$"), "/v1/traces/_/stream"),
    # Normalize chat completions (no IDs currently, but keep stable)
    (re.compile(r"^/v1/chat/completions(?:/[^/]+)?$"), "/v1/chat/completions"),
]


def sanitize_path(path: str) -> str:
    """
    Low-cardinality templating of API paths.
    """
    p = path or "/"
    for rx, repl in _PATH_SUBSTITUTIONS:
        if rx.match(p):
            return repl
    return p


# -----------------------
# Metrics registry
# -----------------------
class _Metrics:
    def __init__(self, service: str, version: str) -> None:
        self.service = service
        self.version = version

        # HTTP level
        self.http_requests_total = Counter(
            "http_requests_total",
            "HTTP requests processed by the gateway",
            ["path", "method", "status", "tenant_id", "service", "version"],
        )
        self.http_request_duration_seconds = Histogram(
            "http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["path", "method", "status", "service", "version"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
        )

        # Domain-specific
        self.chat_requests_total = Counter(
            "chat_requests_total",
            "Chat completion requests received",
            ["tenant_id", "granularity", "featureset", "service", "version"],
        )
        self.traces_status_get_total = Counter(
            "traces_status_get_total",
            "Trace status GET requests",
            ["tenant_id", "service", "version"],
        )
        self.traces_graph_get_total = Counter(
            "traces_graph_get_total",
            "Trace graph GET requests",
            ["tenant_id", "service", "version"],
        )
        self.rate_limit_rejections_total = Counter(
            "rate_limit_rejections_total",
            "Requests rejected by rate limiting",
            ["tenant_id", "category", "service", "version"],
        )
        self.rbac_denied_total = Counter(
            "rbac_denied_total",
            "Requests denied by RBAC",
            ["tenant_id", "service", "version"],
        )

    # Convenience label helpers
    def label_http(self, path: str, method: str, status: int, tenant_id: str) -> dict[str, str]:
        return {
            "path": sanitize_path(path),
            "method": method.upper(),
            "status": str(int(status)),
            "tenant_id": tenant_id,
            "service": self.service,
            "version": self.version,
        }

    def label_http_timing(self, path: str, method: str, status: int) -> dict[str, str]:
        return {
            "path": sanitize_path(path),
            "method": method.upper(),
            "status": str(int(status)),
            "service": self.service,
            "version": self.version,
        }

    def label_tenant(self, tenant_id: str) -> dict[str, str]:
        return {"tenant_id": tenant_id, "service": self.service, "version": self.version}

    def label_chat(self, tenant_id: str, granularity: str, featureset: str) -> dict[str, str]:
        return {
            "tenant_id": tenant_id,
            "granularity": granularity or "unknown",
            "featureset": featureset or "unknown",
            "service": self.service,
            "version": self.version,
        }

    def label_rl(self, tenant_id: str, category: str) -> dict[str, str]:
        c = category if category in ("read", "write") else "unknown"
        return {"tenant_id": tenant_id, "category": c, "service": self.service, "version": self.version}


_METRICS: Optional[_Metrics] = None


def _metrics() -> _Metrics:
    # Initialize on first use to bind service/version at runtime
    global _METRICS
    if _METRICS is None:
        _METRICS = _Metrics(service="gateway", version=os.getenv("GATEWAY_VERSION", "unknown"))
    return _METRICS


# -----------------------
# /metrics rendering
# -----------------------
_MINIMAL_EXPOSITION = """# HELP http_requests_total HTTP requests processed by the gateway
# TYPE http_requests_total counter
# HELP http_request_duration_seconds HTTP request duration in seconds
# TYPE http_request_duration_seconds histogram
# HELP chat_requests_total Chat completion requests received
# TYPE chat_requests_total counter
# HELP traces_status_get_total Trace status GET requests
# TYPE traces_status_get_total counter
# HELP traces_graph_get_total Trace graph GET requests
# TYPE traces_graph_get_total counter
# HELP rate_limit_rejections_total Requests rejected by rate limiting
# TYPE rate_limit_rejections_total counter
# HELP rbac_denied_total Requests denied by RBAC
# TYPE rbac_denied_total counter
"""


def _render_metrics_bytes() -> tuple[bytes, str]:
    """
    Return (payload, content_type) for Prometheus exposition.
    Falls back to a minimal static exposition when prometheus_client is unavailable.
    """
    if _PROM_AVAILABLE:
        try:
            payload = generate_latest()
            return payload, CONTENT_TYPE_LATEST
        except Exception:
            pass
    return _MINIMAL_EXPOSITION.encode("utf-8"), "text/plain; version=0.0.4; charset=utf-8"


# -----------------------
# FastAPI installer
# -----------------------
def setup_otel(app: Any, service_name: str = "gateway") -> None:
    """
    Idempotent setup for:
      - JSON logging
      - OTEL tracing (optional)
      - Prometheus metrics + HTTP middleware
      - /metrics endpoint
    """
    version = os.getenv("GATEWAY_VERSION", "unknown")
    _setup_json_logging(service_name, version)
    _init_tracing(service_name, version)

    # Bind metrics with current service/version
    global _METRICS
    _METRICS = _Metrics(service=service_name, version=version)

    # Optional: try FastAPIInstrumentator if present (kept minimal, default metrics are fine).
    try:
        from prometheus_fastapi_instrumentator import Instrumentator  # type: ignore

        try:
            # Respect env PROMETHEUS_METRICS_EXCLUDE_HANDLER if users opt out
            Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
            _instrumentor_attached = True  # noqa: F841
        except Exception:
            # Fallback to manual /metrics route below
            pass
    except Exception:
        # No instrumentator installed -> manual route below
        pass

    # Manual /metrics route in case instrumentator didn't attach
    try:
        from fastapi import Response

        @app.get("/metrics")
        async def metrics_endpoint() -> Response:  # type: ignore
            payload, ctype = _render_metrics_bytes()
            return Response(content=payload, media_type=ctype)
    except Exception:
        # If FastAPI imports are not available here, the app likely already exposed metrics via instrumentator.
        pass

    # HTTP middleware for common labels and custom counters
    try:
        from fastapi import Request, Response as _FResponse  # type: ignore
    except Exception:
        # If we cannot import FastAPI types, skip middleware installation
        return

    @app.middleware("http")
    async def _metrics_middleware(request: Request, call_next: Callable) -> _FResponse:  # type: ignore
        t0 = time.perf_counter()
        path = str(request.url.path)
        method = str(request.method).upper()
        # Correlation fields
        request_id = uuid.uuid4().hex
        trace_id = request.headers.get("x-trace-id") or ("trc_" + uuid.uuid4().hex[:12])
        # Resolve tenant id (RBAC dependency populates request.state.auth_ctx when available)
        try:
            auth_ctx = getattr(request.state, "auth_ctx", None)
            tenant_id = getattr(auth_ctx, "tenant_id", "anon") or "anon"
        except Exception:
            tenant_id = "anon"

        # Extract domain headers early for chat metrics
        gran = (request.headers.get("x-explain-granularity") or "").strip()
        feats = (request.headers.get("x-explain-features") or "").strip()

        status_code = 500
        response: Optional[_FResponse] = None
        try:
            response = await call_next(request)
            status_code = int(getattr(response, "status_code", 200))
            return response
        except Exception as ex:  # Capture exception status if HTTPException
            try:
                from fastapi import HTTPException  # type: ignore

                if isinstance(ex, HTTPException):
                    status_code = int(ex.status_code)
            except Exception:
                pass
            # Re-raise after recording metrics
            raise
        finally:
            duration = max(0.0, float(time.perf_counter() - t0))
            ptempl = sanitize_path(path)

            # Set correlation headers if we have a response object
            try:
                if response is not None:
                    response.headers.setdefault("x-trace-id", trace_id)
                    response.headers.setdefault("x-request-id", request_id)
            except Exception:
                pass

            # HTTP metrics
            try:
                m = _metrics()
                m.http_requests_total.labels(**m.label_http(ptempl, method, status_code, tenant_id)).inc()
                m.http_request_duration_seconds.labels(**m.label_http_timing(ptempl, method, status_code)).observe(duration)
            except Exception:
                pass

            # RBAC and rate limit signals (approximate)
            try:
                if status_code in (401, 403):
                    _metrics().rbac_denied_total.labels(**_metrics().label_tenant(tenant_id)).inc()
                if status_code == 429:
                    cat = "read" if method == "GET" else "write"
                    _metrics().rate_limit_rejections_total.labels(**_metrics().label_rl(tenant_id, cat)).inc()
            except Exception:
                pass

            # Domain metrics by templated path
            try:
                if method == "GET" and ptempl == "/v1/traces/_/status":
                    _metrics().traces_status_get_total.labels(**_metrics().label_tenant(tenant_id)).inc()
                elif method == "GET" and ptempl == "/v1/traces/_/graph":
                    _metrics().traces_graph_get_total.labels(**_metrics().label_tenant(tenant_id)).inc()
                elif ptempl == "/v1/chat/completions":
                    _metrics().chat_requests_total.labels(**_metrics().label_chat(tenant_id, gran or "unknown", feats or "unknown")).inc()
            except Exception:
                pass

            # Log minimal structured line (JSON formatter will render it in a single line)
            try:
                lg = get_logger("gateway.http")
                lg.info(
                    "http",
                    extra={
                        "trace_id": trace_id,
                        "tenant_id": tenant_id,
                        "request_id": request_id,
                        "path": ptempl,
                        "method": method,
                        "status": status_code,
                    },
                )
            except Exception:
                pass


__all__ = [
    "setup_otel",
    "get_logger",
    "sanitize_path",
]