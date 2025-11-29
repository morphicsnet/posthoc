"""
Lightweight observability helpers for the Explainer service.

- Prometheus metrics exposed via an internal HTTP server (default :9090)
- Optional OpenTelemetry tracing via OTLP (env-driven, import-guarded)
- JSON logging with optional correlation fields

Usage from the worker:
    from services.explainer.src import otel
    otel.setup_otel(service_name="explainer")

Then record metrics as stages progress:
    otel.stage_duration("extract", 0.012)
    otel.job_state("completed", "sentence", "sae-gpt4-2m")
    otel.queue_lag(0.5)
    otel.sae_decode_observe("cuda", 12, batch_size=32, elapsed=0.004, fallback=False)
    otel.update_sae_layer_cache_size(3)
    otel.attribution_observe("acdc", "sentence", duration=0.150, early_stop=False)
    otel.hif_counts(nodes=1024, incidences=4096)
    otel.hif_prune_ratio(0.72)
"""

from __future__ import annotations

import logging
import time
import os
from typing import Optional

# -----------------------
# Prometheus availability
# -----------------------
_PROM_AVAILABLE = True
try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        start_http_server,
    )
except Exception:
    _PROM_AVAILABLE = False

    class _NoopLabels:
        def inc(self, *args, **kwargs) -> None:
            return None

        def observe(self, *args, **kwargs) -> None:
            return None

        def set(self, *args, **kwargs) -> None:
            return None

    class _NoopMetric:
        def labels(self, *args, **kwargs) -> _NoopLabels:
            return _NoopLabels()

        def inc(self, *args, **kwargs) -> None:
            return None

        def observe(self, *args, **kwargs) -> None:
            return None

        def set(self, *args, **kwargs) -> None:
            return None

    Counter = Histogram = Gauge = _NoopMetric  # type: ignore


# -----------------------
# Tracing availability
# -----------------------
def _init_tracing(service_name: str, version: str) -> None:
    endpoint = (os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "") or "").strip()
    if not endpoint:
        return
    try:
        from opentelemetry import trace  # type: ignore
        from opentelemetry.sdk.resources import Resource  # type: ignore
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore

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
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
    except Exception:
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
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(getattr(record, "created", time.time())))
        base = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "service": self.service,
            "version": self.version,
            "msg": record.getMessage(),
        }
        for k in ("trace_id", "tenant_id", "request_id", "stage"):
            v = getattr(record, k, None)
            if v is not None:
                base[k] = v
        try:
            import json as _json
            return _json.dumps(base, separators=(",", ":"), sort_keys=False)
        except Exception:
            return f"{ts} {record.levelname} {record.name} {record.getMessage()}"


def get_logger(name: str = "explainer") -> logging.Logger:
    service = "explainer"
    version = os.getenv("EXPLAINER_VERSION", "unknown")
    if os.getenv("ENABLE_OTEL", "0") == "1":
        try:
            root = logging.getLogger()
            if not root.handlers:
                h = logging.StreamHandler()
                h.setFormatter(_JsonFormatter(service, version))
                root.addHandler(h)
            root.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO))
        except Exception:
            pass
    return logging.getLogger(name)


# -----------------------
# Metrics registry
# -----------------------
class _Metrics:
    def __init__(self) -> None:
        # Worker/job level
        self.explainer_jobs_total = Counter(
            "explainer_jobs_total",
            "Explainer jobs processed by state",
            ["state", "granularity", "featureset"],
        )
        self.explainer_stage_duration_seconds = Histogram(
            "explainer_stage_duration_seconds",
            "Explainer pipeline stage durations",
            ["stage"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
        )
        self.explainer_queue_lag_seconds = Histogram(
            "explainer_queue_lag_seconds",
            "Queue lag from envelope created_at to start time",
            buckets=(0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
        )

        # SAE decode service
        self.sae_decode_requests_total = Counter(
            "sae_decode_requests_total",
            "SAE decode requests by device and layer",
            ["device", "layer"],
        )
        self.sae_decode_latency_seconds = Histogram(
            "sae_decode_latency_seconds",
            "Latency of SAE decode batches by device and layer",
            ["device", "layer"],
            buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
        )
        self.sae_decode_fallbacks_total = Counter(
            "sae_decode_fallbacks_total",
            "Count of GPU->CPU fallback decode executions",
            ["device"],
        )
        self.sae_layer_cache_size = Gauge(
            "sae_layer_cache_size",
            "Number of SAE layers cached in memory",
        )
        self.sae_decode_batch_size = Histogram(
            "sae_decode_batch_size",
            "Observed SAE decode batch sizes",
            buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256),
        )

        # Hypergraph outputs
        self.hif_nodes_count = Gauge("hif_nodes_count", "Number of nodes in HIF output")
        self.hif_incidences_count = Gauge("hif_incidences_count", "Number of incidences (edges) in HIF output")
        self.hif_prune_ratio = Gauge("hif_prune_ratio", "Fraction of incidences pruned (approx)")
        self.hif_supernode_count = Gauge("hif_supernode_count", "Number of circuit_supernode nodes in HIF output")

        # Attribution
        self.attribution_method_total = Counter(
            "attribution_method_total",
            "Attribution invocations by method and granularity",
            ["method", "granularity"],
        )
        self.attribution_latency_seconds = Histogram(
            "attribution_latency_seconds",
            "Attribution latency by method and granularity",
            ["method", "granularity"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
        )
        self.attribution_early_stop_total = Counter(
            "attribution_early_stop_total",
            "Attribution early-stops by method",
            ["method"],
        )


_METRICS: Optional[_Metrics] = None
_METRICS_HTTP_STARTED = False


def _m() -> _Metrics:
    global _METRICS
    if _METRICS is None:
        _METRICS = _Metrics()
    return _METRICS


def _start_metrics_http_if_needed() -> None:
    global _METRICS_HTTP_STARTED
    if _METRICS_HTTP_STARTED:
        return
    if not _PROM_AVAILABLE:
        return
    try:
        host = os.getenv("EXPLAINER_METRICS_HOST", "0.0.0.0")
        port_str = os.getenv("EXPLAINER_METRICS_PORT", "9090")
        try:
            port = int(port_str)
        except Exception:
            port = 9090
        # This starts a background thread HTTP server serving /metrics (Prometheus format)
        start_http_server(port, addr=host)
        _METRICS_HTTP_STARTED = True
    except Exception:
        # fail-safe: no metrics server
        pass


# -----------------------
# Public API
# -----------------------
def setup_otel(service_name: str = "explainer") -> None:
    """
    Enable JSON logging, optional tracing, and start the metrics HTTP endpoint if ENABLE_OTEL=1.
    """
    if os.getenv("ENABLE_OTEL", "0") != "1":
        return
    version = os.getenv("EXPLAINER_VERSION", "unknown")
    # Logging
    try:
        root = logging.getLogger()
        if not root.handlers:
            h = logging.StreamHandler()
            h.setFormatter(_JsonFormatter(service_name, version))
            root.addHandler(h)
        root.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO))
    except Exception:
        pass
    # Tracing
    _init_tracing(service_name, version)
    # Metrics
    _ = _m()  # instantiate families
    _start_metrics_http_if_needed()


def stage_duration(stage: str, seconds: float) -> None:
    try:
        _m().explainer_stage_duration_seconds.labels(stage=str(stage)).observe(float(seconds))
    except Exception:
        pass


def job_state(state: str, granularity: str, featureset: str) -> None:
    try:
        _m().explainer_jobs_total.labels(
            state=str(state or "unknown"),
            granularity=str(granularity or "unknown"),
            featureset=str(featureset or "unknown"),
        ).inc()
    except Exception:
        pass


def queue_lag(seconds: float) -> None:
    try:
        _m().explainer_queue_lag_seconds.observe(float(seconds))
    except Exception:
        pass


def sae_decode_observe(device: str, layer: int, batch_size: int, elapsed: float, fallback: bool = False) -> None:
    try:
        dev = str(device or "unknown")
        lyr = str(int(layer) if isinstance(layer, int) else str(layer or "0"))
        _m().sae_decode_requests_total.labels(dev, lyr).inc()
        _m().sae_decode_latency_seconds.labels(dev, lyr).observe(float(elapsed))
        try:
            _m().sae_decode_batch_size.observe(float(batch_size))
        except Exception:
            pass
        if fallback:
            try:
                _m().sae_decode_fallbacks_total.labels(dev).inc()
            except Exception:
                pass
    except Exception:
        pass


def update_sae_layer_cache_size(n: int) -> None:
    try:
        _m().sae_layer_cache_size.set(float(n))
    except Exception:
        pass


def hif_counts(nodes: int, incidences: int) -> None:
    try:
        _m().hif_nodes_count.set(float(nodes))
        _m().hif_incidences_count.set(float(incidences))
    except Exception:
        pass


def hif_prune_ratio(ratio: float) -> None:
    try:
        r = float(ratio)
        if r < 0.0:
            r = 0.0
        if r > 1.0:
            r = 1.0
        _m().hif_prune_ratio.set(r)
    except Exception:
        pass


def hif_supernode_count(n: int) -> None:
    try:
        _m().hif_supernode_count.set(float(n))
    except Exception:
        pass


def attribution_observe(method: str, granularity: str, duration: float, early_stop: bool) -> None:
    try:
        m = str(method or "unknown")
        g = str(granularity or "unknown")
        _m().attribution_method_total.labels(m, g).inc()
        _m().attribution_latency_seconds.labels(m, g).observe(float(duration))
        if early_stop:
            _m().attribution_early_stop_total.labels(m).inc()
    except Exception:
        pass


__all__ = [
    "setup_otel",
    "get_logger",
    "stage_duration",
    "job_state",
    "queue_lag",
    "sae_decode_observe",
    "update_sae_layer_cache_size",
    "hif_counts",
    "hif_prune_ratio",
    "hif_supernode_count",
    "attribution_observe",
]