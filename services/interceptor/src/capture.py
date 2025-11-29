"""
Interceptor Ingest Service
- Minimal HTTP endpoint to accept finalized completion payloads
- Enqueues envelopes into Redis Streams (XADD) for downstream processing

Env:
  REDIS_URL=redis://redis:6379/0
  REDIS_STREAM=hypergraph:completions
  REDIS_MAXLEN=   # optional integer; when set, use MAXLEN ~ trimming
  HOST=0.0.0.0
  PORT=8080

TODO(version2):
  - auth/signature verification for inbound POSTs.
  - schema validation against HIF ExplanationResponse if needed.
  - retries/backoff to Redis and DLQ.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Optional

# Soft import FastAPI/Starlette; fallback to http.server if unavailable
try:
    from fastapi import FastAPI, Request  # type: ignore
    from starlette.responses import JSONResponse  # type: ignore
except Exception:  # pragma: no cover - soft dependency
    FastAPI = None  # type: ignore
    Request = None  # type: ignore
    JSONResponse = None  # type: ignore

# Soft import redis.asyncio first; fallback to sync redis
try:
    import redis.asyncio as aioredis  # type: ignore
except Exception:  # pragma: no cover - optional
    aioredis = None  # type: ignore

try:
    import redis as redis_sync  # type: ignore
except Exception:  # pragma: no cover - optional
    redis_sync = None  # type: ignore

logger = logging.getLogger(__name__)

# Optional: capture pipeline + utils for dev simulation and future hook integration
try:
    # Absolute import when running from repo root
    from services.interceptor.src.hooks import CaptureConfig, CapturePipeline, register_layer_hook, RedisClientStub  # type: ignore  # pylint: disable=import-error
except Exception:
    try:
        # Relative import when packaged
        from .hooks import CaptureConfig, CapturePipeline, register_layer_hook, RedisClientStub  # type: ignore  # pylint: disable=import-error
    except Exception:
        try:
            # Local import when running the single file
            from hooks import CaptureConfig, CapturePipeline, register_layer_hook, RedisClientStub  # type: ignore  # pylint: disable=import-error
        except Exception:
            CaptureConfig = None  # type: ignore
            CapturePipeline = None  # type: ignore
            register_layer_hook = None  # type: ignore
            RedisClientStub = None  # type: ignore

try:
    from services.interceptor.src.utils import now_ns, ns_to_ms  # type: ignore  # pylint: disable=import-error
except Exception:
    try:
        from .utils import now_ns, ns_to_ms  # type: ignore  # pylint: disable=import-error
    except Exception:
        try:
            from utils import now_ns, ns_to_ms  # type: ignore  # pylint: disable=import-error
        except Exception:
            def now_ns() -> int:  # type: ignore
                import time as _t
                return int(_t.perf_counter_ns())
            def ns_to_ms(ns: int) -> float:  # type: ignore
                return ns / 1e6

# Configuration centralized below in Config class

def _parse_int(val: Optional[str]) -> Optional[int]:
    if val is None or str(val).strip() == "":
        return None
    try:
        return int(str(val).strip())
    except Exception:
        logger.warning("Invalid REDIS_MAXLEN=%r; ignoring", val)
        return None

class Config:
    def __init__(self) -> None:
        self.REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
        self.REDIS_STREAM: str = os.getenv("REDIS_STREAM", "hypergraph:completions")
        self.REDIS_MAXLEN: Optional[int] = _parse_int(os.getenv("REDIS_MAXLEN"))
        self.HOST: str = os.getenv("HOST", "0.0.0.0")
        try:
            self.PORT: int = int(os.getenv("PORT", "8080"))
        except Exception:
            self.PORT = 8080
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    def safe_dict(self) -> dict:
        return {
            "REDIS_URL": self.REDIS_URL,
            "REDIS_STREAM": self.REDIS_STREAM,
            "REDIS_MAXLEN": self.REDIS_MAXLEN,
            "HOST": self.HOST,
            "PORT": self.PORT,
            "LOG_LEVEL": self.LOG_LEVEL,
        }

    def safe_repr(self) -> str:
        try:
            return json.dumps(self.safe_dict(), separators=(",", ":"), sort_keys=True)
        except Exception:
            return str(self.safe_dict())

CONFIG = Config()

try:
    logger.info("config: %s", CONFIG.safe_repr())
except Exception:
    pass

class RedisStreamEnqueuer:
    """
    Small adapter that prefers redis.asyncio when available, otherwise falls
    back to sync redis client. Provides async and blocking enqueue helpers.
    """
    def __init__(self, url: str, stream: str, maxlen: Optional[int]) -> None:
        self.url = url
        self.stream = stream
        self.maxlen = maxlen
        self._client_async = None
        self._client_sync = None
        # Try async first
        if aioredis is not None:
            try:
                if hasattr(aioredis, "from_url"):
                    self._client_async = aioredis.from_url(self.url, encoding="utf-8", decode_responses=True)
                else:
                    self._client_async = aioredis.Redis.from_url(self.url, encoding="utf-8", decode_responses=True)  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning("Failed to init redis.asyncio client: %s", e)
        # Fallback to sync
        if self._client_async is None and redis_sync is not None:
            try:
                if hasattr(redis_sync, "from_url"):
                    self._client_sync = redis_sync.from_url(self.url, encoding="utf-8", decode_responses=True)  # type: ignore[attr-defined]
                else:
                    self._client_sync = redis_sync.Redis.from_url(self.url, encoding="utf-8", decode_responses=True)  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning("Failed to init redis sync client: %s", e)

    @property
    def is_async(self) -> bool:
        return self._client_async is not None

    def _xadd_kwargs(self) -> dict:
        kwargs = {}
        if self.maxlen is not None:
            kwargs["maxlen"] = self.maxlen
            kwargs["approximate"] = True  # MAXLEN ~
        return kwargs

    async def enqueue(self, payload: dict) -> str:
        """
        Enqueue using async client when available, otherwise offload sync to thread.
        """
        data_str = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        try:
            if self._client_async is not None:
                xadd_id = await self._client_async.xadd(self.stream, {"data": data_str}, **self._xadd_kwargs())
                return str(xadd_id)
            if self._client_sync is not None:
                loop = asyncio.get_running_loop()
                def _do() -> str:
                    xid = self._client_sync.xadd(self.stream, {"data": data_str}, **self._xadd_kwargs())
                    return str(xid)
                return await loop.run_in_executor(None, _do)
            raise RuntimeError("No Redis client available. Install 'redis' package.")
        except Exception:
            logger.exception("Redis XADD failed")
            raise

    def enqueue_blocking(self, payload: dict) -> str:
        """
        Blocking helper for environments without an event loop (http.server).
        Will prefer async client if present by running a temporary loop.
        """
        if self._client_async is not None:
            return asyncio.run(self.enqueue(payload))
        if self._client_sync is not None:
            data_str = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
            try:
                xid = self._client_sync.xadd(self.stream, {"data": data_str}, **self._xadd_kwargs())
                return str(xid)
            except Exception:
                logger.exception("Redis XADD failed (sync)")
                raise
        raise RuntimeError("No Redis client available. Install 'redis' package.")

ENQUEUER = RedisStreamEnqueuer(CONFIG.REDIS_URL, CONFIG.REDIS_STREAM, CONFIG.REDIS_MAXLEN)

# Dev-only helper: simulate a token-by-token capture using the new pipeline
def _simulate_capture_trace(trace_id: str = "dev-trace", tokens: int = 8, dim: int = 64, topk: int = 8) -> dict:
    """
    Simulates per-token capture using CapturePipeline and returns an envelope
    with shard keys and metadata. This is not invoked by default server paths.
    """
    if CapturePipeline is None or CaptureConfig is None:
        logger.debug("CapturePipeline unavailable; skipping simulation")
        return {}
    import os as _os, random as _rand
    cfg = CaptureConfig(
        model_name=_os.getenv("DEV_MODEL_NAME", "dev-model"),
        model_hash=_os.getenv("DEV_MODEL_HASH", "dev-hash"),
        topk=int(topk),
        window_size=int(_os.getenv("DEV_WINDOW_SIZE", "16")),
        compress=_os.getenv("DEV_COMPRESS", "json"),
        namespace=_os.getenv("DEV_NAMESPACE", "activations"),
    )
    pipe = CapturePipeline(cfg)
    pipe.start_trace(trace_id, params={"dim": dim, "tokens": tokens})
    hook = register_layer_hook(pipe, trace_id) if register_layer_hook is not None else None

    start_ns = now_ns()
    for t in range(tokens):
        # sparse ~5% non-zeros synthetic activation vector
        v = [0.0] * int(dim)
        nnz = max(1, int(dim * 0.05))
        for _ in range(nnz):
            idx = _rand.randrange(dim)
            v[idx] = _rand.random() * 2 - 1
        if hook:
            hook(t, v)
        else:
            pipe.capture_token(trace_id, t, v)
    env = pipe.flush(trace_id)
    env["capture_ms_total"] = ns_to_ms(now_ns() - start_ns)
    return env

def _validate_payload(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return "invalid json"
    if "request_id" not in payload or "response" not in payload:
        return "missing required fields: request_id and response"
    return None

# FastAPI/Starlette implementation (preferred if available)
if FastAPI is not None:
    app = FastAPI()

    @app.post("/ingest")
    async def ingest(request: Request):
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid json"}, status_code=400)

        err = _validate_payload(payload)
        if err:
            return JSONResponse({"error": err}, status_code=400)

        try:
            xadd_id = await ENQUEUER.enqueue(payload)
        except Exception:
            return JSONResponse({"error": "enqueue_failed"}, status_code=500)

        logger.info("enqueued request_id=%s stream=%s xadd_id=%s", payload.get("request_id"), ENQUEUER.stream, xadd_id)
        return JSONResponse({"status": "enqueued", "xadd_id": str(xadd_id)}, status_code=202)
else:
    app = None  # type: ignore

# Minimal fallback HTTP server using http.server
class IngestHandler(BaseHTTPRequestHandler):
    server_version = "InterceptorIngest/0.1"

    def _write_json(self, status: int, obj: dict) -> None:
        data = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/ingest":
            self._write_json(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except Exception:
            length = 0
        body = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            self._write_json(400, {"error": "invalid json"})
            return

        err = _validate_payload(payload)
        if err:
            self._write_json(400, {"error": err})
            return

        try:
            xadd_id = ENQUEUER.enqueue_blocking(payload)
        except Exception:
            self._write_json(500, {"error": "enqueue_failed"})
            return

        logger.info("enqueued request_id=%s stream=%s xadd_id=%s", payload.get("request_id"), ENQUEUER.stream, xadd_id)
        self._write_json(202, {"status": "enqueued", "xadd_id": str(xadd_id)})

    # Silence default stderr logging; route to logger instead
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401
        try:
            logger.info("%s - %s", self.address_string(), fmt % args)
        except Exception:
            pass


def run(host: str, port: int) -> None:
    httpd = HTTPServer((host, port), IngestHandler)
    logger.info("Starting fallback HTTP server on %s:%s", host, port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            httpd.server_close()
        except Exception:
            pass

if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOGLEVEL", CONFIG.LOG_LEVEL))
    host = CONFIG.HOST
    port = int(getattr(CONFIG, "PORT", 8080))

    if FastAPI is not None:
        try:
            import uvicorn  # type: ignore
        except Exception:
            uvicorn = None  # type: ignore
        if uvicorn is not None:
            uvicorn.run(app, host=host, port=port)  # type: ignore[arg-type]
        else:
            run(host, port)
    else:
        run(host, port)