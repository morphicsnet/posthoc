from fastapi import FastAPI, Header, HTTPException, BackgroundTasks, Response, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exception_handlers import http_exception_handler
from fastapi import status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time, uuid, asyncio, os, json, logging
from datetime import datetime, timezone

try:
    import httpx  # type: ignore
except Exception:
    httpx = None  # type: ignore

try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore

try:
    import psycopg  # type: ignore
    from psycopg import sql  # type: ignore
except Exception:
    psycopg = None  # type: ignore
    sql = None  # type: ignore

from redis import asyncio as aioredis
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()  # loads .env if present; no-op otherwise
except Exception:
    pass

logger = logging.getLogger(__name__)
# Observability (optional): import setup_otel and JSON logger helper
try:
    from services.gateway.src.otel import setup_otel as _setup_otel, get_logger as _otel_get_logger  # type: ignore
except Exception:
    try:
        from otel import setup_otel as _setup_otel, get_logger as _otel_get_logger  # type: ignore
    except Exception:
        _setup_otel = None  # type: ignore

        def _otel_get_logger(name: str = "gateway"):
            return logging.getLogger(name)

# RBAC and rate limiting dependencies
try:
    from services.gateway.src.rbac import AuthContext, rbac_dependency  # type: ignore
except Exception:
    try:
        from rbac import AuthContext, rbac_dependency  # type: ignore
    except Exception:
        AuthContext = None  # type: ignore

        def rbac_dependency(required_scopes):  # type: ignore
            async def _dep(request: Request):
                class _Anon:
                    tenant_id = "anon"
                    scopes = set()
                    subject = "anonymous"
                try:
                    request.state.auth_ctx = _Anon()
                except Exception:
                    pass
                return request.state.auth_ctx
            return _dep

try:
    from services.gateway.src.rate_limit import rate_limit_dependency  # type: ignore
except Exception:
    try:
        from rate_limit import rate_limit_dependency  # type: ignore
    except Exception:
        def rate_limit_dependency(category: str):  # type: ignore
            async def _noop_dep(request: Request):
                return None
            return _noop_dep

# Optional status store (shared abstraction from explainer)
try:
    from services.explainer.src.status_store import (  # type: ignore
        get_status_store_from_env,
        StatusStore as _StatusStore,
        TraceStatusItem as _TraceStatusItem,
    )
except Exception:
    try:
        from status_store import (  # type: ignore
            get_status_store_from_env,
            StatusStore as _StatusStore,
            TraceStatusItem as _TraceStatusItem,
        )
    except Exception:
        get_status_store_from_env = None  # type: ignore
        _StatusStore = None  # type: ignore
        _TraceStatusItem = dict  # type: ignore

class Config:
    def __init__(self) -> None:
        self.LLM_PROXY_URL: Optional[str] = os.getenv("LLM_PROXY_URL")
        self.DEFAULT_PROVIDER: str = os.getenv("DEFAULT_PROVIDER", "openai")
        self.REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
        self.REDIS_STREAM: str = os.getenv("REDIS_STREAM", "hypergraph:completions")
        # Optional int
        try:
            raw = (os.getenv("REDIS_MAXLEN") or "").strip()
            self.REDIS_MAXLEN: Optional[int] = int(raw) if raw else None
        except Exception:
            self.REDIS_MAXLEN = None
        self.GATEWAY_REDACT: str = os.getenv("GATEWAY_REDACT", "0")
        # DB config (defaults preserved)
        self.DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
        self.EXPLAINER_TABLE: str = os.getenv("EXPLAINER_TABLE", "explanations_v2")
        try:
            self.DB_CONNECT_TIMEOUT: int = int(os.getenv("DB_CONNECT_TIMEOUT", "5"))
        except Exception:
            self.DB_CONNECT_TIMEOUT = 5
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
        # Optional provider API key (presence only)
        self.PROVIDER_API_KEY: Optional[str] = os.getenv("PROVIDER_API_KEY")

    @staticmethod
    def _mask_db_url(url: Optional[str]) -> Optional[str]:
        if not url or not isinstance(url, str):
            return url
        try:
            import re as _re
            return _re.sub(r"//([^:/?#]+):([^@]+)@", r"//\1:****@", url)
        except Exception:
            return url

    def safe_dict(self) -> Dict[str, Any]:
        return {
            "LLM_PROXY_URL": self.LLM_PROXY_URL,
            "DEFAULT_PROVIDER": self.DEFAULT_PROVIDER,
            "REDIS_URL": self.REDIS_URL,
            "REDIS_STREAM": self.REDIS_STREAM,
            "REDIS_MAXLEN": self.REDIS_MAXLEN,
            "GATEWAY_REDACT": self.GATEWAY_REDACT,
            "DATABASE_URL": self._mask_db_url(self.DATABASE_URL),
            "EXPLAINER_TABLE": self.EXPLAINER_TABLE,
            "DB_CONNECT_TIMEOUT": self.DB_CONNECT_TIMEOUT,
            "LOG_LEVEL": self.LOG_LEVEL,
            "PROVIDER_API_KEY": "<set>" if self.PROVIDER_API_KEY else "<unset>",
        }

    def safe_repr(self) -> str:
        try:
            return json.dumps(self.safe_dict(), separators=(",", ":"), sort_keys=True)
        except Exception:
            return str(self.safe_dict())

CONFIG = Config()

# Status backend helpers
_STATUS_STORE: Optional[_StatusStore] = None

def _status_backend_enabled() -> bool:
    b = (os.getenv("STATUS_BACKEND", "") or "").strip().lower()
    return b in ("json", "ddb")

def _get_status_store() -> Optional[_StatusStore]:
    global _STATUS_STORE
    if not _status_backend_enabled() or get_status_store_from_env is None:
        return None
    if _STATUS_STORE is None:
        try:
            _STATUS_STORE = get_status_store_from_env()  # type: ignore
        except Exception:
            _STATUS_STORE = None
    return _STATUS_STORE

def _ttl_days() -> int:
    try:
        d = int(os.getenv("TRACE_TTL_DAYS", "7"))
        return d if d > 0 else 7
    except Exception:
        return 7

_redis = None

async def get_redis():
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(CONFIG.REDIS_URL, encoding="utf-8", decode_responses=True)
    return _redis

def redact_messages(messages):
    if CONFIG.GATEWAY_REDACT == "1":
        red = []
        for m in messages or []:
            if isinstance(m, dict):
                role = m.get("role", "")
                content = m.get("content", "")
                red.append({"role": role, "len": len(content) if isinstance(content, str) else 0})
        return red
    return messages

def passthrough_headers(in_headers: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in in_headers.items():
        lk = k.lower()
        if lk in ("authorization", "content-type") or lk.startswith("x-"):
            out[k] = v
    return out

def forward_response_headers(upstream_headers: Dict[str, str]) -> Dict[str, str]:
    exclude = {"content-length", "transfer-encoding", "connection"}
    out: Dict[str, str] = {}
    for k, v in upstream_headers.items():
        if k.lower() in exclude:
            continue
        out[k] = v
    return out

async def xadd_enqueue(payload: Dict[str, Any]) -> Optional[str]:
    try:
        redis = await get_redis()
        fields = {"data": json.dumps(payload, ensure_ascii=False)}
        if CONFIG.REDIS_MAXLEN is not None:
            return await redis.xadd(CONFIG.REDIS_STREAM, fields, maxlen=CONFIG.REDIS_MAXLEN, approximate=True)
        else:
            return await redis.xadd(CONFIG.REDIS_STREAM, fields)
    except Exception as e:
        logger.exception("Redis enqueue failed: %s", e)
        return None

async def enqueue_and_log(payload: Dict[str, Any], req_id: str, provider: str, model: Optional[str]) -> None:
    try:
        xadd_id = await xadd_enqueue(payload)
        logger.info(
            "enqueue completion request_id=%s provider=%s model=%s xadd_id=%s",
            req_id, provider, model, xadd_id
        )
    except Exception as e:
        logger.exception("enqueue_and_log failed: %s", e)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionsRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = True
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: Optional[int] = None


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ExplanationMetadata(BaseModel):
    trace_id: str
    status: str = "processing"
    estimated_wait: Optional[str] = None
    stream_endpoint: str
    granularity: Optional[str] = None
    featureset: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    choices: List[ChatChoice]
    usage: Optional[Usage] = None
    explanation_metadata: ExplanationMetadata


class TraceStatus(BaseModel):
    trace_id: str
    state: str
    progress: Optional[float] = 0.0
    stage: Optional[str] = None
    updated_at: float = Field(default_factory=lambda: time.time())
    s3_key: Optional[str] = None
    error: Optional[str] = None
    granularity: Optional[str] = None
    featureset: Optional[str] = None


class HIFNode(BaseModel):
    id: str
    type: str
    label: Optional[str] = None
    layer: Optional[int] = None
    position: Optional[int] = None
    activation_strength: Optional[float] = None
    attributes: Optional[Dict[str, Any]] = None


class HIFIncidence(BaseModel):
    id: str
    node_ids: List[str]
    weight: float
    metadata: Optional[Dict[str, Any]] = None
    attributes: Optional[Dict[str, Any]] = None


class HIFMeta(BaseModel):
    model_name: Optional[str] = None
    model_hash: Optional[str] = None
    sae_dictionary: Optional[str] = None
    granularity: Optional[str] = None
    created_at: float = Field(default_factory=lambda: time.time())
    limits: Optional[Dict[str, Any]] = None
    version: str = "hif-1"


class HIFGraph(BaseModel):
    network_type: str = Field(alias="network-type", default="directed")
    nodes: List[HIFNode]
    incidences: List[HIFIncidence]
    meta: Optional[HIFMeta] = None

    class Config:
        populate_by_name = True


app = FastAPI(title="Hypergraph Gateway")

# --- Security Audit Logger and PII scrubbing (see docs/security/SECURITY.md) ---
try:
    from libs.sanitize.pii import sanitize_message as _sanitize_message  # type: ignore
except Exception:
    def _sanitize_message(text: str) -> str:  # fallback no-op
        try:
            return text if isinstance(text, str) else str(text)
        except Exception:
            return ""

_AUDIT_ENABLED = (os.getenv("AUDIT_LOG_ENABLE", "0") or "0") == "1"
_AUDIT_PATH = os.getenv("AUDIT_LOG_PATH", "/var/log/hypergraph/audit.log")
_AUDIT_FH = None  # type: ignore
_AUDIT_USING_STDOUT = False

def _audit_init() -> None:
    global _AUDIT_FH, _AUDIT_USING_STDOUT
    if not _AUDIT_ENABLED:
        return
    try:
        d = os.path.dirname(_AUDIT_PATH) or "."
        os.makedirs(d, exist_ok=True)
        _AUDIT_FH = open(_AUDIT_PATH, "a", encoding="utf-8")
    except Exception:
        _AUDIT_FH = None
        _AUDIT_USING_STDOUT = True

def emit_audit(
    event: str,
    request: Request,
    tenant_id: str,
    trace_id: Optional[str] = None,
    subject: Optional[str] = None,
    scopes: Optional[List[str]] = None,
    status: Optional[int] = None,
    reason: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if not _AUDIT_ENABLED:
        return
    try:
        payload = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "service": "gateway",
            "event": event,
            "tenant_id": str(tenant_id or "anon"),
            "trace_id": trace_id,
            "subject": subject,
            "scopes": sorted(list(scopes or [])),
            "path": str(getattr(request.url, "path", "")),
            "method": str(getattr(request, "method", "")),
            "status": int(status) if status is not None else None,
        }
        if reason:
            payload["reason"] = str(reason)
        if isinstance(extra, dict) and extra:
            # Keep cardinality low; attach small scalar fields only
            for k, v in list(extra.items())[:8]:
                if isinstance(v, (str, int, float, bool)) or v is None:
                    payload[k] = v if (k not in ("authorization", "Authorization")) else None
        line = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        if _AUDIT_FH is not None:
            _AUDIT_FH.write(line + "\n")
            _AUDIT_FH.flush()
        else:
            print(line)
    except Exception:
        pass

# initialize audit sink on import
try:
    _audit_init()
except Exception:
    pass
# Initialize observability (metrics, tracing, JSON logs) if enabled
try:
    if os.getenv("ENABLE_OTEL", "0") == "1" and _setup_otel is not None:
        _setup_otel(app, service_name="gateway")
except Exception:
    pass
try:
    logger.info("config: %s", CONFIG.safe_repr())
except Exception:
    pass

# ----------------------------
# Minimal Postgres read layer (psycopg v3)
# ----------------------------
DATABASE_URL = CONFIG.DATABASE_URL
DB_CONNECT_TIMEOUT = CONFIG.DB_CONNECT_TIMEOUT
EXPLAINER_TABLE = CONFIG.EXPLAINER_TABLE
_db_conn = None  # module-level singleton

def get_db():
    """
    Lazy connection helper; autocommit with liveness checks.
    """
    global _db_conn
    if psycopg is None:
        logger.warning("psycopg not available; DB reads disabled")
        return None
    if not DATABASE_URL:
        logger.warning("DATABASE_URL not set; DB reads disabled")
        return None
    try:
        if _db_conn is None or getattr(_db_conn, "closed", True):
            _db_conn = psycopg.connect(CONFIG.DATABASE_URL, connect_timeout=CONFIG.DB_CONNECT_TIMEOUT)
            _db_conn.autocommit = True
        else:
            try:
                with _db_conn.cursor() as cur:
                    cur.execute("SELECT 1;")
            except Exception:
                try:
                    _db_conn.close()
                except Exception:
                    pass
                _db_conn = psycopg.connect(DATABASE_URL, connect_timeout=DB_CONNECT_TIMEOUT)
                _db_conn.autocommit = True
    except Exception as e:
        logger.error(f"DB connect error: {e}")
        _db_conn = None
    return _db_conn

def ensure_table_exists() -> None:
    """
    Ensure explanations table exists to avoid startup races.
    """
    conn = get_db()
    if not conn or sql is None:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {table} (
                      request_id TEXT PRIMARY KEY,
                      status TEXT NOT NULL,
                      provider TEXT,
                      model TEXT,
                      created_at TIMESTAMPTZ,
                      updated_at TIMESTAMPTZ DEFAULT NOW(),
                      hypergraph JSONB
                    );
                    """
                ).format(table=sql.Identifier(CONFIG.EXPLAINER_TABLE))
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {idx} ON {table}(status);").format(
                    idx=sql.Identifier(f"{CONFIG.EXPLAINER_TABLE}_status_idx"),
                    table=sql.Identifier(CONFIG.EXPLAINER_TABLE),
                )
            )
        logger.info(f"Ensured explanation table exists: {CONFIG.EXPLAINER_TABLE}")
    except Exception as e:
        logger.error(f"ensure_table_exists error: {e}")

# App startup ensure
try:
    ensure_table_exists()
except Exception as _e:
    logger.error(f"ensure_table_exists failed at startup: {_e}")

TRACE_STATUS: Dict[str, TraceStatus] = {}
TRACE_GRAPH: Dict[str, HIFGraph] = {}
WEBHOOKS: Dict[str, List[Dict[str, Any]]] = {}


def new_trace_id() -> str:
    return "trc_" + uuid.uuid4().hex[:12]


async def simulate_explanation(trace_id: str, granularity: str, featureset: str):
    # Simulate staged progress and produce a tiny HIF graph
    for pct, stage, delay in [(5, "queued", 0.05), (25, "fetch", 0.05), (60, "decode", 0.1), (85, "attribution", 0.1)]:
        st = TRACE_STATUS.get(trace_id)
        if not st or st.state in ("canceled", "failed"):
            return
        st.state = "running"
        st.stage = stage
        st.progress = pct
        st.updated_at = time.time()
        await asyncio.sleep(delay)
    # Build a tiny graph
    graph = HIFGraph(
        nodes=[
            HIFNode(id="feat_1024", type="sae_feature", label="Geography: Cities", layer=12, activation_strength=4.5),
            HIFNode(id="token_5", type="input_token", label="Paris", position=5),
            HIFNode(id="token_out_1", type="output_token", label="Paris is the capital of France.", position=1)
        ],
        incidences=[
            HIFIncidence(id="e1", node_ids=["feat_1024", "token_out_1"], weight=0.85, metadata={"type": "causal_circuit"})
        ],
        meta=HIFMeta(granularity=granularity, sae_dictionary=featureset, version="hif-1")
    )
    TRACE_GRAPH[trace_id] = graph
    st = TRACE_STATUS.get(trace_id)
    if st:
        st.state = "complete"
        st.stage = "complete"
        st.progress = 100.0
        st.updated_at = time.time()


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: Request,
    background: BackgroundTasks,
    x_explain_mode: Optional[str] = Header(default=None, alias="x-explain-mode"),
    x_explain_granularity: Optional[str] = Header(default="sentence", alias="x-explain-granularity"),
    x_explain_features: Optional[str] = Header(default="sae-gpt4-2m", alias="x-explain-features"),
    x_explain_budget: Optional[int] = Header(default=None, alias="x-explain-budget"),
    x_trace_id: Optional[str] = Header(default=None, alias="x-trace-id"),
    x_idempotency_key: Optional[str] = Header(default=None, alias="x-idempotency-key"),
    x_provider: Optional[str] = Header(default=None, alias="x-provider"),
    auth: Any = Depends(rbac_dependency(["traces:write"])),
    _rl: None = Depends(rate_limit_dependency("write")),
):
    # Optional explainer simulation (preserved behavior)
    if x_explain_mode == "hypergraph":
        trace_id = x_trace_id or new_trace_id()
        granularity = x_explain_granularity or "sentence"
        featureset = x_explain_features or "sae-gpt4-2m"
        TRACE_STATUS[trace_id] = TraceStatus(
            trace_id=trace_id,
            state="queued",
            progress=0.0,
            stage="queued",
            granularity=granularity,
            featureset=featureset,
        )
        # Optionally persist initial status to shared store
        try:
            store = _get_status_store()
            if store is not None:
                now = time.time()
                ttl = float(_ttl_days()) * 86400.0
                item: Dict[str, Any] = {
                    "trace_id": trace_id,
                    "tenant_id": getattr(auth, "tenant_id", "anon"),
                    "state": "queued",
                    "progress": 0.0,
                    "stage": "queued",
                    "granularity": granularity,
                    "featureset": featureset,
                    "created_at": now,
                    "updated_at": now,
                    "expires_at": now + ttl,
                }
                store.put_status(item)  # type: ignore
        except Exception:
            pass
        # audit queued state (low-cardinality, scrubbed)
        try:
            emit_audit(
                event="trace.queued",
                request=request,
                tenant_id=getattr(auth, "tenant_id", "anon"),
                trace_id=trace_id,
                subject=getattr(auth, "subject", None),
                scopes=sorted(list(getattr(auth, "scopes", set()))),
                status=None,
                reason=None,
                extra={"granularity": granularity, "featureset": featureset},
            )
        except Exception:
            pass
        # schedule async simulation without blocking proxy behavior
        asyncio.create_task(simulate_explanation(trace_id, granularity, featureset))

    if not CONFIG.LLM_PROXY_URL:
        return JSONResponse(status_code=500, content={"error": "LLM_PROXY_URL is not configured"})

    # Accept generic JSON body and forward as-is
    try:
        body: Dict[str, Any] = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})
 
    # Audit: user submission when explain mode is enabled (PII-scrubbed, snippet only)
    try:
        if _AUDIT_ENABLED and x_explain_mode:
            snippet = ""
            try:
                for m in (body.get("messages") or []):
                    if isinstance(m, dict) and (m.get("role") or "").lower() == "user":
                        c = m.get("content")
                        if isinstance(c, str) and c:
                            snippet = _sanitize_message(c)[:64]
                            break
            except Exception:
                snippet = ""
            emit_audit(
                event="chat.submit",
                request=request,
                tenant_id=getattr(auth, "tenant_id", "anon"),
                trace_id=x_trace_id or None,
                subject=getattr(auth, "subject", None),
                scopes=sorted(list(getattr(auth, "scopes", set()))),
                status=None,
                reason=None,
                extra={"mode": x_explain_mode, "model": body.get("model"), "snippet": snippet},
            )
    except Exception:
        pass
 
    upstream_url = f"{CONFIG.LLM_PROXY_URL.rstrip('/')}/v1/chat/completions"
    headers_out = passthrough_headers(dict(request.headers))
    stream = bool(body.get("stream", False))
    model = body.get("model")
    messages = body.get("messages") or []
    provider = x_provider or CONFIG.DEFAULT_PROVIDER

    start_time = time.perf_counter()

    # Prefer httpx if available
    if httpx is not None:
        if stream:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", upstream_url, headers=headers_out, json=body) as upstream:
                    status = upstream.status_code
                    ctype = upstream.headers.get("content-type", "text/event-stream")
                    headers_back = forward_response_headers(dict(upstream.headers))
                    if status >= 400:
                        body_bytes = await upstream.aread()
                        return Response(content=body_bytes, status_code=status, media_type=ctype, headers=headers_back)

                    state: Dict[str, Any] = {"upstream_id": None, "assembled_text": ""}

                    async def agen():
                        buf = ""
                        try:
                            async for chunk in upstream.aiter_raw():
                                # Forward chunk to client immediately
                                yield chunk
                                # Best-effort parse SSE lines to accumulate content and id
                                try:
                                    s = chunk.decode("utf-8", errors="ignore")
                                    buf += s
                                    while True:
                                        nl = buf.find("\n")
                                        if nl == -1:
                                            break
                                        line = buf[:nl]
                                        buf = buf[nl + 1 :]
                                        ls = line.strip()
                                        if not ls:
                                            continue
                                        if ls.startswith("data:"):
                                            data_part = ls[5:].strip()
                                            if data_part == "[DONE]":
                                                continue
                                            try:
                                                obj = json.loads(data_part)
                                                if isinstance(obj, dict):
                                                    if not state["upstream_id"] and "id" in obj:
                                                        state["upstream_id"] = obj["id"]
                                                    choices = obj.get("choices")
                                                    if isinstance(choices, list):
                                                        for ch in choices:
                                                            delta = ch.get("delta") or {}
                                                            piece = delta.get("content")
                                                            if piece:
                                                                state["assembled_text"] += piece
                                            except Exception:
                                                # ignore malformed chunk
                                                pass
                                except Exception:
                                    # ignore decoding/parsing errors
                                    pass
                        finally:
                            # After streaming completes, enqueue best-effort payload
                            created_at = datetime.now(timezone.utc).isoformat()
                            duration_ms = int((time.perf_counter() - start_time) * 1000)
                            req_id = state["upstream_id"] or str(uuid.uuid4())
                            payload = {
                                "request_id": req_id,
                                "upstream_id": state["upstream_id"],
                                "provider": provider,
                                "model": model,
                                "messages": redact_messages(messages),
                                "response": state["assembled_text"],
                                "created_at": created_at,
                                "trace": {
                                    "stream": True,
                                    "status_code": status,
                                    "duration_ms": duration_ms,
                                },
                            }
                            xadd_id = await xadd_enqueue(payload)
                            logger.info(
                                "enqueue stream completion request_id=%s provider=%s model=%s xadd_id=%s",
                                req_id,
                                provider,
                                model,
                                xadd_id,
                            )

                    return StreamingResponse(agen(), status_code=status, media_type=ctype, headers=headers_back)
        else:
            async with httpx.AsyncClient(timeout=None) as client:
                upstream = await client.post(upstream_url, headers=headers_out, json=body)
                status = upstream.status_code
                ctype = upstream.headers.get("content-type", "application/json")
                content_bytes = upstream.content
                headers_back = forward_response_headers(dict(upstream.headers))
                if status >= 400:
                    return Response(content=content_bytes, status_code=status, media_type=ctype, headers=headers_back)

                # Parse JSON for id if possible
                try:
                    resp_obj = upstream.json()
                except Exception:
                    resp_obj = None

                upstream_id = resp_obj.get("id") if isinstance(resp_obj, dict) else None
                req_id = upstream_id or str(uuid.uuid4())
                created_at = datetime.now(timezone.utc).isoformat()
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                payload = {
                    "request_id": req_id,
                    "upstream_id": upstream_id,
                    "provider": provider,
                    "model": model,
                    "messages": redact_messages(messages),
                    "response": resp_obj if resp_obj is not None else content_bytes.decode("utf-8", errors="ignore"),
                    "created_at": created_at,
                    "trace": {"stream": False, "status_code": status, "duration_ms": duration_ms},
                }
                # Best-effort enqueue with xadd id logging
                asyncio.create_task(enqueue_and_log(payload, req_id, provider, model))
                return Response(content=content_bytes, status_code=status, media_type=ctype, headers=headers_back)

    # Fallback to requests if httpx unavailable
    if requests is None:
        return JSONResponse(status_code=500, content={"error": "No HTTP client available (httpx/requests missing)"})

    if stream:
        def do_request():
            return requests.post(upstream_url, headers=headers_out, json=body, stream=True)

        r = await asyncio.to_thread(do_request)
        status = r.status_code
        ctype = r.headers.get("content-type", "text/event-stream")
        headers_back = forward_response_headers(dict(r.headers))
        if status >= 400:
            content = await asyncio.to_thread(lambda: r.content)
            return Response(content=content, status_code=status, media_type=ctype, headers=headers_back)

        state: Dict[str, Any] = {"upstream_id": None, "assembled_text": ""}

        def gen():
            buf = ""
            for chunk in r.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                # Forward chunk
                yield chunk
                # Parse best-effort
                try:
                    s = chunk.decode("utf-8", errors="ignore")
                    buf += s
                    while True:
                        nl = buf.find("\n")
                        if nl == -1:
                            break
                        line = buf[:nl]
                        buf = buf[nl + 1 :]
                        ls = line.strip()
                        if not ls:
                            continue
                        if ls.startswith("data:"):
                            data_part = ls[5:].strip()
                            if data_part == "[DONE]":
                                continue
                            try:
                                obj = json.loads(data_part)
                                if isinstance(obj, dict):
                                    if not state["upstream_id"] and "id" in obj:
                                        state["upstream_id"] = obj["id"]
                                    choices = obj.get("choices")
                                    if isinstance(choices, list):
                                        for ch in choices:
                                            delta = ch.get("delta") or {}
                                            piece = delta.get("content")
                                            if piece:
                                                state["assembled_text"] += piece
                            except Exception:
                                pass
                except Exception:
                    pass

        def enqueue_after():
            try:
                req_id = state["upstream_id"] or str(uuid.uuid4())
                created_at = datetime.now(timezone.utc).isoformat()
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                payload = {
                    "request_id": req_id,
                    "upstream_id": state["upstream_id"],
                    "provider": provider,
                    "model": model,
                    "messages": redact_messages(messages),
                    "response": state["assembled_text"],
                    "created_at": created_at,
                    "trace": {"stream": True, "status_code": status, "duration_ms": duration_ms},
                }
                loop = asyncio.get_running_loop()
                loop.create_task(enqueue_and_log(payload, req_id, provider, model))
            except Exception as e:
                logger.exception("enqueue_after failed: %s", e)

        # run enqueue after response fully streamed
        background.add_task(enqueue_after)
        return StreamingResponse(gen(), status_code=status, media_type=ctype, headers=headers_back, background=background)
    else:
        def do_request():
            return requests.post(upstream_url, headers=headers_out, json=body)

        r = await asyncio.to_thread(do_request)
        status = r.status_code
        ctype = r.headers.get("content-type", "application/json")
        content = await asyncio.to_thread(lambda: r.content)
        headers_back = forward_response_headers(dict(r.headers))
        if status >= 400:
            return Response(content=content, status_code=status, media_type=ctype, headers=headers_back)

        try:
            resp_obj = r.json()
        except Exception:
            resp_obj = None

        upstream_id = resp_obj.get("id") if isinstance(resp_obj, dict) else None
        req_id = upstream_id or str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        payload = {
            "request_id": req_id,
            "upstream_id": upstream_id,
            "provider": provider,
            "model": model,
            "messages": redact_messages(messages),
            "response": resp_obj if resp_obj is not None else content.decode("utf-8", errors="ignore"),
            "created_at": created_at,
            "trace": {"stream": False, "status_code": status, "duration_ms": duration_ms},
        }
        asyncio.create_task(enqueue_and_log(payload, req_id, provider, model))
        return Response(content=content, status_code=status, media_type=ctype, headers=headers_back)


@app.get("/v1/traces/{trace_id}/status", response_model=TraceStatus)
async def get_trace_status(
    trace_id: str,
    auth: Any = Depends(rbac_dependency(["traces:read"])),
    _rl: None = Depends(rate_limit_dependency("read")),
):
    # Prefer shared status store if configured
    try:
        store = _get_status_store()
    except Exception:
        store = None
    if store is not None:
        try:
            item = store.get_status(trace_id)  # type: ignore
        except Exception:
            item = None
        if isinstance(item, dict):
            # Map to response model
            return TraceStatus(
                trace_id=str(item.get("trace_id") or trace_id),
                state=str(item.get("state") or "unknown"),
                progress=float(item.get("progress") or 0.0),
                stage=item.get("stage"),
                updated_at=float(item.get("updated_at") or time.time()),
                s3_key=item.get("s3_key"),
                error=item.get("error"),
                granularity=item.get("granularity"),
                featureset=item.get("featureset"),
            )
    # Fallback to in-memory legacy behavior
    st = TRACE_STATUS.get(trace_id)
    if not st:
        raise HTTPException(status_code=404, detail="Trace not found")
    return st


@app.get("/v1/traces/{trace_id}/graph")
async def get_trace_graph(
    trace_id: str,
    auth: Any = Depends(rbac_dependency(["traces:read"])),
    _rl: None = Depends(rate_limit_dependency("read")),
):
    # Prefer shared status store to validate existence/expiry
    item: Optional[Dict[str, Any]] = None
    try:
        store = _get_status_store()
        if store is not None:
            try:
                item = store.get_status(trace_id)  # type: ignore
            except Exception:
                item = None
    except Exception:
        item = None

    if trace_id not in TRACE_STATUS and item is None:
        raise HTTPException(status_code=404, detail="Trace not found")

    # Expiry check (only applicable when STATUS_BACKEND in use)
    try:
        if item is not None:
            now = time.time()
            exp = float(item.get("expires_at") or 0.0)  # type: ignore
            if (item.get("state") == "expired") or (exp and exp < now):
                raise HTTPException(status_code=410, detail="Trace expired")
    except Exception:
        pass

    # Also handle in-memory expiry state
    st_local = TRACE_STATUS.get(trace_id)
    if st_local and getattr(st_local, "state", None) == "expired":
        raise HTTPException(status_code=410, detail="Trace expired")
    graph = TRACE_GRAPH.get(trace_id)
    if not graph:
        # Not ready; return 404 to encourage polling
        raise HTTPException(status_code=404, detail="Graph not ready")
    # Use by_alias to emit "network-type"
    return JSONResponse(content=graph.model_dump(by_alias=True))


@app.get("/v1/traces/{trace_id}/stream")
async def stream_trace(
    trace_id: str,
    auth: Any = Depends(rbac_dependency(["traces:read"])),
    _rl: None = Depends(rate_limit_dependency("read")),
):
    # 404 if unknown trace_id (prefer StatusStore when available)
    exists = trace_id in TRACE_STATUS
    item: Optional[Dict[str, Any]] = None
    try:
        store = _get_status_store()
        if store is not None:
            try:
                item = store.get_status(trace_id)  # type: ignore
            except Exception:
                item = None
    except Exception:
        item = None

    if not exists and item is None:
        raise HTTPException(status_code=404, detail="Trace not found")

    async def event_gen():
        try:
            if item is not None:
                payload = {
                    "trace_id": str(item.get("trace_id") or trace_id),
                    "state": str(item.get("state") or "unknown"),
                    "progress": float(item.get("progress") or 0.0),
                    "stage": item.get("stage"),
                    "updated_at": float(item.get("updated_at") or time.time()),
                    "s3_key": item.get("s3_key"),
                    "error": item.get("error"),
                    "granularity": item.get("granularity"),
                    "featureset": item.get("featureset"),
                }
                state_val = str(item.get("state") or "")
            else:
                st = TRACE_STATUS.get(trace_id)
                payload = st.model_dump() if st else {"trace_id": trace_id, "state": "unknown"}
                state_val = st.state if st else ""
            yield "event: status_update\ndata: " + json.dumps(payload) + "\n\n"
            if state_val == "complete" and trace_id in TRACE_GRAPH:
                yield "event: complete\ndata: " + json.dumps({"trace_id": trace_id}) + "\n\n"
        except Exception:
            return

    return StreamingResponse(event_gen(), media_type="text/event-stream")


class WebhookRegistration(BaseModel):
    url: str
    secret: Optional[str] = None
    events: Optional[List[str]] = None


@app.post("/v1/traces/{trace_id}/webhooks", status_code=201)
async def register_webhook(
    trace_id: str,
    request: Request,
    body: WebhookRegistration,
    auth: Any = Depends(rbac_dependency(["traces:write"])),
    _rl: None = Depends(rate_limit_dependency("write")),
):
    # Require a known trace_id (status store or in-memory)
    exists = trace_id in TRACE_STATUS
    if not exists:
        try:
            store = _get_status_store()
            if store is not None:
                try:
                    exists = store.get_status(trace_id) is not None  # type: ignore
                except Exception:
                    exists = False
        except Exception:
            exists = False
    if not exists:
        raise HTTPException(status_code=404, detail="Trace not found")

    WEBHOOKS.setdefault(trace_id, []).append(body.model_dump())
    try:
        emit_audit(
            event="webhook.register",
            request=request,
            tenant_id=getattr(auth, "tenant_id", "anon"),
            trace_id=trace_id,
            subject=getattr(auth, "subject", None),
            scopes=sorted(list(getattr(auth, "scopes", set()))),
            status=201,
            reason=None,
            extra={"has_secret": bool(body.secret), "events_count": len(body.events or [])},
        )
    except Exception:
        pass
    return {"id": uuid.uuid4().hex[:8], "trace_id": trace_id, "url": body.url}


@app.delete("/v1/traces/{trace_id}")
async def cancel_trace(
    trace_id: str,
    request: Request,
    auth: Any = Depends(rbac_dependency(["traces:write"])),
    _rl: None = Depends(rate_limit_dependency("write")),
):
    # Check backing store first (if enabled)
    store_item: Optional[Dict[str, Any]] = None
    store = None
    try:
        store = _get_status_store()
        if store is not None:
            try:
                store_item = store.get_status(trace_id)  # type: ignore
            except Exception:
                store_item = None
    except Exception:
        store_item = None

    st = TRACE_STATUS.get(trace_id)
    if not st and store_item is None:
        raise HTTPException(status_code=404, detail="Trace not found")

    current_state: Optional[str] = None
    if isinstance(store_item, dict):
        current_state = str(store_item.get("state") or "")
    elif st:
        current_state = st.state

    if current_state in ("complete", "expired", "failed", "canceled"):
        raise HTTPException(status_code=409, detail="Cannot cancel")

    # Update in-memory status
    if st:
        st.state = "canceled"
        st.stage = "canceled"
        st.updated_at = time.time()
    else:
        TRACE_STATUS[trace_id] = TraceStatus(
            trace_id=trace_id,
            state="canceled",
            stage="canceled",
            progress=0.0,
            updated_at=time.time(),
        )

    # Best-effort store update
    try:
        if store is not None:
            store.update_fields(trace_id, state="canceled", stage="canceled")  # type: ignore
    except Exception:
        pass

    st_out = TRACE_STATUS.get(trace_id) or TraceStatus(trace_id=trace_id, state="canceled", stage="canceled")
    try:
        emit_audit(
            event="trace.cancel",
            request=request,
            tenant_id=getattr(auth, "tenant_id", "anon"),
            trace_id=trace_id,
            subject=getattr(auth, "subject", None),
            scopes=sorted(list(getattr(auth, "scopes", set()))),
            status=202,
            reason=None,
            extra=None,
        )
    except Exception:
        pass
    return JSONResponse(content=st_out.model_dump(), status_code=202)

# Audit handler for RBAC denials and rate-limit rejections
@app.exception_handler(HTTPException)
async def _audit_http_exception_handler(request: Request, exc: HTTPException):
    try:
        ctx = getattr(request.state, "auth_ctx", None)
        tenant_id = getattr(ctx, "tenant_id", "anon") if ctx is not None else "anon"
        subject = getattr(ctx, "subject", None) if ctx is not None else None
        scopes = sorted(list(getattr(ctx, "scopes", set()))) if ctx is not None else []
        event = None
        if exc.status_code in (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN):
            event = "rbac.deny"
        elif exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            event = "rate_limited"
        if event:
            reason = None
            try:
                if isinstance(exc.detail, dict):
                    reason = exc.detail.get("code") or exc.detail.get("message")
            except Exception:
                reason = None
            emit_audit(
                event=event,
                request=request,
                tenant_id=tenant_id,
                trace_id=request.headers.get("x-trace-id"),
                subject=subject,
                scopes=scopes,
                status=exc.status_code,
                reason=reason,
                extra=None,
            )
    except Exception:
        pass
    return await http_exception_handler(request, exc)


@app.get("/v1/chat/completions/{id}/explanation")
async def get_completion_explanation(id: str, format: Optional[str] = "hif"):
    """
    Retrieve explanation hypergraph by request_id from Postgres.
    - 404 if not found
    - 202 if not completed
    - 200 with envelope if completed
    """
    if psycopg is None:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "db error"}},
            headers={"Content-Type": "application/json"},
        )
    try:
        conn = get_db()
        if conn is None:
            return JSONResponse(
                status_code=500,
                content={"error": {"message": "db error"}},
                headers={"Content-Type": "application/json"},
            )
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "SELECT request_id, status, provider, model, created_at, hypergraph FROM {table} WHERE request_id=%s"
                ).format(table=sql.Identifier(CONFIG.EXPLAINER_TABLE)),
                (id,),
            )
            row = cur.fetchone()
        if not row:
            return JSONResponse(
                status_code=404,
                content={"error": {"message": "not found"}},
                headers={"Content-Type": "application/json"},
            )
        request_id, status_val, provider_val, model_val, created_at_val, hypergraph_val = row
        logger.info(f"DB fetch explanation request_id={id} status={status_val}")
        if status_val != "completed":
            return JSONResponse(
                status_code=202,
                content={"id": id, "status": "pending", "eta_seconds": 0},
                headers={"Content-Type": "application/json"},
            )
        resp: Dict[str, Any] = {"id": id, "status": "completed"}
        if provider_val:
            resp["provider"] = provider_val
        if model_val:
            resp["model"] = model_val
        if created_at_val:
            try:
                if hasattr(created_at_val, "isoformat"):
                    resp["created_at"] = created_at_val.isoformat()
                elif isinstance(created_at_val, str):
                    resp["created_at"] = created_at_val
            except Exception:
                pass
        # Hypergraph JSON as stored
        if hypergraph_val is not None:
            if isinstance(hypergraph_val, (dict, list)):
                resp["hypergraph"] = hypergraph_val
            else:
                try:
                    resp["hypergraph"] = json.loads(hypergraph_val)
                except Exception:
                    resp["hypergraph"] = None
        else:
            resp["hypergraph"] = None
        return JSONResponse(status_code=200, content=resp, headers={"Content-Type": "application/json"})
    except Exception as e:
        logger.exception("DB error during explanation fetch: %s", e)
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "db error"}},
            headers={"Content-Type": "application/json"},
        )

# TODO(version2): add connection pooling and retry/backoff for DB.
# TODO(version2): enforce JSON schema validation before writing/serving (use libs/hif/validator.py)

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}