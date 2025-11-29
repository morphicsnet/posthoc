"""
Interaction Engine Worker (Redis Streams, Async)

Implements a resilient consumer that:
- Connects to Redis Streams via env vars
- Ensures a consumer group exists
- Blocks on XREADGROUP, parses JSON payload in 'data' field
- Runs placeholder pipeline stages (concept extraction, group testing, verification)
- Builds a minimal HIF v2 hypergraph stub when needed
- Acknowledges messages on success and on parse errors (to avoid poison pill loops)
- Backoffs on Redis connection failures (capped)
- Cleanly shuts down on SIGINT/SIGTERM

Notes:
- TODO(version2): Integrate GLiNER/Spacy for concept extraction.
- TODO(version2): Implement Archipelago-style interaction detection against SHADOW_ENDPOINT (Ollama).
- TODO(version2): Verify top-K edges via OpenAI (logprobs).
- TODO(version2): Validate HIF result with [`libs/hif/validator.py`](libs/hif/validator.py:1) when cross-package import is wired.
- TODO(version2): Persist results to Postgres and wire retrieval by request_id.
- TODO(version2): Add dead-letter queue and claiming of idle messages.

Legacy (kept for compatibility and local DEV):
- A prior DEV_MODE skeleton that generated a v1 HIF artifact to S3 or local gzip still exists below.
- Default behavior now runs the Redis worker; set DEV_MODE=1 explicitly to exercise the legacy DEV driver.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

# PII scrubbing for any log copies (see docs/security/SECURITY.md)
try:
    from libs.sanitize.pii import sanitize_message as _sanitize_message  # type: ignore
except Exception:
    def _sanitize_message(s: str) -> str:
        try:
            return s if isinstance(s, str) else str(s)
        except Exception:
            return ""

# Observability (optional)
try:
    from services.explainer.src import otel as _otel  # type: ignore
except Exception:
    try:
        import otel as _otel  # type: ignore
    except Exception:
        _otel = None  # type: ignore

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass
import signal
import socket
import sys
import time
import uuid
import re
import random
import math
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict
import hashlib

# Backpressure (optional)
try:
    from services.explainer.src.backpressure import BackpressureController, BackpressureConfig, BackpressureState  # type: ignore
except Exception:
    try:
        from backpressure import BackpressureController, BackpressureConfig, BackpressureState  # type: ignore
    except Exception:
        BackpressureController = None  # type: ignore
        BackpressureConfig = None  # type: ignore
        BackpressureState = None  # type: ignore

class _LocalQueueRef:
    def __init__(self) -> None:
        self._per_tenant_running: Dict[str, int] = {}
        self._per_tenant_queued: Dict[str, int] = {}
        self._throughput_ema: Dict[str, float] = {"sentence": 20.0, "token": 10.0}
    def queue_len_tenant(self, tenant_id: str) -> int:
        return int(self._per_tenant_queued.get(str(tenant_id), 0))
    def queue_len_global(self) -> int:
        return int(sum(self._per_tenant_queued.values()))
    def backlog_seconds(self, granularity: str) -> Optional[float]:
        return None
    def throughput_estimate(self, granularity: str) -> Optional[float]:
        return float(self._throughput_ema.get(str(granularity), 10.0))
    def running_for_tenant(self, tenant_id: str) -> int:
        return int(self._per_tenant_running.get(str(tenant_id), 0))
    def incr_running(self, tenant_id: str) -> None:
        t = str(tenant_id)
        self._per_tenant_running[t] = self._per_tenant_running.get(t, 0) + 1
    def decr_running(self, tenant_id: str) -> None:
        t = str(tenant_id)
        self._per_tenant_running[t] = max(0, self._per_tenant_running.get(t, 0) - 1)

_BP_QUEUE = _LocalQueueRef()
_BP_CTRL = None
def _bp_cfg_from_env() -> Optional["BackpressureConfig"]:
    try:
        if BackpressureConfig is None:
            return None
        ladder_env = os.getenv("BP_DEGRADE_LADDER", "token->sentence,reduce-samples,reduce-topk,reduce-layers,saliency-fallback,drop")
        ladder = [s.strip() for s in ladder_env.split(",") if s.strip()]
        return BackpressureConfig(
            max_backlog_seconds_sentence=float(os.getenv("BP_MAX_BACKLOG_SENTENCE", "1.5")),
            max_backlog_seconds_token=float(os.getenv("BP_MAX_BACKLOG_TOKEN", "3.0")),
            max_queue_len_per_tenant=int(os.getenv("BP_MAX_QUEUE_TENANT", "200")),
            max_queue_len_global=int(os.getenv("BP_MAX_QUEUE_GLOBAL", "5000")),
            degrade_ladder=ladder,
            tenant_minimum_guarantee=int(os.getenv("BP_TENANT_MIN_GUARANTEE", "2")),
            burst_multiplier=float(os.getenv("BP_BURST_MULTIPLIER", "1.5")),
        )
    except Exception:
        return BackpressureConfig() if BackpressureConfig is not None else None

try:
    if BackpressureController is not None and BackpressureConfig is not None:
        _BP_CTRL = BackpressureController(_BP_QUEUE, metrics_hook=None, cfg=_bp_cfg_from_env())
except Exception:
    _BP_CTRL = None

# Runtime SAE overrides (set per-envelope by process_envelope)
_BP_RUNTIME_OVERRIDES: Dict[str, Any] = {}

# Attribution strategies (budgeted)
try:
    from services.explainer.src.attribution import AttributionConfig, sentence_attribution, token_attribution  # type: ignore
except Exception:
    try:
        from attribution import AttributionConfig, sentence_attribution, token_attribution  # type: ignore
    except Exception:
        AttributionConfig = None  # type: ignore
        sentence_attribution = None  # type: ignore
        token_attribution = None  # type: ignore

# Hypergraph constructor/pruner (HIF v1 legacy)
try:
    from services.explainer.src.hypergraph import (  # type: ignore
        HypergraphConfig as HGConfig,
        prune_and_group as hg_prune_and_group,
        build_hif as hg_build_hif,
        validate_hif as hg_validate_hif,
    )
except Exception:
    try:
        from hypergraph import (  # type: ignore
            HypergraphConfig as HGConfig,
            prune_and_group as hg_prune_and_group,
            build_hif as hg_build_hif,
            validate_hif as hg_validate_hif,
        )
    except Exception:
        HGConfig = None  # type: ignore
        hg_prune_and_group = None  # type: ignore
        hg_build_hif = None  # type: ignore
        hg_validate_hif = None  # type: ignore

# Optional HTTP clients for future SHADOW_ENDPOINT usage (not used in this task)
try:
    import httpx  # type: ignore
except Exception:
    httpx = None  # type: ignore
try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore

# Redis async preferred, sync fallback
AsyncRedis = None
try:
    from redis.asyncio import Redis as AsyncRedis  # type: ignore
except Exception:
    AsyncRedis = None  # type: ignore

try:
    import redis  # type: ignore
except Exception:
    redis = None  # type: ignore

# Persistence: S3 and StatusStore (optional, import-guarded)
try:
    from services.explainer.src.s3_store import S3Store as NewS3Store  # type: ignore
except Exception:
    try:
        from s3_store import S3Store as NewS3Store  # type: ignore
    except Exception:
        NewS3Store = None  # type: ignore

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

# Expose S3Store symbol for legacy paths (DEV pipeline)
S3Store = NewS3Store  # type: ignore

# Status store helpers
_STATUS_STORE: Optional[_StatusStore] = None

def _get_status_store() -> Optional[_StatusStore]:
    global _STATUS_STORE
    if get_status_store_from_env is None:
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

def _now_epoch() -> float:
    return time.time()

# ----------------------------
# Chaos control (import-guarded)
# ----------------------------
_CHAOS_PATH_DEFAULT = "/tmp/hif/chaos.json"
_CHAOS_CACHE = {"flags": {}, "last_mtime": 0.0, "last_checked": 0.0}

def _read_chaos_flags(path: str = None) -> Dict[str, Any]:
    """Best-effort read of chaos control file. Returns dict of flags."""
    p = path or os.getenv("CHAOS_CONTROL_PATH", _CHAOS_PATH_DEFAULT)
    try:
        st = os.stat(p)
        mtime = float(getattr(st, "st_mtime", 0.0))
    except Exception:
        return {}
    try:
        if _CHAOS_CACHE.get("last_mtime", 0.0) == mtime:
            return dict(_CHAOS_CACHE.get("flags", {}))
    except Exception:
        pass
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            _CHAOS_CACHE["flags"] = data
            _CHAOS_CACHE["last_mtime"] = mtime
            return dict(data)
    except Exception:
        return {}
    return {}

def _chaos_flag(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Return subconfig for a flag or {}."""
    v = cfg.get(name)
    return v if isinstance(v, dict) else {}

def _chaos_hit(prob_percent: float) -> bool:
    """Return True with prob_percent% probability."""
    try:
        p = max(0.0, min(100.0, float(prob_percent)))
    except Exception:
        p = 0.0
    return (p > 0.0) and (random.random() * 100.0 < p)

# ----------------------------
# Environment configuration
# ----------------------------

def _env_int(name: str, default_str: str, minimum: int = 0) -> int:
    val = os.getenv(name, default_str)
    try:
        i = int(val)
        return i if i >= minimum else minimum
    except Exception:
        return int(default_str)


def _env_float(name: str, default_str: str, minimum: float = 0.0) -> float:
    val = os.getenv(name, default_str)
    try:
        f = float(val)
        return f if f >= minimum else minimum
    except Exception:
        return float(default_str)


class Config:
    def __init__(self) -> None:
        # Redis
        self.REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
        self.REDIS_STREAM: str = os.getenv("REDIS_STREAM", "hypergraph:completions")
        self.REDIS_CONSUMER_GROUP: str = os.getenv("REDIS_CONSUMER_GROUP", "explainer")
        default_consumer = socket.gethostname() or "explainer-1"
        self.REDIS_CONSUMER_NAME: str = os.getenv("REDIS_CONSUMER_NAME", default_consumer)
        self.REDIS_BLOCK_MS: int = _env_int("REDIS_BLOCK_MS", "15000", minimum=100)
        self.REDIS_CLAIM_IDLE_MS: int = _env_int("REDIS_CLAIM_IDLE_MS", "60000", minimum=1000)

        # Shadow surrogate (Ollama) configuration
        self.SHADOW_ENDPOINT: str = os.getenv("SHADOW_ENDPOINT", "http://localhost:11434")
        self.SHADOW_MODEL: str = os.getenv("SHADOW_MODEL", "mixtral:8x7b-instruct-q4_K_M")
        self.SHADOW_SAMPLES: int = _env_int("SHADOW_SAMPLES", "10", minimum=1)
        self.SHADOW_TEMPERATURE: float = _env_float("SHADOW_TEMPERATURE", "1.0", minimum=0.0)

        # Interaction discovery configuration
        self.INTERACTION_GROUPS: int = _env_int("INTERACTION_GROUPS", "24", minimum=0)
        _gs = _env_int("INTERACTION_GROUP_SIZE", "2", minimum=2)
        self.INTERACTION_GROUP_SIZE: int = 3 if _gs > 3 else _gs  # clamp to {2,3}
        self.INTERACTION_MAX_EDGES: int = _env_int("INTERACTION_MAX_EDGES", "16", minimum=0)
        self.INTERACTION_MIN_SYNERGY: float = _env_float("INTERACTION_MIN_SYNERGY", "0.05", minimum=0.0)

        # spaCy / Concepts
        self.SPACY_MODEL: str = os.getenv("SPACY_MODEL", "en_core_web_sm")
        try:
            self.CONCEPT_MAX: int = max(0, int(os.getenv("CONCEPT_MAX", "128")))
        except Exception:
            self.CONCEPT_MAX = 128

        # Provider verify / proxy
        self.LLM_PROXY_URL: str = (os.getenv("LLM_PROXY_URL", "") or "").rstrip("/")
        self.VERIFY_TOP_K: int = _env_int("VERIFY_TOP_K", "3", minimum=0)
        self.VERIFY_TIMEOUT_S: float = _env_float("VERIFY_TIMEOUT_S", "20", minimum=1.0)
        self.VERIFY_MODEL: str = os.getenv("VERIFY_MODEL", "gpt-4o-mini")
        self.VERIFY_TEMPERATURE: float = _env_float("VERIFY_TEMPERATURE", "0.0", minimum=0.0)
        self.PROVIDER_HEADER: str = os.getenv("PROVIDER_HEADER", "openai")
        self.PROVIDER_API_KEY: str = os.getenv("PROVIDER_API_KEY", "")

        # DB
        self.DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
        self.EXPLAINER_TABLE: str = os.getenv("EXPLAINER_TABLE", "explanations_v2")
        try:
            self.DB_CONNECT_TIMEOUT: int = int(os.getenv("DB_CONNECT_TIMEOUT", "5"))
        except Exception:
            self.DB_CONNECT_TIMEOUT = 5

        # Logging
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

        # Cost/budget controls
        self.COST_MAX_SHADOW_CALLS: int = _env_int("COST_MAX_SHADOW_CALLS", "500", minimum=0)
        self.COST_MAX_PROVIDER_CALLS: int = _env_int("COST_MAX_PROVIDER_CALLS", "20", minimum=0)
        self.COST_MIN_GAIN: float = _env_float("COST_MIN_GAIN", "0.01", minimum=0.0)
        self.COST_CACHE_SIZE: int = _env_int("COST_CACHE_SIZE", "10000", minimum=0)

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
            "REDIS_URL": self.REDIS_URL,
            "REDIS_STREAM": self.REDIS_STREAM,
            "REDIS_CONSUMER_GROUP": self.REDIS_CONSUMER_GROUP,
            "REDIS_CONSUMER_NAME": self.REDIS_CONSUMER_NAME,
            "REDIS_BLOCK_MS": self.REDIS_BLOCK_MS,
            "REDIS_CLAIM_IDLE_MS": self.REDIS_CLAIM_IDLE_MS,
            "SHADOW_ENDPOINT": self.SHADOW_ENDPOINT,
            "SHADOW_MODEL": self.SHADOW_MODEL,
            "SHADOW_SAMPLES": self.SHADOW_SAMPLES,
            "SHADOW_TEMPERATURE": self.SHADOW_TEMPERATURE,
            "INTERACTION_GROUPS": self.INTERACTION_GROUPS,
            "INTERACTION_GROUP_SIZE": self.INTERACTION_GROUP_SIZE,
            "INTERACTION_MAX_EDGES": self.INTERACTION_MAX_EDGES,
            "INTERACTION_MIN_SYNERGY": self.INTERACTION_MIN_SYNERGY,
            "SPACY_MODEL": self.SPACY_MODEL,
            "CONCEPT_MAX": self.CONCEPT_MAX,
            "LLM_PROXY_URL": self.LLM_PROXY_URL,
            "VERIFY_TOP_K": self.VERIFY_TOP_K,
            "VERIFY_TIMEOUT_S": self.VERIFY_TIMEOUT_S,
            "VERIFY_MODEL": self.VERIFY_MODEL,
            "VERIFY_TEMPERATURE": self.VERIFY_TEMPERATURE,
            "PROVIDER_HEADER": self.PROVIDER_HEADER,
            "PROVIDER_API_KEY": "<set>" if self.PROVIDER_API_KEY else "<unset>",
            "DATABASE_URL": self._mask_db_url(self.DATABASE_URL),
            "EXPLAINER_TABLE": self.EXPLAINER_TABLE,
            "DB_CONNECT_TIMEOUT": self.DB_CONNECT_TIMEOUT,
            "LOG_LEVEL": self.LOG_LEVEL,
            "COST_MAX_SHADOW_CALLS": self.COST_MAX_SHADOW_CALLS,
            "COST_MAX_PROVIDER_CALLS": self.COST_MAX_PROVIDER_CALLS,
            "COST_MIN_GAIN": self.COST_MIN_GAIN,
            "COST_CACHE_SIZE": self.COST_CACHE_SIZE,
        }

    def safe_repr(self) -> str:
        try:
            return json.dumps(self.safe_dict(), separators=(",", ":"), sort_keys=True)
        except Exception:
            return str(self.safe_dict())

CONFIG = Config()

# --- Security Audit Logger (explainer) ---
_AUDIT_ENABLED = (os.getenv("AUDIT_LOG_ENABLE", "0") or "0") == "1"
_AUDIT_PATH = os.getenv("AUDIT_LOG_PATH", "/var/log/hypergraph/audit.log")
_AUDIT_FH = None  # type: ignore
_AUDIT_USING_STDOUT = False

def _audit_init_worker() -> None:
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

def emit_audit_worker(event: str, tenant_id: str, trace_id: str, status: Optional[str] = None, reason: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
    if not _AUDIT_ENABLED:
        return
    try:
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "service": "explainer",
            "event": event,
            "tenant_id": str(tenant_id or "unknown"),
            "trace_id": str(trace_id or "unknown"),
            "status": status,
        }
        if reason:
            rec["reason"] = str(reason)
        if isinstance(extra, dict) and extra:
            for k, v in list(extra.items())[:8]:
                if isinstance(v, (str, int, float, bool)) or v is None:
                    rec[k] = v
        line = json.dumps(rec, separators=(",", ":"), ensure_ascii=False)
        if _AUDIT_FH is not None:
            _AUDIT_FH.write(line + "\n")
            _AUDIT_FH.flush()
        else:
            print(line)
    except Exception:
        pass


def setup_logging(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    # Also set UTC on time module formatting for consistency in logs
    os.environ["TZ"] = "UTC"
    try:
        time.tzset()  # type: ignore[attr-defined]
    except Exception:
        pass
    return logging.getLogger("explainer.worker")


# ----------------------------
# SAE decode microservice (optional)
# ----------------------------

_SAE_SERVICE: Optional[Any] = None

async def _init_sae_service_from_env(logger: logging.Logger) -> Optional[Any]:
    """
    Initialize and start SAEDecodeService if ENABLE_SAE_SERVICE=1 and DICT_ROOT is set.
    Import is guarded and torch is not required.
    """
    try:
        enabled = os.getenv("ENABLE_SAE_SERVICE", "0") == "1"
        dict_root = (os.getenv("DICT_ROOT", "") or "").strip()
        if not enabled or not dict_root:
            return None

        # Import inside function to avoid hard dependency at import time
        try:
            try:
                from services.explainer.src.sae_service import SAEDecodeService, SAEDecodeConfig  # type: ignore
            except Exception:
                from sae_service import SAEDecodeService, SAEDecodeConfig  # type: ignore
        except Exception as e:
            logger.warning(f"SAE service import unavailable: {e}; disabled")
            return None

        def _env_int(name: str, default_val: int) -> int:
            v = os.getenv(name, str(default_val))
            try:
                return int(v)
            except Exception:
                return int(default_val)

        cfg = SAEDecodeConfig(
            dict_root=dict_root,
            dictionary_name=os.getenv("DICT_NAME", "sae-gpt4-2m"),
            device=os.getenv("SAE_DEVICE", "auto"),
            dtype=os.getenv("SAE_DTYPE", "fp16"),
            cache_layers=_env_int("SAE_CACHE_LAYERS", 3),
            batch_size=_env_int("SAE_BATCH_SIZE", 32),
            max_wait_ms=_env_int("SAE_MAX_WAIT_MS", 8),
            topk=_env_int("SAE_TOPK", 256),
            tile_rows=_env_int("SAE_TILE_ROWS", 65536),
        )
        svc = SAEDecodeService(cfg)
        await svc.start()
        global _SAE_SERVICE
        _SAE_SERVICE = svc
        logger.info(f"SAE decode service enabled: device={svc._device} dict={cfg.dictionary_name}")
        return svc
    except Exception as e:
        logger.warning(f"Failed to init SAE service: {e}")
        return None

async def _stop_sae_service() -> None:
    try:
        global _SAE_SERVICE
        if _SAE_SERVICE is not None:
            await _SAE_SERVICE.stop()  # type: ignore
            _SAE_SERVICE = None
    except Exception:
        _SAE_SERVICE = None


# ----------------------------
# Pipeline placeholders
# ----------------------------

def parse_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts and normalizes required fields from the incoming payload dict.
    Required keys (presence tolerated as None if missing): request_id, provider, model, messages, response, created_at, trace
    """
    # Normalize keys with defaults
    return {
        "request_id": payload.get("request_id") or payload.get("id") or f"req_{uuid.uuid4().hex[:12]}",
        "provider": payload.get("provider"),
        "model": payload.get("model"),
        "messages": payload.get("messages") or [],
        "response": payload.get("response") or "",
        "created_at": payload.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trace": payload.get("trace") or {},
    }


def concept_extraction(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Concept extraction using spaCy NER with a regex fallback.

    - Loads spaCy model lazily once (env SPACY_MODEL, default: "en_core_web_sm")
    - Concatenates only user/system message content into a single text block (ignores assistant)
    - Produces HIF concept nodes for unique spans with spans and metadata label
    - Caps total concepts to env CONCEPT_MAX (default: 128)
    - Falls back to regex-based extraction if spaCy is unavailable or fails

    TODO(version2): add GLiNER integration as optional conceptizer.
    TODO(version2): configurable conceptizers chain and per-role weighting.
    """
    logger = logging.getLogger("explainer.worker")
    t0 = time.perf_counter()

    # Determine cap
    try:
        concept_cap = max(0, int(os.getenv("CONCEPT_MAX", "128")))
    except Exception:
        concept_cap = 128

    # Build the input text from messages (user + system only), ignoring assistant
    joined_parts: List[str] = []
    msgs = payload.get("messages") or []
    if isinstance(msgs, list):
        for m in msgs:
            try:
                role = (m.get("role") or "").lower()
            except Exception:
                role = ""
            if role not in ("user", "system"):
                continue
            content = m.get("content")
            if isinstance(content, str):
                joined_parts.append(content)
            elif isinstance(content, list):
                # OpenAI-style content parts
                for part in content:
                    if isinstance(part, dict):
                        text = part.get("text") or part.get("content") or part.get("data")
                        if isinstance(text, str):
                            joined_parts.append(text)
            elif content is not None:
                joined_parts.append(str(content))
    text = "\n\n".join(joined_parts).strip()
    if not text:
        logger.info("concept_extraction: empty input text; returning 0 concepts")
        return []

    # Lazy singleton spaCy loader stored on the function object to avoid globals
    nlp = None  # type: ignore[var-annotated]
    model_name = os.getenv("SPACY_MODEL", "en_core_web_sm")
    try:
        if not hasattr(concept_extraction, "_nlp"):
            setattr(concept_extraction, "_nlp", None)
        if getattr(concept_extraction, "_nlp") is None:
            try:
                import spacy  # type: ignore
                # Try to load with non-NER pipes disabled for speed; fallback to default load
                try:
                    nlp = spacy.load(
                        model_name,
                        disable=["tagger", "parser", "attribute_ruler", "lemmatizer", "textcat"],
                    )
                except Exception:
                    nlp = spacy.load(model_name)
                setattr(concept_extraction, "_nlp", nlp)
                logger.info(f"spaCy model loaded: {model_name}")
            except Exception as e:
                logger.warning(f"spaCy unavailable or failed to load '{model_name}': {e}; using regex fallback")
                setattr(concept_extraction, "_nlp", None)
        nlp = getattr(concept_extraction, "_nlp")
    except Exception:
        nlp = None

    nodes: List[Dict[str, Any]] = []
    seen = set()

    def add_node(start: int, end: int, txt: str, label: str) -> None:
        key = (int(start), int(end), txt)
        if key in seen:
            return
        seen.add(key)
        nodes.append(
            {
                "id": f"c{len(nodes)+1}",
                "type": "CONCEPT",
                "text": txt,
                "span": {"start": int(start), "end": int(end)},
                "metadata": {"label": label},
            }
        )

    used = "regex"
    # Primary: spaCy NER
    if nlp is not None:
        try:
            doc = nlp(text)
            for ent in getattr(doc, "ents", []):
                if len(nodes) >= concept_cap:
                    break
                add_node(ent.start_char, ent.end_char, ent.text, ent.label_)
            used = "spacy"
        except Exception as e:
            logger.warning(f"spaCy NER failed: {e}; falling back to regex")

    # Fallback: regex heuristics if spaCy unavailable or produced zero concepts
    if used != "spacy" or not nodes:
        flags = re.DOTALL | re.MULTILINE

        # Triple backtick code fences (capture inner content)
        for m in re.finditer(r"```(?:\w+)?\s*([\s\S]*?)\s*```", text, flags):
            if len(nodes) >= concept_cap:
                break
            s, e = m.span(1)
            inner = text[s:e].strip()
            if inner:
                add_node(s, e, inner, "FALLBACK")

        # Inline backticks
        if len(nodes) < concept_cap:
            for m in re.finditer(r"`([^`]+)`", text):
                if len(nodes) >= concept_cap:
                    break
                s, e = m.span(1)
                add_node(s, e, text[s:e], "FALLBACK")

        # Double-quoted phrases (minimum length to avoid noise)
        if len(nodes) < concept_cap:
            for m in re.finditer(r'"([^"\n]{3,})"', text):
                if len(nodes) >= concept_cap:
                    break
                s, e = m.span(1)
                add_node(s, e, text[s:e], "FALLBACK")

        # Single-quoted phrases
        if len(nodes) < concept_cap:
            for m in re.finditer(r"'([^'\n]{3,})'", text):
                if len(nodes) >= concept_cap:
                    break
                s, e = m.span(1)
                add_node(s, e, text[s:e], "FALLBACK")

        # Capitalized multi-word sequences (e.g., 'New York City')
        if len(nodes) < concept_cap:
            for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,})\b", text):
                if len(nodes) >= concept_cap:
                    break
                s, e = m.span(1)
                add_node(s, e, text[s:e], "FALLBACK")

    t1 = time.perf_counter()
    logger.info(f"concept_extraction: used={used} concepts={len(nodes)} duration_ms={(t1 - t0)*1000:.2f} cap={concept_cap}")
    return nodes


# ----------------------------
# Shadow surrogate client and helpers
# ----------------------------

_shadow_warned = False

# ----------------------------
# Cost optimizer: Budget + ProbCache + helpers
# ----------------------------
class CostBudget:
    def __init__(self, shadow_limit: int = 0, provider_limit: int = 0) -> None:
        self.shadow_calls: int = 0
        self.provider_calls: int = 0
        self.shadow_limit: int = max(0, int(shadow_limit))
        self.provider_limit: int = max(0, int(provider_limit))

    def can_shadow(self) -> bool:
        return self.shadow_calls < self.shadow_limit

    def can_provider(self) -> bool:
        return self.provider_calls < self.provider_limit

    def incr_shadow(self, n: int = 1) -> None:
        try:
            self.shadow_calls += max(0, int(n))
        except Exception:
            self.shadow_calls += 1

    def incr_provider(self, n: int = 1) -> None:
        try:
            self.provider_calls += max(0, int(n))
        except Exception:
            self.provider_calls += 1


class ProbCache:
    def __init__(self, capacity: int = 10000, name: str = "cache") -> None:
        self.capacity = max(0, int(capacity))
        self._store: "OrderedDict[Tuple[str, str, str, str, float, str], float]" = OrderedDict()
        self.hits: int = 0
        self.misses: int = 0
        self.disabled: bool = False
        self.name = name

    def _evict_if_needed(self) -> None:
        if self.capacity <= 0:
            return
        try:
            while len(self._store) > self.capacity:
                self._store.popitem(last=False)
        except Exception:
            self.disabled = True

    def get(self, key: Tuple[str, str, str, str, float, str]) -> Optional[float]:
        if self.disabled or self.capacity <= 0:
            return None
        try:
            if key in self._store:
                self.hits += 1
                val = self._store.pop(key)
                self._store[key] = val  # move to end (LRU)
                return val
            else:
                self.misses += 1
                return None
        except Exception:
            self.disabled = True
            return None

    def put(self, key: Tuple[str, str, str, str, float, str], value: float) -> None:
        if self.disabled or self.capacity <= 0:
            return
        try:
            self._store[key] = float(value)
            # move to end
            v = self._store.pop(key)
            self._store[key] = v
            self._evict_if_needed()
        except Exception:
            self.disabled = True

    def size(self) -> int:
        try:
            return len(self._store)
        except Exception:
            return 0


# Global caches scoped per payload-processing phase (set by discovery/verification)
_CURRENT_SHADOW_CACHE: Optional[ProbCache] = None
_CURRENT_PROVIDER_CACHE: Optional[ProbCache] = None


def _prompt_hash(prompt: str) -> str:
    try:
        h = hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()
        return h[:16]
    except Exception:
        return "0" * 16


def _cache_get(kind: str, prompt: str, token: str, model: str, temp: float, meta: str) -> Optional[float]:
    try:
        key = (kind, _prompt_hash(prompt), str(token or ""), str(model or ""), float(temp or 0.0), str(meta or ""))
        cache = _CURRENT_SHADOW_CACHE if kind == "shadow" else _CURRENT_PROVIDER_CACHE
        if cache is None:
            return None
        return cache.get(key)
    except Exception:
        return None


def _cache_put(kind: str, prompt: str, token: str, model: str, temp: float, meta: str, value: float) -> None:
    try:
        key = (kind, _prompt_hash(prompt), str(token or ""), str(model or ""), float(temp or 0.0), str(meta or ""))
        cache = _CURRENT_SHADOW_CACHE if kind == "shadow" else _CURRENT_PROVIDER_CACHE
        if cache is None:
            return
        cache.put(key, float(value))
    except Exception:
        try:
            if kind == "shadow" and _CURRENT_SHADOW_CACHE:
                _CURRENT_SHADOW_CACHE.disabled = True
            if kind == "provider" and _CURRENT_PROVIDER_CACHE:
                _CURRENT_PROVIDER_CACHE.disabled = True
        except Exception:
            pass


class ShadowClient:
    """
    Lightweight sync client for Ollama HTTP API.
    - POST {endpoint}/api/generate with JSON:
      {"model": model, "prompt": prompt, "stream": false, "options": {"temperature": temp}}
    - Response JSON includes "response" text.
    """

    def __init__(self, endpoint: str, model: str, logger: Optional[logging.Logger] = None) -> None:
        self.endpoint = (endpoint or "http://localhost:11434").rstrip("/")
        self.model = model or "mixtral:8x7b-instruct-q4_K_M"
        self.logger = logger or logging.getLogger("explainer.worker")
        self._http_lib = "httpx" if httpx is not None else ("requests" if requests is not None else None)
        if self._http_lib is None:
            raise RuntimeError("No HTTP client available for ShadowClient (httpx/requests missing)")

    def _request_once(self, prompt: str, temperature: float) -> Optional[str]:
        url = f"{self.endpoint}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": float(temperature)},
        }
        timeout = 30.0
        try:
            if self._http_lib == "httpx":
                with httpx.Client(timeout=timeout) as client:  # type: ignore
                    resp = client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
            else:
                resp = requests.post(url, json=payload, timeout=timeout)  # type: ignore
                resp.raise_for_status()
                data = resp.json()
            text = data.get("response")
            return text if isinstance(text, str) else None
        except Exception as e:
            if self.logger:
                self.logger.warning(f"ShadowClient request failed: {e}")
            return None

    def generate(self, prompt: str, temperature: float = 1.0) -> str:
        text = self._request_once(prompt, temperature)
        return text or ""

    def estimate_token_freq(self, prompt: str, token: str, samples: int, temperature: float) -> float:
        """
        Estimate probability that the response starts with the given token (whitespace-delimited, case-insensitive).
        Runs N generations and returns frequency / N.
        """
        token_l = (token or "").strip().lower()
        if not token_l:
            return 0.0
        hits = 0
        n = max(1, int(samples))
        for _ in range(n):
            text = self._request_once(prompt, temperature)
            if not text:
                continue
            first = target_token_from_response(text)
            if first == token_l:
                hits += 1
        return hits / float(n)


def build_prompt_from_messages(messages: List[Dict[str, Any]]) -> str:
    """
    Join only user and system messages (ignore assistant) in a simple 'role: text' format.
    """
    parts: List[str] = []
    if isinstance(messages, list):
        for m in messages:
            try:
                role = (m.get("role") or "").lower()
            except Exception:
                role = ""
            if role not in ("system", "user"):
                continue
            content = m.get("content")
            collected: List[str] = []
            if isinstance(content, str):
                collected.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        text = part.get("text") or part.get("content") or part.get("data")
                        if isinstance(text, str):
                            collected.append(text)
            elif content is not None:
                collected.append(str(content))
            if collected:
                parts.append(f"{role}: " + "\n".join(collected).strip())
    return "\n".join(parts).strip()


def _concat_user_system_text(messages: List[Dict[str, Any]]) -> str:
    """
    Rebuilds the exact concatenation used by concept_extraction (user+system only, joined by two newlines, no role labels).
    """
    joined_parts: List[str] = []
    if isinstance(messages, list):
        for m in messages:
            role = (m.get("role") or "").lower()
            if role not in ("user", "system"):
                continue
            content = m.get("content")
            if isinstance(content, str):
                joined_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        text = part.get("text") or part.get("content") or part.get("data")
                        if isinstance(text, str):
                            joined_parts.append(text)
            elif content is not None:
                joined_parts.append(str(content))
    return "\n\n".join(joined_parts).strip()


def target_token_from_response(response_text: str) -> str:
    """
    Return the first non-empty whitespace-delimited token from the assistant response, lowercased.
    """
    if not isinstance(response_text, str):
        return ""
    s = response_text.strip()
    if not s:
        return ""
    return s.split(None, 1)[0].lower()


def mask_text(text: str, spans: List[Tuple[int, int]]) -> str:
    """
    Replace given spans with '[MASK]' in a single pass. Spans are sorted descending by start to avoid offset shifts.
    """
    if not text or not spans:
        return text
    valid_spans: List[Tuple[int, int]] = []
    L = len(text)
    for s, e in spans:
        try:
            s_i, e_i = int(s), int(e)
            if 0 <= s_i < e_i <= L:
                valid_spans.append((s_i, e_i))
        except Exception:
            continue
    if not valid_spans:
        return text
    out = text
    for s, e in sorted(valid_spans, key=lambda x: x[0], reverse=True):
        out = out[:s] + "[MASK]" + out[e:]
    return out


def apply_mask_to_messages(messages: List[Dict[str, Any]], concept_nodes_subset: List[Dict[str, Any]]) -> str:
    """
    Build base prompt with build_prompt_from_messages and apply masking for the subset of concepts.

    Mapping of concept spans assumes the same concatenation rule used in concept_extraction.
    If spans don't align with the built prompt, fallback to naive replace: replace the first
    occurrence of concept['text'] with '[MASK]'. This keeps behavior resilient.
    """
    base_prompt = build_prompt_from_messages(messages)
    if not concept_nodes_subset:
        return base_prompt

    # Try strict span-based masking if the base prompt exactly matches the raw concatenation used during extraction.
    raw_concat = _concat_user_system_text(messages)
    try:
        if raw_concat and base_prompt == raw_concat:
            spans: List[Tuple[int, int]] = []
            for c in concept_nodes_subset:
                sp = c.get("span")
                if isinstance(sp, dict) and "start" in sp and "end" in sp:
                    spans.append((int(sp["start"]), int(sp["end"])))
            if spans:
                return mask_text(base_prompt, spans)
    except Exception:
        # Fall through to naive replace
        pass

    # Fallback: naive first-occurrence replace for each concept's text.
    masked = base_prompt
    for c in concept_nodes_subset:
        txt = c.get("text")
        if not isinstance(txt, str) or not txt:
            continue
        idx = masked.find(txt)
        if idx >= 0:
            masked = masked[:idx] + "[MASK]" + masked[idx + len(txt):]
    return masked


def _extract_assistant_text(payload: Dict[str, Any]) -> str:
    """
    Best-effort extraction of the assistant response text from payload['response'] or messages[].
    """
    resp = payload.get("response")
    if isinstance(resp, str):
        if resp.strip():
            return resp
    elif isinstance(resp, dict):
        try:
            choices = resp.get("choices")
            if isinstance(choices, list) and choices:
                ch0 = choices[0] or {}
                msg = ch0.get("message") or {}
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content
                text_alt = ch0.get("text")
                if isinstance(text_alt, str) and text_alt.strip():
                    return text_alt
        except Exception:
            pass

    msgs = payload.get("messages") or []
    if isinstance(msgs, list):
        for m in reversed(msgs):
            role = (m.get("role") or "").lower()
            if role != "assistant":
                continue
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                return content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        text = part.get("text") or part.get("content") or part.get("data")
                        if isinstance(text, str) and text.strip():
                            return text
            elif content is not None:
                s = str(content)
                if s.strip():
                    return s
    return ""


def estimate_prob(client: "ShadowClient", base_prompt: str, token: str) -> float:
    """
    Estimate P(token | base_prompt) by sampling SHADOW_SAMPLES generations with SHADOW_TEMPERATURE.
    """
    try:
        samples = _env_int("SHADOW_SAMPLES", "10", minimum=1)
        temperature = _env_float("SHADOW_TEMPERATURE", "1.0", minimum=0.0)
        return float(client.estimate_token_freq(base_prompt, token, samples=samples, temperature=temperature))
    except Exception:
        return 0.0


def interaction_discovery(payload: Dict[str, Any], concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Minimal perturbation-based interaction discovery using a shadow surrogate (Ollama) with random group testing.

    - Target the primary output node "out1" by focusing on the first token of the assistant response.
    - For each random group, measure non-linear synergy:
        synergy = Drop_group - sum(Drop_i for i in group),
      where Drop_x = max(P0 - P_x, 0), and P0 = P(y | base_prompt).
    - Keep edges with synergy >= INTERACTION_MIN_SYNERGY and Drop_group > 0.
    - Weight = min(max(synergy / max(P0, 1e-6), 0.0), 1.0).

    Fail-safe: Any HTTP or client errors result in returning [] without breaking the worker.
    """
    logger = logging.getLogger("explainer.worker")
    try:
        if not concepts:
            return []

        messages = payload.get("messages") or []
        base_prompt = build_prompt_from_messages(messages)
        if not base_prompt:
            return []

        response_text = _extract_assistant_text(payload)
        target = target_token_from_response(response_text if isinstance(response_text, str) else "")
        if not target:
            return []

        cfg = Config()

        # Cost controls and cache (per-payload)
        budget = CostBudget(shadow_limit=cfg.COST_MAX_SHADOW_CALLS, provider_limit=cfg.COST_MAX_PROVIDER_CALLS)
        cache = ProbCache(capacity=cfg.COST_CACHE_SIZE, name="shadow")
        global _CURRENT_SHADOW_CACHE
        prev_shadow_cache = _CURRENT_SHADOW_CACHE
        _CURRENT_SHADOW_CACHE = cache
        min_gain = float(cfg.COST_MIN_GAIN)
        early_stop_reason = ""

        # Initialize shadow client
        try:
            client = ShadowClient(cfg.SHADOW_ENDPOINT, cfg.SHADOW_MODEL, logger=logger)
        except Exception as e:
            global _shadow_warned
            if not _shadow_warned:
                logger.warning(f"ShadowClient unavailable: {e}. Returning 0 edges.")
                _shadow_warned = True
            return []

        def shadow_prob(prompt: str) -> Optional[float]:
            meta = f"samples={cfg.SHADOW_SAMPLES}"
            v = _cache_get("shadow", prompt, target, cfg.SHADOW_MODEL, float(cfg.SHADOW_TEMPERATURE), meta)
            if v is not None:
                return float(v)
            if not budget.can_shadow():
                return None
            val = float(estimate_prob(client, prompt, target))
            budget.incr_shadow(1)
            _cache_put("shadow", prompt, target, cfg.SHADOW_MODEL, float(cfg.SHADOW_TEMPERATURE), meta, val)
            return val

        # Baseline probability with cache/budget
        meta = f"samples={cfg.SHADOW_SAMPLES}"
        P0_cached = _cache_get("shadow", base_prompt, target, cfg.SHADOW_MODEL, float(cfg.SHADOW_TEMPERATURE), meta)
        if P0_cached is not None:
            P0 = float(P0_cached)
        else:
            if not budget.can_shadow():
                early_stop_reason = "shadow_budget_exhausted_before_P0"
                logger.info(
                    f"interaction_discovery: early stop ({early_stop_reason}); "
                    f"shadow_calls_used={budget.shadow_calls}/{cfg.COST_MAX_SHADOW_CALLS}"
                )
                return []
            P0 = float(estimate_prob(client, base_prompt, target))
            budget.incr_shadow(1)
            _cache_put("shadow", base_prompt, target, cfg.SHADOW_MODEL, float(cfg.SHADOW_TEMPERATURE), meta, P0)

        logger.info(
            f"interaction_discovery: base P0={P0:.4f} token='{target}' groups={cfg.INTERACTION_GROUPS} "
            f"samples={cfg.SHADOW_SAMPLES} temp={cfg.SHADOW_TEMPERATURE} "
            f"shadow_budget_cap={cfg.COST_MAX_SHADOW_CALLS} min_gain={min_gain}"
        )
        if P0 <= 0.0:
            return []

        # Group testing
        group_size = max(2, min(3, int(cfg.INTERACTION_GROUP_SIZE)))
        if len(concepts) < group_size or cfg.INTERACTION_GROUPS <= 0:
            return []

        def group_priority(g: List[Dict[str, Any]]) -> float:
            try:
                texts = [str((c.get("text") or "")).strip() for c in g]
                score = float(sum(len(t) for t in texts))
                if any(len(t) < 2 for t in texts):
                    score -= 100.0
                if len(set(texts)) < len(texts):
                    score -= 50.0
                return score
            except Exception:
                return 0.0

        edges_by_key: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        groups_tested = 0
        max_groups = int(cfg.INTERACTION_GROUPS)
        consecutive_low_gain = 0
        LOW_GAIN_MAX_STREAK = 5

        for _ in range(max_groups):
            # choose best-of-3 random groups by heuristic
            try:
                num_candidates = 3 if len(concepts) >= group_size * 2 else 1
                cand_groups: List[List[Dict[str, Any]]] = []
                for _i in range(num_candidates):
                    cand_groups.append(random.sample(concepts, group_size))
                group = max(cand_groups, key=group_priority)
            except ValueError:
                break
            groups_tested += 1

            # Individual drops with cache/budget
            drops_individual: List[float] = []
            group_cache_hits_start = cache.hits
            budget_blocked = False
            for c in group:
                masked_prompt_single = apply_mask_to_messages(messages, [c])
                v = _cache_get("shadow", masked_prompt_single, target, cfg.SHADOW_MODEL, float(cfg.SHADOW_TEMPERATURE), meta)
                if v is None:
                    if not budget.can_shadow():
                        budget_blocked = True
                        early_stop_reason = "shadow_budget_exhausted"
                        break
                    Pi = shadow_prob(masked_prompt_single)
                    if Pi is None:
                        budget_blocked = True
                        early_stop_reason = "shadow_budget_exhausted"
                        break
                else:
                    Pi = float(v)
                drop_i = max(P0 - float(Pi), 0.0)
                drops_individual.append(drop_i)
            if budget_blocked:
                break

            # Group drop with cache/budget
            masked_prompt_group = apply_mask_to_messages(messages, group)
            v = _cache_get("shadow", masked_prompt_group, target, cfg.SHADOW_MODEL, float(cfg.SHADOW_TEMPERATURE), meta)
            if v is None:
                if not budget.can_shadow():
                    early_stop_reason = "shadow_budget_exhausted"
                    break
                P_group = shadow_prob(masked_prompt_group)
                if P_group is None:
                    early_stop_reason = "shadow_budget_exhausted"
                    break
            else:
                P_group = float(v)

            drop_group = max(P0 - float(P_group), 0.0)
            synergy = float(drop_group - sum(drops_individual))

            if synergy >= float(cfg.INTERACTION_MIN_SYNERGY) and drop_group > 0.0:
                src_ids = [str(c.get("id") or "") for c in group if isinstance(c, dict)]
                src_ids = [sid for sid in src_ids if sid]
                if len(src_ids) == group_size:
                    key = tuple(sorted(src_ids))
                    weight = float(min(max(synergy / max(P0, 1e-6), 0.0), 1.0))
                    candidate = {
                        "source_nodes": src_ids,
                        "target_node": "out1",
                        "weight": weight,
                        "interaction_type": "SYNERGY",
                        "description": f"Group mask of {len(group)} concepts yielded non-linear drop for token '{target}'.",
                        "evidence": {
                            "P0": float(P0),
                            "P_group": float(P_group),
                            "drops_individual": [float(x) for x in drops_individual],
                            "samples": int(cfg.SHADOW_SAMPLES),
                            "temperature": float(cfg.SHADOW_TEMPERATURE),
                            "shadow_calls_used": int(budget.shadow_calls),
                            "cache_hits": int(max(0, cache.hits - group_cache_hits_start)),
                        },
                    }
                    prev = edges_by_key.get(key)
                    if prev is None or float(prev.get("weight", 0.0)) < weight:
                        edges_by_key[key] = candidate

            # Early stop by marginal gain threshold streak
            if synergy < min_gain:
                consecutive_low_gain += 1
                if consecutive_low_gain >= LOW_GAIN_MAX_STREAK:
                    early_stop_reason = "low_marginal_gain"
                    break
            else:
                consecutive_low_gain = 0

        # Materialize and sort edges
        edges: List[Dict[str, Any]] = []
        sorted_items = sorted(edges_by_key.items(), key=lambda kv: float(kv[1].get("weight", 0.0)), reverse=True)
        for idx, (_, e) in enumerate(sorted_items, start=1):
            he = dict(e)
            he["id"] = f"h{idx}"
            edges.append(he)
            if len(edges) >= int(cfg.INTERACTION_MAX_EDGES):
                break

        # Logging summary
        top_weights = [round(float(e.get("weight", 0.0)), 4) for e in edges[:3]]
        logger.info(
            f"interaction_discovery: groups_tested={groups_tested} edges={len(edges)} top_weights={top_weights} "
            f"shadow_calls_used={budget.shadow_calls}/{cfg.COST_MAX_SHADOW_CALLS} "
            f"cache_entries={cache.size()} cache_hits={cache.hits} cache_misses={cache.misses} "
            f"early_stop_reason={early_stop_reason or 'none'}"
        )

        return edges
    except Exception as e:
        logger.warning(f"interaction_discovery failed (fail-safe returning 0 edges): {e}")
        return []
    finally:
        try:
            global _CURRENT_SHADOW_CACHE
            _CURRENT_SHADOW_CACHE = prev_shadow_cache if 'prev_shadow_cache' in locals() else None
        except Exception:
            pass


class ProviderVerifier:
    """
    Sync HTTP client to query a LiteLLM proxy for Chat Completions with logprobs enabled.
    - Reads env:
        LLM_PROXY_URL (required for use; if missing -> unavailable)
        VERIFY_TIMEOUT_S (default "20")
        VERIFY_MODEL (default "gpt-4o-mini"; can be overridden per-call)
        VERIFY_TEMPERATURE (default "0.0")
        PROVIDER_HEADER (default "openai") -> sent as X-Provider
        PROVIDER_API_KEY (optional) -> Bearer token in Authorization header
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        default_model: Optional[str] = None,
        default_temperature: Optional[float] = None,
    ) -> None:
        self.logger = logger or logging.getLogger("explainer.worker")
        self.base_url: str = (os.getenv("LLM_PROXY_URL", "") or "").rstrip("/")
        # Timeout with minimum of 1 second
        self.timeout_s: float = _env_float("VERIFY_TIMEOUT_S", "20", minimum=1.0)
        # Model/temperature defaults (env takes precedence)
        self.model_default: str = os.getenv("VERIFY_MODEL") or (default_model or "gpt-4o-mini")
        if default_temperature is None:
            self.temperature: float = _env_float("VERIFY_TEMPERATURE", "0.0", minimum=0.0)
        else:
            try:
                self.temperature = float(default_temperature)
            except Exception:
                self.temperature = 0.0
        self.provider_header: str = os.getenv("PROVIDER_HEADER", "openai")
        self.api_key: str = os.getenv("PROVIDER_API_KEY", "")
        self._http_lib: Optional[str] = "httpx" if httpx is not None else ("requests" if requests is not None else None)

    def available(self) -> bool:
        return bool(self.base_url and self._http_lib is not None)

    def _headers(self) -> Dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "X-Provider": self.provider_header,
        }
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    @staticmethod
    def _norm_token(tok: str) -> str:
        # Normalize: lowercase and strip leading whitespace/newlines
        try:
            return (tok or "").lstrip().lower()
        except Exception:
            return ""

    def _extract_token_prob_from_response(self, data: Dict[str, Any], target_norm: str) -> Optional[float]:
        """
        Robustly parse LiteLLM/OpenAI-like chat logprobs payloads.

        Expected shapes:
        - choices[0].logprobs.content[0].top_logprobs -> list[{"token": "...", "logprob": -x.x}, ...]
        - OR choices[0].logprobs.top_logprobs[0] -> list[{"token": "...", "logprob": -x.x}, ...]
        If logprobs present but target token missing: return epsilon (1e-9).
        If shape unexpected or missing: return None (indicates estimation failure).
        """
        try:
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                return None
            ch0 = choices[0] or {}
            lp = ch0.get("logprobs")
            if not isinstance(lp, dict):
                return None

            def iter_top_logprobs() -> List[Dict[str, Any]]:
                # 1) OpenAI Chat style
                content = lp.get("content")
                if isinstance(content, list) and content:
                    entry0 = content[0] or {}
                    top_lp = entry0.get("top_logprobs")
                    if isinstance(top_lp, list) and top_lp:
                        return top_lp  # list[dict]
                # 2) Alt style: top_logprobs at root of logprobs
                tlp = lp.get("top_logprobs")
                if isinstance(tlp, list) and tlp:
                    # Could be list[list[dict]] or list[dict]
                    if isinstance(tlp[0], list):
                        if tlp[0]:
                            return tlp[0]
                    elif isinstance(tlp[0], dict):
                        return tlp  # already list[dict]
                return []

            top_items = iter_top_logprobs()
            if not top_items:
                # No recognizable top_logprobs array -> treat as parse failure
                return None

            # Find target token in candidates
            for item in top_items:
                try:
                    tok = item.get("token") or item.get("string") or item.get("decoded_token") or ""
                    lp_val = item.get("logprob")
                    if lp_val is None:
                        lp_val = item.get("log_prob") or item.get("log_probability")
                    if isinstance(tok, str) and isinstance(lp_val, (float, int)):
                        if self._norm_token(tok) == target_norm:
                            prob = math.exp(float(lp_val))
                            if not (prob > 0.0) or math.isnan(prob) or math.isinf(prob):
                                return 1e-9
                            return float(max(min(prob, 1.0), 1e-9))
                except Exception:
                    continue
            # Token not found among top candidates -> epsilon
            return 1e-9
        except Exception:
            return None

    def estimate_token_prob(self, prompt: str, target_token: str, model_override: Optional[str] = None) -> Optional[float]:
        """
        Estimate P(target_token | prompt) using first-token logprobs.
        Returns:
            float in (0,1] on success,
            1e-9 if token absent from top candidates,
            None on HTTP/parse failure.
        """
        if not self.available():
            return None
        target_norm = self._norm_token(target_token)
        if not target_norm:
            return 0.0

        model = model_override or self.model_default

        # Cache pre-check
        try:
            cached = _cache_get("provider", prompt, target_token, model, float(self.temperature), "logprobs")
            if cached is not None:
                return float(cached)
        except Exception:
            pass

        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(self.temperature),
            "logprobs": True,
            "top_logprobs": 5,
            "max_tokens": 1,
            "stream": False,
        }

        try:
            if self._http_lib == "httpx":
                with httpx.Client(timeout=self.timeout_s) as client:  # type: ignore
                    resp = client.post(url, headers=self._headers(), json=payload)
                    status = resp.status_code
                    if status >= 400:
                        self.logger.warning(f"ProviderVerifier HTTP {status}: {resp.text[:256]}")
                        return None
                    data = resp.json()
            else:
                resp = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout_s)  # type: ignore
                status = resp.status_code  # type: ignore
                if status >= 400:  # type: ignore
                    self.logger.warning(f"ProviderVerifier HTTP {status}: {getattr(resp, 'text', '')[:256]}")  # type: ignore
                    return None
                data = resp.json()  # type: ignore

            prob = self._extract_token_prob_from_response(data, target_norm)
            if prob is not None:
                try:
                    _cache_put("provider", prompt, target_token, model, float(self.temperature), "logprobs", float(prob))
                except Exception:
                    pass
            return prob
        except Exception as e:
            self.logger.warning(f"ProviderVerifier request failed: {e}")
            return None


def verify_top_k(payload: Dict[str, Any], hyperedges: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """
    Verify the top-K hyperedges using an upstream provider (via LiteLLM proxy) with logprobs.
    - Baseline once (P0_v) and per-edge only masked group probability (conservative).
    - Enforce provider budget; cache results to minimize duplicate calls.
    - Annotate edges with provider_calls_used and verifier metadata.
    """
    logger = logging.getLogger("explainer.worker")

    # Nothing to verify
    if not hyperedges:
        return []

    # Clamp K
    try:
        k = int(k)
    except Exception:
        k = 0
    if k <= 0:
        return hyperedges

    # Check proxy availability early
    proxy_url = os.getenv("LLM_PROXY_URL", "").strip()
    if not proxy_url:
        logger.info("verify_top_k: skipping verification (LLM_PROXY_URL not set)")
        top_idx = sorted(range(len(hyperedges)), key=lambda i: float(hyperedges[i].get("weight", 0.0)), reverse=True)[:k]
        for i in top_idx:
            try:
                he = hyperedges[i]
                evid = he.get("evidence") if isinstance(he.get("evidence"), dict) else {}
                evid["verifier"] = "skipped"
                he["evidence"] = evid
                he["verified"] = False
            except Exception:
                continue
        return hyperedges

    # Build base prompt and target token
    messages = payload.get("messages") or []
    base_prompt = build_prompt_from_messages(messages)
    if not base_prompt:
        logger.info("verify_top_k: skipping (empty base_prompt)")
        return hyperedges

    response_text = _extract_assistant_text(payload)
    target = target_token_from_response(response_text)
    if not target:
        logger.info("verify_top_k: skipping (empty target token from assistant response)")
        return hyperedges

    # Determine model
    payload_model = payload.get("model")
    model_to_use = os.getenv("VERIFY_MODEL") or (payload_model if isinstance(payload_model, str) and payload_model.strip() else "gpt-4o-mini")

    # Initialize verifier
    verifier = ProviderVerifier(logger=logger, default_model=model_to_use)
    if not verifier.available():
        logger.info("verify_top_k: skipping (no HTTP client available or invalid proxy URL)")
        return hyperedges

    cfg = Config()
    # Provider budget and cache per-payload
    budget = CostBudget(shadow_limit=cfg.COST_MAX_SHADOW_CALLS, provider_limit=cfg.COST_MAX_PROVIDER_CALLS)
    provider_cache = ProbCache(capacity=cfg.COST_CACHE_SIZE, name="provider")
    global _CURRENT_PROVIDER_CACHE
    prev_provider_cache = _CURRENT_PROVIDER_CACHE
    _CURRENT_PROVIDER_CACHE = provider_cache

    try:
        # Baseline probability P0_v with cache/budget awareness
        meta = "logprobs"
        cached = _cache_get("provider", base_prompt, target, model_to_use, float(verifier.temperature), meta)
        if cached is not None:
            P0_v = float(cached)
        else:
            if not budget.can_provider():
                logger.info(
                    f"verify_top_k: budget exhausted before baseline; provider_calls_used={budget.provider_calls}/{cfg.COST_MAX_PROVIDER_CALLS}"
                )
                top_idx = sorted(range(len(hyperedges)), key=lambda i: float(hyperedges[i].get("weight", 0.0)), reverse=True)[:k]
                for i in top_idx:
                    try:
                        he = hyperedges[i]
                        evid = he.get("evidence") if isinstance(he.get("evidence"), dict) else {}
                        evid.update(
                            {
                                "verifier": "budget_exhausted",
                                "provider": verifier.provider_header,
                                "verifier_model": model_to_use,
                                "verifier_temperature": verifier.temperature,
                                "provider_calls_used": budget.provider_calls,
                            }
                        )
                        he["evidence"] = evid
                        he["verified"] = False
                    except Exception:
                        continue
                return hyperedges
            P0_v = verifier.estimate_token_prob(base_prompt, target, model_override=model_to_use)
            budget.incr_provider(1)

        logger.info(
            f"verify_top_k: provider={verifier.provider_header} model={model_to_use} "
            f"timeout_s={verifier.timeout_s} edges_to_verify={min(k, len(hyperedges))} P0_v={P0_v} "
            f"provider_budget_cap={cfg.COST_MAX_PROVIDER_CALLS}"
        )
        if P0_v is None or P0_v <= 0.0:
            top_idx = sorted(range(len(hyperedges)), key=lambda i: float(hyperedges[i].get("weight", 0.0)), reverse=True)[:k]
            for i in top_idx:
                try:
                    he = hyperedges[i]
                    evid = he.get("evidence") if isinstance(he.get("evidence"), dict) else {}
                    evid["verifier"] = "baseline_failed"
                    evid["P0_provider"] = P0_v if P0_v is not None else "unavailable"
                    evid["provider"] = verifier.provider_header
                    evid["verifier_model"] = model_to_use
                    evid["verifier_temperature"] = verifier.temperature
                    evid["provider_calls_used"] = budget.provider_calls
                    he["evidence"] = evid
                    he["verified"] = False
                except Exception:
                    continue
            return hyperedges

        # Rebuild concepts to map IDs -> concept nodes for masking
        try:
            concepts = concept_extraction(payload)
            id2concept: Dict[str, Dict[str, Any]] = {}
            for c in concepts:
                cid = c.get("id")
                if isinstance(cid, str) and cid:
                    id2concept[cid] = c
        except Exception as e:
            logger.warning(f"verify_top_k: concept rebuild failed, proceeding without verification: {e}")
            return hyperedges

        min_synergy = float(cfg.INTERACTION_MIN_SYNERGY)

        # Choose top-K edges by current weight
        top_indices = sorted(range(len(hyperedges)), key=lambda i: float(hyperedges[i].get("weight", 0.0)), reverse=True)[:k]

        # Annotate non-top edges as skipped
        for idx, edge in enumerate(hyperedges):
            if idx in top_indices:
                continue
            try:
                evid = edge.get("evidence") if isinstance(edge.get("evidence"), dict) else {}
                evid.setdefault("verifier", "skipped")
                edge["evidence"] = evid
                edge["verified"] = False
            except Exception:
                continue

        # Verify top-K (mask only the group; conservative)
        for i in top_indices:
            he = hyperedges[i]
            src_ids = list(he.get("source_nodes") or [])
            concept_subset = [id2concept[sid] for sid in src_ids if sid in id2concept]
            if not concept_subset:
                evid = he.get("evidence") if isinstance(he.get("evidence"), dict) else {}
                evid.update(
                    {
                        "verifier": "skipped_no_concepts",
                        "provider": verifier.provider_header,
                        "P0_provider": P0_v,
                        "verifier_model": model_to_use,
                        "verifier_temperature": verifier.temperature,
                        "provider_calls_used": budget.provider_calls,
                    }
                )
                he["evidence"] = evid
                he["verified"] = False
                logger.info(f"verify_top_k: edge id={he.get('id')} src={src_ids} skipped (no concept mapping)")
                continue

            masked_prompt = apply_mask_to_messages(messages, concept_subset)
            cached_g = _cache_get("provider", masked_prompt, target, model_to_use, float(verifier.temperature), meta)
            if cached_g is not None:
                P_group_v = float(cached_g)
            else:
                if not budget.can_provider():
                    evid = he.get("evidence") if isinstance(he.get("evidence"), dict) else {}
                    evid.update(
                        {
                            "verifier": "budget_exhausted",
                            "provider": verifier.provider_header,
                            "P0_provider": P0_v,
                            "verifier_model": model_to_use,
                            "verifier_temperature": verifier.temperature,
                            "provider_calls_used": budget.provider_calls,
                        }
                    )
                    he["evidence"] = evid
                    he["verified"] = False
                    logger.info("verify_top_k: provider budget exhausted during edge verification")
                    # Mark remaining top edges as budget_exhausted
                    rem = [j for j in top_indices if j > i]
                    for j in rem:
                        try:
                            he2 = hyperedges[j]
                            evid2 = he2.get("evidence") if isinstance(he2.get("evidence"), dict) else {}
                            evid2.update(
                                {
                                    "verifier": "budget_exhausted",
                                    "provider": verifier.provider_header,
                                    "P0_provider": P0_v,
                                    "verifier_model": model_to_use,
                                    "verifier_temperature": verifier.temperature,
                                    "provider_calls_used": budget.provider_calls,
                                }
                            )
                            he2["evidence"] = evid2
                            he2["verified"] = False
                        except Exception:
                            continue
                    return hyperedges
                P_group_v = verifier.estimate_token_prob(masked_prompt, target, model_override=model_to_use)
                budget.incr_provider(1)

            if P_group_v is None:
                evid = he.get("evidence") if isinstance(he.get("evidence"), dict) else {}
                evid.update(
                    {
                        "verifier": "group_failed",
                        "provider": verifier.provider_header,
                        "P0_provider": P0_v,
                        "P_group_provider": "unavailable",
                        "verifier_model": model_to_use,
                        "verifier_temperature": verifier.temperature,
                        "provider_calls_used": budget.provider_calls,
                    }
                )
                he["evidence"] = evid
                he["verified"] = False
                logger.info(f"verify_top_k: edge id={he.get('id')} src={src_ids} group_prob=unavailable verified=False")
                continue

            drop_group_v = max(float(P0_v) - float(P_group_v), 0.0)
            synergy_v = drop_group_v  # conservative
            verified = bool(synergy_v >= min_synergy and drop_group_v > 0.0)

            provider_score = float(min(max(synergy_v / max(float(P0_v), 1e-6), 0.0), 1.0))
            try:
                w_shadow = float(he.get("weight", 0.0))
            except Exception:
                w_shadow = 0.0
            w_blend = 0.5 * w_shadow + 0.5 * provider_score
            he["weight"] = float(w_blend)

            evid = he.get("evidence") if isinstance(he.get("evidence"), dict) else {}
            evid.update(
                {
                    "verifier": verifier.provider_header,
                    "provider": verifier.provider_header,
                    "P0_provider": P0_v,
                    "P_group_provider": P_group_v,
                    "verifier_model": model_to_use,
                    "verifier_temperature": verifier.temperature,
                    "provider_calls_used": budget.provider_calls,
                }
            )
            he["evidence"] = evid
            he["verified"] = verified

            logger.info(
                f"verify_top_k: edge id={he.get('id')} src={src_ids} "
                f"P_group_v={float(P_group_v):.6f} drop_group_v={drop_group_v:.6f} synergy_v={synergy_v:.6f} "
                f"score={provider_score:.4f} w_shadow={w_shadow:.4f} w_blend={w_blend:.4f} verified={verified}"
            )

        # Summary logging
        logger.info(
            f"verify_top_k: provider_calls_used={budget.provider_calls}/{cfg.COST_MAX_PROVIDER_CALLS} "
            f"cache_entries={provider_cache.size()} cache_hits={provider_cache.hits} cache_misses={provider_cache.misses}"
        )
        return hyperedges
    finally:
        try:
            global _CURRENT_PROVIDER_CACHE
            _CURRENT_PROVIDER_CACHE = prev_provider_cache if prev_provider_cache is not None else None
        except Exception:
            pass


def _first_output_snippet(response_text: str, limit: int = 64) -> str:
    if not isinstance(response_text, str):
        return "..."
    s = response_text.strip()
    if not s:
        return "..."
    return s[:limit]


def build_hypergraph(concepts: List[Dict[str, Any]], edges: List[Dict[str, Any]], response_text: str) -> Dict[str, Any]:
    """
    Build a minimal HIF v2 hypergraph.
    - nodes: array of HIFNode {id, type: CONCEPT|OUTPUT, text, span?, metadata?}
    - hyperedges: array of HIFHyperedge {id, source_nodes, target_node, weight, interaction_type, ...}

    If concepts/edges are empty, produce a minimal valid result mapping to a single OUTPUT node with no edges:
      { "id": "out1", "type": "OUTPUT", "text": <first token or response snippet> }
    """
    nodes: List[Dict[str, Any]] = []
    out_id = "out1"
    out_node = {"id": out_id, "type": "OUTPUT", "text": _first_output_snippet(response_text)}
    nodes.append(out_node)

    # Keep only valid HIFNode fields for concepts
    for idx, c in enumerate(concepts, 1):
        node = {
            "id": c.get("id") or f"c{idx}",
            "type": "CONCEPT",
            "text": c.get("text") or f"concept-{idx}",
        }
        # Optional span and metadata if provided in correct shape
        span = c.get("span")
        if isinstance(span, dict) and "start" in span and "end" in span:
            node["span"] = {"start": int(span["start"]), "end": int(span["end"])}
        meta = c.get("metadata")
        if isinstance(meta, dict):
            node["metadata"] = meta
        nodes.append(node)

    # Keep edges as provided if any, else default to empty
    hyperedges: List[Dict[str, Any]] = []
    for j, e in enumerate(edges, 1):
        # Validate minimal shape; coerce interaction_type default
        he = {
            "id": e.get("id") or f"h{j}",
            "source_nodes": list(e.get("source_nodes") or []),
            "target_node": e.get("target_node") or out_id,
            "weight": float(e.get("weight") or 0.0),
            "interaction_type": e.get("interaction_type") or "SYNERGY",
        }
        if "description" in e and isinstance(e["description"], str):
            he["description"] = e["description"]
        if "evidence" in e and isinstance(e["evidence"], dict):
            he["evidence"] = e["evidence"]
        hyperedges.append(he)

    # TODO(version2): from libs.hif.validator import validate_hypergraph; validate_hypergraph(hypergraph)
    hypergraph = {"nodes": nodes, "hyperedges": hyperedges}
    return hypergraph


# ----------------------------
# Minimal Postgres persistence (psycopg v3)
# ----------------------------
try:
    import psycopg  # type: ignore
    from psycopg import sql  # type: ignore
except Exception:
    psycopg = None  # type: ignore
    sql = None  # type: ignore

_DB_CONN = None  # module-level singleton connection

def _get_table_name() -> str:
    return getattr(CONFIG, "EXPLAINER_TABLE", "explanations_v2")

def get_db():
    """
    Lazy-connect to Postgres with autocommit and basic liveness check.
    Reads DATABASE_URL and DB_CONNECT_TIMEOUT.
    """
    global _DB_CONN
    if psycopg is None:
        logging.getLogger("explainer.worker").warning("psycopg not available; skipping DB persistence")
        return None
    dsn = getattr(CONFIG, "DATABASE_URL", None)
    if not dsn:
        logging.getLogger("explainer.worker").warning("DATABASE_URL not set; skipping DB persistence")
        return None
    timeout = int(getattr(CONFIG, "DB_CONNECT_TIMEOUT", 5))
    try:
        if _DB_CONN is None or getattr(_DB_CONN, "closed", True):
            _DB_CONN = psycopg.connect(dsn, connect_timeout=timeout)
            _DB_CONN.autocommit = True
        else:
            try:
                with _DB_CONN.cursor() as cur:
                    cur.execute("SELECT 1;")
            except Exception:
                try:
                    _DB_CONN.close()
                except Exception:
                    pass
                _DB_CONN = psycopg.connect(dsn, connect_timeout=timeout)
                _DB_CONN.autocommit = True
    except Exception as e:
        logging.getLogger("explainer.worker").error(f"DB connect error: {e}")
        _DB_CONN = None
    return _DB_CONN

def ensure_table_exists() -> None:
    """
    Ensure the explanations table exists with required indexes.
    """
    conn = get_db()
    if not conn or sql is None:
        return
    table = _get_table_name()
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
                ).format(table=sql.Identifier(table))
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {idx} ON {table}(status);").format(
                    idx=sql.Identifier(f"{table}_status_idx"),
                    table=sql.Identifier(table),
                )
            )
        logging.getLogger("explainer.worker").info(f"Ensured explanation table exists: {table}")
    except Exception as e:
        logging.getLogger("explainer.worker").error(f"ensure_table_exists error: {e}")

def upsert_explanation(
    request_id: str,
    status: str,
    provider: Optional[str],
    model: Optional[str],
    created_at: Optional[Any],
    hypergraph_json: Optional[str],
) -> None:
    """
    Upsert the explanation record.
    """
    conn = get_db()
    if not conn or sql is None:
        return
    table = _get_table_name()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {table} (request_id, status, provider, model, created_at, updated_at, hypergraph)
                    VALUES (%s, %s, %s, %s, %s, NOW(), CAST(%s AS jsonb))
                    ON CONFLICT (request_id) DO UPDATE
                    SET status=EXCLUDED.status,
                        provider=COALESCE(EXCLUDED.provider, {table}.provider),
                        model=COALESCE(EXCLUDED.model, {table}.model),
                        created_at=COALESCE(EXCLUDED.created_at, {table}.created_at),
                        updated_at=NOW(),
                        hypergraph=EXCLUDED.hypergraph;
                    """
                ).format(table=sql.Identifier(table)),
                (request_id, status, provider, model, created_at, hypergraph_json),
            )
    except Exception as e:
        logging.getLogger("explainer.worker").error(f"upsert_explanation error: {e}")

# TODO(version2): add connection pooling and retry/backoff for DB.
# TODO(version2): enforce JSON schema validation before writing/serving (use libs/hif/validator.py)

def persist_result(
    request_id: str,
    status: str,
    hypergraph: Optional[Dict[str, Any]] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    created_at: Optional[Any] = None,
) -> None:
    """
    Best-effort persistence to Postgres. Does not raise.
    """
    logger = logging.getLogger("explainer.worker")
    try:
        # Chaos: drop-s3 -> simulate persistence failures (caught below)
        try:
            _flags = _read_chaos_flags()
            _drop = _chaos_flag(_flags, "drop-s3")
            if _drop.get("enabled", True) and _chaos_hit(_drop.get("percent", 0)):
                raise RuntimeError("chaos_drop_persist")
        except Exception:
            # ignore control file errors; only intentional raises should bubble
            pass

        hypergraph_json = json.dumps(hypergraph) if isinstance(hypergraph, dict) else None
        upsert_explanation(
            request_id=request_id,
            status=status,
            provider=provider,
            model=model,
            created_at=created_at,
            hypergraph_json=hypergraph_json,
        )
    except Exception as e:
        logger.error(f"[persist] DB write failed for request_id={request_id}: {e}")
        # Reflect failure in status store when available
        try:
            _st = _get_status_store()
            if _st is not None:
                _st.update_fields(str(request_id), state="failed", error=str(e))  # type: ignore
        except Exception:
            pass
    finally:
        if status == "completed":
            logger.info(
                f"[persist] request_id={request_id} status={status} "
                f"nodes={len(hypergraph.get('nodes', [])) if isinstance(hypergraph, dict) else 0} "
                f"edges={len(hypergraph.get('hyperedges', [])) if isinstance(hypergraph, dict) else 0}"
            )
        else:
            logger.info(f"[persist] request_id={request_id} status={status}")


# ----------------------------
# Async worker implementation
# ----------------------------

class AsyncWorker:
    def __init__(self, cfg: Config, logger: logging.Logger) -> None:
        self.cfg = cfg
        self.logger = logger
        self._redis: Optional[Any] = None
        self._stop_evt: asyncio.Event = asyncio.Event()
        self._backoff_base = 0.5
        self._backoff_max = 5.0

    async def start(self) -> None:
        await self._connect_with_backoff()
        await self._ensure_consumer_group()

    async def stop(self) -> None:
        self._stop_evt.set()
        await self._close_redis()

    def request_stop(self, signame: str = "") -> None:
        self.logger.info(f"Shutdown requested via {signame or 'external'}")
        self._stop_evt.set()

    async def _connect_with_backoff(self) -> None:
        delay = self._backoff_base
        while not self._stop_evt.is_set():
            try:
                if AsyncRedis is None:
                    raise RuntimeError("redis.asyncio is not available")
                self._redis = AsyncRedis.from_url(self.cfg.REDIS_URL, decode_responses=True, health_check_interval=30)
                # Sanity ping
                await self._redis.ping()
                self.logger.info("Connected to Redis")
                return
            except Exception as e:
                self.logger.error(f"Redis connect failed: {e}. Retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
                delay = min(self._backoff_max, delay * 2)
        raise asyncio.CancelledError("Stopped before connecting")

    async def _ensure_consumer_group(self) -> None:
        assert self._redis is not None
        stream = self.cfg.REDIS_STREAM
        group = self.cfg.REDIS_CONSUMER_GROUP
        try:
            # Create group starting at '$' (new messages), mkstream to create stream if missing
            await self._redis.xgroup_create(stream, groupname=group, id="$", mkstream=True)
            self.logger.info(f"Created consumer group '{group}' on stream '{stream}'")
        except Exception as e:
            if "BUSYGROUP" in str(e):
                self.logger.info(f"Consumer group '{group}' already exists on stream '{stream}'")
            else:
                raise

    async def _close_redis(self) -> None:
        if self._redis is None:
            return
        try:
            # redis-py provides async close; pool disconnect as extra safety
            if hasattr(self._redis, "close"):
                await self._redis.close()  # type: ignore[arg-type]
            if hasattr(self._redis, "connection_pool"):
                await self._redis.connection_pool.disconnect()  # type: ignore[func-returns-value]
        except Exception as e:
            self.logger.warning(f"Error during Redis close: {e}")
        finally:
            self._redis = None

    async def run(self) -> None:
        await self.start()
        assert self._redis is not None
        stream = self.cfg.REDIS_STREAM
        group = self.cfg.REDIS_CONSUMER_GROUP
        consumer = self.cfg.REDIS_CONSUMER_NAME
        block_ms = self.cfg.REDIS_BLOCK_MS
        k = self.cfg.VERIFY_TOP_K

        # Signal handling for clean shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.request_stop, sig.name)
            except NotImplementedError:
                # Windows or restricted environment
                pass

        self.logger.info(
            f"Worker started: stream='{stream}' group='{group}' consumer='{consumer}' block_ms={block_ms}"
        )

        try:
            while not self._stop_evt.is_set():
                try:
                    results = await self._redis.xreadgroup(
                        groupname=group,
                        consumername=consumer,
                        streams={stream: ">"},
                        count=1,
                        block=block_ms,
                    )
                except Exception as e:
                    self.logger.error(f"XREADGROUP failed: {e}")
                    await asyncio.sleep(0.5)
                    continue

                if not results:
                    # Timeout, loop again to check stop event
                    continue

                # redis-py returns: [(stream_name, [(message_id, {field: value, ...}), ...])]
                for stream_name, messages in results:
                    for message_id, fields in messages:
                        acked = False
                        try:
                            raw = fields.get("data")
                            if raw is None:
                                self.logger.warning(f"Message {message_id} missing 'data' field. XACK to discard.")
                                await self._redis.xack(stream, group, message_id)
                                acked = True
                                self.logger.info(f"XACK message_id={message_id}")
                                continue

                            # Parse JSON
                            try:
                                payload_raw = json.loads(raw) if isinstance(raw, str) else raw
                                if not isinstance(payload_raw, dict):
                                    raise ValueError("payload JSON is not an object")
                            except Exception as pe:
                                self.logger.error(f"JSON parse error for message_id={message_id}: {pe}. XACK to avoid poison pill.")
                                # TODO(version2): send to dead-letter queue
                                await self._redis.xack(stream, group, message_id)
                                acked = True
                                self.logger.info(f"XACK message_id={message_id}")
                                try:
                                    emit_audit_worker("trace.complete", tenant_id="unknown", trace_id=str(req_id), status="complete")
                                except Exception:
                                    pass
                                try:
                                    emit_audit_worker("trace.complete", tenant_id="unknown", trace_id=str(req_id), status="complete")
                                except Exception:
                                    pass
                                try:
                                    emit_audit_worker("trace.complete", tenant_id="unknown", trace_id=str(req_id), status="complete")
                                except Exception:
                                    pass
                                continue

                            payload = parse_payload(payload_raw)
                            req_id = payload["request_id"]
                            provider = payload.get("provider")
                            model = payload.get("model")
                            # Queue lag observation from created_at
                            try:
                                created = payload.get("created_at")
                                lag_s = 0.0
                                if isinstance(created, str) and created:
                                    try:
                                        from datetime import datetime
                                        if created.endswith("Z"):
                                            dt = datetime.fromisoformat(created.replace("Z","+00:00"))
                                        else:
                                            dt = datetime.fromisoformat(created)
                                        lag_s = max(0.0, (time.time() - dt.timestamp()))
                                    except Exception:
                                        pass
                                if _otel is not None:
                                    _otel.queue_lag(lag_s)
                            except Exception:
                                pass

                            self.logger.info(f"Processing request_id={req_id} provider={provider} model={model} message_id={message_id}")
                            try:
                                emit_audit_worker(
                                    "trace.running",
                                    tenant_id="unknown",
                                    trace_id=str(req_id),
                                    status="running",
                                    extra={"provider": str(provider), "model": str(model)},
                                )
                            except Exception:
                                pass

                            # Stage timings
                            t0 = time.perf_counter()
                            concepts = concept_extraction(payload)
                            t1 = time.perf_counter()
                            self.logger.info(f"Stage concept_extraction duration_ms={(t1 - t0)*1000:.2f}")
                            try:
                                if _otel is not None:
                                    _otel.stage_duration("extract", float(t1 - t0))
                            except Exception:
                                pass

                            hyperedges = interaction_discovery(payload, concepts)
                            t2 = time.perf_counter()
                            self.logger.info(f"Stage interaction_discovery duration_ms={(t2 - t1)*1000:.2f}")
                            try:
                                if _otel is not None:
                                    _otel.stage_duration("decode", float(t2 - t1))
                            except Exception:
                                pass

                            verified = verify_top_k(payload, hyperedges, k)
                            t3 = time.perf_counter()
                            self.logger.info(f"Stage verify_top_k duration_ms={(t3 - t2)*1000:.2f}")
                            try:
                                if _otel is not None:
                                    _otel.stage_duration("attribution", float(t3 - t2))
                            except Exception:
                                pass

                            hypergraph = build_hypergraph(concepts, verified, response_text=payload.get("response", ""))
                            t4 = time.perf_counter()
                            self.logger.info(f"Stage build_hypergraph duration_ms={(t4 - t3)*1000:.2f}")
                            try:
                                if _otel is not None:
                                    _otel.stage_duration("prune", float(t4 - t3))
                                    try:
                                        nn = len(hypergraph.get("nodes", [])) if isinstance(hypergraph, dict) else 0
                                        ee = len(hypergraph.get("hyperedges", [])) if isinstance(hypergraph, dict) else 0
                                        _otel.hif_counts(nn, ee)
                                    except Exception:
                                        pass
                            except Exception:
                                pass

                            persist_result(
                                req_id,
                                status="completed",
                                hypergraph=hypergraph,
                                provider=provider,
                                model=model,
                                created_at=payload.get("created_at"),
                            )
                            t5 = time.perf_counter()
                            self.logger.info(f"Stage persist_result duration_ms={(t5 - t4)*1000:.2f}")
                            try:
                                if _otel is not None:
                                    _otel.stage_duration("persist", float(t5 - t4))
                            except Exception:
                                pass

                            # Success: XACK
                            await self._redis.xack(stream, group, message_id)
                            acked = True
                            self.logger.info(f"XACK message_id={message_id}")
                            try:
                                if _otel is not None:
                                    _otel.job_state("completed", "unknown", "unknown")
                            except Exception:
                                pass

                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            # Pipeline error => do not XACK to allow redelivery (unless parse error handled above)
                            self.logger.exception(f"Processing error for message_id={message_id}: {e}")
                            try:
                                rid = fields.get("request_id") or "unknown"
                                persist_result(str(rid), status="failed", hypergraph=None)
                            except Exception:
                                pass
                            try:
                                if _otel is not None:
                                    _otel.job_state("failed", "unknown", "unknown")
                            except Exception:
                                pass
                        finally:
                            if self._stop_evt.is_set():
                                # If we're stopping and still have a message not acked due to failure we leave it pending
                                pass
        finally:
            await self.stop()


# ----------------------------
# Sync fallback worker
# ----------------------------

class SyncWorker:
    def __init__(self, cfg: Config, logger: logging.Logger) -> None:
        self.cfg = cfg
        self.logger = logger
        self._redis: Optional[Any] = None
        self._stop = False
        self._backoff_base = 0.5
        self._backoff_max = 5.0

    def request_stop(self, signame: str = "") -> None:
        self.logger.info(f"Shutdown requested via {signame or 'external'}")
        self._stop = True

    def start(self) -> None:
        self._connect_with_backoff()
        self._ensure_consumer_group()

    def stop(self) -> None:
        try:
            if self._redis is not None:
                if hasattr(self._redis, "close"):
                    self._redis.close()  # type: ignore[attr-defined]
                if hasattr(self._redis, "connection_pool"):
                    self._redis.connection_pool.disconnect()  # type: ignore[attr-defined]
        finally:
            self._redis = None

    def _connect_with_backoff(self) -> None:
        delay = self._backoff_base
        while not self._stop:
            try:
                if redis is None:
                    raise RuntimeError("redis client is not available")
                self._redis = redis.Redis.from_url(self.cfg.REDIS_URL, decode_responses=True, health_check_interval=30)  # type: ignore[call-arg]
                self._redis.ping()
                self.logger.info("Connected to Redis (sync fallback)")
                return
            except Exception as e:
                self.logger.error(f"Redis connect failed (sync): {e}. Retrying in {delay:.1f}s")
                time.sleep(delay)
                delay = min(self._backoff_max, delay * 2)
        raise RuntimeError("Stopped before connecting (sync)")

    def _ensure_consumer_group(self) -> None:
        assert self._redis is not None
        stream = self.cfg.REDIS_STREAM
        group = self.cfg.REDIS_CONSUMER_GROUP
        try:
            self._redis.xgroup_create(stream, groupname=group, id="$", mkstream=True)  # type: ignore[call-arg]
            self.logger.info(f"Created consumer group '{group}' on stream '{stream}' (sync)")
        except Exception as e:
            if "BUSYGROUP" in str(e):
                self.logger.info(f"Consumer group '{group}' already exists on stream '{stream}' (sync)")
            else:
                raise

    def run(self) -> None:
        self.start()
        assert self._redis is not None
        stream = self.cfg.REDIS_STREAM
        group = self.cfg.REDIS_CONSUMER_GROUP
        consumer = self.cfg.REDIS_CONSUMER_NAME
        block_ms = self.cfg.REDIS_BLOCK_MS
        k = self.cfg.VERIFY_TOP_K

        # Signal handling
        try:
            signal.signal(signal.SIGINT, lambda *_: self.request_stop("SIGINT"))
            signal.signal(signal.SIGTERM, lambda *_: self.request_stop("SIGTERM"))
        except Exception:
            pass

        self.logger.info(
            f"(sync) Worker started: stream='{stream}' group='{group}' consumer='{consumer}' block_ms={block_ms}"
        )

        try:
            while not self._stop:
                try:
                    results = self._redis.xreadgroup(  # type: ignore[attr-defined]
                        groupname=group,
                        consumername=consumer,
                        streams={stream: ">"},
                        count=1,
                        block=block_ms,
                    )
                except Exception as e:
                    self.logger.error(f"(sync) XREADGROUP failed: {e}")
                    time.sleep(0.5)
                    continue

                if not results:
                    continue

                for stream_name, messages in results:
                    for message_id, fields in messages:
                        acked = False
                        try:
                            raw = fields.get("data")
                            if raw is None:
                                self.logger.warning(f"(sync) Message {message_id} missing 'data'. XACK to discard.")
                                self._redis.xack(stream, group, message_id)  # type: ignore[attr-defined]
                                acked = True
                                self.logger.info(f"(sync) XACK message_id={message_id}")
                                continue

                            try:
                                payload_raw = json.loads(raw) if isinstance(raw, str) else raw
                                if not isinstance(payload_raw, dict):
                                    raise ValueError("payload JSON is not an object")
                            except Exception as pe:
                                self.logger.error(f"(sync) JSON parse error for message_id={message_id}: {pe}. XACK to avoid poison pill.")
                                # TODO(version2): send to dead-letter queue
                                self._redis.xack(stream, group, message_id)  # type: ignore[attr-defined]
                                acked = True
                                self.logger.info(f"(sync) XACK message_id={message_id}")
                                continue

                            payload = parse_payload(payload_raw)
                            req_id = payload["request_id"]
                            provider = payload.get("provider")
                            model = payload.get("model")
                            self.logger.info(f"(sync) Processing request_id={req_id} provider={provider} model={model} message_id={message_id}")

                            t0 = time.perf_counter()
                            concepts = concept_extraction(payload)
                            t1 = time.perf_counter()
                            self.logger.info(f"(sync) Stage concept_extraction duration_ms={(t1 - t0)*1000:.2f}")

                            hyperedges = interaction_discovery(payload, concepts)
                            t2 = time.perf_counter()
                            self.logger.info(f"(sync) Stage interaction_discovery duration_ms={(t2 - t1)*1000:.2f}")

                            verified = verify_top_k(payload, hyperedges, k)
                            t3 = time.perf_counter()
                            self.logger.info(f"(sync) Stage verify_top_k duration_ms={(t3 - t2)*1000:.2f}")

                            hypergraph = build_hypergraph(concepts, verified, response_text=payload.get("response", ""))
                            t4 = time.perf_counter()
                            self.logger.info(f"(sync) Stage build_hypergraph duration_ms={(t4 - t3)*1000:.2f}")

                            persist_result(
                                req_id,
                                status="completed",
                                hypergraph=hypergraph,
                                provider=provider,
                                model=model,
                                created_at=payload.get("created_at"),
                            )
                            t5 = time.perf_counter()
                            self.logger.info(f"(sync) Stage persist_result duration_ms={(t5 - t4)*1000:.2f}")

                            self._redis.xack(stream, group, message_id)  # type: ignore[attr-defined]
                            acked = True
                            self.logger.info(f"(sync) XACK message_id={message_id}")

                        except Exception as e:
                            self.logger.exception(f"(sync) Processing error for message_id={message_id}: {e}")
                            try:
                                rid = fields.get("request_id") or "unknown"
                                persist_result(str(rid), status="failed", hypergraph=None)
                            except Exception:
                                pass
                        finally:
                            if self._stop:
                                pass
        finally:
            self.stop()


async def _run_worker_async(cfg: Config, logger: logging.Logger) -> None:
    # Optional SAE decode microservice startup (runs in-process)
    service = await _init_sae_service_from_env(logger)

    if AsyncRedis is None:
        logger.warning("redis.asyncio not available; falling back to sync worker")
        try:
            sw = SyncWorker(cfg, logger)
            sw.run()
        finally:
            if service is not None:
                await _stop_sae_service()
        return

    worker = AsyncWorker(cfg, logger)
    try:
        await worker.run()
    finally:
        if service is not None:
            await _stop_sae_service()


def worker_main() -> None:
    logger = setup_logging(CONFIG.LOG_LEVEL)
    logger.info("Launching Interaction Engine Worker")
    try:
        logger.info("config: %s", CONFIG.safe_repr())
    except Exception:
        pass
    # Ensure table exists once at startup (best-effort)
    try:
        ensure_table_exists()
        logger.info(f"Ensured explanation table: {_get_table_name()}")
    except Exception as e:
        logger.error(f"ensure_table_exists failed: {e}")
    try:
        asyncio.run(_run_worker_async(CONFIG, logger))
    except KeyboardInterrupt:
        # In some environments, asyncio loop interruption bubbles here
        logger.info("Worker terminated via KeyboardInterrupt")
    except Exception as e:
        logger.exception(f"Worker terminated with error: {e}")


# ======================================================================================
# Legacy DEV skeleton retained for compatibility (produces HIF v1 artifacts via S3/local)
# ======================================================================================

import gzip
import pathlib

from typing import Any as _Any  # type: ignore  # alias to avoid confusion in legacy code
from typing import Dict as _Dict  # type: ignore


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# S3Store moved to services.explainer.src.s3_store (imported above as NewS3Store)


def build_hif(
    *,
    model_name: str,
    model_hash: str,
    sae_dictionary: str,
    granularity: str,
    nodes: list[dict],
    incidences: list[dict],
    limits: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Construct a HIF v1 JSON object.
    """
    return {
        "network-type": "directed",
        "nodes": nodes,
        "incidences": incidences,
        "meta": {
            "model_name": model_name,
            "model_hash": model_hash,
            "sae_dictionary": sae_dictionary,
            "granularity": granularity,
            "created_at": now_iso(),
            "limits": limits
            or {
                "min_edge_weight": 0.01,
                "max_nodes": 5000,
                "max_incidences": 20000,
            },
            "version": "hif-1",
        },
    }


async def fetch_activation_shards_stub(trace_id: str) -> Dict[str, Any]:
    """
    Placeholder for reading from Redis/NVMe.
    Returns a minimal structure so the pipeline can proceed.
    """
    await asyncio.sleep(0.01)
    return {"trace_id": trace_id, "shards": 1, "sample": True}


async def sae_decode_stub(activations: Dict[str, Any], featureset: str) -> Dict[str, float]:
    """
    SAE decode adapter.

    Behavior:
    - Preferred (if enabled): Use in-process SAEDecodeService for batched projection.
    - Optional: if USE_SAE_LOADER=1 and DICT_ROOT is valid, use CPU project_topk loader path.
    - Else: legacy synthetic features.
    """
    logger = logging.getLogger("explainer.worker")

    # Chaos: slow-sae -> add artificial latency/jitter before decode
    try:
        _flags = _read_chaos_flags()
        _slow = _chaos_flag(_flags, "slow-sae")
        if _slow.get("enabled", True) and _chaos_hit(_slow.get("percent", 0)):
            base_ms = float(_slow.get("jitter_ms", 150) or 150.0)
            low = max(0.0, base_ms * 0.5)
            high = max(low, base_ms * 1.5)
            delay = random.uniform(low, high) / 1000.0
            await asyncio.sleep(delay)
    except Exception:
        # never break pipeline on chaos control read errors
        pass

    # 1) Service-enabled path
    try:
        if os.getenv("ENABLE_SAE_SERVICE", "0") == "1":
            # Ensure service initialized
            try:
                if _SAE_SERVICE is None:
                    await _init_sae_service_from_env(logger)
            except Exception:
                pass
            if _SAE_SERVICE is not None:
                try:
                    layer_idx = int(os.getenv("SAE_LAYER", "12"))
                except Exception:
                    layer_idx = 12
                # Probe hidden_dim using service's dictionary representation
                try:
                    lw = _SAE_SERVICE._get_layer_rep(layer_idx)  # type: ignore[attr-defined]
                    _, hidden_dim = lw.shape
                except Exception as e:
                    logger.warning(f"SAE service layer probe failed; falling back to loader/stub: {e}")
                    raise
                # Deterministic probe vector from trace_id
                seed_src = str(activations.get("trace_id") or "0")
                seed_int = int(hashlib.sha256(seed_src.encode("utf-8")).hexdigest()[:8], 16)
                try:
                    import numpy as _np  # type: ignore
                    rng = _np.random.default_rng(seed_int)  # type: ignore
                    probe = rng.standard_normal(hidden_dim, dtype=_np.float32)  # type: ignore
                except Exception:
                    r = random.Random(seed_int)
                    probe = [r.uniform(-1.0, 1.0) for _ in range(hidden_dim)]
                res = await _SAE_SERVICE.decode(layer_idx, probe)  # type: ignore
                return {f"feat_{int(i)}": float(s) for (i, s) in res.items()}
    except Exception as e:
        logger.warning(f"SAE service path failed; continuing with optional loader: {e}")

    # 2) Optional loader CPU fallback (legacy)
    try:
        use_loader = os.getenv("USE_SAE_LOADER", "0") == "1"
        dict_root = (os.getenv("DICT_ROOT", "") or "").strip()
        dict_name = (featureset or "sae-gpt4-2m") if isinstance(featureset, str) and featureset else "sae-gpt4-2m"
        if use_loader and dict_root:
            try:
                from libs.sae.loader import SAEConfig, load_dictionary, project_topk  # type: ignore
                cfg = SAEConfig(root_path=dict_root, dictionary_name=dict_name, device="auto", dtype="fp16", prefer_sparse=True, cache_layers=2)
                dct = load_dictionary(cfg)
                try:
                    layer_idx = int(os.getenv("SAE_LAYER", "12"))
                except Exception:
                    layer_idx = 12
                layer = dct.get_layer(layer_idx)
                _, hidden_dim = layer.shape
                seed_src = str(activations.get("trace_id") or "0")
                seed_int = int(hashlib.sha256(seed_src.encode("utf-8")).hexdigest()[:8], 16)
                try:
                    import numpy as _np  # type: ignore
                    rng = _np.random.default_rng(seed_int)  # type: ignore
                    probe = rng.standard_normal(hidden_dim, dtype=_np.float32)  # type: ignore
                except Exception:
                    r = random.Random(seed_int)
                    probe = [r.uniform(-1.0, 1.0) for _ in range(hidden_dim)]
                topk = 256
                try:
                    topk = max(1, int(os.getenv("SAE_TOPK", "256")))
                except Exception:
                    topk = 256
                # Apply backpressure override if present
                try:
                    ov = _BP_RUNTIME_OVERRIDES.get("sae_topk")
                    if ov is not None:
                        topk = max(1, min(int(topk), int(ov)))
                except Exception:
                    pass
                top = project_topk(layer, probe, k=topk)
                return {f"feat_{int(i)}": float(s) for (i, s) in top}
            except Exception as e:
                logger.warning(f"SAE loader unavailable or failed (falling back to stub): {e}")
        elif use_loader and not dict_root:
            logger.info("USE_SAE_LOADER=1 but DICT_ROOT not set; using stub path")
    except Exception as e:
        logger.warning(f"sae_decode_stub optional loader path encountered an error; using stub: {e}")

    # 3) Legacy stub
    await asyncio.sleep(0.02)
    return {
        "feat_1024": 4.5,
        "feat_77": 2.2,
        "feat_302": 1.1,
    }


async def attribution_stub(features: Dict[str, float], granularity: str) -> list[dict]:
    """
    Placeholder attribution. Returns weighted incidences to a single output token.
    """
    await asyncio.sleep(0.02)
    out_token = {"id": "token_out_1", "type": "output_token", "label": "Paris is the capital of France.", "position": 1}
    nodes = [{"id": fid, "type": "sae_feature"} for fid in features.keys()] + [out_token]
    edges = []
    for fid, score in features.items():
        edges.append(
            {
                "id": "e_" + fid,
                "node_ids": [fid, out_token["id"]],
                "weight": round(min(1.0, 0.15 + score / 10), 3),
                "metadata": {"type": "causal_circuit", "method": "stub", "window": "sent-1"},
            }
        )
    return nodes, edges


def prune_graph_stub(nodes: list[dict], incidences: list[dict], min_w: float = 0.01, max_nodes: int = 5000, max_edges: int = 20000):
    """
    Simple pruning logic aligned with guardrails.
    """
    keep_edges = [e for e in incidences if e.get("weight", 0) >= min_w]
    if len(keep_edges) > max_edges:
        keep_edges = keep_edges[:max_edges]
    # collect referenced node ids
    ref_ids = set()
    for e in keep_edges:
        for nid in e.get("node_ids", []):
            ref_ids.add(nid)
    keep_nodes = [n for n in nodes if n.get("id") in ref_ids]
    if len(keep_nodes) > max_nodes:
        keep_nodes = keep_nodes[:max_nodes]
    return keep_nodes, keep_edges


async def process_envelope(envelope: Dict[str, Any], store: S3Store) -> Dict[str, Any]:
    """
    Legacy core processing pipeline (v1).
    """
    trace_id = envelope["trace_id"]
    granularity = envelope.get("granularity", "sentence")
    featureset = envelope.get("featureset", "sae-gpt4-2m")
    model_hash = envelope.get("model_hash", "unknown")
    model_name = envelope.get("model_name", "unknown")
    tenant_id = str(envelope.get("tenant_id") or "anon")

    # Backpressure evaluation (early) and runtime overrides
    bp = None
    try:
        if _BP_CTRL is not None:
            bp = _BP_CTRL.evaluate(tenant_id, granularity)  # type: ignore
    except Exception:
        bp = None

    # Chaos: rate-limit-spike -> escalate backpressure severity (soft->hard)
    try:
        _flags = _read_chaos_flags()
        _rl_cfg = _chaos_flag(_flags, "rate-limit-spike")
        if _rl_cfg.get("enabled", True) and (_rl_cfg.get("severity") in ("soft", "hard") or _rl_cfg):
            target = str(_rl_cfg.get("severity") or "hard").lower()
            # Create minimal stub if controller/state types not available
            if bp is None:
                class _BPStub:
                    def __init__(self, level: str, actions: List[str]) -> None:
                        self.level = level
                        self.actions = actions
                bp = _BPStub(level="normal", actions=[])  # type: ignore
            # Escalate
            lvl = getattr(bp, "level", "normal")
            if target == "hard":
                setattr(bp, "level", "hard")
                acts = set(list(getattr(bp, "actions", [])) + ["reduce-samples", "reduce-topk", "reduce-layers", "saliency-fallback", "drop"])
                setattr(bp, "actions", list(acts))
            elif target == "soft" and lvl == "normal":
                setattr(bp, "level", "soft")
                acts = set(list(getattr(bp, "actions", [])) + ["reduce-samples", "reduce-topk"])
                setattr(bp, "actions", list(acts))
    except Exception:
        pass

    # Apply token->sentence downgrade if advised
    granularity_downgraded = False
    try:
        if bp is not None and "token->sentence" in bp.actions and str(granularity).lower() == "token":
            granularity = "sentence"
            granularity_downgraded = True
    except Exception:
        pass

    # SAE runtime overrides for service/loader paths
    try:
        _BP_RUNTIME_OVERRIDES.clear()
        if bp is not None:
            if "reduce-topk" in bp.actions:
                _BP_RUNTIME_OVERRIDES["sae_topk"] = 64 if bp.level == "hard" else 128
            if "reduce-layers" in bp.actions:
                _BP_RUNTIME_OVERRIDES["sae_cache_layers"] = 1 if bp.level == "hard" else 2
    except Exception:
        pass

    # Surface bp state and optionally shed early
    try:
        _st_tmp = _get_status_store()
        if _st_tmp is not None and bp is not None:
            _st_tmp.update_fields(  # type: ignore
                trace_id,
                bp_level=bp.level,
                bp_actions=list(bp.actions),
                granularity_downgraded=granularity_downgraded,
            )
    except Exception:
        pass

    if bp is not None and "drop" in bp.actions:
        try:
            _st_tmp2 = _get_status_store()
            if _st_tmp2 is not None:
                _st_tmp2.update_fields(trace_id, state="shed", error="backpressure_dropped")  # type: ignore
        except Exception:
            pass
        try:
            emit_audit_worker("trace.canceled", tenant_id=tenant_id, trace_id=trace_id, status="canceled", reason="backpressure_dropped")
        except Exception:
            pass
        return {
            "trace_id": trace_id,
            "state": "shed",
            "granularity": granularity,
            "featureset": featureset,
            "created_at": now_iso(),
        }

    # Optional status store initialization and initial status
    _st = _get_status_store()
    _now = _now_epoch()
    _ttl = float(_ttl_days()) * 86400.0
    _created = _now
    try:
        if _st is not None:
            existing = _st.get_status(trace_id)  # type: ignore
            if isinstance(existing, dict) and "created_at" in existing:
                try:
                    _created = float(existing.get("created_at") or _now)
                except Exception:
                    _created = _now
            _st.put_status({  # type: ignore
                "trace_id": trace_id,
                "state": "running",
                "progress": 0.0,
                "stage": "fetch",
                "granularity": granularity,
                "featureset": featureset,
                "created_at": _created,
                "updated_at": _now,
                "expires_at": _created + _ttl,
            })
            try:
                emit_audit_worker("trace.running", tenant_id=tenant_id, trace_id=trace_id, status="running", extra={"granularity": granularity, "featureset": featureset})
            except Exception:
                pass
    except Exception:
        pass

    # Build meta for persistence key layout
    store_meta = {
        "granularity": granularity,
        "sae_dictionary": featureset,
        "model_hash": model_hash,
    }
    # Optional status store initialization and initial status
    _st = _get_status_store()
    _now = _now_epoch()
    _ttl = float(_ttl_days()) * 86400.0
    _created = _now
    try:
        if _st is not None:
            existing = _st.get_status(trace_id)  # type: ignore
            if isinstance(existing, dict) and "created_at" in existing:
                try:
                    _created = float(existing.get("created_at") or _now)
                except Exception:
                    _created = _now
            _st.put_status({  # type: ignore
                "trace_id": trace_id,
                "state": "running",
                "progress": 0.0,
                "stage": "fetch",
                "granularity": granularity,
                "featureset": featureset,
                "created_at": _created,
                "updated_at": _now,
                "expires_at": _created + _ttl,
            })
    except Exception:
        pass

    # Build meta for persistence key layout
    store_meta = {
        "granularity": granularity,
        "sae_dictionary": featureset,
        "model_hash": model_hash,
    }

    # 1) Fetch activations
    activations = await fetch_activation_shards_stub(trace_id)
    try:
        if _st is not None:
            _st.update_fields(trace_id, state="running", stage="decode", progress=25.0)  # type: ignore
    except Exception:
        pass
    try:
        if _st is not None:
            _st.update_fields(trace_id, state="running", stage="decode", progress=25.0)  # type: ignore
    except Exception:
        pass

    # 2) SAE decode
    feats = await sae_decode_stub(activations, featureset)
    try:
        if _st is not None:
            _st.update_fields(trace_id, state="running", stage="attribution", progress=60.0)  # type: ignore
    except Exception:
        pass

    # 3) Attribution (budgeted, env-driven; non-breaking fallback)
    out_text = os.getenv("DEV_OUTPUT_SENTENCE", "Paris is the capital of France.")
    toks = [t for t in out_text.strip().split() if t]

    # Determine token index/window and target output node id
    if granularity == "token":
        tok_idx = max(0, len(toks) - 1) if toks else 0  # last token
        out_id = f"token_out_{tok_idx+1}"
        window_tag = f"tok-{tok_idx}"
    else:
        tok_idx = 0
        out_id = "token_out_1"
        window_tag = "sent-1"

    # Build nodes (features + single output token node for this window)
    out_token = {"id": out_id, "type": "output_token", "label": out_text, "position": tok_idx + 1}
    nodes = [{"id": fid, "type": "sae_feature"} for fid in feats.keys()] + [out_token]

    def _activation_fallback_edges(feat_map: Dict[str, float], target_id: str, cap: int, min_w: float, window: str) -> list[dict]:
        # Normalize activations to [0,1] and cap/prune
        vals = [float(v) for v in feat_map.values()] or [1.0]
        m = max(abs(x) for x in vals) or 1.0
        items = sorted(((fid, max(0.0, float(v) / m)) for fid, v in feat_map.items()), key=lambda kv: kv[1], reverse=True)
        edges: list[dict] = []
        for i, (fid, w) in enumerate(items, start=1):
            if w < float(min_w):
                continue
            edges.append(
                {
                    "id": f"att_fallback_{i}",
                    "node_ids": [fid, target_id],
                    "weight": float(min(1.0, w)),
                    "metadata": {"type": "causal_circuit", "method": "fallback", "window": window},
                }
            )
            if len(edges) >= int(cap):
                break
        return edges

    # Env-driven attribution config
    method_sent = (os.getenv("ATTR_METHOD_SENTENCE", "acdc") or "acdc").lower()
    method_tok = (os.getenv("ATTR_METHOD_TOKEN", "shapley") or "shapley").lower()
    sent_budget = _env_int("ATTR_SENTENCE_BUDGET_MS", "900", minimum=0)
    tok_budget = _env_int("ATTR_TOKEN_BUDGET_MS", "3500", minimum=0)
    max_samples = _env_int("ATTR_MAX_SAMPLES", "512", minimum=1)
    early_delta = _env_float("ATTR_EARLY_STOP_DELTA", "0.01", minimum=0.0)
    cap_edges = _env_int("ATTR_PER_TOKEN_CAP", "256", minimum=1)
    min_edge_w = _env_float("ATTR_MIN_EDGE_WEIGHT", "0.01", minimum=0.0)
    seed_env = os.getenv("ATTR_RANDOM_SEED")
    seed_val = int(seed_env) if isinstance(seed_env, str) and seed_env.strip().lstrip("-").isdigit() else None

    edges: list[dict] = []
    try:
        # Chaos: fail-attribution -> optionally force fallback or hard fail
        try:
            _flags = _read_chaos_flags()
            _fail = _chaos_flag(_flags, "fail-attribution")
            if _fail.get("enabled", True) and _chaos_hit(_fail.get("percent", 0)):
                if str(_fail.get("mode", "fallback")).lower() in ("fail", "hard"):
                    try:
                        if _st is not None:
                            _st.update_fields(trace_id, state="failed", error="chaos_fail_attribution")  # type: ignore
                    except Exception:
                        pass
                    raise RuntimeError("chaos_fail_attribution_hard")
                else:
                    # soft mode: raise to hit fallback block below
                    raise RuntimeError("chaos_fail_attribution")
        except RuntimeError:
            # propagate intentional chaos failures
            raise
        except Exception:
            # ignore control read issues
            pass

        if granularity == "sentence":
            if sentence_attribution is None or AttributionConfig is None:
                raise RuntimeError("attribution module unavailable")
            cfg = AttributionConfig(
                method="acdc" if method_sent in ("acdc", "auto") else method_sent,
                max_samples=max_samples,
                early_stop_delta=early_delta,
                max_ms_budget=sent_budget or 900,
                random_seed=seed_val,
                min_edge_weight=min_edge_w,
                per_token_incident_cap=cap_edges,
            )
            edges = sentence_attribution(feats, toks, cfg)
        else:
            if token_attribution is None or AttributionConfig is None:
                raise RuntimeError("attribution module unavailable")
            cfg = AttributionConfig(
                method="shapley" if method_tok in ("shapley", "auto") else method_tok,
                max_samples=max_samples,
                early_stop_delta=early_delta,
                max_ms_budget=tok_budget or 3500,
                random_seed=seed_val,
                min_edge_weight=min_edge_w,
                per_token_incident_cap=cap_edges,
            )
            edges = token_attribution(feats, tok_idx, toks, cfg)
    except Exception:
        # Keep existing fallback behavior on exceptions or budget exhaustion
        edges = _activation_fallback_edges(feats, out_id, cap_edges, min_edge_w, window_tag)

    # 4) Augment nodes with labels/attrs
    for n in nodes:
        if n["type"] == "sae_feature" and "label" not in n:
            # Placeholder feature label
            n["label"] = f"Feature {n['id'].split('_')[-1]}"

    # 5) Prune and group (standardized)
    logger = logging.getLogger("explainer.worker")
    cfg_hg = None
    try:
        if HGConfig is not None:
            cfg_hg = HGConfig(
                min_edge_weight=_env_float("HG_MIN_EDGE_WEIGHT", "0.01", minimum=0.0),
                per_token_incident_cap=_env_int("HG_PER_TOKEN_CAP", "256", minimum=1),
                max_nodes=_env_int("HG_MAX_NODES", "5000", minimum=1),
                max_incidences=_env_int("HG_MAX_INCIDENCES", "20000", minimum=1),
                grouping=(os.getenv("HG_GROUPING", "supernode") or "supernode").lower(),
                supernode_min_group=_env_int("HG_SUPERNODE_MIN_GROUP", "3", minimum=2),
                supernode_label_delim=os.getenv("HG_LABEL_DELIM", ":") or ":",
            )
    except Exception as e:
        logger.warning(f"HypergraphConfig init failed; falling back to stub: {e}")
        cfg_hg = None

    grouping_meta = {"supernodes": {}}
    try:
        if hg_prune_and_group is not None and cfg_hg is not None:
            pn, pe, grouping_meta = hg_prune_and_group(nodes, edges, cfg_hg)
        else:
            pn, pe = prune_graph_stub(
                nodes,
                edges,
                min_w=_env_float("HG_MIN_EDGE_WEIGHT", "0.01", minimum=0.0),
                max_nodes=_env_int("HG_MAX_NODES", "5000", minimum=1),
                max_edges=_env_int("HG_MAX_INCIDENCES", "20000", minimum=1),
            )
    except Exception as e:
        logger.warning(f"prune_and_group failed; using stub pruning: {e}")
        pn, pe = prune_graph_stub(
            nodes,
            edges,
            min_w=_env_float("HG_MIN_EDGE_WEIGHT", "0.01", minimum=0.0),
            max_nodes=_env_int("HG_MAX_NODES", "5000", minimum=1),
            max_edges=_env_int("HG_MAX_INCIDENCES", "20000", minimum=1),
        )

    # 6) Assemble HIF (legacy v1)
    meta = {
        "model_name": model_name,
        "model_hash": model_hash,
        "sae_dictionary": featureset,
        "granularity": granularity,
    }
    try:
        if hg_build_hif is not None and cfg_hg is not None:
            hif = hg_build_hif(pn, pe, meta, cfg_hg)
            # Optional validation hook
            try:
                if hg_validate_hif is not None:
                    hg_validate_hif(hif)
            except Exception as ve:
                logger.info(f"HIF validation skipped/failed: {ve}")
        else:
            # Fallback to legacy constructor
            hif = build_hif(
                model_name=model_name,
                model_hash=model_hash,
                sae_dictionary=featureset,
                granularity=granularity,
                nodes=pn,
                incidences=pe,
            )
    except Exception as e:
        # Final safety net - fallback to minimal legacy constructor on any error
        logger.warning(f"build_hif failed; using fallback legacy constructor: {e}")
        hif = build_hif(
            model_name=model_name,
            model_hash=model_hash,
            sae_dictionary=featureset,
            granularity=granularity,
            nodes=pn,
            incidences=pe,
        )

    # Emit grouping summary in logs (schema-compliant HIF omits this)
    try:
        supernode_count = sum(1 for n in hif.get("nodes", []) if n.get("type") == "circuit_supernode")
        collapsed_features = sum(len(v) for v in (grouping_meta.get("supernodes", {}) or {}).values())
        logger.info(f"grouping_summary: supernodes={supernode_count} collapsed_features={collapsed_features}")
    except Exception:
        pass

    # 7) Persist
    artifact_uri = store.put_json_gz(trace_id, hif)

    result = {
        "trace_id": trace_id,
        "state": "complete",
        "artifact_uri": artifact_uri,
        "granularity": granularity,
        "featureset": featureset,
        "created_at": now_iso(),
    }
    print(f"[worker] processed {trace_id}: {result}")
    return result


async def dev_main():
    """
    DEV_MODE driver: generate a sample envelope and process it once.
    """
    trace_id = "trc_" + uuid.uuid4().hex[:12]
    envelope = {
        "trace_id": trace_id,
        "granularity": os.getenv("DEV_GRANULARITY", "sentence"),
        "featureset": os.getenv("DEV_FEATURESET", "sae-gpt4-2m"),
        "model_name": os.getenv("DEV_MODEL_NAME", "gpt-4-turbo"),
        "model_hash": os.getenv("DEV_MODEL_HASH", "devhash1234"),
    }
    store = S3Store(bucket=os.getenv("S3_BUCKET"), prefix=os.getenv("S3_PREFIX", "traces"))
    await process_envelope(envelope, store)


if __name__ == "__main__":
    # Default to running the Redis worker; enable legacy DEV mode explicitly with DEV_MODE=1
    if os.getenv("DEV_MODE", "0") == "1":
        asyncio.run(dev_main())
    else:
        worker_main()