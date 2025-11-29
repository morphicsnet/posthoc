from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from fastapi import HTTPException, Request, status


@dataclass(frozen=True)
class _CategoryCfg:
    burst: int
    rps: float


class TokenBucketLimiter:
    """
    Simple token bucket limiter:
      - capacity: max tokens (burst)
      - refill_rate_per_sec: tokens added per second
    """

    def __init__(self, capacity: int, refill_rate_per_sec: float) -> None:
        self.capacity = float(max(0, capacity))
        self.refill_rate = float(max(0.0, refill_rate_per_sec))
        self.tokens = float(capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def _refill_locked(self) -> None:
        now = time.monotonic()
        dt = max(0.0, now - self._last)
        if dt > 0.0 and self.refill_rate > 0.0:
            self.tokens = min(self.capacity, self.tokens + dt * self.refill_rate)
        self._last = now

    def allow(self, n: int = 1) -> bool:
        if n <= 0:
            return True
        with self._lock:
            self._refill_locked()
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False

    def retry_after_seconds(self, n: int = 1) -> int:
        with self._lock:
            self._refill_locked()
            if self.tokens >= n:
                return 0
            missing = max(0.0, n - self.tokens)
            if self.refill_rate <= 0.0:
                # If never refilling, advise a conservative retry-after
                return int(max(1.0, self.capacity or 1.0))
            wait = missing / self.refill_rate
            # HTTP Retry-After should be integer seconds
            return max(1, int(wait + 0.999))

    def export_state(self) -> Dict[str, float]:
        with self._lock:
            # wall clock timestamp for persistence recovery
            return {"tokens": float(self.tokens), "ts": float(time.time())}

    def import_state(self, state: Dict[str, float]) -> None:
        try:
            tokens = float(state.get("tokens", 0.0))
            ts = float(state.get("ts", time.time()))
        except Exception:
            return
        now = time.time()
        elapsed = max(0.0, now - ts)
        with self._lock:
            # Recover and refill based on wall clock delta
            recovered = min(self.capacity, max(0.0, tokens) + elapsed * self.refill_rate)
            self.tokens = recovered
            self._last = time.monotonic()


class RateLimiterRegistry:
    """
    Keeps per-tenant buckets for categories:
      - write (completions, webhooks, cancel)
      - read (status, graph, stream)
    Optional JSON persistence via RL_COUNTERS_JSON_PATH.
    """

    def __init__(self) -> None:
        self._buckets: Dict[str, TokenBucketLimiter] = {}
        self._lock = threading.Lock()

        self.cfg_read = _CategoryCfg(
            burst=int(os.getenv("RATE_LIMIT_READ_BURST", "200") or "200"),
            rps=float(os.getenv("RATE_LIMIT_READ_RPS", "50") or "50"),
        )
        self.cfg_write = _CategoryCfg(
            burst=int(os.getenv("RATE_LIMIT_WRITE_BURST", "20") or "20"),
            rps=float(os.getenv("RATE_LIMIT_WRITE_RPS", "5") or "5"),
        )

        self._persist_path: Optional[str] = os.getenv("RL_COUNTERS_JSON_PATH") or None
        self._loaded = False

    def _load_persisted(self) -> Dict[str, Dict[str, float]]:
        if self._persist_path is None or self._loaded:
            return {}
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._loaded = True
                return {str(k): v for k, v in data.items() if isinstance(v, dict)}
        except Exception:
            pass
        self._loaded = True
        return {}

    def _save_persisted(self) -> None:
        if self._persist_path is None:
            return
        try:
            os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
            snap: Dict[str, Dict[str, float]] = {}
            for k, bucket in self._buckets.items():
                snap[k] = bucket.export_state()
            tmp = f"{self._persist_path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(snap, f, separators=(",", ":"), ensure_ascii=False)
            os.replace(tmp, self._persist_path)
        except Exception:
            # best-effort only
            pass

    def _category_cfg(self, category: str) -> _CategoryCfg:
        if category == "write":
            return self.cfg_write
        return self.cfg_read

    def _bucket_key(self, category: str, tenant_id: str) -> str:
        return f"{category}:{tenant_id}"

    def _get_or_create_bucket(self, category: str, tenant_id: str) -> TokenBucketLimiter:
        key = self._bucket_key(category, tenant_id)
        with self._lock:
            b = self._buckets.get(key)
            if b is not None:
                return b
            cfg = self._category_cfg(category)
            b = TokenBucketLimiter(cfg.burst, cfg.rps)
            # import persisted state if any
            persisted = self._load_persisted()
            state = persisted.get(key)
            if isinstance(state, dict):
                b.import_state(state)
            self._buckets[key] = b
            return b

    def allow(self, category: str, tenant_id: str, n: int = 1) -> Tuple[bool, int]:
        b = self._get_or_create_bucket(category, tenant_id)
        ok = b.allow(n)
        retry_after = 0 if ok else b.retry_after_seconds(n)
        # best-effort persist (cheap write)
        self._save_persisted()
        return ok, retry_after


_REGISTRY: Optional[RateLimiterRegistry] = None


def _get_registry() -> RateLimiterRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = RateLimiterRegistry()
    return _REGISTRY


def rate_limit_dependency(category: str):
    """
    FastAPI dependency factory. Uses tenant_id from request.state.auth_ctx if available,
    defaults to 'anon'. On limit exceeded returns 429 with Retry-After header.
    """

    async def _dep(request: Request) -> None:
        tenant_id = "anon"
        try:
            ctx = getattr(request.state, "auth_ctx", None)
            if ctx is not None:
                tenant_id = getattr(ctx, "tenant_id", None) or "anon"
        except Exception:
            tenant_id = "anon"

        ok, retry_after = _get_registry().allow(category, tenant_id, 1)
        if not ok:
            hdrs = {"Retry-After": str(max(1, int(retry_after or 1)))}
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={"code": "rate_limited", "message": "Rate limit exceeded"},
                headers=hdrs,
            )

    return _dep


__all__ = ["TokenBucketLimiter", "RateLimiterRegistry", "rate_limit_dependency"]