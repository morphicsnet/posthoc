#!/usr/bin/env python3
# tests/load/load_runner.py
# Async load/stress harness for the Gateway.
# - Drives POST /v1/chat/completions at high concurrency
# - Optionally attaches explanations (x-explain-mode: hypergraph)
# - Collects per-request JSONL and summary JSON
# - Concurrently polls /v1/traces/{id}/status and /graph for attached traces

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import string
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# Prefer httpx; fallback to requests in threads
try:
    import httpx  # type: ignore
except Exception:
    httpx = None  # type: ignore
try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore


RESULTS_DIR_DEFAULT = "tests/load/results"
CHAT_RESULTS_FILE = "chat_results.jsonl"
EXPL_RESULTS_FILE = "explanation_results.jsonl"
SUMMARY_FILE = "summary.json"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _monotonic_ms() -> int:
    return int(time.monotonic() * 1000)


def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * pct
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def _gen_text(n: int = 16) -> str:
    alph = string.ascii_letters + "     "
    return "".join(random.choice(alph) for _ in range(max(4, n)))


@dataclass
class Args:
    base_url: str
    duration_seconds: int
    concurrency: int
    attach_rate: float
    granularity: str
    features: str
    rps_limit: float
    auth_token: Optional[str]
    tenant_id: Optional[str]
    timeout_seconds: float
    model: str


@dataclass
class ChatResult:
    t0_ms: int
    t1_ms: int
    latency_ms: int
    status: str  # "ok" or "error"
    http_status: int
    error: Optional[str]
    request_id: Optional[str]
    trace_id: Optional[str]
    attached: bool
    granularity: Optional[str]
    features: Optional[str]


@dataclass
class ExplResult:
    chat_t1_ms: int
    trace_id: str
    state: str  # "complete" | "failed" | "expired" | "canceled" | "unknown"
    granularity: Optional[str]
    duration_ms: Optional[int]  # from chat_t1_ms to observation of complete/failed/expired
    final_http: Optional[int]
    error: Optional[str]


class TokenBucket:
    """Simple async token bucket for global RPS limit."""
    def __init__(self, rate_per_sec: float, capacity: Optional[int] = None) -> None:
        self.rate = max(0.0, float(rate_per_sec))
        self.capacity = int(capacity if capacity is not None else max(1, int(self.rate))) if self.rate > 0 else 0
        self._tokens = float(self.capacity)
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, n: int = 1) -> None:
        if self.rate <= 0.0:
            return
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = max(0.0, now - self._last)
                if elapsed > 0.0:
                    self._tokens = min(float(self.capacity), self._tokens + elapsed * self.rate)
                    self._last = now
                if self._tokens >= n:
                    self._tokens -= n
                    return
            # Sleep a small quantum to avoid busy wait
            await asyncio.sleep(0.001)


class Writers:
    """Thread-safe writers with an asyncio lock to avoid interleaved lines."""
    def __init__(self, chat_path: str, expl_path: str) -> None:
        self._chat_f = open(chat_path, "a", encoding="utf-8")
        self._expl_f = open(expl_path, "a", encoding="utf-8")
        self._lock = asyncio.Lock()

    async def write_chat(self, rec: Dict[str, Any]) -> None:
        line = json.dumps(rec, separators=(",", ":"), ensure_ascii=False)
        async with self._lock:
            self._chat_f.write(line + "\n")
            self._chat_f.flush()

    async def write_expl(self, rec: Dict[str, Any]) -> None:
        line = json.dumps(rec, separators=(",", ":"), ensure_ascii=False)
        async with self._lock:
            self._expl_f.write(line + "\n")
            self._expl_f.flush()

    def close(self) -> None:
        try:
            self._chat_f.close()
        except Exception:
            pass
        try:
            self._expl_f.close()
        except Exception:
            pass


class HTTP:
    """Minimal HTTP client wrapper supporting async httpx and sync requests fallback."""
    def __init__(self, timeout: float):
        self.timeout = float(timeout)

    async def post_json(self, url: str, headers: Dict[str, str], body: Dict[str, Any]) -> Tuple[int, Dict[str, Any], Dict[str, str], Optional[str]]:
        # returns (status_code, json_or_empty_dict, headers, text_on_error)
        if httpx is not None:
            async with httpx.AsyncClient(timeout=self.timeout) as client:  # type: ignore
                resp = await client.post(url, headers=headers, json=body)
                try:
                    data = resp.json()
                except Exception:
                    data = {}
                return int(resp.status_code), data, dict(resp.headers), None if resp.status_code < 400 else (resp.text[:512] if hasattr(resp, "text") else "")
        if requests is None:
            return 599, {}, {}, "no_http_client"
        def _do():
            r = requests.post(url, headers=headers, json=body, timeout=self.timeout)  # type: ignore
            try:
                dj = r.json()
            except Exception:
                dj = {}
            return int(r.status_code), dj, dict(r.headers), None if r.status_code < 400 else (getattr(r, "text", "")[:512])
        return await asyncio.to_thread(_do)

    async def get_json(self, url: str, headers: Dict[str, str]) -> Tuple[int, Dict[str, Any]]:
        if httpx is not None:
            async with httpx.AsyncClient(timeout=self.timeout) as client:  # type: ignore
                resp = await client.get(url, headers=headers)
                try:
                    data = resp.json()
                except Exception:
                    data = {}
                return int(resp.status_code), data
        if requests is None:
            return 599, {}
        def _do():
            r = requests.get(url, headers=headers, timeout=self.timeout)  # type: ignore
            try:
                dj = r.json()
            except Exception:
                dj = {}
            return int(r.status_code), dj
        return await asyncio.to_thread(_do)


async def poll_trace(http: HTTP, base_url: str, trace_id: str, chat_t1_ms: int, auth: Optional[str], writers: Writers, poll_cap_s: float = 180.0, poll_interval_s: float = 0.5) -> ExplResult:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = f"Bearer {auth}"
    state = "unknown"
    gran: Optional[str] = None
    final_http: Optional[int] = None
    err: Optional[str] = None
    started = time.monotonic()
    duration_ms: Optional[int] = None

    status_url = f"{base_url.rstrip('/')}/v1/traces/{trace_id}/status"
    graph_url = f"{base_url.rstrip('/')}/v1/traces/{trace_id}/graph"

    try:
        while (time.monotonic() - started) < float(poll_cap_s):
            st_code, st_json = await http.get_json(status_url, headers)
            final_http = st_code
            if st_code == 200 and isinstance(st_json, dict):
                state = str(st_json.get("state") or state)
                gran = st_json.get("granularity") or gran
                if state in ("complete", "failed", "expired", "canceled"):
                    duration_ms = max(0, _now_ms() - chat_t1_ms)
                    break
            elif st_code == 404:
                # Not yet known; continue polling
                pass
            elif st_code == 410:
                state = "expired"
                duration_ms = max(0, _now_ms() - chat_t1_ms)
                break

            # Try graph readiness as a proxy for completion
            g_code, _ = await http.get_json(graph_url, headers)
            if g_code == 200 and duration_ms is None:
                state = "complete"
                duration_ms = max(0, _now_ms() - chat_t1_ms)
                break
            await asyncio.sleep(poll_interval_s)
        else:
            state = "expired"
            duration_ms = max(0, _now_ms() - chat_t1_ms)
    except Exception as e:
        err = str(e)[:256]
        if duration_ms is None:
            duration_ms = max(0, _now_ms() - chat_t1_ms)

    rec = ExplResult(
        chat_t1_ms=chat_t1_ms,
        trace_id=trace_id,
        state=state,
        granularity=gran,
        duration_ms=duration_ms,
        final_http=final_http,
        error=err,
    )
    await writers.write_expl(asdict(rec))
    return rec


async def worker_loop(
    args: Args,
    http: HTTP,
    writers: Writers,
    end_monotonic_ms: int,
    trace_queue: "asyncio.Queue[Tuple[str,int]]",
    limiter: Optional[TokenBucket],
    agg_latencies: List[int],
    agg_http_status: List[int],
    agg_errors: List[str],
    attach_counter: List[int],
) -> None:
    base_url = args.base_url.rstrip("/")
    url = f"{base_url}/v1/chat/completions"

    while _monotonic_ms() < end_monotonic_ms:
        # RPS global control
        if limiter is not None:
            await limiter.acquire(1)

        t0_ms = _now_ms()
        body = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Say hello and mention a color: {_gen_text(12)}"},
            ],
            "stream": False,
            "temperature": 0.7,
        }

        # Decide attachment
        do_attach = (random.random() < max(0.0, min(1.0, args.attach_rate)))
        gran = args.granularity.lower().strip()
        if gran == "mix":
            gran = ("sentence" if random.random() < 0.8 else "token")
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
        }
        if args.tenant_id:
            headers["x-tenant-id"] = str(args.tenant_id)
        if args.auth_token:
            headers["Authorization"] = f"Bearer {args.auth_token}"
        trace_id: Optional[str] = None
        if do_attach:
            headers["x-explain-mode"] = "hypergraph"
            headers["x-explain-granularity"] = gran
            headers["x-explain-features"] = args.features
            # client-provided trace id for easier correlation
            trace_id = "trc_" + "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(12))
            headers["x-trace-id"] = trace_id

        # Send
        status_code: int = 0
        error_text: Optional[str] = None
        request_id: Optional[str] = None
        try:
            status_code, data, resp_headers, error_text = await http.post_json(url, headers, body)
            # request id from body if available
            if isinstance(data, dict):
                rid = data.get("id")
                if isinstance(rid, str):
                    request_id = rid
        except Exception as e:
            status_code = 599
            error_text = str(e)[:512]

        t1_ms = _now_ms()
        latency_ms = max(0, t1_ms - t0_ms)
        agg_latencies.append(latency_ms)
        agg_http_status.append(status_code)
        if status_code >= 400 or error_text:
            agg_errors.append(error_text or f"http_{status_code}")

        chat_rec = ChatResult(
            t0_ms=t0_ms,
            t1_ms=t1_ms,
            latency_ms=latency_ms,
            status="ok" if (status_code < 400 and not error_text) else "error",
            http_status=status_code,
            error=error_text,
            request_id=request_id,
            trace_id=trace_id,
            attached=bool(do_attach),
            granularity=(gran if do_attach else None),
            features=(args.features if do_attach else None),
        )
        await writers.write_chat(asdict(chat_rec))

        if do_attach and trace_id:
            # enqueue for polling (carry gran weight for optional downstream logic if needed)
            attach_counter.append(1)
            try:
                await trace_queue.put((trace_id, t1_ms))
            except Exception:
                pass

    # loop ends when time is up


async def poller_loop(
    args: Args,
    http: HTTP,
    writers: Writers,
    trace_queue: "asyncio.Queue[Tuple[str,int]]",
    expl_results_out: List[ExplResult],
) -> None:
    while True:
        try:
            item = await trace_queue.get()
        except asyncio.CancelledError:
            return
        except Exception:
            break
        if item is None:
            trace_queue.task_done()
            break
        trace_id, chat_t1_ms = item
        try:
            res = await poll_trace(http, args.base_url, trace_id, chat_t1_ms, args.auth_token, writers)
            expl_results_out.append(res)
        except Exception:
            # Even if polling fails, mark task done
            pass
        finally:
            trace_queue.task_done()


def _summarize(
    latencies: List[int],
    http_statuses: List[int],
    chat_count: int,
    test_duration_s: float,
    expl_results: List[ExplResult],
) -> Dict[str, Any]:
    # Chat metrics
    ok_lat = sorted([float(x) for x in latencies if x >= 0])
    p50 = _percentile(ok_lat, 0.50)
    p95 = _percentile(ok_lat, 0.95)
    p99 = _percentile(ok_lat, 0.99)
    rps = (chat_count / test_duration_s) if test_duration_s > 0 else 0.0
    c4xx = sum(1 for s in http_statuses if 400 <= int(s) < 500)
    c5xx = sum(1 for s in http_statuses if int(s) >= 500)
    err_rate = (c4xx + c5xx) / float(chat_count or 1)

    # Explanation metrics by granularity (only completed)
    by_gran: Dict[str, List[int]] = {"sentence": [], "token": []}
    expired = 0
    failed = 0
    for r in expl_results:
        if r.state == "complete" and r.duration_ms is not None:
            g = (r.granularity or "sentence").lower()
            if g in by_gran:
                by_gran[g].append(int(r.duration_ms))
        elif r.state in ("expired", "failed"):
            if r.state == "expired":
                expired += 1
            else:
                failed += 1
    p95_sentence = _percentile(sorted([float(x) for x in by_gran["sentence"]]), 0.95) if by_gran["sentence"] else 0.0
    p95_token = _percentile(sorted([float(x) for x in by_gran["token"]]), 0.95) if by_gran["token"] else 0.0
    expl_total = len(expl_results) if expl_results else 0
    exp_pct = (expired / float(expl_total or 1)) if expl_total else 0.0
    fail_pct = (failed / float(expl_total or 1)) if expl_total else 0.0

    return {
        "chat": {
            "count": chat_count,
            "rps": rps,
            "latency_ms": {"p50": p50, "p95": p95, "p99": p99},
            "errors": {"4xx": c4xx, "5xx": c5xx, "rate": err_rate},
        },
        "explanation": {
            "count": expl_total,
            "sla_p95_ms": {"sentence": p95_sentence, "token": p95_token},
            "expired_pct": exp_pct,
            "failed_pct": fail_pct,
        },
    }


async def run_load_async(args: Args) -> Dict[str, Any]:
    results_dir = RESULTS_DIR_DEFAULT
    _ensure_dir(results_dir)
    chat_path = os.path.join(results_dir, CHAT_RESULTS_FILE)
    expl_path = os.path.join(results_dir, EXPL_RESULTS_FILE)
    writers = Writers(chat_path, expl_path)

    http = HTTP(timeout=args.timeout_seconds)
    end_monotonic_ms = _monotonic_ms() + int(max(1, args.duration_seconds) * 1000)
    limiter = TokenBucket(rate_per_sec=args.rps_limit, capacity=int(args.rps_limit)) if args.rps_limit and args.rps_limit > 0 else None

    # Aggregators
    agg_lat: List[int] = []
    agg_status: List[int] = []
    agg_errs: List[str] = []
    attach_counter: List[int] = []

    # Trace polling infra
    trace_q: "asyncio.Queue[Tuple[str,int]]" = asyncio.Queue(maxsize=max(1000, args.concurrency * 2))
    expl_results: List[ExplResult] = []
    poller_count = min(32, max(4, args.concurrency // 8))
    pollers = [asyncio.create_task(poller_loop(args, http, writers, trace_q, expl_results)) for _ in range(poller_count)]

    # Workers
    workers = [
        asyncio.create_task(
            worker_loop(args, http, writers, end_monotonic_ms, trace_q, limiter, agg_lat, agg_status, agg_errs, attach_counter)
        )
        for _ in range(args.concurrency)
    ]

    t_start = time.monotonic()
    await asyncio.gather(*workers, return_exceptions=True)

    # Close trace queue and wait for pollers to drain with a bounded grace time
    await trace_q.join()
    for p in pollers:
        try:
            p.cancel()
        except Exception:
            pass
    # Give cancellation some time
    await asyncio.gather(*pollers, return_exceptions=True)

    t_end = time.monotonic()
    writers.close()

    summary = _summarize(
        latencies=agg_lat,
        http_statuses=agg_status,
        chat_count=len(agg_status),
        test_duration_s=max(0.001, (t_end - t_start)),
        expl_results=expl_results,
    )
    # Persist summary
    with open(os.path.join(results_dir, SUMMARY_FILE), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    # Also print to stdout (single-line for easier scraping)
    print(json.dumps(summary, separators=(",", ":"), sort_keys=True))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Async load/stress harness for Gateway")
    p.add_argument("--base-url", default="http://localhost:8080", help="Gateway base URL")
    p.add_argument("--duration-seconds", type=int, default=300, help="Test duration (seconds)")
    p.add_argument("--concurrency", type=int, default=1000, help="Concurrent workers")
    p.add_argument("--attach-rate", type=float, default=0.3, help="Fraction of chats that attach explanations [0-1]")
    p.add_argument("--granularity", choices=["sentence", "token", "mix"], default="sentence", help="Explanation granularity")
    p.add_argument("--features", default="sae-gpt4-2m", help="Featureset header (x-explain-features)")
    p.add_argument("--rps-limit", type=float, default=0.0, help="Global RPS limit across all workers (0=unlimited)")
    p.add_argument("--auth-token", default=None, help="Optional Bearer token")
    p.add_argument("--tenant-id", default=None, help="Optional tenant id (for logs)")
    p.add_argument("--timeout-seconds", type=float, default=15.0, help="HTTP timeout seconds")
    p.add_argument("--model", default="gpt-4o-mini", help="Model name for chat body")
    return p


def parse_args(argv: Optional[List[str]] = None) -> Args:
    ns = build_arg_parser().parse_args(argv)
    return Args(
        base_url=str(ns.base_url),
        duration_seconds=int(ns.duration_seconds),
        concurrency=int(ns.concurrency),
        attach_rate=float(ns.attach_rate),
        granularity=str(ns.granularity),
        features=str(ns.features),
        rps_limit=float(ns.rps_limit),
        auth_token=(str(ns.auth_token) if ns.auth_token else None),
        tenant_id=(str(ns.tenant_id) if ns.tenant_id else None),
        timeout_seconds=float(ns.timeout_seconds),
        model=str(ns.model),
    )


def run_load(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    return asyncio.run(run_load_async(args))


if __name__ == "__main__":
    _ = run_load()