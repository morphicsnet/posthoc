#!/usr/bin/env python3
# tests/load/report.py
# Aggregate JSONL outputs from load_runner and optional Prometheus /metrics snapshots
# to produce a human-readable markdown report and a structured JSON summary.

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Optional HTTP clients (prefer httpx)
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
REPORT_JSON = "report.json"
REPORT_MD = "report.md"

CHAOS_PATH_DEFAULT = "/tmp/hif/chaos.json"


@dataclass
class ChatRec:
    t0_ms: int
    t1_ms: int
    latency_ms: int
    status: str
    http_status: int
    error: Optional[str]
    request_id: Optional[str]
    trace_id: Optional[str]
    attached: bool
    granularity: Optional[str]
    features: Optional[str]


@dataclass
class ExplRec:
    chat_t1_ms: int
    trace_id: str
    state: str
    granularity: Optional[str]
    duration_ms: Optional[int]
    final_http: Optional[int]
    error: Optional[str]


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                    if isinstance(obj, dict):
                        out.append(obj)
                except Exception:
                    continue
    except FileNotFoundError:
        return []
    except Exception:
        return []
    return out


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


def _coerce_chat(rec: Dict[str, Any]) -> Optional[ChatRec]:
    try:
        return ChatRec(
            t0_ms=int(rec.get("t0_ms") or 0),
            t1_ms=int(rec.get("t1_ms") or 0),
            latency_ms=int(rec.get("latency_ms") or 0),
            status=str(rec.get("status") or ""),
            http_status=int(rec.get("http_status") or 0),
            error=(rec.get("error")),
            request_id=(rec.get("request_id")),
            trace_id=(rec.get("trace_id")),
            attached=bool(rec.get("attached")),
            granularity=(rec.get("granularity")),
            features=(rec.get("features")),
        )
    except Exception:
        return None


def _coerce_expl(rec: Dict[str, Any]) -> Optional[ExplRec]:
    try:
        return ExplRec(
            chat_t1_ms=int(rec.get("chat_t1_ms") or 0),
            trace_id=str(rec.get("trace_id") or ""),
            state=str(rec.get("state") or "unknown"),
            granularity=(rec.get("granularity")),
            duration_ms=(int(rec.get("duration_ms")) if rec.get("duration_ms") is not None else None),
            final_http=(int(rec.get("final_http")) if rec.get("final_http") is not None else None),
            error=(rec.get("error")),
        )
    except Exception:
        return None


def _load_chaos_flags(path: str = CHAOS_PATH_DEFAULT) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


async def _fetch_text_async(url: str, timeout: float = 5.0) -> Optional[str]:
    if not url:
        return None
    if httpx is not None:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:  # type: ignore
                resp = await client.get(url)
                if resp.status_code >= 400:
                    return None
                return resp.text
        except Exception:
            return None
    if requests is not None:
        try:
            def _do():
                r = requests.get(url, timeout=timeout)  # type: ignore
                return r.text if r.status_code < 400 else None
            import asyncio
            return await asyncio.to_thread(_do)
        except Exception:
            return None
    return None


def _parse_prometheus_text(text: str) -> Dict[str, Any]:
    """
    Very small Prometheus text parser for selected metrics:
      - backpressure_actions_total{action="X"} N
      - backpressure_level{tenant="T",granularity="G"} V
      - explainer_jobs_total{state="completed",...} N
    Returns:
      {
        "backpressure_actions_total": {"reduce-topk": 123, ...},
        "backpressure_level": [{"tenant":"t","granularity":"sentence","value":2.0}, ...],
        "explainer_jobs_total": {"completed": 100, "failed": 2, ...}
      }
    """
    out: Dict[str, Any] = {
        "backpressure_actions_total": {},
        "backpressure_level": [],
        "explainer_jobs_total": {},
    }
    if not isinstance(text, str) or not text:
        return out
    # Remove comments and help/type lines
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    # Regex for metric with labels and value
    m_re = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\{([^}]*)\}\s+([+-]?[0-9.]+)(?:\s+[0-9]+)?$')
    for ln in lines:
        try:
            m = m_re.match(ln.strip())
            if not m:
                continue
            metric, labels_str, val_str = m.group(1), m.group(2), m.group(3)
            value = float(val_str)
            # Parse labels
            labels: Dict[str, str] = {}
            for part in labels_str.split(","):
                if not part.strip():
                    continue
                if "=" not in part:
                    continue
                k, v = part.split("=", 1)
                v = v.strip().strip('"')
                labels[k.strip()] = v
            if metric == "backpressure_actions_total":
                act = labels.get("action") or ""
                if act:
                    out["backpressure_actions_total"][act] = float(value)
            elif metric == "backpressure_level":
                t = labels.get("tenant") or ""
                g = labels.get("granularity") or ""
                out["backpressure_level"].append({"tenant": t, "granularity": g, "value": float(value)})
            elif metric == "explainer_jobs_total":
                s = labels.get("state") or ""
                if s:
                    prev = float(out["explainer_jobs_total"].get(s, 0.0))
                    out["explainer_jobs_total"][s] = prev + float(value)
        except Exception:
            continue
    return out


def _compute_chat_summary(chats: List[ChatRec]) -> Dict[str, Any]:
    if not chats:
        return {"count": 0, "rps": 0.0, "latency_ms": {"p50": 0.0, "p95": 0.0, "p99": 0.0}, "errors": {"4xx": 0, "5xx": 0, "rate": 0.0}}
    # Compute time window
    t0 = min(c.t0_ms for c in chats)
    t1 = max(c.t1_ms for c in chats)
    dur_s = max(0.001, (t1 - t0) / 1000.0)
    lat_sorted = sorted(float(c.latency_ms) for c in chats if c.latency_ms >= 0)
    p50 = _percentile(lat_sorted, 0.50)
    p95 = _percentile(lat_sorted, 0.95)
    p99 = _percentile(lat_sorted, 0.99)
    c4xx = sum(1 for c in chats if 400 <= int(c.http_status) < 500)
    c5xx = sum(1 for c in chats if int(c.http_status) >= 500)
    err_rate = (c4xx + c5xx) / float(len(chats) or 1)
    return {
        "count": len(chats),
        "rps": (len(chats) / dur_s),
        "latency_ms": {"p50": p50, "p95": p95, "p99": p99},
        "errors": {"4xx": c4xx, "5xx": c5xx, "rate": err_rate},
    }


def _compute_expl_summary(expls: List[ExplRec]) -> Dict[str, Any]:
    if not expls:
        return {
            "count": 0,
            "sla_p95_ms": {"sentence": 0.0, "token": 0.0},
            "expired_pct": 0.0,
            "failed_pct": 0.0,
        }
    sent = sorted(float(r.duration_ms) for r in expls if (r.state == "complete" and r.duration_ms is not None and (r.granularity or "").lower() == "sentence"))
    tok = sorted(float(r.duration_ms) for r in expls if (r.state == "complete" and r.duration_ms is not None and (r.granularity or "").lower() == "token"))
    p95_sent = _percentile(sent, 0.95) if sent else 0.0
    p95_tok = _percentile(tok, 0.95) if tok else 0.0
    total = len(expls)
    expired = sum(1 for r in expls if r.state == "expired")
    failed = sum(1 for r in expls if r.state == "failed")
    return {
        "count": total,
        "sla_p95_ms": {"sentence": p95_sent, "token": p95_tok},
        "expired_pct": (expired / float(total or 1)),
        "failed_pct": (failed / float(total or 1)),
    }


def _compute_time_buckets(chats: List[ChatRec], expls: List[ExplRec]) -> Dict[str, Any]:
    if not chats:
        return {"chat_p95_ms": {"pre": 0, "during": 0, "post": 0, "delta_pre_post": 0}, "expl_p95_ms": {"sentence": {}, "token": {}}}
    tmin = min(c.t1_ms for c in chats)
    tmax = max(c.t1_ms for c in chats)
    span = max(1, tmax - tmin)
    pre_end = tmin + span / 3.0
    dur_end = tmin + 2.0 * span / 3.0

    def _bucket(vals: Iterable[float]) -> Dict[str, float]:
        arr = sorted(float(x) for x in vals)
        return {
            "p50": _percentile(arr, 0.50) if arr else 0.0,
            "p95": _percentile(arr, 0.95) if arr else 0.0,
        }

    pre_lat = [c.latency_ms for c in chats if c.t1_ms <= pre_end]
    dur_lat = [c.latency_ms for c in chats if pre_end < c.t1_ms <= dur_end]
    post_lat = [c.latency_ms for c in chats if c.t1_ms > dur_end]
    chat_buckets = {
        "pre": _bucket(pre_lat),
        "during": _bucket(dur_lat),
        "post": _bucket(post_lat),
        "delta_pre_post": ( (_bucket(post_lat)["p95"] - _bucket(pre_lat)["p95"]) if pre_lat and post_lat else 0.0 ),
    }

    # Explanation buckets (by chat_t1_ms)
    def _expl_filter(gran: str, t_lo: float, t_hi: float) -> List[int]:
        return [
            int(r.duration_ms) for r in expls
            if r.duration_ms is not None
            and r.state == "complete"
            and (r.granularity or "").lower() == gran
            and t_lo < r.chat_t1_ms <= t_hi
        ]

    out_expl: Dict[str, Dict[str, Dict[str, float]]] = {}
    for gran in ("sentence", "token"):
        pre = _bucket(_expl_filter(gran, tmin - 1, pre_end))
        dur = _bucket(_expl_filter(gran, pre_end, dur_end))
        post = _bucket(_expl_filter(gran, dur_end, tmax + 1))
        out_expl[gran] = {
            "pre": pre,
            "during": dur,
            "post": post,
            "delta_pre_post": {"p95": (post["p95"] - pre["p95"]) if pre["p95"] and post["p95"] else 0.0},
        }

    return {"chat_p95_ms": chat_buckets, "expl_p95_ms": out_expl}


def _summarize_backpressure(metrics: Dict[str, Any]) -> Dict[str, Any]:
    acts: Dict[str, float] = metrics.get("backpressure_actions_total") or {}
    levels: List[Dict[str, Any]] = metrics.get("backpressure_level") or []
    soft = sum(1 for x in levels if float(x.get("value", 0.0)) >= 1.0 and float(x.get("value", 0.0)) < 2.0)
    hard = sum(1 for x in levels if float(x.get("value", 0.0)) >= 2.0)
    return {"actions_total": acts, "level_counts": {"soft": soft, "hard": hard}}


def _render_markdown(summary: Dict[str, Any]) -> str:
    chat = summary.get("chat", {})
    expl = summary.get("explanation", {})
    buckets = summary.get("time_buckets", {})
    bp = summary.get("backpressure", {})
    chaos = summary.get("chaos_flags", {})
    lines: List[str] = []
    lines.append("# Load/Stress/Chaos Report")
    lines.append("")
    lines.append("## Chat lane")
    lines.append(f"- Requests: {int(chat.get('count', 0))}")
    lines.append(f"- RPS: {chat.get('rps', 0.0):.2f}")
    lat = chat.get("latency_ms", {})
    lines.append(f"- Latency p50/p95/p99 (ms): {lat.get('p50', 0):.1f} / {lat.get('p95', 0):.1f} / {lat.get('p99', 0):.1f}")
    errs = chat.get("errors", {})
    lines.append(f"- Errors: 4xx={int(errs.get('4xx',0))} 5xx={int(errs.get('5xx',0))} rate={float(errs.get('rate',0.0)):.4f}")
    lines.append("")
    lines.append("## Explanation lane")
    lines.append(f"- Total attached traces: {int(expl.get('count', 0))}")
    sla = (expl.get("sla_p95_ms") or {})
    lines.append(f"- SLA p95 (ms): sentence={float(sla.get('sentence',0.0)):.1f} token={float(sla.get('token',0.0)):.1f}")
    lines.append(f"- Expired %: {float(expl.get('expired_pct',0.0))*100:.2f}%  Failed %: {float(expl.get('failed_pct',0.0))*100:.2f}%")
    lines.append("")
    lines.append("## Time-bucket SLOs (pre / during / post)")
    chat_b = (buckets.get("chat_p95_ms") or {})
    lines.append(f"- Chat p95 (ms): pre={chat_b.get('pre',{}).get('p95',0):.1f} during={chat_b.get('during',{}).get('p95',0):.1f} post={chat_b.get('post',{}).get('p95',0):.1f} delta_post-pre={chat_b.get('delta_pre_post',0):.1f}")
    expl_b = (buckets.get("expl_p95_ms") or {})
    for gran in ("sentence", "token"):
        g = expl_b.get(gran, {})
        lines.append(f"- Explanation p95 (ms) [{gran}]: pre={g.get('pre',{}).get('p95',0):.1f} during={g.get('during',{}).get('p95',0):.1f} post={g.get('post',{}).get('p95',0):.1f} delta_post-pre={g.get('delta_pre_post',{}).get('p95',0):.1f}")
    lines.append("")
    lines.append("## Backpressure (from Prometheus metrics)")
    acts = (bp.get("actions_total") or {})
    if acts:
        act_parts = [f"{k}={v:.0f}" for k, v in sorted(acts.items())]
        lines.append(f"- Actions total: {', '.join(act_parts)}")
    lvl = (bp.get("level_counts") or {})
    lines.append(f"- Level counts: soft={int(lvl.get('soft',0))} hard={int(lvl.get('hard',0))}")
    lines.append("")
    lines.append("## Chaos flags (current)")
    if chaos:
        try:
            lines.append("```json")
            lines.append(json.dumps(chaos, indent=2, sort_keys=True))
            lines.append("```")
        except Exception:
            lines.append(str(chaos))
    else:
        lines.append("- none detected")
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Report generator for load/stress/chaos runs")
    p.add_argument("--results-dir", default=RESULTS_DIR_DEFAULT, help="Directory with chat_results.jsonl and explanation_results.jsonl")
    p.add_argument("--metrics-url-gateway", default=None, help="Optional Prometheus metrics URL for Gateway (e.g., http://localhost:9091/metrics)")
    p.add_argument("--metrics-url-explainer", default=None, help="Optional Prometheus metrics URL for Explainer (e.g., http://localhost:9090/metrics)")
    p.add_argument("--out-md", default=None, help="Override path for markdown output (default results/report.md)")
    p.add_argument("--out-json", default=None, help="Override path for json output (default results/report.json)")
    return p


async def main_async(ns: argparse.Namespace) -> Dict[str, Any]:
    # Load JSONL
    chat_path = os.path.join(ns.results_dir, CHAT_RESULTS_FILE)
    expl_path = os.path.join(ns.results_dir, EXPL_RESULTS_FILE)
    chat_raw = _read_jsonl(chat_path)
    expl_raw = _read_jsonl(expl_path)
    chats: List[ChatRec] = [c for c in (_coerce_chat(x) for x in chat_raw) if c is not None]
    expls: List[ExplRec] = [e for e in (_coerce_expl(x) for x in expl_raw) if e is not None]

    # Compute summaries
    chat_summary = _compute_chat_summary(chats)
    expl_summary = _compute_expl_summary(expls)
    time_buckets = _compute_time_buckets(chats, expls)

    # Optional /metrics fetch
    metrics_texts: List[str] = []
    for url in filter(None, [ns.metrics_url_gateway, ns.metrics_url_explainer]):
        try:
            txt = await _fetch_text_async(str(url))
            if isinstance(txt, str) and txt:
                metrics_texts.append(txt)
        except Exception:
            continue
    metrics_agg: Dict[str, Any] = {"backpressure_actions_total": {}, "backpressure_level": [], "explainer_jobs_total": {}}
    for txt in metrics_texts:
        parsed = _parse_prometheus_text(txt)
        # merge actions
        for k, v in (parsed.get("backpressure_actions_total") or {}).items():
            metrics_agg["backpressure_actions_total"][k] = metrics_agg["backpressure_actions_total"].get(k, 0.0) + float(v)
        # extend levels
        for item in (parsed.get("backpressure_level") or []):
            metrics_agg["backpressure_level"].append(item)
        # merge jobs
        for s, v in (parsed.get("explainer_jobs_total") or {}).items():
            metrics_agg["explainer_jobs_total"][s] = metrics_agg["explainer_jobs_total"].get(s, 0.0) + float(v)

    backpressure_summary = _summarize_backpressure(metrics_agg)

    # Chaos snapshot (current)
    chaos_flags = _load_chaos_flags()

    # Compose final summary
    summary: Dict[str, Any] = {
        "chat": chat_summary,
        "explanation": expl_summary,
        "time_buckets": time_buckets,
        "backpressure": backpressure_summary,
        "metrics_raw_counts": metrics_agg,  # keep raw for debugging
        "chaos_flags": chaos_flags,
    }

    # Persist
    os.makedirs(ns.results_dir, exist_ok=True)
    out_json = ns.out_json or os.path.join(ns.results_dir, REPORT_JSON)
    out_md = ns.out_md or os.path.join(ns.results_dir, REPORT_MD)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    md = _render_markdown(summary)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    # Print a short single-line summary for convenience
    print(json.dumps({"ok": True, "out_json": out_json, "out_md": out_md}, separators=(",", ":"), sort_keys=True))
    return summary


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ns = build_arg_parser().parse_args(argv)
    import asyncio
    return asyncio.run(main_async(ns))


if __name__ == "__main__":
    _ = main()