#!/usr/bin/env python3
"""
Attach-rate analyzer for Hypergraph explanations.

Inputs:
  --status-json: LocalJSONStatusStore file (default: /tmp/hif/status.json)
  --audit-log:   Gateway audit JSONL (optional). If missing, analyzer proceeds with StatusStore only.
  --window-minutes: Rolling window to analyze (default: 60)
  --bucket-seconds: Aggregation bucket size in seconds (default: 60)
  --output:      JSON summary output path (default: tools/analysis/results/attach_rate_summary.json)
  --report-output: Markdown report output path (default: tools/analysis/results/attach_rate_report.md)
  --token-mix:   Fraction of explanation traffic that is token-level (default: 0.1)
  --throughput-sentence-range: "min,max" explanations/sec/GPU for sentence path (default: "3,6")
  --throughput-token-range:    "min,max" explanations/sec/GPU for token path (default: "0.4,0.8")
  --concurrency: Comma-separated concurrency targets for capacity calc (default: "1000,2000")

Behavior:
- Reads StatusStore JSON (mapping of trace_id -> TraceStatusItem).
  Each item is treated as a chat outcome in the observation window; items with a non-empty 'granularity'
  field are counted as explanation-attached chats.
- Optionally reads Audit JSON lines. The analyzer will:
  * Treat event=="chat.submit" as a chat (and as attached if payload includes mode=="hypergraph").
  * Treat event=="trace.queued" as an attached explanation (helps when StatusStore lags).
  Aggregate counts are deduplicated by trace_id when present.
- Computes per-bucket attach_rate = explanations/chats (clamped to [0,1]) and summarizes
  mean, median, and p95 across buckets in the window.
- Produces capacity recommendations at provided concurrency targets using throughput assumptions.

Notes/Assumptions:
- If the audit log is unavailable and StatusStore only captures explanation traces, the analyzer
  will conservatively treat each store item as both a chat and (if granularity present) an attached explanation,
  which can bias attach-rate upward. The markdown report will flag this condition.
- Throughput ranges are used to compute conservative (min throughput) and optimistic
  (max throughput) GPU replica counts.

Example:
  python3 tools/analysis/attach_rate_analyzer.py \\
    --status-json /tmp/hif/status.json \\
    --audit-log /var/log/hypergraph/audit.log \\
    --window-minutes 120 \\
    --bucket-seconds 60 \\
    --output tools/analysis/results/attach_rate_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


def _debug(msg: str) -> None:
    if os.getenv("ATTACH_ANALYZER_DEBUG", "0") == "1":
        try:
            print(f"[attach_rate_analyzer] {msg}", file=sys.stderr)
        except Exception:
            pass


def _ensure_dir_for(path: str) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)


def _parse_iso8601(ts: str) -> Optional[float]:
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts).timestamp()
    except Exception:
        return None


def _floor_bucket(ts: float, bucket: int) -> int:
    b = int(bucket) if bucket > 0 else 60
    return int(ts // b) * b


def _load_status_items(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            out: List[Dict[str, Any]] = []
            for _, v in data.items():
                if isinstance(v, dict):
                    out.append(v)
            return out
        return []
    except FileNotFoundError:
        return []
    except Exception:
        return []


def _iter_audit_lines(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = (ln or "").strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                    if isinstance(obj, dict):
                        yield obj
                except Exception:
                    # ignore malformed line
                    continue
    except FileNotFoundError:
        return
    except Exception:
        return


def _parse_range(s: str, default_min: float, default_max: float) -> Tuple[float, float]:
    try:
        parts = [p.strip() for p in (s or "").split(",") if p.strip()]
        if len(parts) == 2:
            a = float(parts[0]); b = float(parts[1])
            lo, hi = (a, b) if a <= b else (b, a)
            return (max(1e-9, lo), max(1e-9, hi))
        elif len(parts) == 1:
            v = float(parts[0])
            return (max(1e-9, v), max(1e-9, v))
    except Exception:
        pass
    return (default_min, default_max)


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    v = sorted(values)
    if len(v) == 1:
        return v[0]
    # Nearest-rank method
    k = max(1, int(math.ceil((p / 100.0) * len(v))))
    return v[min(len(v)-1, k-1)]


def compute_attach_stats(
    status_items: List[Dict[str, Any]],
    audit_path: Optional[str],
    window_minutes: int,
    bucket_seconds: int,
) -> Tuple[Dict[int, Dict[str, int]], Dict[str, Any]]:
    now = time.time()
    window_start = now - float(max(1, window_minutes)) * 60.0
    buckets: Dict[int, Dict[str, int]] = {}

    # Seen sets to reduce double-counting when trace_id available
    seen_chat_trace: Dict[int, set] = {}
    seen_explain_trace: Dict[int, set] = {}

    def _add(bucket: int, chat: int, explain: int, trace_id: Optional[str]):
        b = buckets.setdefault(bucket, {"chats": 0, "explanations": 0})
        # Dedup only when trace_id is available
        if trace_id:
            sc = seen_chat_trace.setdefault(bucket, set())
            se = seen_explain_trace.setdefault(bucket, set())
            if chat:
                if trace_id not in sc:
                    b["chats"] += 1
                    sc.add(trace_id)
            if explain:
                if trace_id not in se:
                    b["explanations"] += 1
                    se.add(trace_id)
        else:
            b["chats"] += int(chat)
            b["explanations"] += int(explain)

    # StatusStore-derived counts
    for it in status_items:
        try:
            created = float(it.get("created_at") or 0.0)
            updated = float(it.get("updated_at") or 0.0)
            ts = created or updated
            if ts <= 0:
                continue
            if ts < window_start:
                continue
            bucket = _floor_bucket(ts, bucket_seconds)
            gran = it.get("granularity")
            tid = str(it.get("trace_id") or "")
            has_tid = tid if tid else None
            # Treat each item as a chat; count as explanation if granularity is present
            _add(bucket, chat=1, explain=1 if gran else 0, trace_id=has_tid)
        except Exception:
            continue

    # Audit-derived counts (optional)
    if audit_path:
        for obj in _iter_audit_lines(audit_path):
            try:
                ev = str(obj.get("event") or "")
                ts_raw = obj.get("ts")
                ts = None
                if isinstance(ts_raw, str):
                    ts = _parse_iso8601(ts_raw)
                elif isinstance(ts_raw, (int, float)):
                    ts = float(ts_raw)
                if ts is None:
                    # fallback to now if missing
                    ts = now
                if ts < window_start:
                    continue
                bucket = _floor_bucket(ts, bucket_seconds)
                tid = str(obj.get("trace_id") or "") or None
                mode = (obj.get("mode") or obj.get("x-explain-mode") or "")
                if ev == "chat.submit":
                    # This is a chat; count as attached if mode==hypergraph
                    _add(bucket, chat=1, explain=1 if str(mode) == "hypergraph" else 0, trace_id=tid)
                elif ev == "trace.queued":
                    # Explanation queued; count as attached explanation, but do not infer a chat beyond this signal
                    _add(bucket, chat=0, explain=1, trace_id=tid)
                # Other events ignored
            except Exception:
                continue

    # Clamp explanations <= chats per bucket
    for b, d in buckets.items():
        if d["explanations"] > d["chats"] and d["chats"] > 0:
            d["explanations"] = d["chats"]

    # Compute aggregate rates
    rates: List[float] = []
    for _, d in sorted(buckets.items()):
        chats = d["chats"]
        expl = d["explanations"]
        if chats <= 0:
            continue
        r = max(0.0, min(1.0, float(expl) / float(chats)))
        rates.append(r)

    mean = (sum(rates) / len(rates)) if rates else 0.0
    median = (sorted(rates)[len(rates)//2] if rates else 0.0) if (len(rates) % 2 == 1) else (
        (sorted(rates)[len(rates)//2 - 1] + sorted(rates)[len(rates)//2]) / 2.0 if rates else 0.0
    )
    p95 = _percentile(rates, 95.0) or 0.0

    meta = {
        "now": int(now),
        "window_start": int(window_start),
        "bucket_seconds": int(bucket_seconds),
        "buckets_analyzed": len(buckets),
        "rates_count": len(rates),
    }
    return buckets, {
        "mean": round(mean, 6),
        "median": round(median, 6),
        "p95": round(p95, 6),
        "meta": meta,
    }


def _capacity_recommendations(
    attach_rate_mean: float,
    attach_rate_p95: float,
    token_mix: float,
    throughput_sentence_range: Tuple[float, float],
    throughput_token_range: Tuple[float, float],
    concurrency_targets: List[int],
) -> List[Dict[str, Any]]:
    """
    For each concurrency target:
      - explanation_qps = concurrency * attach_rate
      - split by mix (sentence vs token)
      - compute replicas using throughput ranges:
         conservative: use min throughput (more GPUs)
         optimistic:   use max throughput (fewer GPUs)
    """
    out: List[Dict[str, Any]] = []
    token_mix = max(0.0, min(1.0, float(token_mix)))
    sent_mix = 1.0 - token_mix
    s_min, s_max = throughput_sentence_range
    t_min, t_max = throughput_token_range

    for c in concurrency_targets:
        c = int(c)
        m_qps = c * attach_rate_mean
        p_qps = c * attach_rate_p95

        def _split(qps: float) -> Tuple[float, float]:
            return qps * sent_mix, qps * token_mix

        def _replicas(qps: float, thr_lo: float, thr_hi: float) -> Tuple[int, int]:
            # conservative uses low throughput => more replicas
            cons = int(math.ceil(qps / max(1e-9, thr_lo)))
            opti = int(math.ceil(qps / max(1e-9, thr_hi)))
            return cons, max(1, opti)

        m_s_qps, m_t_qps = _split(m_qps)
        p_s_qps, p_t_qps = _split(p_qps)

        mean_cons_s, mean_opti_s = _replicas(m_s_qps, s_min, s_max)
        mean_cons_t, mean_opti_t = _replicas(m_t_qps, t_min, t_max)
        p95_cons_s, p95_opti_s = _replicas(p_s_qps, s_min, s_max)
        p95_cons_t, p95_opti_t = _replicas(p_t_qps, t_min, t_max)

        out.append({
            "concurrency": c,
            "explanation_qps": {"mean": round(m_qps, 3), "p95": round(p_qps, 3)},
            "mix": {"sentence": round(1.0 - token_mix, 3), "token": round(token_mix, 3)},
            "replicas": {
                "mean": {
                    "conservative": {
                        "sentence": mean_cons_s,
                        "token": mean_cons_t,
                        "total": mean_cons_s + mean_cons_t
                    },
                    "optimistic": {
                        "sentence": mean_opti_s,
                        "token": mean_opti_t,
                        "total": mean_opti_s + mean_opti_t
                    }
                },
                "p95": {
                    "conservative": {
                        "sentence": p95_cons_s,
                        "token": p95_cons_t,
                        "total": p95_cons_s + p95_cons_t
                    },
                    "optimistic": {
                        "sentence": p95_opti_s,
                        "token": p95_opti_t,
                        "total": p95_opti_s + p95_opti_t
                    }
                }
            }
        })
    return out


def _write_json(path: str, obj: Any) -> None:
    _ensure_dir_for(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _write_report(path: str, summary: Dict[str, Any]) -> None:
    _ensure_dir_for(path)
    lines: List[str] = []
    lines.append("# Attach-rate analysis report")
    lines.append("")
    meta = summary.get("meta") or {}
    lines.append(f"- Window start: {datetime.fromtimestamp(meta.get('window_start', 0), tz=timezone.utc).isoformat()}")
    lines.append(f"- Now:          {datetime.fromtimestamp(meta.get('now', 0), tz=timezone.utc).isoformat()}")
    lines.append(f"- Bucket size:  {meta.get('bucket_seconds', 60)} seconds")
    lines.append(f"- Buckets:      {meta.get('buckets_analyzed', 0)}")
    lines.append("")
    att = summary.get("attach_rate", {})
    lines.append("## Baseline attach-rate")
    lines.append("")
    lines.append(f"- Mean:   {att.get('mean')}")
    lines.append(f"- Median: {att.get('median')}")
    lines.append(f"- P95:    {att.get('p95')}")
    lines.append("")
    assumptions = summary.get("assumptions", {})
    lines.append("## Assumptions")
    lines.append("")
    lines.append(f"- Token mix: {assumptions.get('token_mix')} (sentence={round(1.0 - float(assumptions.get('token_mix', 0.0)), 3)})")
    thr_s = assumptions.get("throughput_sentence_per_gpu", {})
    thr_t = assumptions.get("throughput_token_per_gpu", {})
    lines.append(f"- Sentence throughput (explanations/sec/GPU) min={thr_s.get('min')} max={thr_s.get('max')}")
    lines.append(f"- Token throughput    (explanations/sec/GPU) min={thr_t.get('min')} max={thr_t.get('max')}")
    if assumptions.get("data_quality_note"):
        lines.append(f"- Note: {assumptions.get('data_quality_note')}")
    lines.append("")
    lines.append("## Capacity recommendations")
    lines.append("")
    lines.append("| Concurrency | Mean total GPUs (cons/opt) | P95 total GPUs (cons/opt) |")
    lines.append("|-------------|-----------------------------|----------------------------|")
    for r in summary.get("capacity_recommendations", []):
        c = r.get("concurrency")
        mean = r.get("replicas", {}).get("mean", {})
        p95 = r.get("replicas", {}).get("p95", {})
        m_total = mean.get("conservative", {}).get("total", 0)
        m_total_opt = mean.get("optimistic", {}).get("total", 0)
        p_total = p95.get("conservative", {}).get("total", 0)
        p_total_opt = p95.get("optimistic", {}).get("total", 0)
        lines.append(f"| {c} | {m_total}/{m_total_opt} | {p_total}/{p_total_opt} |")
    lines.append("")
    lines.append("Apply via Karpenter node groups and KEDA min/max replicas. See:")
    lines.append("- Helm KEDA scaler: [manifests/helm/hypergraph/templates/keda-scalers.yaml](manifests/helm/hypergraph/templates/keda-scalers.yaml)")
    lines.append("- Helm values: [manifests/helm/hypergraph/values.yaml](manifests/helm/hypergraph/values.yaml)")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze attach-rate baseline and spikes, produce capacity guidance.")
    ap.add_argument("--status-json", default="/tmp/hif/status.json", help="Path to StatusStore JSON (default: /tmp/hif/status.json)")
    ap.add_argument("--audit-log", default=None, help="Path to Gateway audit JSONL (optional)")
    ap.add_argument("--window-minutes", type=int, default=60, help="Rolling window in minutes (default: 60)")
    ap.add_argument("--bucket-seconds", type=int, default=60, help="Bucket size in seconds (default: 60)")
    ap.add_argument("--output", default="tools/analysis/results/attach_rate_summary.json", help="Summary JSON output path")
    ap.add_argument("--report-output", default="tools/analysis/results/attach_rate_report.md", help="Markdown report output path")
    ap.add_argument("--token-mix", type=float, default=0.1, help="Fraction of explanations that are token-level (default: 0.1)")
    ap.add_argument("--throughput-sentence-range", default="3,6", help="Sentence path throughput range as 'min,max' (default: 3,6)")
    ap.add_argument("--throughput-token-range", default="0.4,0.8", help="Token path throughput range as 'min,max' (default: 0.4,0.8)")
    ap.add_argument("--concurrency", default="1000,2000", help="Comma-separated concurrency targets (default: 1000,2000)")

    args = ap.parse_args()

    status_items = _load_status_items(args.status_json)
    buckets, ar = compute_attach_stats(status_items, args.audit_log, args.window_minutes, args.bucket_seconds)

    thr_s = _parse_range(args.throughput_sentence_range, 3.0, 6.0)
    thr_t = _parse_range(args.throughput_token_range, 0.4, 0.8)
    conc = [int(p.strip()) for p in str(args.concurrency).split(",") if p.strip().isdigit()]
    if not conc:
        conc = [1000, 2000]

    cap = _capacity_recommendations(
        attach_rate_mean=float(ar["mean"]),
        attach_rate_p95=float(ar["p95"]),
        token_mix=float(args.token_mix),
        throughput_sentence_range=thr_s,
        throughput_token_range=thr_t,
        concurrency_targets=conc,
    )

    data_quality_note = None
    if not os.path.exists(args.status_json):
        data_quality_note = "StatusStore JSON missing; results derived from audit only."
    elif (len(status_items) > 0) and (args.audit_log is None or not os.path.exists(str(args.audit_log))):
        # Likely overestimation if StatusStore contains only explanation traces
        data_quality_note = "Audit log not provided; if StatusStore contains only explanation traces, attach-rate may be biased upward."

    summary = {
        "attach_rate": ar,
        "assumptions": {
            "token_mix": float(args.token_mix),
            "throughput_sentence_per_gpu": {"min": thr_s[0], "max": thr_s[1]},
            "throughput_token_per_gpu": {"min": thr_t[0], "max": thr_t[1]},
            "data_quality_note": data_quality_note,
        },
        "capacity_recommendations": cap,
        "meta": ar.get("meta", {}),
    }

    _write_json(args.output, summary)
    _write_report(args.report_output, {
        "attach_rate": ar,
        "assumptions": summary["assumptions"],
        "capacity_recommendations": cap,
        "meta": summary["meta"],
    })

    # Also print a concise stdout line for quick inspection
    try:
        print(json.dumps({
            "mean": ar["mean"], "median": ar["median"], "p95": ar["p95"], "buckets": summary["meta"].get("buckets_analyzed", 0)
        }))
    except Exception:
        pass


if __name__ == "__main__":
    main()
