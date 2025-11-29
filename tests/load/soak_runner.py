#!/usr/bin/env python3
# tests/load/soak_runner.py
# Soak orchestrator that performs repeated runs of tests.load.load_runner with different profiles.
# - Rotates attach_rate and granularity across runs (token bursts)
# - Supports profiles: default | heavy | spiky
# - Aggregates time-series SLOs into tests/load/results/soak_summary.json

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

# Local imports
try:
    from tests.load.load_runner import Args as LoadArgs, run_load_async as run_load_async  # type: ignore
except Exception:
    # Relative fallback when executed as module
    try:
        from load_runner import Args as LoadArgs, run_load_async as run_load_async  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Cannot import load_runner: {e}")

RESULTS_DIR_DEFAULT = "tests/load/results"
SOAK_SUMMARY_FILE = "soak_summary.json"


@dataclass
class SoakRunConfig:
    idx: int
    duration_seconds: int
    concurrency: int
    rps_limit: float
    attach_rate: float
    granularity: str
    features: str


@dataclass
class SoakRunResult:
    idx: int
    ts: float
    profile: str
    cfg: SoakRunConfig
    summary: Dict[str, Any]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _choose_attach_rate() -> float:
    # Rotate in 0.30..0.50
    return round(random.uniform(0.30, 0.50), 2)


def _choose_granularity(run_idx: int) -> str:
    # Token bursts every 5th run; otherwise mix
    return "token" if (run_idx % 5 == 0) else "mix"


def _build_segment_plan(hours: int, profile: str) -> List[SoakRunConfig]:
    total_seconds = max(1, int(hours)) * 3600
    # 10-minute segments
    seg = 600
    n = max(1, total_seconds // seg)
    plan: List[SoakRunConfig] = []
    for i in range(1, n + 1):
        if profile == "spiky":
            # Alternate high/low RPS or concurrency
            hi = (i % 2 == 1)
            concurrency = 800 if hi else 200
            rps_limit = 0.0 if hi else 50.0
        elif profile == "heavy":
            concurrency = 1000
            rps_limit = 0.0
        else:
            concurrency = 400
            rps_limit = 0.0

        cfg = SoakRunConfig(
            idx=i,
            duration_seconds=seg,
            concurrency=concurrency,
            rps_limit=rps_limit,
            attach_rate=_choose_attach_rate(),
            granularity=_choose_granularity(i),
            features="sae-gpt4-2m",
        )
        plan.append(cfg)
    return plan


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Soak orchestrator for Gateway load/stress")
    p.add_argument("--base-url", default="http://localhost:8080", help="Gateway base URL")
    p.add_argument("--hours", type=int, default=2, help="Soak duration in hours (1-4 recommended)")
    p.add_argument("--profile", choices=["default", "heavy", "spiky"], default="default", help="Traffic profile")
    p.add_argument("--auth-token", default=None, help="Optional Bearer token")
    p.add_argument("--tenant-id", default=None, help="Optional tenant id")
    p.add_argument("--timeout-seconds", type=float, default=15.0, help="HTTP timeout")
    p.add_argument("--model", default="gpt-4o-mini", help="Model for chat requests")
    p.add_argument("--results-dir", default=RESULTS_DIR_DEFAULT, help="Directory to write results/soak_summary.json")
    return p


async def run_segment(base_url: str, common_args: argparse.Namespace, cfg: SoakRunConfig) -> Dict[str, Any]:
    args = LoadArgs(
        base_url=base_url,
        duration_seconds=cfg.duration_seconds,
        concurrency=cfg.concurrency,
        attach_rate=cfg.attach_rate,
        granularity=cfg.granularity,
        features=cfg.features,
        rps_limit=cfg.rps_limit,
        auth_token=(str(common_args.auth_token) if common_args.auth_token else None),
        tenant_id=(str(common_args.tenant_id) if common_args.tenant_id else None),
        timeout_seconds=float(common_args.timeout_seconds),
        model=str(common_args.model),
    )
    return await run_load_async(args)


def _extract_key_slos(segment_summary: Dict[str, Any]) -> Dict[str, Any]:
    try:
        chat = segment_summary.get("chat", {})
        expl = segment_summary.get("explanation", {})
        return {
            "chat_rps": float(chat.get("rps") or 0.0),
            "chat_p50_ms": float((chat.get("latency_ms") or {}).get("p50") or 0.0),
            "chat_p95_ms": float((chat.get("latency_ms") or {}).get("p95") or 0.0),
            "err_4xx": int((chat.get("errors") or {}).get("4xx") or 0),
            "err_5xx": int((chat.get("errors") or {}).get("5xx") or 0),
            "err_rate": float((chat.get("errors") or {}).get("rate") or 0.0),
            "expl_p95_sentence_ms": float((expl.get("sla_p95_ms") or {}).get("sentence") or 0.0),
            "expl_p95_token_ms": float((expl.get("sla_p95_ms") or {}).get("token") or 0.0),
            "expl_expired_pct": float(expl.get("expired_pct") or 0.0),
            "expl_failed_pct": float(expl.get("failed_pct") or 0.0),
        }
    except Exception:
        return {}


async def run_soak_async(ns: argparse.Namespace) -> Dict[str, Any]:
    base_url = str(ns.base_url)
    plan = _build_segment_plan(ns.hours, ns.profile)
    results_dir = ns.results_dir or RESULTS_DIR_DEFAULT
    _ensure_dir(results_dir)

    runs: List[SoakRunResult] = []
    t0 = time.time()

    for cfg in plan:
        t_run = time.time()
        seg_summary = await run_segment(base_url, ns, cfg)
        runs.append(
            SoakRunResult(
                idx=cfg.idx,
                ts=t_run,
                profile=str(ns.profile),
                cfg=cfg,
                summary=seg_summary,
            )
        )

    # Aggregate series
    series = [
        {
            "idx": r.idx,
            "ts": r.ts,
            "profile": r.profile,
            "config": asdict(r.cfg),
            "slos": _extract_key_slos(r.summary),
        }
        for r in runs
    ]
    aggregate: Dict[str, Any] = {
        "started_at": t0,
        "ended_at": time.time(),
        "profile": str(ns.profile),
        "base_url": base_url,
        "runs": series,
    }
    with open(os.path.join(results_dir, SOAK_SUMMARY_FILE), "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)
    print(json.dumps({"runs": len(series), "profile": ns.profile}, separators=(",", ":"), sort_keys=True))
    return aggregate


def run_soak(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ns = build_arg_parser().parse_args(argv)
    return asyncio.run(run_soak_async(ns))


if __name__ == "__main__":
    _ = run_soak()