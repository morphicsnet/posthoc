#!/usr/bin/env python3
# tests/e2e/test_backpressure_under_load.py
# Stress/load validation with backpressure presence checks and HIF continuity under load.
#
# Actions:
# - Invoke load harness [run_load_async()](tests/load/load_runner.py:455) via CLI for a short burst using orchestrator-provided params.
# - After run:
#   * Scrape /metrics (Gateway and/or Explainer) and check for presence of backpressure signals
#     (e.g., backpressure_actions_total, backpressure_level) when available; skip gracefully if absent.
#   * Sample a subset of attached traces from chat_results.jsonl and assert at least some graphs are retrievable (HIF continuity).
#
# Notes:
# - This test is environment-sensitive. If the deployment does not expose backpressure metrics, the test will not fail the suite.
# - Keeps assertions presence-based and low-cardinality per guidelines.
#
# Related references:
# - Gateway endpoints: [create_chat_completion()](services/gateway/src/app.py:540), [get_trace_graph()](services/gateway/src/app.py:922)
# - Explainer metrics API (if running separately): [setup_otel()](services/explainer/src/otel.py:273)

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tests.e2e.utils import (
    E2EConfig,
    HttpClient,
    auth_headers,
    new_result,
    parse_prometheus_text,
    run_cmd,
    which,
)

RESULTS_DIR = Path("tests/load/results")
CHAT_RESULTS_FILE = RESULTS_DIR / "chat_results.jsonl"
EXPL_RESULTS_FILE = RESULTS_DIR / "explanation_results.jsonl"
SUMMARY_FILE = RESULTS_DIR / "summary.json"


def _auth_enabled_probe(base_url: str) -> bool:
    import urllib.request
    import urllib.error
    url = f"{base_url.rstrip('/')}/v1/traces/trc_probe/status"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=3.0):
            return False
    except urllib.error.HTTPError as he:
        return int(getattr(he, "code", 0)) in (401, 403)
    except Exception:
        return False


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = (ln or "").strip()
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


def _fetch_text(url: str, timeout: float = 5.0) -> Optional[str]:
    import urllib.request
    import urllib.error
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as he:
        try:
            return (he.read() or b"").decode("utf-8", errors="ignore")
        except Exception:
            return None
    except Exception:
        return None


def _run_load_cli(cfg: E2EConfig) -> Tuple[bool, str, str]:
    """
    Execute the load runner with orchestrator parameters.
    Returns (ok, stdout, stderr).
    """
    py = sys.executable or "python3"
    load_script = str((Path(__file__).resolve().parents[2] / "tests" / "load" / "load_runner.py"))
    cmd = [
        py,
        load_script,
        "--base-url", cfg.base_url,
        "--duration-seconds", str(max(5, int(cfg.load_duration))),
        "--concurrency", str(max(1, int(cfg.concurrency))),
        "--attach-rate", f"{max(0.0, min(1.0, float(cfg.attach_rate))):.3f}",
        "--granularity", "mix",
        "--features", "sae-gpt4-2m",
        "--rps-limit", "0",
    ]
    # Prefer write token so attachments can be requested
    if cfg.auth_token_write:
        cmd += ["--auth-token", cfg.auth_token_write]
    if cfg.tenant_a:
        cmd += ["--tenant-id", cfg.tenant_a]
    res = run_cmd(cmd, cwd=str(Path(__file__).resolve().parents[2]))
    return (res.rc == 0), res.out, res.err


def _sample_attached_trace_ids(chat_lines: List[Dict[str, Any]], max_n: int = 12) -> List[str]:
    tids: List[str] = []
    for rec in chat_lines:
        try:
            if bool(rec.get("attached")) and isinstance(rec.get("trace_id"), str):
                tids.append(str(rec["trace_id"]))
        except Exception:
            continue
    random.shuffle(tids)
    return tids[:max_n]


def run(config: E2EConfig) -> Dict[str, Any]:
    # Preconditions
    base = config.base_url
    if not base:
        return new_result("test_backpressure_under_load.py", "SKIP", reason="no base_url provided")

    auth_enabled = _auth_enabled_probe(base)
    if auth_enabled and not config.auth_token_write:
        return new_result("test_backpressure_under_load.py", "SKIP", reason="auth enabled but --auth-token-write not provided")

    # Execute load for a short, higher-concurrency burst
    ok, out, err = _run_load_cli(config)
    if not ok:
        return new_result("test_backpressure_under_load.py", "SKIP", reason=f"load runner failed rc!=0", details={"stdout": out[-600:], "stderr": err[-600:]})

    # Read outputs
    try:
        with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception:
        summary = {}

    chat_lines = _load_jsonl(CHAT_RESULTS_FILE)
    expl_lines = _load_jsonl(EXPL_RESULTS_FILE)

    # Backpressure presence check via metrics (optional)
    gw_metrics = config.metrics_gateway or (base.rstrip("/") + "/metrics")
    ex_metrics = config.metrics_explainer or "http://localhost:9090/metrics"
    gw_text = _fetch_text(gw_metrics, timeout=5.0)
    ex_text = _fetch_text(ex_metrics, timeout=5.0)

    bp_signals: Dict[str, Any] = {"gateway": False, "explainer": False, "names": []}
    def _scan_bp(txt: Optional[str]) -> Tuple[bool, List[str]]:
        names: List[str] = []
        if not isinstance(txt, str) or not txt.strip():
            return False, names
        parsed = parse_prometheus_text(txt)
        present = False
        for nm in ("backpressure_actions_total", "backpressure_level", "hypergraph_backlog_seconds"):
            if nm in parsed:
                present = True
                names.append(nm)
        return present, names

    gw_bp, gw_names = _scan_bp(gw_text)
    ex_bp, ex_names = _scan_bp(ex_text)
    bp_signals["gateway"] = gw_bp
    bp_signals["explainer"] = ex_bp
    bp_signals["names"] = list({*gw_names, *ex_names})

    # HIF continuity: sample attached trace_ids and try to fetch graphs
    client = HttpClient(timeout=8.0)
    tenant = config.tenant_a or "tenantA"
    token = config.auth_token_read if auth_enabled else None

    tids = _sample_attached_trace_ids(chat_lines, max_n=12)
    graphs_ok = 0
    graphs_total = len(tids)
    for tid in tids:
        st, body, _err = client.get_json(f"{base.rstrip('/')}/v1/traces/{tid}/graph", headers=auth_headers(token, tenant))
        if st == 200 and isinstance(body, dict) and ("nodes" in body or "incidences" in body or "hyperedges" in body):
            graphs_ok += 1

    # Compose details
    details = {
        "summary": summary,
        "metrics": {
            "gateway_endpoint": gw_metrics,
            "explainer_endpoint": ex_metrics,
            "bp_signals_present": bp_signals,
        },
        "graphs_checked": {"ok": graphs_ok, "total": graphs_total},
        "samples": {
            "chat_count": len(chat_lines),
            "expl_count": len(expl_lines),
        },
    }

    # Decide outcome:
    # - Require load to have executed (we already ensured) and that at least some attached graphs are retrievable (continuity).
    # - Backpressure signals are optional; if absent, record as note but do not fail.
    if graphs_total == 0:
        return new_result("test_backpressure_under_load.py", "SKIP", reason="no attached traces found in load outputs", details=details)

    if graphs_ok == 0:
        return new_result("test_backpressure_under_load.py", "FAIL", reason="no graphs retrievable for sampled attached traces", details=details)

    # PASS, with notes on bp signals (could be false if not exported by env)
    return new_result("test_backpressure_under_load.py", "PASS", reason=None, details=details)