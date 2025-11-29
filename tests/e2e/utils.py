#!/usr/bin/env python3
# tests/e2e/utils.py
# Stdlib-first helper utilities for the E2E harness.
#
# References:
# - Gateway RBAC dependency: [rbac_dependency()](services/gateway/src/rbac.py:62)
# - HIF validator entrypoint: [validate_hif()](libs/hif/validator.py:117)
# - Chaos injector CLI: [main()](tests/chaos/chaos_injector.py:90)
# - Load runner harness: [run_load_async()](tests/load/load_runner.py:455)
# - Attach-rate analyzer: [main()](tools/analysis/attach_rate_analyzer.py:396)

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

# Optional httpx support (import-guarded; stdlib urllib remains the default)
try:
    import httpx  # type: ignore
except Exception:
    httpx = None  # type: ignore


# --------------------------
# Filesystem and JSON helpers
# --------------------------

def ensure_dir(path: str | Path) -> None:
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)


def read_json_file(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    data = p.read_text(encoding="utf-8")
    return json.loads(data)


def write_json_file(path: str | Path, obj: Any) -> None:
    p = Path(path)
    if p.parent:
        ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_text_file(path: str | Path, text: str) -> None:
    p = Path(path)
    if p.parent:
        ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")


def now_ms() -> int:
    return int(time.time() * 1000)


def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# --------------------------
# HTTP client (stdlib-first)
# --------------------------

class HttpClient:
    """
    Minimal HTTP client with stdlib urllib and optional httpx acceleration.

    Notes:
    - Uses JSON bodies and returns tuples for simple branching without raising.
    - Avoids third-party deps unless httpx is present.
    """

    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = float(timeout)

    def _headers(self, headers: Optional[Dict[str, str]]) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if headers:
            h.update(headers)
        return h

    def get_json(self, url: str, headers: Optional[Dict[str, str]] = None) -> Tuple[int, Dict[str, Any], str]:
        # Returns (status_code, json_or_empty, raw_text_on_error_or_empty)
        if httpx is not None:
            try:
                with httpx.Client(timeout=self.timeout) as client:  # type: ignore
                    resp = client.get(url, headers=self._headers(headers))
                    try:
                        data = resp.json()
                        if isinstance(data, dict):
                            return int(resp.status_code), data, ""
                        return int(resp.status_code), {}, resp.text[:2048]
                    except Exception:
                        return int(resp.status_code), {}, resp.text[:2048]
            except Exception as e:
                return 599, {}, str(e)[:2048]

        # urllib fallback
        req = urllib.request.Request(url, headers=self._headers(headers), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                status = getattr(r, "status", 200)
                body = r.read()
                try:
                    obj = json.loads(body.decode("utf-8", errors="ignore"))
                    return int(status), (obj if isinstance(obj, dict) else {}), ""
                except Exception:
                    return int(status), {}, body[:2048].decode("utf-8", errors="ignore")
        except urllib.error.HTTPError as he:
            try:
                body = he.read()
            except Exception:
                body = b""
            return int(he.code), {}, body[:2048].decode("utf-8", errors="ignore")
        except Exception as e:
            return 599, {}, str(e)[:2048]

    def post_json(
        self, url: str, body: Mapping[str, Any], headers: Optional[Dict[str, str]] = None
    ) -> Tuple[int, Dict[str, Any], Dict[str, str], str]:
        # Returns (status_code, json_or_empty, resp_headers, raw_text_on_error_or_empty)
        if httpx is not None:
            try:
                with httpx.Client(timeout=self.timeout) as client:  # type: ignore
                    resp = client.post(url, json=body, headers=self._headers(headers))
                    try:
                        data = resp.json()
                        if not isinstance(data, dict):
                            data = {}
                    except Exception:
                        data = {}
                    return int(resp.status_code), data, dict(resp.headers), "" if resp.status_code < 400 else resp.text[:2048]
            except Exception as e:
                return 599, {}, {}, str(e)[:2048]

        # urllib fallback
        data_bytes = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes, headers=self._headers(headers), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                status = getattr(r, "status", 200)
                resp_headers = {k: v for k, v in getattr(r, "headers", {}).items()}
                body_bytes = r.read()
                try:
                    obj = json.loads(body_bytes.decode("utf-8", errors="ignore"))
                    return int(status), (obj if isinstance(obj, dict) else {}), resp_headers, ""
                except Exception:
                    return int(status), {}, resp_headers, body_bytes[:2048].decode("utf-8", errors="ignore")
        except urllib.error.HTTPError as he:
            try:
                body_bytes = he.read()
            except Exception:
                body_bytes = b""
            return int(he.code), {}, dict(getattr(he, "headers", {})), body_bytes[:2048].decode("utf-8", errors="ignore")
        except Exception as e:
            return 599, {}, {}, str(e)[:2048]


# ---------------------------------
# Prometheus exposition text parsing
# ---------------------------------

_METRIC_LINE_RE = re.compile(r"""^
    (?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)
    (?:\{
        (?P<labels>[^}]*)  # key="value",key2="value2"
    \})?
    \s+
    (?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)
    (?:\s+(?P<ts>\d+))?
    \s*$
""", re.VERBOSE)

def _parse_labels(lbl: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not lbl:
        return out
    # split on comma not inside quotes
    parts: List[str] = []
    buf, in_q, esc = "", False, False
    for ch in lbl:
        if esc:
            buf += ch
            esc = False
            continue
        if ch == "\\":
            buf += ch
            esc = True
            continue
        if ch == '"':
            in_q = not in_q
            buf += ch
            continue
        if ch == "," and not in_q:
            if buf.strip():
                parts.append(buf.strip())
            buf = ""
            continue
        buf += ch
    if buf.strip():
        parts.append(buf.strip())

    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] == '"':
            try:
                v = bytes(v[1:-1], "utf-8").decode("unicode_escape")
            except Exception:
                v = v[1:-1]
        out[k] = v
    return out


def parse_prometheus_text(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Return dict: name -> list of samples {labels: {}, value: float, ts: Optional[int]}
    Ignores HELP/TYPE/comment lines. Low-cardinality presence checks only for E2E.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC_LINE_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        labels = _parse_labels(m.group("labels") or "")
        try:
            val = float(m.group("value"))
        except Exception:
            val = math.nan
        ts = m.group("ts")
        sample = {"labels": labels, "value": val}
        if ts:
            try:
                sample["ts"] = int(ts)
            except Exception:
                pass
        out.setdefault(name, []).append(sample)
    return out


# --------------------------
# Subprocess helpers
# --------------------------

@dataclass
class CmdResult:
    rc: int
    out: str
    err: str
    elapsed_ms: int


def run_cmd(cmd: List[str], cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None, timeout: Optional[int] = None) -> CmdResult:
    t0 = monotonic_ms()
    try:
        cp = subprocess.run(
            cmd,
            cwd=cwd,
            env=env or os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            text=True,
        )
        return CmdResult(rc=int(cp.returncode), out=cp.stdout, err=cp.stderr, elapsed_ms=monotonic_ms() - t0)
    except subprocess.TimeoutExpired as te:
        return CmdResult(rc=124, out=te.stdout or "", err=te.stderr or "timeout", elapsed_ms=monotonic_ms() - t0)
    except FileNotFoundError as fnf:
        return CmdResult(rc=127, out="", err=str(fnf), elapsed_ms=monotonic_ms() - t0)
    except Exception as e:
        return CmdResult(rc=1, out="", err=str(e), elapsed_ms=monotonic_ms() - t0)


def which(bin_name: str) -> Optional[str]:
    for p in (os.getenv("PATH") or "").split(os.pathsep):
        cand = Path(p) / bin_name
        if cand.exists() and os.access(cand, os.X_OK):
            return str(cand)
    return None


# --------------------------
# Audit log helpers (JSONL)
# --------------------------

def iter_audit_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
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
                    continue
    except FileNotFoundError:
        return
    except Exception:
        return


# --------------------------
# S3/local artifact helpers
# --------------------------

def read_artifact(path_or_uri: str) -> Optional[bytes]:
    """
    Supports:
    - file paths
    - file:// URIs
    - s3:// URIs (skipped; returns None with a note to caller for graceful skip)
    """
    if not path_or_uri:
        return None
    # Normalize file://
    if path_or_uri.startswith("file://"):
        local = urllib.parse.urlparse(path_or_uri)
        p = local.path
        try:
            return Path(p).read_bytes()
        except Exception:
            return None
    if path_or_uri.startswith("s3://"):
        # No boto deps allowed; caller should SKIP gracefully.
        return None
    # Plain path
    try:
        return Path(path_or_uri).read_bytes()
    except Exception:
        return None


def read_json_maybe_gz(path: str | Path) -> Optional[Any]:
    p = Path(path)
    try:
        data = p.read_bytes()
    except Exception:
        return None
    try:
        if p.suffix == ".gz":
            data = gzip.decompress(data)
        return json.loads(data.decode("utf-8", errors="ignore"))
    except Exception:
        return None


# --------------------------
# Lightweight YAML sanity (string-based)
# --------------------------

def yaml_contains_kinds(text: str, required_kinds: Iterable[str]) -> Dict[str, bool]:
    """Very light check for 'kind: XYZ' presence without full YAML parsing."""
    found: Dict[str, bool] = {}
    lines = [ln.strip() for ln in (text or "").splitlines()]
    for rk in required_kinds:
        target = f"kind: {rk}"
        found[rk] = any(ln == target for ln in lines)
    return found


def grep_yaml_keys(text: str, key_regex: str) -> List[str]:
    pat = re.compile(key_regex)
    out: List[str] = []
    for ln in (text or "").splitlines():
        if pat.search(ln):
            out.append(ln.strip())
    return out


# --------------------------
# Polling utility
# --------------------------

def poll_until(fn, timeout_s: float = 60.0, interval_s: float = 0.5) -> Tuple[bool, Any]:
    """
    Repeatedly call fn() -> (done_bool, value) until done or timeout.
    Returns (done, last_value).
    """
    t0 = time.monotonic()
    last_val: Any = None
    while (time.monotonic() - t0) < float(timeout_s):
        try:
            done, val = fn()
            last_val = val
            if done:
                return True, val
        except Exception as e:
            last_val = e
        time.sleep(interval_s)
    return False, last_val


# --------------------------
# Percentiles
# --------------------------

def percentile(sorted_vals: List[float], pct: float) -> float:
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


# --------------------------
# Orchestrator config model
# --------------------------

@dataclass
class E2EConfig:
    base_url: str
    metrics_gateway: Optional[str]
    metrics_explainer: Optional[str]
    auth_token_write: Optional[str]
    auth_token_read: Optional[str]
    tenant_a: Optional[str]
    tenant_b: Optional[str]
    status_json: Optional[str]
    audit_log: Optional[str]
    s3_check: Optional[str]
    helm_chart: Optional[str]
    argocd_manifest: Optional[str]
    keda_template: Optional[str]
    karpenter_file: Optional[str]
    load_duration: int
    concurrency: int
    attach_rate: float
    token_mix: float
    chaos_control: Optional[str]
    output_json: str
    output_md: str

    @staticmethod
    def from_args(ns: argparse.Namespace) -> "E2EConfig":
        out = E2EConfig(
            base_url=ns.base_url,
            metrics_gateway=ns.metrics_gateway,
            metrics_explainer=ns.metrics_explainer,
            auth_token_write=ns.auth_token_write,
            auth_token_read=ns.auth_token_read,
            tenant_a=ns.tenant_a,
            tenant_b=ns.tenant_b,
            status_json=ns.status_json,
            audit_log=ns.audit_log,
            s3_check=ns.s3_check,
            helm_chart=ns.helm_chart,
            argocd_manifest=ns.argocd_manifest,
            keda_template=ns.keda_template,
            karpenter_file=ns.karpenter_file,
            load_duration=int(ns.load_duration),
            concurrency=int(ns.concurrency),
            attach_rate=float(ns.attach_rate),
            token_mix=float(ns.token_mix),
            chaos_control=ns.chaos_control,
            output_json=ns.output,
            output_md=str(ns.output).rsplit(".", 1)[0] + ".md",
        )
        return out


# --------------------------
# Results aggregation helpers
# --------------------------

def new_result(test_name: str, status: str, reason: Optional[str] = None, details: Optional[Dict[str, Any]] = None, duration_ms: Optional[int] = None) -> Dict[str, Any]:
    return {
        "test": test_name,
        "status": status,  # PASS | FAIL | SKIP
        "reason": reason,
        "details": details or {},
        "duration_ms": duration_ms,
        "ts": int(time.time()),
    }


def write_report(outputs_dir: str, json_path: str, md_path: str, items: List[Dict[str, Any]], env: Dict[str, Any]) -> None:
    ensure_dir(outputs_dir)
    summary = {
        "environment": env,
        "results": items,
        "stats": {
            "total": len(items),
            "passed": sum(1 for x in items if x.get("status") == "PASS"),
            "failed": sum(1 for x in items if x.get("status") == "FAIL"),
            "skipped": sum(1 for x in items if x.get("status") == "SKIP"),
        },
        "generated_at": int(time.time()),
    }
    write_json_file(json_path, summary)

    # Markdown
    lines: List[str] = []
    lines.append("# Async Sidecar E2E Report")
    lines.append("")
    lines.append("## Environment")
    for k, v in env.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Test | Status | Duration(ms) | Reason |")
    lines.append("|------|--------|--------------|--------|")
    for it in items:
        lines.append(f"| {it.get('test')} | {it.get('status')} | {it.get('duration_ms') or ''} | {it.get('reason') or ''} |")
    lines.append("")
    write_text_file(md_path, "\n".join(lines) + "\n")


# --------------------------
# Module discovery helper
# --------------------------

def import_test_module(mod_path: str) -> Any:
    """
    Import a test module by file path without installing as a package.
    Requires each module to expose run(config: E2EConfig) -> Dict[str, Any]
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(Path(mod_path).stem, mod_path)
    assert spec and spec.loader, f"cannot load {mod_path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# --------------------------
# Small helpers for endpoints
# --------------------------

def auth_headers(token: Optional[str], tenant: Optional[str] = None) -> Dict[str, str]:
    h: Dict[str, str] = {}
    if token:
        h["Authorization"] = f"Bearer {token}"
    if tenant:
        h["x-tenant-id"] = str(tenant)
    return h


def post_chat(client: HttpClient, base_url: str, token: Optional[str], tenant: Optional[str], explain: bool = True, granularity: str = "sentence", features: str = "sae-gpt4-2m") -> Tuple[int, Dict[str, Any], Dict[str, str], str]:
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Briefly say hello and mention a random color."},
        ],
        "stream": False,
        "temperature": 0.2,
    }
    headers = auth_headers(token, tenant)
    if explain:
        headers["x-explain-mode"] = "hypergraph"
        headers["x-explain-granularity"] = granularity
        headers["x-explain-features"] = features
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    return client.post_json(url, body, headers=headers)


def get_status(client: HttpClient, base_url: str, trace_id: str, token: Optional[str], tenant: Optional[str]) -> Tuple[int, Dict[str, Any], str]:
    url = f"{base_url.rstrip('/')}/v1/traces/{trace_id}/status"
    return client.get_json(url, headers=auth_headers(token, tenant))


def get_graph(client: HttpClient, base_url: str, trace_id: str, token: Optional[str], tenant: Optional[str]) -> Tuple[int, Dict[str, Any], str]:
    url = f"{base_url.rstrip('/')}/v1/traces/{trace_id}/graph"
    return client.get_json(url, headers=auth_headers(token, tenant))