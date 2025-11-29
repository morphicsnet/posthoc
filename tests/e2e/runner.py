#!/usr/bin/env python3
# tests/e2e/runner.py
# Master orchestrator for Async Sidecar end-to-end checks.
#
# Subtests are simple Python scripts exposing: run(config: E2EConfig) -> Dict[str, Any]
# Shared helpers: [E2E utils](tests/e2e/utils.py)
#
# Key references for validations:
# - RBAC dependency: [rbac_dependency()](services/gateway/src/rbac.py:62)
# - Gateway endpoints: [create_chat_completion()](services/gateway/src/app.py:540), [get_trace_status()](services/gateway/src/app.py:886), [get_trace_graph()](services/gateway/src/app.py:922)
# - Explainer metrics: [setup_otel()](services/explainer/src/otel.py:273)
# - HIF validator: [validate_hif()](libs/hif/validator.py:117)
# - Chaos toggles CLI: [main()](tests/chaos/chaos_injector.py:90)
# - Load harness: [run_load_async()](tests/load/load_runner.py:455)
# - Attach-rate analyzer: [main()](tools/analysis/attach_rate_analyzer.py:396)

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Local helpers (stdlib-first)
try:
    # When executed from repo root: python tests/e2e/runner.py ...
    ROOT = Path(__file__).resolve().parents[2]
except Exception:
    ROOT = Path(os.getcwd()).resolve()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.e2e.utils import (  # noqa: E402
    E2EConfig,
    HttpClient,
    ensure_dir,
    import_test_module,
    new_result,
    write_report,
    which,
    run_cmd,
)


DEFAULT_RESULTS_DIR = ROOT / "tests" / "e2e" / "results"
DEFAULT_OUTPUT_JSON = DEFAULT_RESULTS_DIR / "e2e_report.json"


SUBTEST_FILES = [
    "tests/e2e/test_gateway_rbac.py",
    "tests/e2e/test_trace_workflow.py",
    "tests/e2e/test_security_tenant_isolation.py",
    "tests/e2e/test_observability.py",
    "tests/e2e/test_backpressure_under_load.py",
    "tests/e2e/test_slo_adherence.py",
    "tests/e2e/test_deploy_artifacts.py",
    "tests/e2e/test_autoscaling_policies.py",
    "tests/e2e/test_chaos_recovery.py",
    "tests/e2e/test_capacity_attach_rate.py",
    "tests/e2e/test_audit_pii.py",
    "tests/e2e/test_api_versioning.py",
]


def _env_snapshot(cfg: E2EConfig) -> Dict[str, Any]:
    # Detect external tools and minimal env for reproducibility
    helm_bin = which("helm")
    kubectl_bin = which("kubectl")
    argocd_bin = which("argocd")
    git_bin = which("git")

    helm_ver = None
    if helm_bin:
        res = run_cmd([helm_bin, "version", "--short"])
        if res.rc == 0:
            helm_ver = (res.out or "").strip()

    kubectl_ver = None
    if kubectl_bin:
        res = run_cmd([kubectl_bin, "version", "--client", "--output", "yaml"])
        if res.rc == 0:
            kubectl_ver = (res.out or "").strip()

    argocd_ver = None
    if argocd_bin:
        res = run_cmd([argocd_bin, "version", "--client"])
        if res.rc == 0:
            argocd_ver = (res.out or "").strip()

    git_rev = None
    if git_bin:
        res = run_cmd([git_bin, "rev-parse", "--short", "HEAD"], cwd=str(ROOT))
        if res.rc == 0:
            git_rev = (res.out or "").strip()

    return {
        "base_url": cfg.base_url,
        "metrics_gateway": cfg.metrics_gateway,
        "metrics_explainer": cfg.metrics_explainer,
        "auth_write_set": bool(cfg.auth_token_write),
        "auth_read_set": bool(cfg.auth_token_read),
        "tenant_a": cfg.tenant_a,
        "tenant_b": cfg.tenant_b,
        "status_json": cfg.status_json,
        "audit_log": cfg.audit_log,
        "s3_check": cfg.s3_check,
        "helm_chart": cfg.helm_chart,
        "argocd_manifest": cfg.argocd_manifest,
        "keda_template": cfg.keda_template,
        "karpenter_file": cfg.karpenter_file,
        "load_duration": cfg.load_duration,
        "concurrency": cfg.concurrency,
        "attach_rate": cfg.attach_rate,
        "token_mix": cfg.token_mix,
        "chaos_control": cfg.chaos_control,
        "tools": {
            "helm": {"path": helm_bin, "version": helm_ver},
            "kubectl": {"path": kubectl_bin, "version": kubectl_ver},
            "argocd": {"path": argocd_bin, "version": argocd_ver},
            "git": {"path": git_bin, "revision": git_rev},
        },
    }


def _filter_tests(all_tests: List[str], only: List[str], skip: List[str]) -> List[str]:
    out: List[str] = []
    for t in all_tests:
        name = Path(t).name
        if only:
            if not any(fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(t, pat) for pat in only):
                continue
        if skip and any(fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(t, pat) for pat in skip):
            continue
        out.append(t)
    return out


def _load_module(path: Path):
    try:
        return import_test_module(str(path))
    except FileNotFoundError:
        return None
    except AssertionError:
        return None
    except Exception:
        return None


def _run_one(test_path: Path, cfg: E2EConfig) -> Dict[str, Any]:
    t0 = time.perf_counter()
    name = test_path.name
    if not test_path.exists():
        return new_result(name, "SKIP", reason="test file missing", duration_ms=int((time.perf_counter() - t0) * 1000))

    mod = _load_module(test_path)
    if mod is None:
        return new_result(name, "SKIP", reason="import failed", duration_ms=int((time.perf_counter() - t0) * 1000))

    run_fn = getattr(mod, "run", None)
    if run_fn is None or not callable(run_fn):
        return new_result(name, "SKIP", reason="no run(config) exported", duration_ms=int((time.perf_counter() - t0) * 1000))

    try:
        res = run_fn(cfg)  # type: ignore[call-arg]  # pylint: disable=not-callable
        # Enforce schema with defaults
        if not isinstance(res, dict):
            return new_result(name, "FAIL", reason="run() did not return dict", duration_ms=int((time.perf_counter() - t0) * 1000))
        # Ensure test name and duration present
        res.setdefault("test", name)
        res.setdefault("status", "FAIL" if res.get("status") not in ("PASS", "SKIP") else res.get("status"))
        res.setdefault("details", {})
        res.setdefault("reason", None)
        res.setdefault("duration_ms", int((time.perf_counter() - t0) * 1000))
        res.setdefault("ts", int(time.time()))
        return res
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return new_result(name, "FAIL", reason=f"exception: {e}", duration_ms=int((time.perf_counter() - t0) * 1000))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="E2E Orchestrator for Hypergraph Async Sidecar (stdlib-only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base-url", default="http://localhost:8080", help="Gateway base URL")
    p.add_argument("--metrics-gateway", default="http://localhost:8080/metrics", help="Gateway /metrics URL")
    p.add_argument("--metrics-explainer", default="http://localhost:9090/metrics", help="Explainer /metrics URL")
    p.add_argument("--auth-token-write", default=None, help="Bearer token with traces:write")
    p.add_argument("--auth-token-read", default=None, help="Bearer token with traces:read")
    p.add_argument("--tenant-a", default=None, help="Tenant A id")
    p.add_argument("--tenant-b", default=None, help="Tenant B id")
    p.add_argument("--status-json", default="/tmp/hif/status.json", help="Local StatusStore JSON path")
    p.add_argument("--audit-log", default="/var/log/hypergraph/audit.log", help="Gateway audit JSONL (optional)")
    p.add_argument("--s3-check", default=None, help="file:///path or s3://bucket/prefix to verify persistence (optional)")
    p.add_argument("--helm-chart", default=str(ROOT / "manifests" / "helm" / "hypergraph"), help="Path to Helm chart")
    p.add_argument("--argocd-manifest", default=str(ROOT / "manifests" / "argocd" / "argocd-apps.yaml"), help="ArgoCD Applications YAML")
    p.add_argument("--keda-template", default=str(ROOT / "manifests" / "helm" / "hypergraph" / "templates" / "keda-scalers.yaml"), help="KEDA ScaledObject template")
    p.add_argument("--karpenter-file", default=str(ROOT / "manifests" / "karpenter" / "karpenter-provisioners.yaml"), help="Karpenter NodePools file")
    p.add_argument("--load-duration", type=int, default=120, help="Load test duration seconds")
    p.add_argument("--concurrency", type=int, default=300, help="Load test concurrency")
    p.add_argument("--attach-rate", type=float, default=0.3, help="Explanation attach rate [0-1]")
    p.add_argument("--token-mix", type=float, default=0.1, help="Fraction of token-level explanations [0-1]")
    p.add_argument("--chaos-control", default="/tmp/hif/chaos.json", help="Chaos control JSON path")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT_JSON), help="Path to consolidated JSON report")
    p.add_argument("--only", default=None, help="Comma-separated patterns to include (e.g. test_*slo*')")
    p.add_argument("--skip", default=None, help="Comma-separated patterns to exclude")
    return p


def parse_args(argv: Optional[List[str]] = None) -> Tuple[E2EConfig, List[str], List[str], Path, Path]:
    ns = build_arg_parser().parse_args(argv)

    # Derive MD path
    out_json = Path(ns.output)
    out_md = out_json.with_suffix(".md")

    cfg = E2EConfig.from_args(ns)
    only = [p.strip() for p in (ns.only or "").split(",") if p.strip()]
    skip = [p.strip() for p in (ns.skip or "").split(",") if p.strip()]
    return cfg, only, skip, out_json, out_md


def main(argv: Optional[List[str]] = None) -> int:
    cfg, only, skip, out_json, out_md = parse_args(argv)
    results_dir = out_json.parent
    ensure_dir(results_dir)

    # Discover test files and filter
    all_tests = [str((ROOT / p).resolve()) for p in SUBTEST_FILES]
    selected = _filter_tests(all_tests, only, skip)

    # Snapshot environment
    env = _env_snapshot(cfg)

    # Run subtests in order, aggregate results
    items: List[Dict[str, Any]] = []
    for tp in selected:
        res = _run_one(Path(tp), cfg)
        items.append(res)

    # Persist consolidated report (JSON + Markdown)
    write_report(
        outputs_dir=str(results_dir),
        json_path=str(out_json),
        md_path=str(out_md),
        items=items,
        env=env,
    )

    # Emit concise stdout summary
    summary = {
        "total": len(items),
        "passed": sum(1 for x in items if x.get("status") == "PASS"),
        "failed": sum(1 for x in items if x.get("status") == "FAIL"),
        "skipped": sum(1 for x in items if x.get("status") == "SKIP"),
        "output_json": str(out_json),
        "output_md": str(out_md),
    }
    try:
        print(json.dumps(summary, separators=(",", ":"), sort_keys=True))
    except Exception:
        pass

    # Non-zero exit if any FAIL
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())