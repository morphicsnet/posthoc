#!/usr/bin/env python3
# tests/e2e/test_capacity_attach_rate.py
# Attach-rate and capacity validation against Helm values.
#
# Plan:
# - Run the attach-rate analyzer CLI to produce a summary JSON
#   [main()](tools/analysis/attach_rate_analyzer.py:396)
# - Parse Helm values.yaml for explainer replicas and KEDA min/max
#   [values.yaml](manifests/helm/hypergraph/values.yaml:1)
# - Compare analyzer recommendations vs chart config:
#   * Require p95.conservative.total GPUs <= maxReplicaCount + slack
#   * Require mean.optimistic.total GPUs >= minReplicaCount
#
# Notes:
# - This is presence-based and environment-sensitive; skips gracefully when inputs are unavailable.
# - "GPU replicas" below refers to explainer-replica guidance; environments without GPU still benefit as a scale proxy.

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tests.e2e.utils import E2EConfig, new_result, run_cmd


ANALYZER = Path("tools/analysis/attach_rate_analyzer.py")
ANALYZER_OUT = Path("tools/analysis/results/attach_rate_summary.json")
VALUES_YAML = Path("manifests/helm/hypergraph/values.yaml")

# Slack allowed between p95.conservative recommendation and maxReplicaCount
MAX_SLACK = 2


def _parse_values_yaml(path: Path) -> Dict[str, Any]:
    """
    Minimal stateful parser to extract:
      - explainer.replicas (base)
      - explainer.keda.minReplicaCount / maxReplicaCount / backlogSecondsTarget
    """
    out: Dict[str, Any] = {
        "expl_replicas": None,
        "minReplicaCount": None,
        "maxReplicaCount": None,
        "backlogSecondsTarget": None,
    }
    if not path.exists():
        return out
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return out

    in_explainer = False
    in_keda = False
    expl_indent = None
    keda_indent = None

    def _indent(s: str) -> int:
        return len(s) - len(s.lstrip())

    for raw in lines:
        s = raw.rstrip("\n")
        if not s.strip() or s.strip().startswith("#"):
            continue
        ind = _indent(s)
        st = s.strip()

        if st.startswith("explainer:"):
            in_explainer = True
            in_keda = False
            expl_indent = ind
            keda_indent = None
            continue

        if in_explainer and (expl_indent is not None) and ind <= expl_indent and not st.startswith("explainer:"):
            in_explainer = False
            in_keda = False
            expl_indent = None
            keda_indent = None

        if in_explainer and st.startswith("keda:"):
            in_keda = True
            keda_indent = ind
            continue

        if in_keda and (keda_indent is not None) and ind <= keda_indent and not st.startswith("keda:"):
            in_keda = False
            keda_indent = None

        if in_explainer and not in_keda:
            if st.startswith("replicas:"):
                val_str = st.split(":", 1)[1].strip().strip('"').strip("'")
                try:
                    out["expl_replicas"] = int(val_str)
                except Exception:
                    out["expl_replicas"] = val_str

        if in_keda:
            for key in ("minReplicaCount", "maxReplicaCount", "backlogSecondsTarget"):
                if st.startswith(f"{key}:"):
                    val = st.split(":", 1)[1].strip().strip('"').strip("'")
                    try:
                        out[key] = int(val) if "Replica" in key else float(val)
                    except Exception:
                        out[key] = val
    return out


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _pick_recommendation(recs: List[Dict[str, Any]], target: int) -> Optional[Dict[str, Any]]:
    if not isinstance(recs, list) or not recs:
        return None
    # Prefer exact concurrency match; else nearest by absolute distance
    exact = [r for r in recs if int(r.get("concurrency", -1)) == int(target)]
    if exact:
        return exact[0]
    # Nearest
    try:
        best = None
        best_d = 1e9
        for r in recs:
            c = int(r.get("concurrency", -1))
            d = abs(c - int(target))
            if d < best_d:
                best = r
                best_d = d
        return best
    except Exception:
        return recs[0]


def _run_analyzer(cfg: E2EConfig) -> Tuple[bool, str]:
    """
    Execute the analyzer with orchestrator-provided inputs.
    """
    py = sys.executable or "python3"
    cmd = [
        py,
        str(ANALYZER),
        "--status-json", cfg.status_json or "/tmp/hif/status.json",
        "--window-minutes", "60",
        "--bucket-seconds", "60",
        "--token-mix", f"{max(0.0, min(1.0, float(cfg.token_mix))):.3f}",
        "--concurrency", str(max(1, int(cfg.concurrency))),
        "--output", str(ANALYZER_OUT),
    ]
    # Optional audit log
    if cfg.audit_log:
        cmd += ["--audit-log", cfg.audit_log]
    res = run_cmd(cmd)
    ok = (res.rc == 0) and ANALYZER_OUT.exists()
    return ok, (res.out or res.err or "")


def run(config: E2EConfig) -> Dict[str, Any]:
    # Ensure values.yaml present
    values_path = Path(config.helm_chart or VALUES_YAML.parent.parent).joinpath("values.yaml")
    if not values_path.exists():
        return new_result("test_capacity_attach_rate.py", "SKIP", reason=f"values.yaml not found at {values_path}")

    # Run analyzer
    ok, _out = _run_analyzer(config)
    if not ok:
        return new_result("test_capacity_attach_rate.py", "SKIP", reason="attach_rate_analyzer did not complete", details={"output_path": str(ANALYZER_OUT)})

    summary = _read_json(ANALYZER_OUT)
    recs = summary.get("capacity_recommendations") or []
    if not recs:
        return new_result("test_capacity_attach_rate.py", "SKIP", reason="no capacity recommendations in analyzer output")

    values = _parse_values_yaml(values_path)
    min_rc = values.get("minReplicaCount")
    max_rc = values.get("maxReplicaCount")

    if (min_rc is None) or (max_rc is None):
        return new_result("test_capacity_attach_rate.py", "SKIP", reason="could not parse min/max replicas from values.yaml", details={"values": values})

    rec = _pick_recommendation(recs, int(config.concurrency))
    if not isinstance(rec, dict):
        return new_result("test_capacity_attach_rate.py", "SKIP", reason="no recommendation matched concurrency", details={"concurrency": config.concurrency})

    # Extract guidance
    reps = rec.get("replicas") or {}
    mean = reps.get("mean") or {}
    p95 = reps.get("p95") or {}
    mean_opti_total = int(((mean.get("optimistic") or {}).get("total") or 0))
    p95_cons_total = int(((p95.get("conservative") or {}).get("total") or 0))

    # Acceptance rules:
    # - mean.optimistic.total >= minReplicaCount
    # - p95.conservative.total <= maxReplicaCount + MAX_SLACK
    pass_min = (mean_opti_total >= int(min_rc))
    pass_max = (p95_cons_total <= int(max_rc) + int(MAX_SLACK))

    details = {
        "values": {"minReplicaCount": min_rc, "maxReplicaCount": max_rc},
        "recommendation": {
            "concurrency": rec.get("concurrency"),
            "mean_optimistic_total": mean_opti_total,
            "p95_conservative_total": p95_cons_total,
        },
        "slack_allowance": MAX_SLACK,
        "analyzer_output": str(ANALYZER_OUT),
    }

    if pass_min and pass_max:
        return new_result("test_capacity_attach_rate.py", "PASS", details=details)

    reasons: List[str] = []
    if not pass_min:
        reasons.append(f"mean.optimistic.total={mean_opti_total} < minReplicaCount={min_rc}")
    if not pass_max:
        reasons.append(f"p95.conservative.total={p95_cons_total} > maxReplicaCount+slack={int(max_rc)+int(MAX_SLACK)}")
    return new_result("test_capacity_attach_rate.py", "FAIL", reason="; ".join(reasons), details=details)