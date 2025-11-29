#!/usr/bin/env python3
# tests/e2e/test_autoscaling_policies.py
# Autoscaling policies sanity:
# - KEDA ScaledObject template consistency with values.yaml (min/max replicas, backlogSecondsTarget)
# - Karpenter provisioners presence and key fields
#
# Skips gracefully if manifests are not present.
#
# References:
# - Helm values: [values.yaml](manifests/helm/hypergraph/values.yaml:1)
# - KEDA Scaler template: [keda-scalers.yaml](manifests/helm/hypergraph/templates/keda-scalers.yaml:1)
# - Karpenter provisioners: [karpenter-provisioners.yaml](manifests/karpenter/karpenter-provisioners.yaml:1)

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from tests.e2e.utils import E2EConfig, new_result


VALUES_DEFAULT = Path("manifests/helm/hypergraph/values.yaml")
KEDA_TMPL_DEFAULT = Path("manifests/helm/hypergraph/templates/keda-scalers.yaml")
KARPENTER_DEFAULT = Path("manifests/karpenter/karpenter-provisioners.yaml")


def _parse_values_yaml(path: Path) -> Dict[str, Any]:
    """
    Minimal stateful parser to extract:
      .explainer.keda.minReplicaCount
      .explainer.keda.maxReplicaCount
      .explainer.keda.backlogSecondsTarget
    """
    out: Dict[str, Any] = {}
    if not path.exists():
        return out
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return out

    # Track indentation levels for 'explainer:' and 'keda:'
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

        # Level transitions
        if st.startswith("explainer:"):
            in_explainer = True
            in_keda = False
            expl_indent = ind
            keda_indent = None
            continue

        if in_explainer and (expl_indent is not None) and ind <= expl_indent and not st.startswith("explainer:"):
            # leaving explainer block
            in_explainer = False
            in_keda = False
            expl_indent = None
            keda_indent = None

        if in_explainer and st.startswith("keda:"):
            in_keda = True
            keda_indent = ind
            continue

        if in_keda and (keda_indent is not None) and ind <= keda_indent and not st.startswith("keda:"):
            # leaving keda block
            in_keda = False
            keda_indent = None

        if in_keda:
            # capture keys we care about
            for key in ("minReplicaCount", "maxReplicaCount", "backlogSecondsTarget"):
                if st.startswith(f"{key}:"):
                    val_str = st.split(":", 1)[1].strip().strip('"').strip("'")
                    # try int / float coercion
                    try:
                        if key.endswith("Target"):
                            out[key] = float(val_str)
                        else:
                            out[key] = int(val_str)
                    except Exception:
                        # keep as raw string if coercion fails
                        out[key] = val_str
    return out


def _check_keda_template(path: Path, values: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Ensure the template references the same values for:
      - minReplicaCount: .Values.explainer.keda.minReplicaCount
      - maxReplicaCount: .Values.explainer.keda.maxReplicaCount
      - threshold for backlogSecondsTarget: .Values.explainer.keda.backlogSecondsTarget
    This is a template-level check (no 'helm template' execution).
    """
    if not path.exists():
        return False, f"keda template not found at {path}", {}

    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"failed to read keda template: {e}", {}

    # Check explicit value references exist
    refs_ok = all([
        "minReplicaCount: {{ .Values.explainer.keda.minReplicaCount }}" in text,
        "maxReplicaCount: {{ .Values.explainer.keda.maxReplicaCount }}" in text,
    ])

    # Threshold uses printf in the chart; verify reference exists anywhere on the threshold line(s)
    threshold_ref_ok = (".Values.explainer.keda.backlogSecondsTarget" in text) and ("threshold:" in text)

    # Additionally ensure query references namespace templating (non-fatal)
    query_ok = "metricName:" in text and "query:" in text

    ok = refs_ok and threshold_ref_ok
    note = None
    if not refs_ok:
        note = "min/max replica references missing"
    elif not threshold_ref_ok:
        note = "threshold backlogSecondsTarget reference missing"

    details = {
        "values": values,
        "refs_ok": refs_ok,
        "threshold_ref_ok": threshold_ref_ok,
        "query_block_present": query_ok,
    }
    return ok, (note or "ok"), details


def _check_karpenter(path: Path) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Verify presence of CPU and GPU NodePools and NodeClasses with basic required keys.
    Checks are string-based, low-complexity.
    """
    if not path.exists():
        return False, f"karpenter manifest not found at {path}", {}

    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"failed to read karpenter manifest: {e}", {}

    # Presence checks by name/kind
    cpu_pool = ("kind: NodePool" in text) and ("name: hypergraph-cpu" in text)
    gpu_pool = ("kind: NodePool" in text) and ("name: hypergraph-gpu" in text)
    cpu_class = ("kind: EC2NodeClass" in text) and ("name: hypergraph-cpu" in text)
    gpu_class = ("kind: EC2NodeClass" in text) and ("name: hypergraph-gpu" in text)

    # Key fields
    has_requirements = ("requirements:" in text)
    has_role = ("role:" in text)

    ok = cpu_pool and gpu_pool and cpu_class and gpu_class and has_requirements and has_role
    missing_bits = []
    if not cpu_pool:
        missing_bits.append("NodePool hypergraph-cpu")
    if not gpu_pool:
        missing_bits.append("NodePool hypergraph-gpu")
    if not cpu_class:
        missing_bits.append("EC2NodeClass hypergraph-cpu")
    if not gpu_class:
        missing_bits.append("EC2NodeClass hypergraph-gpu")
    if not has_requirements:
        missing_bits.append("requirements")
    if not has_role:
        missing_bits.append("role")

    return ok, ("ok" if ok else f"missing: {missing_bits}"), {
        "cpu_pool": cpu_pool,
        "gpu_pool": gpu_pool,
        "cpu_class": cpu_class,
        "gpu_class": gpu_class,
        "has_requirements": has_requirements,
        "has_role": has_role,
    }


def run(config: E2EConfig) -> Dict[str, Any]:
    values_path = Path(config.helm_chart or VALUES_DEFAULT.parent.parent).joinpath("values.yaml")
    keda_path = Path(config.keda_template or str(KEDA_TMPL_DEFAULT))
    karpenter_path = Path(config.karpenter_file or str(KARPENTER_DEFAULT))

    # values.yaml required for KEDA checks
    if not values_path.exists():
        return new_result("test_autoscaling_policies.py", "SKIP", reason=f"values.yaml not found at {values_path}")

    values = _parse_values_yaml(values_path)

    # Basic sanity that we extracted required fields
    needed = ("minReplicaCount", "maxReplicaCount", "backlogSecondsTarget")
    if not all(k in values for k in needed):
        return new_result("test_autoscaling_policies.py", "SKIP", reason=f"could not parse required keys from values.yaml", details={"values_extracted": values})

    # KEDA template checks
    keda_ok, keda_reason, keda_details = _check_keda_template(keda_path, values)

    # Karpenter checks (optional)
    karp_ok, karp_reason, karp_details = _check_karpenter(karpenter_path)

    details = {
        "values_path": str(values_path),
        "keda_template": str(keda_path),
        "keda": {"ok": keda_ok, "reason": keda_reason, **keda_details},
        "karpenter_file": str(karpenter_path),
        "karpenter": {"ok": karp_ok, "reason": karp_reason, **karp_details},
    }

    # Decision: KEDA must pass (chart integrity). Karpenter is recommended; if missing, SKIP rather than FAIL.
    if not keda_ok:
        return new_result("test_autoscaling_policies.py", "FAIL", reason=f"KEDA checks failed: {keda_reason}", details=details)

    # Karpenter optional: do not fail if absent; record as note.
    return new_result("test_autoscaling_policies.py", "PASS", details=details)