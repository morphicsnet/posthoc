#!/usr/bin/env python3
# tests/e2e/test_deploy_artifacts.py
# Deployment artifacts and GitOps sanity checks.
#
# Validations:
# - If helm binary is available:
#   * Run "helm template" on the Hypergraph chart and assert presence of:
#       - Deployments
#       - Services
#       - PodDisruptionBudgets
#       - NetworkPolicies
#       - ScaledObject (KEDA)
# - If ArgoCD manifest file exists:
#   * Perform minimal YAML sanity for Applications presence.
#
# Skips gracefully with clear reasons when external deps (helm) are unavailable.
#
# References:
# - Helm chart root: manifests/helm/hypergraph/ (Chart.yaml at [Chart.yaml](manifests/helm/hypergraph/Chart.yaml:1))
# - KEDA template: [keda-scalers.yaml](manifests/helm/hypergraph/templates/keda-scalers.yaml:1)
# - NetworkPolicy: [networkpolicy.yaml](manifests/helm/hypergraph/templates/networkpolicy.yaml:1)
# - PodDisruptionBudgets: [poddisruptionbudgets.yaml](manifests/helm/hypergraph/templates/poddisruptionbudgets.yaml:1)

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from tests.e2e.utils import (
    E2EConfig,
    new_result,
    which,
    run_cmd,
    yaml_contains_kinds,
)


REQUIRED_KINDS = [
    "Deployment",
    "Service",
    "PodDisruptionBudget",
    "NetworkPolicy",
    "ScaledObject",  # via KEDA
]


def _helm_template(chart_dir: str) -> Dict[str, Any]:
    helm = which("helm")
    if not helm:
        return {"available": False, "reason": "helm not found in PATH"}

    # Helm requires a release name; namespace does not matter for these presence checks.
    cmd = [helm, "template", "hypergraph", chart_dir]
    res = run_cmd(cmd, cwd=str(Path(chart_dir).resolve()))
    if res.rc != 0:
        return {"available": True, "ok": False, "rc": res.rc, "stderr": (res.err or "")[-800:]}
    return {"available": True, "ok": True, "stdout": res.out or ""}


def _check_required_kinds(rendered_yaml: str, required: Iterable[str]) -> List[str]:
    present = yaml_contains_kinds(rendered_yaml, required)
    missing = [k for k, ok in present.items() if not ok]
    return missing


def _argocd_applications_present(path: str) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    try:
        txt = p.read_text(encoding="utf-8")
        return "kind: Application" in txt
    except Exception:
        return False


def run(config: E2EConfig) -> Dict[str, Any]:
    # Defaults from orchestrator CLI if not provided
    chart_dir = config.helm_chart or str(Path("manifests") / "helm" / "hypergraph")
    argo_path = config.argocd_manifest or str(Path("manifests") / "argocd" / "argocd-apps.yaml")

    chart_path = Path(chart_dir)
    if not chart_path.exists():
        return new_result("test_deploy_artifacts.py", "SKIP", reason=f"helm chart path not found: {chart_dir}")

    # Helm templating
    h = _helm_template(chart_dir)
    details: Dict[str, Any] = {"chart_dir": chart_dir, "helm": {"available": h.get("available", False)}}

    if not h.get("available", False):
        # Helm not installed; skip with reason
        return new_result("test_deploy_artifacts.py", "SKIP", reason=str(h.get("reason") or "helm unavailable"), details=details)

    if not h.get("ok", False):
        return new_result("test_deploy_artifacts.py", "FAIL", reason="helm template failed", details={**details, "helm_err": h.get("stderr")})

    rendered = str(h.get("stdout") or "")
    missing = _check_required_kinds(rendered, REQUIRED_KINDS)

    # ArgoCD Applications sanity (optional)
    apps_present = _argocd_applications_present(argo_path)
    details.update({
        "required_kinds_missing": missing,
        "argocd_manifest": argo_path,
        "argocd_applications_present": apps_present,
    })

    if missing:
        return new_result("test_deploy_artifacts.py", "FAIL", reason=f"rendered chart missing kinds: {missing}", details=details)

    # PASS (ArgoCD presence is optional; report as note)
    return new_result("test_deploy_artifacts.py", "PASS", details=details)