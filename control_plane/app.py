#!/usr/bin/env python3
# control_plane/app.py
# FastAPI control plane for Async Sidecar GUI: orchestrates E2E runs, scrapes metrics,
# reads Helm/Argo/KEDA/Karpenter manifests, serves compliance/audit endpoints, and exports reports.
#
# Key integrations (repo anchors):
# - E2E orchestrator: [runner.py](tests/e2e/runner.py:1), [utils.py](tests/e2e/utils.py:1)
# - Metrics endpoints: [setup_otel()](services/gateway/src/otel.py:330), [setup_otel()](services/explainer/src/otel.py:273)
# - Helm chart & templates: [Chart.yaml](manifests/helm/hypergraph/Chart.yaml:1), [values.yaml](manifests/helm/hypergraph/values.yaml:1), [keda-scalers.yaml](manifests/helm/hypergraph/templates/keda-scalers.yaml:1)
# - Argo manifest: [argocd-apps.yaml](manifests/argocd/argocd-apps.yaml:1)
# - Karpenter: [karpenter-provisioners.yaml](manifests/karpenter/karpenter-provisioners.yaml:1)
# - Compliance: [hypergraph-api.yaml](api/openapi/hypergraph-api.yaml:1), [validate_hif()](libs/hif/validator.py:117)
# - Security/PII: [sanitize_message()](libs/sanitize/pii.py:72)
#
# Notes:
# - Read-only by design (no cluster mutations). Helm interactions run `helm template` if available, else fall back to textual checks.
# - RBAC middleware supports AUTH_MODE=none|static with AUTH_TOKENS_JSON for token->scopes map (aligned with gateway semantics).

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
import time
import uuid
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure repository root import path for tests/e2e/utils and libs/*
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Optional imports (guarded)
try:
    from tests.e2e.utils import parse_prometheus_text as _parse_prom_text  # type: ignore
except Exception:
    _parse_prom_text = None  # type: ignore

try:
    from libs.hif.validator import validate_hif as _validate_hif  # type: ignore
except Exception:
    _validate_hif = None  # type: ignore


# ----------------------------
# Configuration & constants
# ----------------------------

DEFAULT_RESULTS_DIR = ROOT / "tests" / "e2e" / "results"
DEFAULT_E2E_JSON = DEFAULT_RESULTS_DIR / "e2e_report.json"
DEFAULT_E2E_MD = DEFAULT_RESULTS_DIR / "e2e_report.md"

MANIFESTS_DIR = ROOT / "manifests"
HELM_CHART_DIR = MANIFESTS_DIR / "helm" / "hypergraph"
HELM_VALUES_PATH = HELM_CHART_DIR / "values.yaml"
KEDA_TEMPLATE_PATH = HELM_CHART_DIR / "templates" / "keda-scalers.yaml"
ARGO_APPS_PATH = MANIFESTS_DIR / "argocd" / "argocd-apps.yaml"
KARPENTER_PATH = MANIFESTS_DIR / "karpenter" / "karpenter-provisioners.yaml"

OPENAPI_SPEC = ROOT / "api" / "openapi" / "hypergraph-api.yaml"

AUDIT_LOG_DEFAULT = "/var/log/hypergraph/audit.log"

# ----------------------------
# RBAC middleware (static)
# ----------------------------

class AuthContext(BaseModel):
    tenant_id: str = "anon"
    subject: Optional[str] = None
    scopes: List[str] = Field(default_factory=list)


def _load_static_tokens() -> Dict[str, Dict[str, Any]]:
    raw = os.getenv("AUTH_TOKENS_JSON", "") or ""
    if not raw.strip():
        return {}
    try:
        # Support base64 or plain JSON
        s = raw.strip()
        if re.fullmatch(r"[A-Za-z0-9+/=]+", s):
            try:
                s = base64.b64decode(s.encode("utf-8")).decode("utf-8", errors="ignore")
            except Exception:
                pass
        data = json.loads(s)
        if isinstance(data, dict):
            return {str(k): (v if isinstance(v, dict) else {}) for k, v in data.items()}
        return {}
    except Exception:
        return {}


def rbac_guard(required_scopes: List[str] | Tuple[str, ...] = ()):
    async def _dep(authorization: Optional[str] = None, x_tenant_id: Optional[str] = None) -> AuthContext:
        mode = (os.getenv("AUTH_MODE", "none") or "none").strip().lower()
        if mode == "none":
            return AuthContext(tenant_id=x_tenant_id or "anon", subject=None, scopes=[])

        # static mode: Authorization: Bearer <token>
        if not authorization:
            raise HTTPException(status_code=401, detail={"code": "unauthorized", "message": "Missing Authorization"})
        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise HTTPException(status_code=401, detail={"code": "unauthorized", "message": "Invalid Authorization scheme"})
        token = parts[1]
        tokens_map = _load_static_tokens()
        identity = tokens_map.get(token)
        if not isinstance(identity, dict):
            raise HTTPException(status_code=403, detail={"code": "forbidden", "message": "Invalid token"})

        tenant_id = str(identity.get("tenant_id") or "").strip() or "unknown"
        scopes_val = identity.get("scopes")
        if isinstance(scopes_val, str):
            scopes = [s.strip() for s in scopes_val.split(",") if s.strip()]
        elif isinstance(scopes_val, (list, tuple, set)):
            scopes = [str(s) for s in scopes_val]
        else:
            scopes = []

        # Validate required scopes
        missing = [s for s in required_scopes if s not in scopes]
        if missing:
            raise HTTPException(status_code=403, detail={"code": "missing_scope", "message": f"Missing required scopes: {missing}"})

        # Optional admin can use X-Tenant-ID for read-only cross-tenant (no write paths in this control plane)
        effective_tenant = x_tenant_id or tenant_id
        return AuthContext(tenant_id=str(effective_tenant), subject=str(identity.get("subject") or None), scopes=scopes)
    return _dep


# ----------------------------
# FastAPI app
# ----------------------------

app = FastAPI(title="Async Sidecar Control Plane", version="0.1.0", description="Read-only GUI backend for Async Sidecar operations")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("CORS_ALLOW_ORIGINS", "*")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve UI static files (SPA) at root path.
# This allows the static UI to call control plane APIs on the same origin.
app.mount("/", StaticFiles(directory=str(ROOT / "ui"), html=True), name="ui")

# ----------------------------
# In-memory TestRun registry
# ----------------------------

class RunArgs(BaseModel):
    base_url: str = "http://localhost:8080"
    metrics_gateway: str = "http://localhost:8080/metrics"
    metrics_explainer: str = "http://localhost:9090/metrics"
    auth_token_write: Optional[str] = None
    auth_token_read: Optional[str] = None
    tenant_a: Optional[str] = None
    tenant_b: Optional[str] = None
    status_json: str = "/tmp/hif/status.json"
    audit_log: str = AUDIT_LOG_DEFAULT
    s3_check: Optional[str] = "file:///tmp/hif"
    helm_chart: str = str(HELM_CHART_DIR)
    argocd_manifest: str = str(ARGO_APPS_PATH)
    keda_template: str = str(KEDA_TEMPLATE_PATH)
    karpenter_file: str = str(KARPENTER_PATH)
    load_duration: int = 60
    concurrency: int = 100
    attach_rate: float = 0.3
    token_mix: float = 0.1
    chaos_control: str = "/tmp/hif/chaos.json"
    only: Optional[str] = None
    skip: Optional[str] = None
    output: str = str(DEFAULT_E2E_JSON)


class TestRun(BaseModel):
    id: str
    created_at: float
    created_by: Optional[str] = None
    tenant_id: Optional[str] = None
    args: RunArgs
    status: str = "queued"  # queued|running|completed|failed
    exit_code: Optional[int] = None
    output_json: Optional[str] = None
    output_md: Optional[str] = None
    error: Optional[str] = None


RUNS: Dict[str, TestRun] = {}
RUNS_LOCK = asyncio.Lock()


def _build_runner_cmd(args: RunArgs) -> List[str]:
    py = sys.executable or "python3"
    runner = str(ROOT / "tests" / "e2e" / "runner.py")
    cmd = [
        py, runner,
        "--base-url", args.base_url,
        "--metrics-gateway", args.metrics_gateway,
        "--metrics-explainer", args.metrics_explainer,
        "--status-json", args.status_json,
        "--audit-log", args.audit_log,
        "--helm-chart", args.helm_chart,
        "--argocd-manifest", args.argocd_manifest,
        "--keda-template", args.keda_template,
        "--karpenter-file", args.karpenter_file,
        "--load-duration", str(int(args.load_duration)),
        "--concurrency", str(int(args.concurrency)),
        "--attach-rate", str(float(args.attach_rate)),
        "--token-mix", str(float(args.token_mix)),
        "--chaos-control", args.chaos_control,
        "--output", args.output,
    ]
    if args.tenant_a:
        cmd += ["--tenant-a", args.tenant_a]
    if args.tenant_b:
        cmd += ["--tenant-b", args.tenant_b]
    if args.auth_token_write:
        cmd += ["--auth-token-write", args.auth_token_write]
    if args.auth_token_read:
        cmd += ["--auth-token-read", args.auth_token_read]
    if args.s3_check:
        cmd += ["--s3-check", args.s3_check]
    if args.only:
        cmd += ["--only", args.only]
    if args.skip:
        cmd += ["--skip", args.skip]
    return cmd


async def _run_e2e_subprocess(run_id: str) -> None:
    # Launch runner.py and update RUNS registry
    tr = RUNS.get(run_id)
    if not tr:
        return
    tr.status = "running"
    args = tr.args
    cmd = _build_runner_cmd(args)
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        tr.exit_code = int(proc.returncode)
        # The runner prints a concise JSON summary to stdout on completion
        # Capture output_json/output_md from summary
        try:
            if stdout:
                line = stdout.decode("utf-8", errors="ignore").strip().splitlines()[-1]
                summary = json.loads(line)
                tr.output_json = summary.get("output_json") or str(DEFAULT_E2E_JSON)
                tr.output_md = summary.get("output_md") or str(DEFAULT_E2E_MD)
        except Exception:
            # Fallback to defaults
            tr.output_json = tr.output_json or str(DEFAULT_E2E_JSON)
            tr.output_md = tr.output_md or str(DEFAULT_E2E_MD)
        if tr.exit_code == 0:
            tr.status = "completed"
        else:
            tr.status = "failed"
            err_tail = (stderr or b"").decode("utf-8", errors="ignore")[-800:]
            tr.error = f"runner exit {tr.exit_code}; {err_tail}"
    except Exception as e:
        tr.status = "failed"
        tr.exit_code = -1
        tr.error = str(e)


# ----------------------------
# Routes: health & runs
# ----------------------------

@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    return {"ok": True, "ts": int(time.time())}


@app.post("/api/testruns", response_model=TestRun)
async def start_testrun(req: RunArgs, auth: AuthContext = Depends(rbac_guard(["tests.read"]))):
    run_id = uuid.uuid4().hex[:12]
    tr = TestRun(
        id=run_id,
        created_at=time.time(),
        created_by=auth.subject,
        tenant_id=auth.tenant_id,
        args=req,
        status="queued",
    )
    async with RUNS_LOCK:
        RUNS[run_id] = tr
    # Fire-and-forget background task
    asyncio.create_task(_run_e2e_subprocess(run_id))
    return tr


@app.get("/api/testruns", response_model=List[TestRun])
async def list_testruns(auth: AuthContext = Depends(rbac_guard(["tests.read"]))):
    # Tenant filter (admin/all allowed read-only)
    return sorted(RUNS.values(), key=lambda x: x.created_at, reverse=True)


@app.get("/api/testruns/{run_id}", response_model=TestRun)
async def get_testrun(run_id: str, auth: AuthContext = Depends(rbac_guard(["tests.read"]))):
    tr = RUNS.get(run_id)
    if not tr:
        raise HTTPException(status_code=404, detail="run_id not found")
    return tr


@app.get("/api/testruns/{run_id}/report.json")
async def get_report_json(run_id: str, auth: AuthContext = Depends(rbac_guard(["tests.read"]))):
    tr = RUNS.get(run_id)
    if not tr or not tr.output_json:
        raise HTTPException(status_code=404, detail="report not available")
    p = Path(tr.output_json)
    if not p.exists():
        raise HTTPException(status_code=404, detail="report file missing")
    return FileResponse(str(p), media_type="application/json")


@app.get("/api/testruns/{run_id}/report.md")
async def get_report_md(run_id: str, auth: AuthContext = Depends(rbac_guard(["tests.read"]))):
    tr = RUNS.get(run_id)
    if not tr or not tr.output_md:
        raise HTTPException(status_code=404, detail="report not available")
    p = Path(tr.output_md)
    if not p.exists():
        raise HTTPException(status_code=404, detail="report file missing")
    return FileResponse(str(p), media_type="text/markdown")


@app.websocket("/ws/testruns/{run_id}/stream")
async def ws_stream_run(ws: WebSocket, run_id: str):
    await ws.accept()
    try:
        while True:
            tr = RUNS.get(run_id)
            if not tr:
                await ws.send_text(json.dumps({"type": "error", "message": "run_id not found"}))
                break
            ev = {
                "type": "testrun.update",
                "id": tr.id,
                "status": tr.status,
                "exit_code": tr.exit_code,
                "output_json": tr.output_json,
                "output_md": tr.output_md,
                "error": tr.error,
                "ts": int(time.time()),
            }
            await ws.send_text(json.dumps(ev))
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return
    except Exception:
        try:
            await ws.close()
        except Exception:
            pass


# ----------------------------
# Routes: metrics aggregator
# ----------------------------

class MetricsQuery(BaseModel):
    endpoints: List[str]


@app.post("/api/metrics/aggregate")
async def metrics_aggregate(q: MetricsQuery, auth: AuthContext = Depends(rbac_guard(["observability.read"]))):
    out: Dict[str, Any] = {"sources": [], "parsed": {}}
    for url in q.endpoints:
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=5.0) as r:
                text = r.read().decode("utf-8", errors="ignore")
        except Exception as e:
            out["sources"].append({"url": url, "ok": False, "error": str(e)})
            continue
        out["sources"].append({"url": url, "ok": True, "length": len(text)})
        if _parse_prom_text is not None:
            try:
                parsed = _parse_prom_text(text)  # type: ignore
                # merge by metric name (names only to keep low-cardinality)
                for name, samples in parsed.items():
                    out["parsed"].setdefault(name, 0)
                    out["parsed"][name] += len(samples or [])
            except Exception:
                pass
    return out


# ----------------------------
# Routes: Helm/Argo/KEDA/Karpenter (read-only)
# ----------------------------

def _which(bin_name: str) -> Optional[str]:
    for p in (os.getenv("PATH") or "").split(os.pathsep):
        cand = Path(p) / bin_name
        if cand.exists() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def _helm_template(chart_dir: Path) -> Tuple[bool, str]:
    helm = _which("helm")
    if not helm:
        return False, "helm not found"
    try:
        cp = subprocess.run([helm, "template", "hypergraph", str(chart_dir)], cwd=str(chart_dir), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        if cp.returncode != 0:
            return False, f"helm template failed: {cp.stderr[-400:]}"
        return True, cp.stdout
    except Exception as e:
        return False, str(e)


def _yaml_contains_kinds(text: str, kinds: List[str]) -> Dict[str, bool]:
    found: Dict[str, bool] = {}
    lines = [ln.strip() for ln in (text or "").splitlines()]
    for k in kinds:
        found[k] = any(ln == f"kind: {k}" for ln in lines)
    return found


@app.get("/api/helm/checks")
async def helm_checks(auth: AuthContext = Depends(rbac_guard(["deployments.read"]))):
    if not HELM_CHART_DIR.exists():
        raise HTTPException(status_code=404, detail="helm chart dir missing")
    ok, rendered = _helm_template(HELM_CHART_DIR)
    if not ok:
        # fallback to textual checks using templates present
        try:
            pieces = []
            for p in (HELM_CHART_DIR / "templates").glob("*.yaml"):
                pieces.append(p.read_text(encoding="utf-8"))
            rendered = "\n---\n".join(pieces)
        except Exception:
            rendered = ""
    kinds = ["Deployment", "Service", "PodDisruptionBudget", "NetworkPolicy", "ScaledObject"]
    present = _yaml_contains_kinds(rendered, kinds)
    return {"chart_dir": str(HELM_CHART_DIR), "rendered": ok, "kinds": present}


def _parse_values_yaml(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
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
                try:
                    out["expl_replicas"] = int(st.split(":", 1)[1].strip().strip("'").strip('"'))
                except Exception:
                    pass
        if in_keda:
            for key in ("minReplicaCount", "maxReplicaCount", "backlogSecondsTarget"):
                if st.startswith(f"{key}:"):
                    val = st.split(":", 1)[1].strip().strip("'").strip('"')
                    try:
                        out[key] = int(val) if "Replica" in key else float(val)
                    except Exception:
                        out[key] = val
    return out


@app.get("/api/keda/scalers")
async def keda_scalers(auth: AuthContext = Depends(rbac_guard(["autoscaling.read"]))):
    vals = _parse_values_yaml(HELM_VALUES_PATH)
    keda_tpl = ""
    try:
        keda_tpl = KEDA_TEMPLATE_PATH.read_text(encoding="utf-8")
    except Exception:
        pass
    return {"values_path": str(HELM_VALUES_PATH), "values": vals, "keda_template_excerpt": (keda_tpl[:2000] if keda_tpl else "")}


@app.get("/api/karpenter/nodepools")
async def karpenter_nodepools(auth: AuthContext = Depends(rbac_guard(["autoscaling.read"]))):
    if not KARPENTER_PATH.exists():
        raise HTTPException(status_code=404, detail="karpenter manifest missing")
    text = KARPENTER_PATH.read_text(encoding="utf-8")
    cpu_pool = ("kind: NodePool" in text) and ("name: hypergraph-cpu" in text)
    gpu_pool = ("kind: NodePool" in text) and ("name: hypergraph-gpu" in text)
    cpu_class = ("kind: EC2NodeClass" in text) and ("name: hypergraph-cpu" in text)
    gpu_class = ("kind: EC2NodeClass" in text) and ("name: hypergraph-gpu" in text)
    return {"cpu_pool": cpu_pool, "gpu_pool": gpu_pool, "cpu_class": cpu_class, "gpu_class": gpu_class}


@app.get("/api/argocd/apps")
async def argocd_apps(auth: AuthContext = Depends(rbac_guard(["deployments.read"]))):
    if not ARGO_APPS_PATH.exists():
        raise HTTPException(status_code=404, detail="argocd apps manifest missing")
    txt = ARGO_APPS_PATH.read_text(encoding="utf-8")
    # Minimal surface: return raw text and presence checks
    apps = len(re.findall(r"kind:\s*Application", txt))
    return {"path": str(ARGO_APPS_PATH), "applications": apps, "preview": txt[:2000]}


# ----------------------------
# Routes: Audit & Compliance
# ----------------------------

@app.get("/api/audit/search")
async def audit_search(q: str = "", path: str = AUDIT_LOG_DEFAULT, auth: AuthContext = Depends(rbac_guard(["security.read"]))):
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="audit log not found")
    results: List[Dict[str, Any]] = []
    try:
        with p.open("r", encoding="utf-8") as f:
            for idx, ln in enumerate(f):
                s = (ln or "").strip()
                if not s:
                    continue
                if q and (q.lower() not in s.lower()):
                    continue
                try:
                    obj = json.loads(s)
                    results.append(obj)
                except Exception:
                    results.append({"line": idx + 1, "raw": s})
                if len(results) >= 200:
                    break
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"audit read failed: {e}")
    return {"count": len(results), "items": results}


@app.post("/api/compliance/openapi/check")
async def compliance_openapi(auth: AuthContext = Depends(rbac_guard(["compliance.read"]))):
    if not OPENAPI_SPEC.exists():
        raise HTTPException(status_code=404, detail="openapi spec not found")
    text = OPENAPI_SPEC.read_text(encoding="utf-8")
    # Minimal path extraction
    lines = text.splitlines()
    paths: List[str] = []
    in_paths = False
    base_indent = None

    def _indent(s: str) -> int:
        return len(s) - len(s.lstrip(" "))

    for raw in lines:
        s = raw.rstrip("\n")
        if not s.strip():
            continue
        if s.strip() == "paths:":
            in_paths = True
            base_indent = _indent(s)
            continue
        if in_paths:
            ind = _indent(s)
            if base_indent is not None and ind <= base_indent and not s.strip().startswith("paths:"):
                break
            m = re.match(r"^\s{2,}(/[^:\s]+):\s*$", s)
            if m:
                paths.append(m.group(1))
    non_v1 = [p for p in paths if not p.startswith("/v1/")]
    return {"paths_total": len(paths), "non_v1_paths": non_v1, "ok": (len(paths) > 0 and not non_v1)}


class HIFPayload(BaseModel):
    payload: Dict[str, Any]


@app.post("/api/compliance/hif/validate")
async def compliance_hif_validate(req: HIFPayload, auth: AuthContext = Depends(rbac_guard(["compliance.read"]))):
    if _validate_hif is None:
        # fallback minimal check
        obj = req.payload
        try:
            meta = obj.get("meta") if isinstance(obj, dict) else None
            ver = meta.get("version") if isinstance(meta, dict) else None
            return {"valid": ver == "hif-1", "message": None if ver == "hif-1" else f"unexpected meta.version={ver!r}"}
        except Exception as e:
            return {"valid": False, "message": str(e)}
    try:
        _validate_hif(req.payload)  # type: ignore
        return {"valid": True}
    except Exception as e:
        return JSONResponse(status_code=400, content={"valid": False, "message": str(e)})


# ----------------------------
# Routes: Reports export
# ----------------------------

@app.get("/api/reports/export")
async def export_report(format: str = Query("json", pattern="^(json|md)$"), auth: AuthContext = Depends(rbac_guard(["tests.read"]))):
    # Export latest e2e report if available
    p = DEFAULT_E2E_JSON if format == "json" else DEFAULT_E2E_MD
    if not p.exists():
        raise HTTPException(status_code=404, detail="report artifact not found")
    if format == "json":
        return FileResponse(str(p), media_type="application/json")
    return FileResponse(str(p), media_type="text/markdown")


# ----------------------------
# Entrypoint
# ----------------------------

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("CP_HOST", "0.0.0.0")
    port = int(os.getenv("CP_PORT", "8088"))
    uvicorn.run("control_plane.app:app", host=host, port=port, reload=True)