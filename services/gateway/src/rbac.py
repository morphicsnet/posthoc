from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Set

from fastapi import HTTPException, Request, status

# Centralized scope constants/mapping
TRACES_READ = "traces:read"
TRACES_WRITE = "traces:write"
SCOPES = {
    "TRACES_READ": TRACES_READ,
    "TRACES_WRITE": TRACES_WRITE,
}


@dataclass
class AuthContext:
    tenant_id: str
    scopes: Set[str]
    subject: str


def _load_static_tokens() -> Dict[str, Dict[str, Any]]:
    """
    Parse AUTH_TOKENS_JSON env var:
      {"tokenA":{"tenant_id":"t1","scopes":["traces:read","traces:write"],"subject":"userA"}}
    Returns a dict[token] -> identity payload
    """
    raw = os.getenv("AUTH_TOKENS_JSON", "") or ""
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            # normalize inner values
            out: Dict[str, Dict[str, Any]] = {}
            for k, v in data.items():
                if isinstance(k, str) and isinstance(v, dict):
                    out[k] = v
            return out
        return {}
    except Exception:
        return {}


def _ensure_scope_set(scopes_val: Any) -> Set[str]:
    if isinstance(scopes_val, (list, set, tuple)):
        return {str(s) for s in scopes_val}
    if isinstance(scopes_val, str):
        # allow comma-separated string
        return {s.strip() for s in scopes_val.split(",") if s.strip()}
    return set()


def _missing_scopes(required: Iterable[str], granted: Set[str]) -> Set[str]:
    return {s for s in required if s not in granted}


def rbac_dependency(required_scopes: Iterable[str]) -> Callable[[Request], AuthContext]:
    """
    Returns a FastAPI dependency that:
      - Extracts tenant/scopes from Authorization when AUTH_MODE=static
      - Allows anonymous when AUTH_MODE=none (default)
      - Validates required_scopes
    On failure:
      - 401 if missing/invalid token
      - 403 if token known but insufficient scopes
    """
    req_scopes = [str(s) for s in required_scopes]

    async def _dep(request: Request) -> AuthContext:
        mode = (os.getenv("AUTH_MODE", "none") or "none").strip().lower()

        if mode == "none":
            ctx = AuthContext(tenant_id="anon", scopes=set(), subject="anonymous")
            try:
                request.state.auth_ctx = ctx
            except Exception:
                pass
            return ctx

        if mode == "static":
            authz = request.headers.get("authorization") or request.headers.get("Authorization")
            if not authz:
                # Missing credentials
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail={"code": "unauthorized", "message": "Missing Authorization header"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            parts = authz.split()
            if len(parts) != 2 or parts[0].lower() != "bearer":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail={"code": "unauthorized", "message": "Invalid Authorization scheme"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            token = parts[1]
            tokens_map = _load_static_tokens()
            identity = tokens_map.get(token)
            if not isinstance(identity, dict):
                # Unknown token
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={"code": "forbidden", "message": "Invalid token"},
                )

            tenant_id = str(identity.get("tenant_id") or "").strip()
            if not tenant_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={"code": "forbidden", "message": "Invalid token (missing tenant_id)"},
                )
            scopes = _ensure_scope_set(identity.get("scopes"))
            subject = str(identity.get("subject") or tenant_id or "user")

            # Optional static audience/issuer checks (defense-in-depth)
            try:
                exp_iss = os.getenv("AUTH_EXPECT_ISS")
                exp_aud = os.getenv("AUTH_EXPECT_AUD")
                tok_iss = identity.get("iss")
                tok_aud = identity.get("aud")
                if exp_iss and tok_iss and str(tok_iss) != str(exp_iss):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail={"code": "forbidden", "message": "Invalid token (issuer mismatch)"},
                    )
                if exp_aud and tok_aud:
                    if isinstance(tok_aud, (list, set, tuple)):
                        if str(exp_aud) not in {str(x) for x in tok_aud}:
                            raise HTTPException(
                                status_code=status.HTTP_403_FORBIDDEN,
                                detail={"code": "forbidden", "message": "Invalid token (audience mismatch)"},
                            )
                    else:
                        if str(tok_aud) != str(exp_aud):
                            raise HTTPException(
                                status_code=status.HTTP_403_FORBIDDEN,
                                detail={"code": "forbidden", "message": "Invalid token (audience mismatch)"},
                            )
            except HTTPException:
                raise
            except Exception:
                # ignore env parsing errors
                pass

            # Check scopes
            missing = _missing_scopes(req_scopes, scopes)
            if missing:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "code": "missing_scope",
                        "message": f"Missing required scopes: {sorted(list(missing))}",
                    },
                )

            ctx = AuthContext(tenant_id=tenant_id, scopes=scopes, subject=subject)
            try:
                request.state.auth_ctx = ctx
            except Exception:
                pass
            return ctx

        # Fallback: treat as none
        ctx = AuthContext(tenant_id="anon", scopes=set(), subject="anonymous")
        try:
            request.state.auth_ctx = ctx
        except Exception:
            pass
        return ctx

    return _dep


__all__ = ["AuthContext", "rbac_dependency", "TRACES_READ", "TRACES_WRITE", "SCOPES"]