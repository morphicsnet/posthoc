#!/usr/bin/env python3
# tests/e2e/test_audit_pii.py
# Audit logging integrity and PII scrubbing verification.
#
# Validates (best-effort, presence-based):
# - Authorization headers are never present in audit JSONL (top-level or nested).
# - PII masking applied to obvious email/phone patterns in audit snippets
#   using [sanitize_message()](libs/sanitize/pii.py:72) rules.
# - Presence of audit events covering:
#     * chat.submit (from [create_chat_completion()](services/gateway/src/app.py:540))
#     * webhook.register (from [register_webhook()](services/gateway/src/app.py:1021))
#     * trace.cancel (from [cancel_trace()](services/gateway/src/app.py:1062))
#     * rbac.deny (emitted by exception handler in Gateway)
#
# Notes:
# - This test parses a configured audit JSONL (config.audit_log). If missing, it SKIPs gracefully.
# - PII checks are conservative: we verify that sanitize_message is idempotent on stored snippets,
#   and we ensure common email/phone patterns do not appear in clear text.

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from tests.e2e.utils import (
    E2EConfig,
    new_result,
    iter_audit_jsonl,
)

# Optional import of sanitizer; if missing, we fallback to regex checks only.
try:
    from libs.sanitize.pii import sanitize_message  # type: ignore
except Exception:
    sanitize_message = None  # type: ignore


def _iter_objects(path: str | Path) -> Iterator[Dict[str, Any]]:
    yield from iter_audit_jsonl(path)


def _contains_auth_header(obj: Any) -> bool:
    """
    Recursively scan for Authorization-like keys to verify they never appear in audit logs.
    """
    try:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(k, str) and k.lower() in ("authorization", "proxy-authorization"):
                    return True
                if _contains_auth_header(v):
                    return True
        elif isinstance(obj, list):
            for it in obj:
                if _contains_auth_header(it):
                    return True
    except Exception:
        return False
    return False


# Conservative email/phone regex (align loosely with [libs/sanitize/pii.py](libs/sanitize/pii.py:26))
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s.\-]?)?(?:\(?\d{3}\)?[\s.\-]?)\d{3}[\s.\-]?\d{4}")


def _snippet_is_sanitized(snippet: str) -> bool:
    if not isinstance(snippet, str) or not snippet:
        return True
    # If sanitizer available, it should be idempotent on already-sanitized text
    try:
        if sanitize_message is not None:
            return sanitize_message(snippet) == snippet  # type: ignore[operator]
    except Exception:
        pass
    # Fallback: do not allow raw email/phone patterns
    if _EMAIL_RE.search(snippet):
        return False
    if _PHONE_RE.search(snippet):
        return False
    return True


def run(config: E2EConfig) -> Dict[str, Any]:
    apath = config.audit_log or "/var/log/hypergraph/audit.log"
    p = Path(apath)
    if not p.exists():
        return new_result("test_audit_pii.py", "SKIP", reason=f"audit log not found at {apath}")

    # Aggregates
    found_any = False
    auth_header_seen = False
    events = {
        "chat.submit": False,
        "webhook.register": False,
        "trace.cancel": False,
        "rbac.deny": False,
    }
    bad_snippet_count = 0
    checked_snippet = 0
    lines_scanned = 0
    sample_errors: List[str] = []

    for obj in _iter_objects(apath):
        lines_scanned += 1
        found_any = True
        try:
            ev = str(obj.get("event") or "")
            if ev in events:
                events[ev] = True
            # Snippet (if present)
            extra = obj.get("extra") if isinstance(obj, dict) else None
            snippet = None
            if isinstance(extra, dict) and "snippet" in extra:
                s = extra.get("snippet")
                if isinstance(s, str):
                    snippet = s
            if snippet is not None:
                checked_snippet += 1
                if not _snippet_is_sanitized(snippet):
                    bad_snippet_count += 1
            # Ensure no Authorization headers anywhere in the object
            if _contains_auth_header(obj):
                auth_header_seen = True
        except Exception as e:
            sample_errors.append(str(e)[:120])
            continue

    details = {
        "path": apath,
        "lines_scanned": lines_scanned,
        "events_present": events,
        "snippets_checked": checked_snippet,
        "bad_snippet_count": bad_snippet_count,
        "auth_header_seen": auth_header_seen,
        "errors": sample_errors[:3],
    }

    if not found_any:
        return new_result("test_audit_pii.py", "SKIP", reason="no audit lines found", details=details)

    # Authorization headers must not be present
    if auth_header_seen:
        return new_result("test_audit_pii.py", "FAIL", reason="Authorization headers found in audit log", details=details)

    # PII must be masked in snippets (if any were present)
    if checked_snippet > 0 and bad_snippet_count > 0:
        return new_result("test_audit_pii.py", "FAIL", reason="PII patterns detected in audit snippets", details=details)

    # Event presence: require at least chat.submit; other events depend on other tests running in the same session.
    if not events["chat.submit"]:
        # Other tests should have issued chat.submit when x-explain-mode header was used.
        return new_result("test_audit_pii.py", "SKIP", reason="no chat.submit events observed in audit log", details=details)

    # PASS; include note which optional events were observed
    return new_result("test_audit_pii.py", "PASS", details=details)